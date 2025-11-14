from __future__ import annotations

import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Optional, Tuple

import math

import cv2
import numpy as np
import torch
import torch.serialization as torch_serialization
from PIL import Image
from ultralytics import YOLO


DEFAULT_POSE_MODEL_PATH = Path("models/yolov8m-pose.pt")
"""Đường dẫn YOLOv8 pose mặc định. Chỉnh hằng số này khi thay đổi weights."""


def _compute_confidence(prob_tensor: Optional[torch.Tensor]) -> float:
    if prob_tensor is None:
        return 0.0
    probs = prob_tensor.detach()
    if probs.numel() == 0:
        return 0.0
    if probs.numel() > 1:
        probs = probs[:-1]
    return float(probs.mean().item())


def _ensure_parseq_on_path(parseq_root: Path) -> None:
    parseq_root = parseq_root.resolve()
    if str(parseq_root) not in sys.path:
        sys.path.append(str(parseq_root))


if not hasattr(torch_serialization, "add_safe_globals"):
    def _noop_add_safe_globals(*args, **kwargs):
        return None
    torch_serialization.add_safe_globals = _noop_add_safe_globals  # type: ignore

try:
    from torch.optim.swa_utils import SWALR  # type: ignore
    torch_serialization.add_safe_globals([SWALR])
except Exception:
    pass


@dataclass
class OCRReading:
    text: str
    confidence: float


@dataclass
class StableOCRDecision:
    text: Optional[str]
    consensus: float
    votes: int
    total: int
    mean_confidence: float
    weighted_votes: float = 0.0
    candidate: Optional[str] = None
    is_confirmed: bool = False
    misses: int = 0


@dataclass
class _TrackConsensusState:
    confirmed_text: Optional[str] = None
    consensus: float = 0.0
    miss_count: int = 0
    last_frame: int = -1
    pending_text: Optional[str] = None
    last_top: Optional[str] = None
    top_repeat: int = 0


@dataclass
class JerseyRecogniser:
    """Thin wrapper around PARSeq inference for jersey number recognition."""

    parseq_root: Path = Path("parseq")
    checkpoint_path: Optional[Path] = None
    device: Optional[str] = None
    pose_model_path: Optional[Path] = None
    pose_conf_threshold: float = 0.5
    pose_imgsz: int = 256
    debug: bool = False

    crop_debug_dir: Optional[Path] = None
    crop_debug_limit: int = 0

    history_window: int = 40
    confidence_threshold: float = 0.6
    vote_min_confidence: float = 0.5
    vote_min_support: int = 1
    vote_high_threshold: float = 0.6
    vote_count_min: int = 3
    vote_count_margin: int = 1
    vote_low_threshold: float = 0.4
    switch_margin: float = 0.15
    vote_weight_gamma: float = 2.0
    vote_time_decay_lambda: float = 0.1
    switch_margin_base: float = 0.05
    switch_margin_bonus: float = 0.05
    switch_margin_discount: float = 0.02
    switch_margin_reid_threshold: float = 0.5
    history_compact_consensus: float = 0.7
    history_compact_min: int = 3
    max_miss_streak: int = 6

    _history: Dict[int, Deque[Tuple[OCRReading, int]]] = field(default_factory=dict)
    _states: Dict[int, _TrackConsensusState] = field(default_factory=dict)
    _frame_index: int = 0
    _crop_debug_count: int = field(default=0, init=False)
    _crop_debug_enabled: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        _ensure_parseq_on_path(self.parseq_root)

        from strhub.data.module import SceneTextDataModule  # type: ignore
        from strhub.models.utils import load_from_checkpoint  # type: ignore

        self.SceneTextDataModule = SceneTextDataModule
        self._load_from_checkpoint = load_from_checkpoint

        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = self._resolve_checkpoint(self.checkpoint_path)

        import torch.serialization
        torch.serialization.add_safe_globals([getattr])
        self.model = self._load_from_checkpoint(str(self.checkpoint_path)).eval().to(self.device)
        self.transform = self.SceneTextDataModule.get_transform(self.model.hparams.img_size)
        print(f"[JerseyOCR] PARSeq model loaded ({self.checkpoint_path}) on device {self.device}.")

        self.pose_model = None
        pose_candidate = self.pose_model_path or DEFAULT_POSE_MODEL_PATH
        resolved_pose = self._resolve_pose_model(pose_candidate)
        if resolved_pose is not None:
            try:
                self.pose_model = YOLO(str(resolved_pose))
                self.pose_model_path = resolved_pose
                print(f"[JerseyOCR] YOLOv8 pose model loaded ({resolved_pose}) on device {self.device}.")
            except Exception as exc:
                print(f"[JerseyOCR] Không thể tải YOLOv8 pose model ({resolved_pose}): {exc}")
        else:
            print(
                f"[JerseyOCR] Khong tim thay file pose model tai {pose_candidate}. "
                "Dieu chinh DEFAULT_POSE_MODEL_PATH trong ocr/jersey_recognizer.py "
                "hoac truyen --jersey-pose-model de chi dinh duong dan khac."
            )

        if self.crop_debug_dir is not None:
            self.enable_crop_debug(self.crop_debug_dir, limit=self.crop_debug_limit or 0, reset=False)

    def enable_crop_debug(self, output_dir: Path, *, limit: int = 0, reset: bool = True) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.crop_debug_dir = output_path
        self.crop_debug_limit = max(0, int(limit))
        if reset:
            self._crop_debug_count = 0
        self._crop_debug_enabled = True
        if self.crop_debug_limit:
            print(
                f"[JerseyOCR] Saving up to {self.crop_debug_limit} OCR crops to {self.crop_debug_dir}."
            )
        else:
            print(f"[JerseyOCR] Saving OCR crops to {self.crop_debug_dir} (no limit).")

    def disable_crop_debug(self) -> None:
        self._crop_debug_enabled = False

    def _save_debug_patch(
        self,
        patch: np.ndarray,
        frame_idx: int,
        source: str,
        bbox: Tuple[int, int, int, int],
    ) -> None:
        if not self._crop_debug_enabled or self.crop_debug_dir is None:
            return
        if self.crop_debug_limit and self._crop_debug_count >= self.crop_debug_limit:
            return
        frame_id = int(frame_idx)
        filename = (
            self.crop_debug_dir
            / (
                f"{frame_id:06d}_{self._crop_debug_count:03d}_{source}"
                f"_{bbox[0]}_{bbox[1]}_{bbox[2]}_{bbox[3]}.jpg"
            )
        )
        cv2.imwrite(str(filename), patch)
        self._crop_debug_count += 1
        if self.crop_debug_limit and self._crop_debug_count >= self.crop_debug_limit:
            print(
                f"[JerseyOCR] Crop debug limit reached ({self.crop_debug_limit}); disabling capture."
            )
            self._crop_debug_enabled = False

    def extract_team_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """Trả về patch áo (ưu tiên pose) phục vụ chia đội/kmeans."""
        patch = self._crop_with_pose(frame, bbox)
        if patch is not None:
            return patch
        return self.crop_jersey_region(frame, bbox)

    def _resolve_checkpoint(self, checkpoint: Optional[Path]) -> Path:
        if checkpoint:
            candidate = (self.parseq_root / checkpoint).resolve() if not checkpoint.is_absolute() else checkpoint
            if not candidate.exists():
                raise FileNotFoundError(f"Cannot find PARSeq checkpoint: {candidate}")
            return candidate

        test_ckpts = sorted(
            self.parseq_root.glob("outputs/**/checkpoints/test.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if test_ckpts:
            return test_ckpts[0]

        last_ckpts = sorted(
            self.parseq_root.glob("outputs/**/checkpoints/last.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if last_ckpts:
            return last_ckpts[0]

        any_ckpt = sorted(
            self.parseq_root.glob("outputs/**/checkpoints/*.ckpt"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if any_ckpt:
            return any_ckpt[0]

        raise FileNotFoundError("No PARSeq checkpoint found under parseq/outputs/**/checkpoints/")

    def _resolve_pose_model(self, pose_path: Optional[Path]) -> Optional[Path]:
        if pose_path is None:
            return None
        path_obj = Path(pose_path)
        candidate = path_obj.resolve()
        if candidate.exists():
            return candidate
        return None

    @staticmethod
    def crop_jersey_region(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w - 1)
        y2 = min(int(y2), h - 1)

        if x2 <= x1 or y2 <= y1:
            return None

        height = y2 - y1
        width = x2 - x1

        # Focus on mid-lower torso where jersey digits usually appear.
        top_margin = int(height * 0.20)
        bottom_margin = int(height * 0.05)
        side_margin = int(width * 0.15)

        y_top = y1 + top_margin
        y_bottom = y2 - bottom_margin
        x_left = x1 + side_margin
        x_right = x2 - side_margin

        # If margins collapse the crop, relax them gradually.
        if x_right <= x_left:
            center_x = (x1 + x2) // 2
            half_width = max(width // 4, 12)
            x_left = max(center_x - half_width, x1)
            x_right = min(center_x + half_width, x2)

        if y_bottom <= y_top:
            center_y = (y1 + y2) // 2
            half_height = max(height // 4, 12)
            y_top = max(center_y - half_height, y1)
            y_bottom = min(center_y + half_height, y2)

        y_top = max(y_top, y1)
        y_bottom = min(y_bottom, y2)
        x_left = max(x_left, x1)
        x_right = min(x_right, x2)

        if x_right <= x_left or y_bottom <= y_top:
            print(f"[JerseyOCR] Invalid crop for bbox={bbox}")
            return None

        patch = frame[y_top:y_bottom, x_left:x_right]
        if patch.size == 0:
            print(f"[JerseyOCR] Crop empty for bbox={bbox}")
            return None
        if patch.shape[0] < 12 or patch.shape[1] < 12:
            pad_y = max(int(height * 0.1), 4)
            pad_x = max(int(width * 0.1), 4)
            y_a = max(y1 - pad_y, 0)
            y_b = min(y2 + pad_y, h)
            x_a = max(x1 - pad_x, 0)
            x_b = min(x2 + pad_x, w)
            fallback = frame[y_a:y_b, x_a:x_b]
            if fallback.size == 0:
                print(f"[JerseyOCR] Fallback crop empty for bbox={bbox}")
                return None
            return fallback
        return patch

    def _crop_with_pose(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        if self.pose_model is None:
            return None

        x1, y1, x2, y2 = bbox
        h, w = frame.shape[:2]
        x1 = max(int(x1), 0)
        y1 = max(int(y1), 0)
        x2 = min(int(x2), w)
        y2 = min(int(y2), h)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return None

        person_crop = frame[y1:y2, x1:x2]
        if person_crop.size == 0:
            return None

        try:
            results = self.pose_model.predict(
                person_crop,
                conf=self.pose_conf_threshold,
                imgsz=self.pose_imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:
            print(f"[JerseyOCR] Lỗi YOLOv8 pose: {exc}")
            return None

        if not results:
            return None

        result = results[0]
        keypoints = getattr(result, "keypoints", None)
        if keypoints is None or keypoints.xy is None or len(keypoints.xy) == 0:
            return None

        if keypoints.xy.shape[0] > 1:
            boxes = getattr(result, "boxes", None)
            if boxes is not None and getattr(boxes, "conf", None) is not None and len(boxes.conf) > 0:
                best_idx = int(boxes.conf.argmax().item())
            else:
                best_idx = 0
        else:
            best_idx = 0

        xy = keypoints.xy[best_idx]
        conf = keypoints.conf[best_idx] if keypoints.conf is not None else None

        if isinstance(xy, torch.Tensor):
            xy = xy.detach().cpu().numpy()
        else:
            xy = np.asarray(xy)

        if conf is None:
            conf_arr = np.ones(xy.shape[0], dtype=np.float32)
        else:
            if isinstance(conf, torch.Tensor):
                conf_arr = conf.detach().cpu().numpy()
            else:
                conf_arr = np.asarray(conf)

        shoulder_indices = [5, 6]
        hip_indices = [11, 12]

        shoulders = [
            xy[idx]
            for idx in shoulder_indices
            if idx < xy.shape[0] and conf_arr[idx] >= self.pose_conf_threshold
        ]
        hips = [
            xy[idx]
            for idx in hip_indices
            if idx < xy.shape[0] and conf_arr[idx] >= self.pose_conf_threshold
        ]

        if not shoulders or not hips:
            return None

        x_values = [point[0] for point in shoulders + hips]
        shoulder_y = [point[1] for point in shoulders]
        hip_y = [point[1] for point in hips]

        min_x = min(x_values)
        max_x = max(x_values)
        center_x = (min_x + max_x) / 2.0
        torso_top = min(shoulder_y)
        torso_bottom = max(hip_y)

        if torso_bottom <= torso_top:
            return None

        torso_height = torso_bottom - torso_top
        torso_width = max_x - min_x
        half_width = max(
            torso_width * 0.6,
            torso_height * 0.35,
            20.0,
        )

        left = int(max(center_x - half_width, 0))
        right = int(min(center_x + half_width, person_crop.shape[1]))

        top = int(max(torso_top - torso_height * 0.15, 0))
        bottom = int(min(torso_bottom + torso_height * 0.4, person_crop.shape[0]))

        if right - left < 4 or bottom - top < 4:
            return None

        global_x1 = int(max(min(x1 + left, w), 0))
        global_x2 = int(max(min(x1 + right, w), 0))
        global_y1 = int(max(min(y1 + top, h), 0))
        global_y2 = int(max(min(y1 + bottom, h), 0))

        # Expand a little in global coordinates to keep jersey edges.
        expand_y = max(int((global_y2 - global_y1) * 0.05), 2)
        expand_x = max(int((global_x2 - global_x1) * 0.05), 2)

        global_x1 = max(global_x1 - expand_x, 0)
        global_x2 = min(global_x2 + expand_x, w)
        global_y1 = max(global_y1 - expand_y, 0)
        global_y2 = min(global_y2 + expand_y, h)

        if global_x2 - global_x1 < 4 or global_y2 - global_y1 < 4:
            return None

        return frame[global_y1:global_y2, global_x1:global_x2]

    def read_number(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        *,
        frame_idx: Optional[int] = None,
    ) -> Optional[OCRReading]:
        if frame_idx is not None:
            self._frame_index = frame_idx
        else:
            self._frame_index += 1
            frame_idx = self._frame_index

        crop_source = "pose"
        patch = self._crop_with_pose(frame, bbox)
        if patch is not None:
            if self.debug:
                print(
                    f"[JerseyOCR] Frame {frame_idx}: using pose crop "
                    f"(shape={patch.shape})."
                )
        else:
            if self.pose_model is not None and self.debug:
                print(
                    f"[JerseyOCR] Frame {frame_idx}: pose crop unavailable "
                    "or low confidence; falling back to bbox crop."
                )
            crop_source = "bbox"
            patch = self.crop_jersey_region(frame, bbox)
            if patch is not None and self.debug:
                print(
                    f"[JerseyOCR] Frame {frame_idx}: using bbox crop "
                    f"(shape={patch.shape})."
                )
        if patch is None:
            if self.debug:
                print(
                    f"[JerseyOCR] Frame {frame_idx}: could not create crop from bbox {bbox}."
                )
            return None

        self._save_debug_patch(patch, frame_idx, crop_source, bbox)

        rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_patch)
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(tensor)
            probs = logits.softmax(-1)
            labels, confidences = self.model.tokenizer.decode(probs)

        if not labels:
            print("[JerseyOCR] Model returned an empty sequence.")
            return None

        text = labels[0]
        if text.strip() == "~":
            print("[JerseyOCR] OCR produced placeholder '~'; treating as empty result.")
            return None
        confidence = _compute_confidence(confidences[0]) if confidences else 0.0
        if confidence < self.confidence_threshold:
            print(
                f"[JerseyOCR] Ignore '{text}' because confidence {confidence:.2f} "
                f"< {self.confidence_threshold}."
            )
            return None

        clean = "".join(ch for ch in text if ch.isdigit())
        if not clean:
            print(f"[JerseyOCR] Parsed text '{text}' does not contain digits.")
            return None

        if not clean.isdigit():
            return None
        value = int(clean)
        if value <= 0 or value > 99:
            print(f"[JerseyOCR] Value '{value}' is outside the jersey range 1-99.")
            return None

        candidate = f"{value:02d}"
        print(f"[JerseyOCR] OCR read '{candidate}' with confidence {confidence:.2f}.")
        return OCRReading(text=candidate, confidence=confidence)

    def confirm_number(
        self,
        track_id: int,
        reading: Optional[OCRReading],
        *,
        frame_idx: Optional[int] = None,
        reid_similarity: Optional[float] = None,
    ) -> Optional[StableOCRDecision]:
        history = self._history.setdefault(track_id, deque(maxlen=self.history_window))
        state = self._states.setdefault(track_id, _TrackConsensusState())

        if frame_idx is not None:
            state.last_frame = frame_idx

        current_frame = state.last_frame if state.last_frame >= 0 else 0

        if reading is None:
            state.miss_count += 1
        else:
            sample_frame = frame_idx if frame_idx is not None else state.last_frame
            if sample_frame is None or sample_frame < 0:
                sample_frame = current_frame
            history.append((reading, sample_frame))
            state.miss_count = 0

        if not history:
            if state.confirmed_text and state.miss_count <= self.max_miss_streak:
                return StableOCRDecision(
                    text=state.confirmed_text,
                    consensus=state.consensus,
                    votes=0,
                    total=0,
                    mean_confidence=0.0,
                    weighted_votes=0.0,
                    candidate=state.pending_text,
                    is_confirmed=True,
                    misses=state.miss_count,
                )
            if state.miss_count > self.max_miss_streak:
                state.confirmed_text = None
                state.consensus = 0.0
            if self.debug:
                print(f"[JerseyOCR] Track {track_id}: empty history.")
            return None

        valid_samples = [
            (sample, sample_frame)
            for sample, sample_frame in history
            if sample.confidence >= self.vote_min_confidence and sample.text and sample.text != "~"
        ]
        if not valid_samples:
            if state.confirmed_text and state.miss_count <= self.max_miss_streak:
                return StableOCRDecision(
                    text=state.confirmed_text,
                    consensus=state.consensus,
                    votes=0,
                    total=len(history),
                    mean_confidence=0.0,
                    weighted_votes=0.0,
                    candidate=state.pending_text,
                    is_confirmed=True,
                    misses=state.miss_count,
                )
            if state.miss_count > self.max_miss_streak:
                state.confirmed_text = None
                state.consensus = 0.0
            if self.debug:
                print(f"[JerseyOCR] Track {track_id}: no valid OCR samples.")
            return None

        weight_map: Dict[str, float] = {}
        vote_map: Dict[str, int] = Counter()
        conf_map: Dict[str, float] = Counter()
        total_weight = 0.0

        for sample, sample_frame in valid_samples:
            age_frames = max(0, current_frame - sample_frame) if current_frame is not None else 0
            conf_weight = sample.confidence ** self.vote_weight_gamma
            time_weight = math.exp(-self.vote_time_decay_lambda * age_frames)
            weight = conf_weight * time_weight

            weight_map[sample.text] = weight_map.get(sample.text, 0.0) + weight
            vote_map[sample.text] += 1
            conf_map[sample.text] += sample.confidence
            total_weight += weight

        if total_weight <= 0:
            return None

        top_text, top_weight = max(
            weight_map.items(),
            key=lambda item: (item[1], vote_map[item[0]]),
        )
        top_votes = vote_map[top_text]
        second_votes = max(
            (count for jersey, count in vote_map.items() if jersey != top_text),
            default=0,
        )
        count_margin = top_votes - second_votes
        majority_ready = top_votes >= self.vote_count_min and count_margin >= self.vote_count_margin
        consensus = top_weight / total_weight if total_weight else 0.0
        mean_conf_candidate = conf_map[top_text] / max(top_votes, 1)

        prev_text = state.confirmed_text
        prev_weight = weight_map.get(prev_text, 0.0) if prev_text else 0.0
        prev_consensus = prev_weight / total_weight if total_weight and prev_text else 0.0

        state.pending_text = top_text

        if state.last_top == top_text:
            state.top_repeat += 1
        else:
            state.last_top = top_text
            state.top_repeat = 1

        if reid_similarity is not None and reid_similarity < self.switch_margin_reid_threshold:
            consensus *= 0.95

        if (
            state.top_repeat >= self.history_compact_min
            and consensus >= self.history_compact_consensus
            and len(weight_map) > 1
        ):
            filtered = [(sample, sample_frame) for sample, sample_frame in history if sample.text == top_text]
            history.clear()
            history.extend(filtered)

        enough_support = top_votes >= self.vote_min_support

        margin = self.switch_margin_base
        if reid_similarity is not None:
            if reid_similarity < self.switch_margin_reid_threshold:
                margin = max(0.0, margin - self.switch_margin_bonus)
            else:
                margin += self.switch_margin_discount

        score_gap = (weight_map[top_text] - prev_weight) / max(total_weight, 1e-6)
        should_switch = True
        if (
            prev_text
            and prev_text != top_text
            and score_gap < margin
            and state.miss_count <= self.max_miss_streak
        ):
            should_switch = False

        candidate_ready = (
            (consensus >= self.vote_high_threshold and enough_support)
            or majority_ready
        )
        if candidate_ready and should_switch:
            confirm_consensus = consensus
            if majority_ready and confirm_consensus < self.vote_high_threshold:
                confirm_consensus = max(confirm_consensus, self.vote_high_threshold)
            state.confirmed_text = top_text
            state.consensus = confirm_consensus
            state.miss_count = 0
            if self.debug:
                print(
                    f"[JerseyOCR] Track {track_id}: confirm '{top_text}' "
                    f"(consensus={confirm_consensus:.2f}, votes={top_votes}, margin={count_margin}, mean_conf={mean_conf_candidate:.2f})."
                )
            return StableOCRDecision(
                text=top_text,
                consensus=confirm_consensus,
                votes=top_votes,
                total=len(history),
                mean_confidence=mean_conf_candidate,
                weighted_votes=top_weight,
                candidate=top_text,
                is_confirmed=True,
                misses=state.miss_count,
            )

        if prev_text:
            maintain = (
                prev_consensus >= self.vote_low_threshold
                or state.miss_count <= self.max_miss_streak
                or (prev_text == top_text and top_votes >= self.vote_count_min)
            )
            if maintain:
                mean_conf_prev = (
                    conf_map[prev_text] / max(vote_map.get(prev_text, 1), 1)
                    if prev_text in conf_map
                    else state.consensus
                )
                state.consensus = max(prev_consensus, state.consensus)
                if self.debug:
                    print(
                        f"[JerseyOCR] Track {track_id}: keep '{prev_text}' "
                        f"(consensus={state.consensus:.2f}, miss={state.miss_count}, gap={score_gap:.3f})."
                    )
                return StableOCRDecision(
                    text=prev_text,
                    consensus=state.consensus,
                    votes=vote_map.get(prev_text, 0),
                    total=len(history),
                    mean_confidence=mean_conf_prev,
                    weighted_votes=weight_map.get(prev_text, 0.0),
                    candidate=top_text,
                    is_confirmed=True,
                    misses=state.miss_count,
                )

        if prev_text and self.debug:
            print(
                f"[JerseyOCR] Track {track_id}: release '{prev_text}' "
                f"(prev_consensus={prev_consensus:.2f}, new='{top_text}' {consensus:.2f}, gap={score_gap:.3f})."
            )
        if prev_text and state.miss_count > self.max_miss_streak and self.debug:
            print(
                f"[JerseyOCR] Track {track_id}: streak {state.miss_count} misses, drop '{prev_text}'."
            )
        state.confirmed_text = None
        state.consensus = 0.0

        return StableOCRDecision(
            text=None,
            consensus=consensus,
            votes=top_votes,
            total=len(history),
            mean_confidence=mean_conf_candidate,
            weighted_votes=top_weight,
            candidate=top_text,
            is_confirmed=False,
            misses=state.miss_count,
        )

    def reset_history(self, track_id: int) -> None:
        self._history.pop(track_id, None)
        self._states.pop(track_id, None)

    def reset_all(self) -> None:
        self._history.clear()
        self._states.clear()

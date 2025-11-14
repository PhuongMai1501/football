#!/usr/bin/env python3
"""
Video inference script that combines a YOLO player detector with the PARSeq OCR model
to read jersey numbers from the central region of each bounding box.

Predictions are stabilised by keeping a rolling window of OCR results per track and
confirming a jersey number only after it appears consistently with sufficient confidence.
"""

from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO

from strhub.data.module import SceneTextDataModule
from strhub.models.utils import load_from_checkpoint


@dataclass
class TrackState:
    track_id: int
    box: np.ndarray
    stable_text: Optional[str] = None
    best_conf: float = 0.0
    missed: int = 0
    last_text: Optional[str] = None
    last_confirmed_text: Optional[str] = None
    stability_misses: int = 0


class JerseyOCRBuffer:
    def __init__(self, window: int = 7):
        self.window = max(1, window)
        self.buffer: dict[int, deque[tuple[str, float]]] = {}

    def update(self, track_id: int, text: str, conf: float) -> None:
        if track_id not in self.buffer:
            self.buffer[track_id] = deque(maxlen=self.window)
        self.buffer[track_id].append((text, conf))

    def get_stable_text(
        self,
        track_id: int,
        *,
        conf_thresh: float,
        min_count: int,
    ) -> Optional[tuple[str, float, float]]:
        if track_id not in self.buffer:
            return None

        samples = [
            (text, conf)
            for text, conf in self.buffer[track_id]
            if conf >= conf_thresh and text not in (None, "", "~")
        ]
        if not samples:
            return None

        text_counts = Counter(text for text, _ in samples)
        most_common = text_counts.most_common(1)
        if not most_common:
            return None

        top_text, freq = most_common[0]
        if freq < min_count:
            return None

        confs = [conf for text, conf in samples if text == top_text]
        if not confs:
            return None

        avg_conf = float(sum(confs) / len(confs))
        max_conf = float(max(confs))
        return top_text, avg_conf, max_conf

    def prune(self, active_ids: set[int]) -> None:
        stale = [track_id for track_id in self.buffer if track_id not in active_ids]
        for track_id in stale:
            self.buffer.pop(track_id, None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect players with YOLO and read jersey numbers with PARSeq."
    )
    parser.add_argument("--video", type=str, required=True, help="Path to input video file.")
    parser.add_argument(
        "--yolo",
        type=str,
        default="model/football_20102025.pt",
        help="YOLO weights path or model name (default: model/football_20102025.pt).",
    )
    parser.add_argument(
        "--pose",
        type=str,
        default="model/yolov8m-pose.pt",
        help="YOLOv8 pose weights path or model name used to refine jersey crops (default: model/yolov8m-pose.pt).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="outputs/parseq/03_11_2025/checkpoints/test.ckpt",
        help="PARSeq checkpoint or alias (default: pretrained=parseq).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device for inference (default: cpu).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Optional path to save the annotated video. Container is inferred from extension.",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.3,
        help="YOLO confidence threshold for detections (default: 0.3).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="YOLO NMS IoU threshold (default: 0.45).",
    )
    parser.add_argument(
        "--pose_conf",
        type=float,
        default=0.25,
        help="Minimum confidence threshold for YOLO pose detections (default: 0.25).",
    )
    parser.add_argument(
        "--pose_keypoint_conf",
        type=float,
        default=0.3,
        help="Minimum confidence required for pose keypoints when extracting the back region (default: 0.3).",
    )
    parser.add_argument(
        "--pose_padding",
        type=int,
        default=8,
        help="Pixel padding applied to each player bounding box before running the pose model (default: 8).",
    )
    parser.add_argument(
        "--track_iou",
        type=float,
        default=0.3,
        help="IoU threshold to associate detections with existing tracks (default: 0.3).",
    )
    parser.add_argument(
        "--min_consecutive",
        type=int,
        default=3,
        help=(
            "Minimum occurrences inside the rolling OCR window before confirming the jersey number "
            "(default: 3)."
        ),
    )
    parser.add_argument(
        "--min_text_conf",
        type=float,
        default=0.5,
        help="Minimum mean token confidence from PARSeq to accept a prediction (default: 0.5).",
    )
    parser.add_argument(
        "--max_missed",
        type=int,
        default=2,
        help="Maximum frames to keep a track without a matching detection (default: 2).",
    )
    parser.add_argument(
        "--crop_width",
        type=float,
        default=0.6,
        help="Relative width (0-1] of the jersey crop within the player bounding box (default: 0.6).",
    )
    parser.add_argument(
        "--crop_height",
        type=float,
        default=0.6,
        help="Relative height (0-1] of the jersey crop within the player bounding box (default: 0.6).",
    )
    parser.add_argument(
        "--ocr_window",
        type=int,
        default=7,
        help="Number of recent OCR predictions retained per track for smoothing (default: 7).",
    )
    parser.add_argument(
        "--ocr_grace",
        type=int,
        default=5,
        help="Frames to keep the last confirmed jersey when predictions temporarily drop (default: 5).",
    )
    parser.add_argument(
        "--digits_only",
        action="store_true",
        help="Keep only numeric characters from OCR predictions.",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="Disable on-screen display even if a GUI is available.",
    )
    parser.add_argument(
        "--frame_stride",
        type=int,
        default=2,
        help="Process every N-th frame to speed up inference (default: 2, i.e. skip every other frame).",
    )
    return parser.parse_args()


def ensure_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA is not available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_arg)


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, (xa2 - xa1)) * max(0.0, (ya2 - ya1))
    area_b = max(0.0, (xb2 - xb1)) * max(0.0, (yb2 - yb1))
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def pad_box(box: np.ndarray, pad: int, frame_shape: tuple[int, int, int]) -> np.ndarray:
    """Pad a bounding box by a fixed pixel amount, clamped to frame bounds."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box.astype(float)
    pad = float(max(0, pad))
    padded = np.array(
        [
            max(0.0, x1 - pad),
            max(0.0, y1 - pad),
            min(float(w), x2 + pad),
            min(float(h), y2 + pad),
        ],
        dtype=float,
    )
    return padded


def extract_back_crop(
    frame: np.ndarray,
    padded_box: np.ndarray,
    pose_model: Optional[YOLO],
    *,
    pose_conf: float,
    keypoint_conf: float,
    device: torch.device,
) -> Optional[np.ndarray]:
    """Use pose keypoints to crop the jersey area within the padded box."""
    if pose_model is None:
        return None

    h, w = frame.shape[:2]
    x1, y1, x2, y2 = padded_box.astype(int)
    x1 = int(np.clip(x1, 0, w))
    y1 = int(np.clip(y1, 0, h))
    x2 = int(np.clip(x2, 0, w))
    y2 = int(np.clip(y2, 0, h))
    if x2 <= x1 or y2 <= y1:
        return None

    pose_crop = frame[y1:y2, x1:x2]
    if pose_crop.size == 0:
        return None

    try:
        pose_results = pose_model(
            pose_crop,
            conf=pose_conf,
            verbose=False,
            device=str(device),
        )
    except Exception:
        return None

    if not pose_results:
        return None

    pose_result = pose_results[0]
    if pose_result.boxes is None or pose_result.keypoints is None:
        return None
    if len(pose_result.boxes) == 0 or pose_result.keypoints.xy is None:
        return None

    boxes_conf = pose_result.boxes.conf
    if boxes_conf is None or boxes_conf.numel() == 0:
        return None
    best_idx = int(torch.argmax(boxes_conf).item())

    keypoints_tensor = pose_result.keypoints.xy[best_idx]
    if keypoints_tensor is None:
        return None

    keypoints = keypoints_tensor.detach().cpu().numpy()
    kp_conf_tensor = pose_result.keypoints.conf
    kp_conf = (
        kp_conf_tensor[best_idx].detach().cpu().numpy()
        if kp_conf_tensor is not None
        else np.ones(len(keypoints), dtype=float)
    )

    def _get_point(idx: int) -> Optional[np.ndarray]:
        if idx >= len(keypoints):
            return None
        point = keypoints[idx]
        if np.isnan(point).any():
            return None
        conf_val = kp_conf[idx] if idx < len(kp_conf) else 1.0
        if conf_val < keypoint_conf:
            return None
        return point

    shoulder_indices = (5, 6)
    hip_indices = (11, 12)
    shoulders = [pt for idx in shoulder_indices if (pt := _get_point(idx)) is not None]
    hips = [pt for idx in hip_indices if (pt := _get_point(idx)) is not None]

    if not shoulders or not hips:
        return None

    left_x = min(pt[0] for pt in shoulders + hips)
    right_x = max(pt[0] for pt in shoulders + hips)
    top_y = min(pt[1] for pt in shoulders)
    bottom_y = max(pt[1] for pt in hips)

    if right_x <= left_x or bottom_y <= top_y:
        return None

    width = right_x - left_x
    height = bottom_y - top_y
    margin_x = max(4.0, 0.1 * width)
    margin_y_top = max(4.0, 0.05 * height)
    margin_y_bottom = max(4.0, 0.15 * height)

    crop_w = pose_crop.shape[1]
    crop_h = pose_crop.shape[0]

    roi_x1 = int(np.clip(left_x - margin_x, 0, crop_w))
    roi_x2 = int(np.clip(right_x + margin_x, 0, crop_w))
    roi_y1 = int(np.clip(top_y - margin_y_top, 0, crop_h))
    roi_y2 = int(np.clip(bottom_y + margin_y_bottom, 0, crop_h))

    if roi_x2 <= roi_x1 or roi_y2 <= roi_y1:
        return None

    global_x1 = x1 + roi_x1
    global_x2 = x1 + roi_x2
    global_y1 = y1 + roi_y1
    global_y2 = y1 + roi_y2

    global_x1 = int(np.clip(global_x1, 0, w))
    global_x2 = int(np.clip(global_x2, 0, w))
    global_y1 = int(np.clip(global_y1, 0, h))
    global_y2 = int(np.clip(global_y2, 0, h))

    if global_x2 <= global_x1 or global_y2 <= global_y1:
        return None

    return frame[global_y1:global_y2, global_x1:global_x2]


def crop_central_region(
    frame: np.ndarray,
    box: np.ndarray,
    width_ratio: float,
    height_ratio: float,
) -> Optional[np.ndarray]:
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)
    crop_w = min(box_w, box_w * width_ratio)
    crop_h = min(box_h, box_h * height_ratio)

    crop_x1 = int(max(0, cx - crop_w * 0.5))
    crop_y1 = int(max(0, cy - crop_h * 0.5))
    crop_x2 = int(min(w, cx + crop_w * 0.5))
    crop_y2 = int(min(h, cy + crop_h * 0.5))

    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None

    return frame[crop_y1:crop_y2, crop_x1:crop_x2]


def normalize_text(text: str, digits_only: bool) -> str:
    cleaned = text.strip()
    if cleaned == "~":
        return ""
    if digits_only:
        cleaned = "".join(ch for ch in cleaned if ch.isdigit())
    else:
        cleaned = cleaned.replace(" ", "")
    return cleaned


def mean_token_conf(prob_tensor: torch.Tensor) -> float:
    if prob_tensor.numel() == 0:
        return 0.0
    probs = prob_tensor.detach().cpu()
    if probs.numel() > 1:
        probs = probs[:-1]  # drop EOS probability
    return float(probs.mean().item())


def load_parseq(model_ref: str, device: torch.device, refine_iters: int = 1, decode_ar: bool = True):
    model_kwargs = {"refine_iters": refine_iters, "decode_ar": decode_ar}
    model = load_from_checkpoint(model_ref, **model_kwargs).eval().to(device)
    transform = SceneTextDataModule.get_transform(model.hparams.img_size)
    return model, transform


def draw_tracks(frame: np.ndarray, tracks: list[TrackState]) -> None:
    for track in tracks:
        if track.missed > 0:
            continue
        color = (0, 255, 0) if track.stable_text is not None else (0, 165, 255)
        x1, y1, x2, y2 = track.box.astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track.stable_text is not None:
            display_text = track.stable_text if track.stable_text else "None"
            label = f"{display_text} ({track.best_conf:.2f})"
            text_pos = (x1, max(0, y1 - 10))
            cv2.putText(
                frame,
                label,
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


def main():
    args = parse_args()
    torch_device = ensure_device(args.device)

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Video path does not exist: {args.video}")

    print("=" * 60)
    print("YOLO + PARSeq Jersey Reader")
    print("=" * 60)

    print(f"[INFO] Loading YOLO model: {args.yolo}")
    yolo_model = YOLO(args.yolo)
    yolo_model.to(str(torch_device))

    pose_model: Optional[YOLO] = None
    if args.pose:
        print(f"[INFO] Loading YOLO pose model: {args.pose}")
        pose_model = YOLO(args.pose)
        pose_model.to(str(torch_device))

    print(f"[INFO] Loading PARSeq model: {args.model}")
    parseq_model, img_transform = load_parseq(args.model, torch_device)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        print(f"[INFO] Writing annotated video to: {output_path}")

    tracks: list[TrackState] = []
    next_track_id = 0
    frame_idx = -1
    ocr_buffer = JerseyOCRBuffer(window=args.ocr_window)
    crop_preview_dir = Path("croped")
    crop_preview_dir.mkdir(parents=True, exist_ok=True)
    for stale_crop in crop_preview_dir.glob("crop_*.png"):
        try:
            stale_crop.unlink()
        except OSError:
            pass
    crop_preview_limit = 500
    saved_crop_count = 0
    crop_limit_notified = False
    if crop_preview_limit > 0:
        print(
            f"[INFO] Saving up to {crop_preview_limit} crop previews to: {crop_preview_dir.resolve()}"
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            if args.frame_stride > 1 and (frame_idx % args.frame_stride) != 0:
                draw_tracks(frame, tracks)
                if writer:
                    writer.write(frame)
                if not args.no_display:
                    cv2.imshow("jersey_reader", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
                continue

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            with torch.inference_mode():
                detections = yolo_model(
                    rgb_frame,
                    conf=args.conf,
                    iou=args.iou,
                    classes=[0],  # person class in COCO
                    verbose=False,
                    device=str(torch_device),
                )[0]

            parsed_detections = []
            crops = []
            crop_meta = []

            if detections.boxes is not None and len(detections.boxes) > 0:
                for box in detections.boxes:
                    xyxy = box.xyxy[0].detach().cpu().numpy()
                    padded_xyxy = pad_box(xyxy, args.pose_padding, frame.shape)
                    crop = extract_back_crop(
                        frame,
                        padded_xyxy,
                        pose_model,
                        pose_conf=args.pose_conf,
                        keypoint_conf=args.pose_keypoint_conf,
                        device=torch_device,
                    )
                    if crop is None:
                        crop = crop_central_region(
                            frame,
                            padded_xyxy,
                            width_ratio=args.crop_width,
                            height_ratio=args.crop_height,
                        )
                    if crop is None or crop.size == 0:
                        continue
                    if saved_crop_count < crop_preview_limit:
                        crop_path = crop_preview_dir / f"crop_{saved_crop_count:03d}.png"
                        if cv2.imwrite(str(crop_path), crop):
                            saved_crop_count += 1
                            if (
                                saved_crop_count == crop_preview_limit
                                and not crop_limit_notified
                            ):
                                print(
                                    f"[INFO] Reached crop preview limit ({crop_preview_limit}); samples available in {crop_preview_dir.resolve()}"
                                )
                                crop_limit_notified = True
                    pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                    tensor = img_transform(pil_img).unsqueeze(0)
                    crops.append(tensor)
                    crop_meta.append(xyxy)

            if crops:
                batch = torch.cat(crops, dim=0).to(torch_device)
                with torch.inference_mode():
                    logits = parseq_model(batch)
                    probs = logits.softmax(-1)
                    texts, seq_probs = parseq_model.tokenizer.decode(probs)

                for xyxy, text, prob_vec in zip(crop_meta, texts, seq_probs):
                    normalized = normalize_text(text, digits_only=args.digits_only)
                    conf_score = mean_token_conf(prob_vec)
                    parsed_detections.append(
                        {
                            "box": xyxy,
                            "text": normalized,
                            "conf": conf_score,
                        }
                    )

            matched_tracks: set[int] = set()

            for det in parsed_detections:
                box = det["box"]
                text = det["text"]
                conf = det["conf"]

                best_track: Optional[TrackState] = None
                best_iou = args.track_iou
                for track in tracks:
                    if track.track_id in matched_tracks:
                        continue
                    overlap = compute_iou(track.box, box)
                    if overlap > best_iou:
                        best_iou = overlap
                        best_track = track

                if best_track is None:
                    assigned_track = TrackState(
                        track_id=next_track_id,
                        box=box,
                        stable_text=None,
                        best_conf=0.0,
                        missed=0,
                    )
                    tracks.append(assigned_track)
                    next_track_id += 1
                else:
                    assigned_track = best_track
                    assigned_track.box = box

                matched_tracks.add(assigned_track.track_id)
                assigned_track.missed = 0
                assigned_track.last_text = text

                if text:
                    ocr_buffer.update(assigned_track.track_id, text, conf)

                stable_result = ocr_buffer.get_stable_text(
                    assigned_track.track_id,
                    conf_thresh=args.min_text_conf,
                    min_count=args.min_consecutive,
                )

                if stable_result:
                    stable_text, avg_conf, _ = stable_result
                    assigned_track.stable_text = stable_text
                    assigned_track.best_conf = avg_conf
                    assigned_track.last_confirmed_text = stable_text
                    assigned_track.stability_misses = 0
                else:
                    if assigned_track.last_confirmed_text:
                        assigned_track.stability_misses += 1
                        if assigned_track.stability_misses <= args.ocr_grace:
                            assigned_track.stable_text = assigned_track.last_confirmed_text
                        else:
                            assigned_track.stable_text = None
                            assigned_track.last_confirmed_text = None
                            assigned_track.best_conf = 0.0
                    else:
                        assigned_track.stability_misses = 0
                        assigned_track.stable_text = None
                        assigned_track.best_conf = 0.0

            for track in tracks:
                if track.track_id not in matched_tracks:
                    track.missed += 1
                    track.stability_misses += 1
                    if track.stability_misses > args.ocr_grace:
                        track.stable_text = None
                        track.last_confirmed_text = None
                        track.best_conf = 0.0

            draw_tracks(frame, tracks)
            tracks = [track for track in tracks if track.missed <= args.max_missed]
            ocr_buffer.prune({track.track_id for track in tracks})

            if writer:
                writer.write(frame)

            if not args.no_display:
                cv2.imshow("jersey_reader", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

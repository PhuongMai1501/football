from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from ultralytics import YOLO


class TorsoCropper:
    """Utility để crop phần thân áo dựa trên YOLOv8 pose."""

    def __init__(
        self,
        pose_model_path: Path,
        *,
        device: str = "cpu",
        conf_threshold: float = 0.5,
        imgsz: int = 256,
    ) -> None:
        pose_path = Path(pose_model_path)
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose model not found: {pose_path}")
        self.pose_model = YOLO(str(pose_path))
        self.device = device
        self.conf_threshold = conf_threshold
        self.imgsz = imgsz

    def crop(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
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
                conf=self.conf_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - log callee
            print(f"[TorsoCropper] YOLOv8 pose error: {exc}")
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
            if idx < xy.shape[0] and conf_arr[idx] >= self.conf_threshold
        ]
        hips = [
            xy[idx]
            for idx in hip_indices
            if idx < xy.shape[0] and conf_arr[idx] >= self.conf_threshold
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
        half_width = max(torso_width * 0.6, torso_height * 0.35, 20.0)

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

        expand_y = max(int((global_y2 - global_y1) * 0.05), 2)
        expand_x = max(int((global_x2 - global_x1) * 0.05), 2)

        global_x1 = max(global_x1 - expand_x, 0)
        global_x2 = min(global_x2 + expand_x, w)
        global_y1 = max(global_y1 - expand_y, 0)
        global_y2 = min(global_y2 + expand_y, h)

        if global_x2 - global_x1 < 4 or global_y2 - global_y1 < 4:
            return None

        return frame[global_y1:global_y2, global_x1:global_x2]

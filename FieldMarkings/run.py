import os
import json
import concurrent.futures as cf
from collections import deque
import cv2
import torch
import numpy as np
import sys

sys.path.append('FieldMarkings')
from tqdm import tqdm

# --- FIXED: compatibility layer for legacy torch-argus ---
try:
    from argus import load_model  # nếu là bản torch-argus cũ
except (ImportError, AttributeError):
    # tự định nghĩa lại load_model nếu không tồn tại
    import torch.nn as nn

    def load_model(model_path, loss=None, optimizer=None, device='cpu'):
        """Fallback loader for Argus models (.pth checkpoints)."""
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        # Nếu checkpoint chứa key 'model_state_dict'
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # Tạo model trống tương ứng với cấu trúc HRNet
            from src.models.hrnet.metamodel import HRNetMetaModel
            model = HRNetMetaModel()
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            model.to(device)
            model.eval()
            return model
        elif isinstance(checkpoint, nn.Module):
            return checkpoint.to(device).eval()
        else:
            raise RuntimeError(f"Unsupported checkpoint format in {model_path}")

from torchvision import transforms as T
from baseline.camera import unproject_image_point
from baseline.baseline_cameras import draw_pitch_homography
from src.datatools.ellipse import PITCH_POINTS
from src.models.hrnet.metamodel import HRNetMetaModel
from src.models.hrnet.prediction import CameraCreator


MODEL_PATH = 'models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth'
LINES_FILE = None # '/workdir/data/result/line_model_result.pkl'
DEVICE = 'cuda:0'  # Device for running inference


class CamCalib:
    def __init__(self, keypoints_path, lines_path, *, keep_last_valid=True,
                 smoothing_window: int = 3, max_cam_position_delta: float = 40.0,
                 max_cam_rotation_delta: float = 0.5, persist_heatmap: bool = False,
                 persistence_decay: float = 0.85, persistence_alpha: float = 0.6,
                 show_labels: bool = True):
        self.IMG_W = 960
        self.IMG_H = 540
        self.f = 6
        self.model = load_model(keypoints_path, loss=None, optimizer=None, device=DEVICE)

        self.calibrator = CameraCreator(
            PITCH_POINTS, conf_thresh=0.5, conf_threshs=[0.5, 0.35, 0.2],
            algorithm='iterative_voter',
            lines_file=lines_path, max_rmse=55.0, max_rmse_rel=5.0,
            min_points=5, min_focal_length=10.0, min_points_per_plane=6,
            min_points_for_refinement=6, reliable_thresh=57
        )

        self.H = None
        self.keep_last_valid = keep_last_valid
        self.smoothing_window = max(1, int(smoothing_window))
        self._H_buffer = deque(maxlen=self.smoothing_window)
        self._last_cam_pose = None
        self.max_cam_position_delta = max_cam_position_delta
        self.max_cam_rotation_delta = max_cam_rotation_delta
        self.show_labels = show_labels
        self.persistence_enabled = persist_heatmap
        self.persistence_decay = float(np.clip(persistence_decay, 0.0, 0.999))
        self.persistence_alpha = float(np.clip(persistence_alpha, 0.0, 1.0))
        self.persistence_layer = None
        self._pitch_polygon = None
        self.last_metadata = []
        self.last_keypoint_count = 0

    def _should_accept_cam(self, cam) -> bool:
        if cam is None:
            return False
        if not self.keep_last_valid:
            return True
        if self._last_cam_pose is None:
            return True
        prev_pos = self._last_cam_pose.get("position")
        prev_rot = self._last_cam_pose.get("rotation")
        try:
            if prev_pos is not None and getattr(cam, "position", None) is not None:
                shift = np.linalg.norm(cam.position - prev_pos)
                if shift > self.max_cam_position_delta:
                    return False
            if prev_rot is not None and getattr(cam, "rotation", None) is not None:
                rot_delta = np.linalg.norm(cam.rotation - prev_rot)
                if rot_delta > self.max_cam_rotation_delta:
                    return False
        except Exception:
            # Nếu không thể so sánh (shape lệch, v.v.) thì chấp nhận để tránh kẹt.
            return True
        return True

    def _record_cam_pose(self, cam):
        self._last_cam_pose = {
            "position": getattr(cam, "position", None),
            "rotation": getattr(cam, "rotation", None)
        }

    def _smooth_h(self):
        if not self._H_buffer:
            return None
        if len(self._H_buffer) == 1:
            return self._H_buffer[0]
        stacked = np.stack(self._H_buffer, axis=0)
        return np.mean(stacked, axis=0)

    def _ensure_pitch_polygon(self):
        if self._pitch_polygon is None:
            margin = 8
            self._pitch_polygon = np.array([
                [margin, margin],
                [self.IMG_W - margin, margin],
                [self.IMG_W - margin, self.IMG_H - margin],
                [margin, self.IMG_H - margin]
            ], dtype=np.float32)

    def __call__(self, img):
        to_tensor = T.ToTensor()
        img = cv2.resize(img, (self.IMG_W, self.IMG_H))
        tensor = to_tensor(img).unsqueeze(0).to(DEVICE)
        pred = self.model.predict(tensor).cpu().numpy()[0]
        keypoint_conf_mask = pred[:, 2] > getattr(self.calibrator, "conf_thresh", 0.5)
        conf_count = int(np.count_nonzero(keypoint_conf_mask))
        self.last_keypoint_count = conf_count
        print(f"[CamCalib] frame keypoints>=thresh: {conf_count}/{pred.shape[0]}")

        cam = self.calibrator(pred)
        
        if not self._should_accept_cam(cam):
            return

        if cam is not None:
            candidate_H = cam.calibration @ cam.rotation @ np.concatenate(
                (np.eye(3)[:, :2], -cam.position.reshape(3, 1)), axis=1)
            det_candidate = float(np.linalg.det(candidate_H))
            trace_candidate = float(np.trace(candidate_H))
            if det_candidate < 1e-6:
                print(f"[CamCalib] Reject H (det too small: {det_candidate:.4e})")
                return
            self._record_cam_pose(cam)
            self._H_buffer.append(candidate_H)
            smoothed = self._smooth_h()
            if smoothed is not None:
                current_det = float(np.linalg.det(smoothed))
                if current_det < 1e-6:
                    print(f"[CamCalib] Smoothed H rejected (det={current_det:.4e})")
                    return
                if self.H is not None:
                    old_det = float(np.linalg.det(self.H))
                    if current_det < old_det * 0.5:
                        print(f"[CamCalib] Smoothed H worse than existing (old det={old_det:.4e}, new={current_det:.4e}), skip")
                        return
                self.H = smoothed
                det_val = float(np.linalg.det(self.H))
                trace_val = float(np.trace(self.H))
                print(f"[CamCalib] det(H)={det_val:.4e}, trace(H)={trace_val:.4e}")

    def reset(self):
        self.H = None
        self._H_buffer.clear()
        self._last_cam_pose = None

    def calibrate_player_feet(self, xyxyn):
        if self.H is None:
            return None

        x1, y1, x2, y2 = xyxyn
        x1 *= self.IMG_W
        y1 *= self.IMG_H
        x2 *= self.IMG_W
        y2 *= self.IMG_H
        point2D = np.array([x1 + (x2 - x1) / 2, y2, 1])

        top_view_h = np.array([[self.f, 0, self.IMG_W/2], [0, self.f, self.IMG_H/2], [0, 0, 1]])
        
        feet = unproject_image_point(self.H, point2D=point2D)
        imaged_feets = top_view_h @ np.array([feet[0], feet[1], 1])
        imaged_feets /= imaged_feets[2]

        return imaged_feets

    def draw(self, img, colors, feets, labels=None, metadata=None, show_labels=None):
        if self.H is None:
            self.last_metadata = []
            return None, []

        black_img = np.zeros((self.IMG_H, self.IMG_W, 3), dtype=np.uint8)
        top_view_h = np.array([[self.f, 0, self.IMG_W/2], [0, self.f, self.IMG_H/2], [0, 0, 1]])
        drawn = draw_pitch_homography(black_img, top_view_h)
        self._ensure_pitch_polygon()

        labels = labels or []
        metadata = metadata or []
        show_labels = self.show_labels if show_labels is None else show_labels

        overlay_layer = np.zeros_like(drawn)
        valid_entries = []
        limit = min(len(colors), len(feets))

        for idx in range(limit):
            feet = feets[idx]
            if feet is None:
                continue

            feet = np.array(feet, dtype=np.float64).flatten()

            if feet.size < 2 or not np.isfinite(feet[0]) or not np.isfinite(feet[1]):
                continue

            x = float(feet[0])
            y = float(feet[1])

            if self._pitch_polygon is not None:
                inside = cv2.pointPolygonTest(self._pitch_polygon, (x, y), False)
                if inside < 0:
                    continue

            color = colors[idx]
            color_arr = np.array(color, dtype=np.float32).flatten()
            bgr = tuple(int(np.clip(c, 0, 255)) for c in color_arr[:3])

            center = (int(round(x)), int(round(y)))
            cv2.circle(overlay_layer, center, 8, bgr, -1)

            label_text = None
            if idx < len(labels):
                label_text = labels[idx]

            if show_labels and label_text not in (None, ""):
                label_text = str(label_text)
                text_size, _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                text_w, text_h = text_size
                text_origin = (center[0] - text_w // 2, center[1] - 10)
                cv2.rectangle(
                    overlay_layer,
                    (text_origin[0] - 2, text_origin[1] - text_h - 2),
                    (text_origin[0] + text_w + 2, text_origin[1] + 2),
                    (0, 0, 0),
                    -1
                )
                cv2.putText(
                    overlay_layer,
                    label_text,
                    text_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

            entry = {
                "x": x,
                "y": y,
                "display_color": list(bgr)
            }

            if idx < len(metadata) and isinstance(metadata[idx], dict):
                for key, value in metadata[idx].items():
                    if value is not None:
                        entry[key] = value

            if label_text not in (None, ""):
                entry["label"] = label_text

            valid_entries.append(entry)

        overlay = overlay_layer
        if self.persistence_enabled:
            if self.persistence_layer is None:
                self.persistence_layer = np.zeros_like(overlay_layer)
            self.persistence_layer = cv2.addWeighted(
                self.persistence_layer, self.persistence_decay,
                overlay_layer, 1.0 - self.persistence_decay, 0
            )
            overlay = cv2.addWeighted(
                overlay_layer, 0.5,
                self.persistence_layer, 0.5, 0
            )

        drawn = cv2.add(drawn, overlay)
        self.last_metadata = valid_entries
        return drawn, valid_entries

def blend(img1, img2, scale=0.5, alpha=0.5):
    img2 = cv2.resize(img2, (int(img2.shape[1]*scale), int(img2.shape[0]*scale)))
    img2_h, img2_w = img2.shape[:2]

    img1_h, img1_w = img1.shape[:2]
    x1, y1 = img1_w//2 - img2_w//2, img1_h - img2_h
    x2, y2 = img1_w//2 + img2_w//2, img1_h
    roi = img1[y1:y2, x1:x2]
    img1[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, img2, alpha, 0)

    return img1

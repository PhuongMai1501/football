import numpy as np
import torch
from ultralytics import YOLO
from boxmot import BoostTrack
from pathlib import Path

class YOLOBoostTrack:
    def __init__(self, model_path, reid_weights="osnet_x0_25_msmt17.pt", device=None):
        self.model = YOLO(model_path)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tracker = BoostTrack(reid_weights=Path(reid_weights), device=self.device, half=False)

    def predict(self, frames, conf=0.25, verbose=False):
        """
        frames: 1 frame (np.array) hoặc list các frame
        return: list kết quả, mỗi phần tử có bbox + id + cls + conf
        """

        # Run YOLO
        results = self.model.predict(frames, conf=conf, verbose=verbose)

        all_outputs = []
        for r in results:
            detections = []
            if r.boxes is not None and len(r.boxes) > 0:
                boxes = r.boxes.xyxy.cpu().numpy()      # [x1, y1, x2, y2]
                scores = r.boxes.conf.cpu().numpy()     # conf
                labels = r.boxes.cls.cpu().numpy()      # cls
                detections = np.concatenate([boxes, scores[:, None], labels[:, None]], axis=1)

            # Update tracker (tracking từng frame)
            outputs = self.tracker.update(detections, r.orig_img)
            # Format: [x1, y1, x2, y2, id, conf, cls, ind]

            all_outputs.append(outputs)

        return all_outputs

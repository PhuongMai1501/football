import cv2
import numpy as np
from utils.bbox_utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:

    def __init__(self, first_frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15,15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,10,0.03)
        )

        first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        h, w = first_gray.shape[:2]

        # chỉ lấy đặc trưng ở 2 mép ảnh 
        mask = np.zeros_like(first_gray, dtype=np.uint8)
        m = max(20, int(0.1 * w))
        mask[:, :m] = 1
        mask[:, w-m:] = 1

        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask
        )

        self.prev_gray = first_gray
        self.prev_pts = cv2.goodFeaturesToTrack(first_gray, **self.feature_params)

    def update(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, **self.feature_params)

        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_gray = gray
            return 0.0, 0.0

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray, gray, self.prev_pts, None, **self.lk_params
        )

        dx = dy = 0.0
        max_distance = 0.0
        if new_pts is not None and status is not None:
            for new, old in zip(new_pts, self.prev_pts):
                new_p, old_p = new.ravel(), old.ravel()
                dist = measure_distance(new_p, old_p)
                if dist > max_distance:
                    max_distance = dist
                    dx, dy = measure_xy_distance(old_p, new_p)

        if max_distance > self.minimum_distance:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self.feature_params)
        self.prev_gray = gray
        return dx, dy

    @staticmethod
    def draw_overlay(frame, dx, dy):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, f"Camera Movement X: {dx:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Camera Movement Y: {dy:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        return frame

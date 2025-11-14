# team_assigner/team_assigner.py
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from sklearn.cluster import KMeans

ColorSample = Tuple[str, np.ndarray]


class TeamAssigner:
    """Phân chia đội dựa trên màu áo đã được OCR crop sẵn.

    Thay vì trích xuất màu cho từng frame, module chỉ cần các mẫu màu
    (jersey_number, color_vector). Các mẫu này được gom định kỳ (VD: mỗi 300 frame)
    để fit KMeans k=2. Sau đó ta giữ lại mapping `jersey -> team_id` và
    `team_colors[team_id]` nhằm tô màu cho overlay.
    """

    MAX_HISTORY = 10

    def __init__(self):
        self.team_colors: Dict[int, Tuple[float, float, float]] = {}
        self.entity_to_team: Dict[str, int] = {}
        self.kmeans: Optional[KMeans] = None
        self._color_history: Dict[str, List[np.ndarray]] = defaultdict(list)
        self.last_fit_frame: int = -9999

    @staticmethod
    def _color_from_patch(patch: np.ndarray) -> Optional[np.ndarray]:
        if patch is None or patch.size == 0 or patch.ndim != 3 or patch.shape[2] != 3:
            return None
        resized = cv2.resize(patch, (30, 30), interpolation=cv2.INTER_AREA)
        blurred = cv2.GaussianBlur(resized, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))
        if cv2.countNonZero(mask) < 10:
            return None
        mean_bgr = cv2.mean(blurred, mask=mask)[:3]
        mean_vec = np.array(mean_bgr, dtype=np.float32)
        lab_patch = cv2.cvtColor(
            np.clip(mean_vec, 0, 255).reshape(1, 1, 3).astype(np.uint8),
            cv2.COLOR_BGR2LAB,
        )
        return lab_patch.reshape(3).astype(np.float32)

    def add_samples(self, samples: Iterable[ColorSample]) -> None:
        for jersey, color in samples:
            if jersey and color is not None:
                history = self._color_history[jersey]
                history.append(color)
                if len(history) > self.MAX_HISTORY:
                    del history[0]

    def _prepare_training_matrix(self) -> Tuple[np.ndarray, List[str]]:
        jersey_keys = []
        color_vectors = []
        for jersey, vectors in self._color_history.items():
            if not vectors:
                continue
            stacked = np.stack(vectors, axis=0)
            mean_color = stacked.mean(axis=0)
            jersey_keys.append(jersey)
            color_vectors.append(mean_color)
        if not color_vectors:
            return np.empty((0, 3)), []
        return np.stack(color_vectors, axis=0), jersey_keys

    def fit(self) -> bool:
        X, jersey_keys = self._prepare_training_matrix()
        if X.shape[0] < 2:
            self.kmeans = None
            self.team_colors.clear()
            self.entity_to_team.clear()
            return False
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init="auto", random_state=42)
        kmeans.fit(X)

        self.kmeans = kmeans
        self.team_colors = {}
        for idx, center in enumerate(kmeans.cluster_centers_):
            lab_center = np.clip(center.reshape(1, 1, 3), 0, 255).astype(np.uint8)
            bgr = cv2.cvtColor(lab_center, cv2.COLOR_Lab2BGR).reshape(3)
            self.team_colors[idx + 1] = tuple(map(float, bgr))
        self.entity_to_team = {}
        for jersey, vec in zip(jersey_keys, X):
            team_idx = int(kmeans.predict(vec.reshape(1, -1))[0]) + 1
            self.entity_to_team[jersey] = team_idx
        return True

    def assign_team_by_key(self, key: Optional[str]) -> Optional[int]:
        if key is None:
            return None
        return self.entity_to_team.get(key)

    def get_team_color(self, team_id: Optional[int]) -> Tuple[int, int, int]:
        if team_id is None:
            return (0, 0, 255)
        color = self.team_colors.get(team_id, (0.0, 0.0, 255.0))
        return tuple(int(np.clip(c, 0, 255)) for c in color)

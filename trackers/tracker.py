from ultralytics import YOLO
import supervision as sv
import pickle
import os
import sys
import cv2
from types import SimpleNamespace
import numpy as np
import pandas as pd
import logging
from pathlib import Path
import torch

sys.path.append('../')
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

# Import BoT-SORT
sys.path.append("BoT-SORT")
from tracker.bot_sort import BoTSORT


class Tracker:
    def __init__(self, model_path, use_boost=False, device=None,
                 reid_config=None, reid_weights=None, reid_backend="torchreid",
                 reid_batch=16, half=None):
        self.logger = logging.getLogger("Tracker")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        self.device = self._resolve_device(device)
        self.using_cuda = self.device.startswith('cuda') and torch.cuda.is_available()
        self.use_fp16 = half if half is not None else self.using_cuda

        self.logger.info("Initializing tracker on device=%s (cuda_available=%s)", self.device, torch.cuda.is_available())

        self.model = YOLO(model_path)
        self.model.to(self.device)
        self._predict_kwargs = {"device": self.device}
        if self.use_fp16:
            self._predict_kwargs["half"] = True

        self.use_boost = use_boost
        if self.use_boost:
            self.max_player_ids = 22
            self.free_player_ids = list(range(1, self.max_player_ids + 1))
        else:
            self.max_player_ids = None
            self.free_player_ids = []
        self.player_id_map = {}
        self.display_to_track = {}
        self.jersey_to_display = {}
        self.display_to_jersey = {}
        self.track_to_jersey = {}
        self.reserved_display_ids = set()

        self.reid_backend = (reid_backend or "torchreid").lower()
        self.reid_config = Path(reid_config).expanduser().resolve() if reid_config else None
        self.reid_weights = Path(reid_weights).expanduser().resolve() if reid_weights else None
        self.reid_batch = reid_batch
        self.reid_enabled = False

        if self.reid_backend not in {"fastreid", "torchreid"}:
            self.logger.warning("Unknown ReID backend '%s'. ReID disabled.", self.reid_backend)
            self.reid_backend = None
        else:
            if self.reid_backend == "fastreid":
                if self.reid_config and self.reid_weights:
                    if self.reid_config.exists() and self.reid_weights.exists():
                        self.reid_enabled = True
                        self.logger.info("BoT-SORT ReID (FastReID) enabled: config=%s | weights=%s",
                                         self.reid_config, self.reid_weights)
                    else:
                        self.logger.warning("FastReID files missing (config=%s, weights=%s). ReID disabled.",
                                            self.reid_config, self.reid_weights)
                else:
                    self.logger.warning("FastReID backend requires both config and weights.")
            elif self.reid_backend == "torchreid":
                if self.reid_weights and self.reid_weights.exists():
                    self.reid_enabled = True
                    self.logger.info("BoT-SORT ReID (TorchreID) enabled: weights=%s", self.reid_weights)
                else:
                    self.logger.warning("TorchreID weight file not found (%s). ReID disabled.", self.reid_weights)

        if self.use_boost:
            fast_cfg = str(self.reid_config) if self.reid_backend == "fastreid" and self.reid_config else None
            fast_weights = str(self.reid_weights) if self.reid_backend == "fastreid" and self.reid_weights else None
            torchreid_weights = str(self.reid_weights) if self.reid_backend == "torchreid" and self.reid_weights else None

            args = SimpleNamespace(
                track_high_thresh=0.55, # giảm 
                track_low_thresh=0.1, # increase in b5
                new_track_thresh=0.8, # tăng
                track_buffer=240, # -> giảm ID switch = tăng track_buffer + giảm match_thresh 
                match_thresh=0.7,
                proximity_thresh=0.5, # khi có reID
                appearance_thresh=0.25, # reID
                with_reid=self.reid_enabled,
                reid_backend=self.reid_backend or "none",
                fast_reid_config=fast_cfg,
                fast_reid_weights=fast_weights,
                torchreid_weights=torchreid_weights,
                reid_batch=self.reid_batch,
                device=self.device,
                cmc_method="orb",
                name="exp",
                ablation=False,
                mot20=True   # True nếu có nhiều đối tượng chen lấn
            )
            if self.reid_enabled and not self.using_cuda:
                self.logger.warning("ReID is enabled but CUDA is unavailable. ReID will fall back to CPU.")
            self.tracker = BoTSORT(args, frame_rate=30)
        else:
            self.tracker = sv.ByteTrack()
            if self.reid_enabled:
                self.logger.info("ReID settings ignored because use_boost=False (ByteTrack).")


    """
    Pending
    """
    # def interpolate_ball_positions(self, ball_positions):
    #     import pandas as pd
    #     ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
    #     df_ball_positions = pd.DataFrame(ball_positions, columns=['x1','y1','x2','y2'])
    #     df_ball_positions = df_ball_positions.interpolate()
    #     df_ball_positions = df_ball_positions.bfill()
    #     ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
    #     return ball_positions

   

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(
                    frames[i:i + batch_size],
                    imgsz=960,     
                    conf=0.3,    
                    iou=0.45,
                    **self._predict_kwargs
                )
            detections += detections_batch
        return detections


    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "refs": [], "ball": []}

        for frame_num, (frame, detection) in enumerate(zip(frames, detections)):

            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # Convert Goalkeeper → Player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

                # Update
            if self.use_boost:
                det_input = []
                det_cls_map = {}
                for idx, (xyxy, score, cls_id) in enumerate(zip(
                    detection_supervision.xyxy,
                    detection_supervision.confidence,
                    detection_supervision.class_id
                )):
                    x1, y1, x2, y2 = xyxy.tolist()
                    det_input.append([x1, y1, x2, y2, float(score), int(cls_id)])
                    det_cls_map[idx] = int(cls_id)   # lưu cls gốc theo thứ tự

                det_input = np.array(det_input) if len(det_input) > 0 else np.empty((0, 6))
                tracks_active = self.tracker.update(det_input, frame)

                detection_with_tracks = []
                for track in tracks_active:
                    if not track.is_activated:
                        continue

                    tlwh = track.tlwh
                    x1, y1, w, h = tlwh
                    x2, y2 = x1 + w, y1 + h
                    bbox = [int(x1), int(y1), int(x2), int(y2)]
                    track_id = track.track_id
                    
                    # Lấy cls_id từ det_input (BoT-SORT không lưu class -> tự map)

                    cls_id = track.cls if hasattr(track, "cls") and track.cls is not None else cls_names_inv["player"]

                    detection_with_tracks.append((bbox, cls_id, track_id))
        
            else:
                # ByteTrack
                detection_with_tracks = []
                for det in self.tracker.update_with_detections(detection_supervision):
                    bbox = det[0]
                    cls_id = det[3]
                    track_id = det[4]
                    detection_with_tracks.append((bbox, cls_id, track_id))

            tracks["players"].append({})
            tracks["refs"].append({})
            tracks["ball"].append({})

            for bbox, cls_id, track_id in detection_with_tracks:
                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv['ref']:
                    tracks["refs"][frame_num][track_id] = {"bbox": bbox}

            # bóng thì lấy trực tiếp từ detection
            for bbox, cls_id in zip(detection_supervision.xyxy, detection_supervision.class_id):
                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox.tolist()}

            if self.use_boost and self.max_player_ids is not None:
                players_frame = tracks["players"][frame_num]
                current_track_ids = set(players_frame.keys())

                for tid in list(self.player_id_map.keys()):
                    if tid not in current_track_ids:
                        old_display = self.player_id_map.pop(tid)
                        self.display_to_track.pop(old_display, None)
                        self.track_to_jersey.pop(tid, None)
                        if old_display not in self.display_to_jersey:
                            self.free_player_ids.append(old_display)
                if self.free_player_ids:
                    self.free_player_ids = sorted(set(self.free_player_ids))

                remapped_players = {}
                for tid in sorted(players_frame.keys()):
                    if tid not in self.player_id_map:
                        if not self.free_player_ids:
                            continue
                        assigned = self.free_player_ids.pop(0)
                        self.player_id_map[tid] = assigned
                    assigned_id = self.player_id_map[tid]
                    self.display_to_track[assigned_id] = tid
                    info = players_frame[tid]
                    info["source_track_id"] = tid
                    remapped_players[assigned_id] = info

                tracks["players"][frame_num] = remapped_players


        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

    def sync_display_assignments(self, jersey_display_map, player_id_map):
        if not self.use_boost or self.max_player_ids is None:
            return
        self.jersey_to_display = dict(jersey_display_map)
        self.display_to_jersey = {display_id: jersey for jersey, display_id in jersey_display_map.items()}
        self.reserved_display_ids = set(self.display_to_jersey.keys())
        self.player_id_map = dict(player_id_map)
        self.display_to_track = {display_id: track_id for track_id, display_id in player_id_map.items()}
        self.track_to_jersey = {}
        for track_id, display_id in player_id_map.items():
            jersey = self.display_to_jersey.get(display_id)
            if jersey:
                self.track_to_jersey[track_id] = jersey
        all_ids = set(range(1, self.max_player_ids + 1))
        occupied = set(self.display_to_track.keys()) | self.reserved_display_ids
        self.free_player_ids = sorted(list(all_ids - occupied))

    def get_display_to_jersey(self):
        return dict(self.display_to_jersey)

    def get_display_to_track(self):
        return dict(self.display_to_track)

    def _resolve_device(self, device):
        if device is None:
            return 'cuda:0' if torch.cuda.is_available() else 'cpu'

        if isinstance(device, torch.device):
            if device.type == 'cuda' and device.index is not None:
                return f'cuda:{device.index}'
            return device.type

        if isinstance(device, int):
            return f'cuda:{device}' if device >= 0 else 'cpu'

        device_str = str(device).lower()
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but torch.cuda.is_available() is False.")
        return device_str
    

    
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center, y2),  
            axes=(int(0.8 * width), int(0.3 * width)),  
            angle=0.0,
            startAngle=-45,
            endAngle=235,  
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        # Hiển thị track_id (nếu có)
        if track_id is not None:
            rectangle_width = 30
            rectangle_height = 16
            x1_rect = x_center - rectangle_width // 2
            x2_rect = x_center + rectangle_width // 2
            y1_rect = y2 + 10
            y2_rect = y1_rect + rectangle_height

            cv2.rectangle(
                frame,
                (int(x1_rect), int(y1_rect)),
                (int(x2_rect), int(y2_rect)),
                color,
                cv2.FILLED
            )

            brightness = sum(color) / 3
            text_color = (0, 0, 0) if brightness > 127 else (255, 255, 255)

            cv2.putText(
                frame,
                f"{track_id}",
                (x1_rect + 8, y2_rect - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                text_color,
                2
            )

        return frame

    # Draw
    def draw_bbox_with_id(self, frame, bbox, color, track_id=None):
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if track_id is not None:
            cv2.rectangle(frame, (x1, y1 - 20), (x1 + 40, y1), color, cv2.FILLED)
            cv2.putText(frame, f"{track_id}", (x1 + 5, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        return frame



    def draw_annotations_frame(self, frame, tracks_frame):

        out = frame.copy()
        player_dict = tracks_frame.get("players", {}) or {}
        ref_dict    = tracks_frame.get("refs", {}) or {}
        ball_dict   = tracks_frame.get("ball", {}) or {}

        # Players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            out = self.draw_ellipse(out, player["bbox"], color, track_id)
            jersey_text = player.get("jersey_number")
            if jersey_text:
                x1, y1, _, _ = player["bbox"]
                label = f"#{jersey_text}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                thickness = 2
                text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
                text_w, text_h = text_size
                text_x = int(x1)
                text_y = max(text_h + 4, int(y1) - 6)
                cv2.rectangle(
                    out,
                    (text_x - 4, text_y - text_h - 6),
                    (text_x + text_w + 4, text_y + 4),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    out,
                    label,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                    cv2.LINE_AA,
                )

        # Refs
        for _, ref in ref_dict.items():
            out = self.draw_ellipse(out, ref["bbox"], (0, 255, 255))

        # Ball
        for _, ball in ball_dict.items():
            out = self.draw_bbox_with_id(out, ball["bbox"], (0, 255, 0))

        return out



from pathlib import Path
import math
from collections import deque
from typing import Deque, Dict, Set, Tuple
import torch
import cv2
import numpy as np

from trackers import Tracker
from team_assigner.team_assigner import TeamAssigner
from team_assigner.torso_cropper import TorsoCropper
from utils.video_utils import open_video, iter_frames, create_writer, write_frame, close_video, close_writer
from utils.heatmap_logger import HeatmapMetadataLogger
from gsr_adapter import GameStateAdapter
from camera_movement_estimator import CameraMovementEstimator

from FieldMarkings.run import CamCalib, MODEL_PATH, LINES_FILE, blend

from BallAction.Test_Visual import BallActionSpot
from ocr.jersey_recognizer import JerseyRecogniser

REID_WEIGHTS_PATH = Path("BoT-SORT/reid/log/osnet_x1_0_market1501_softmax/model/model.pth.tar-30")
PARSEQ_CKPT_PATH = Path("outputs/parseq/03_11_2025/checkpoints/test.ckpt")
POSE_MODEL_PATH = Path("models/yolov8m-pose.pt")
OCR_FRAME_STRIDE = 2
HEATMAP_METADATA_PATH = Path("output_videos/heatmap_points.jsonl")
JERSEY_CACHE_TTL_FRAMES = 150
JERSEY_CACHE_MIN_CONFIDENCE = 0.58
TEAM_REFIT_INTERVAL = 300
CROP_DEBUG_ROOT = Path("output_videos/croped_output")
CAM_RESET_MOV_THRESHOLD = 80.0
CAM_RESET_COOLDOWN = 30
CAM_PAN_SETTLE_THRESHOLD = 5.0
CAM_RESET_SPEED_THRESHOLD = 90.0
CAM_SMALL_PAN_STEP = 5.0
CAM_SMALL_PAN_ACCUM_THRESHOLD = 120.0
KEYPOINT_MIN_VALID = 20
KEYPOINT_LOW_STREAK = 4
CAM_LOG_PATH = Path("output_videos/camera_movement.log")


def log_cuda_state(tracker: Tracker) -> None:
    available = torch.cuda.is_available()
    print(f"[CUDA] torch.cuda.is_available(): {available}")
    print(f"[CUDA] Requested tracker device: {getattr(tracker, 'device', 'unknown')}")
    if available:
        device_count = torch.cuda.device_count()
        current_device = torch.cuda.current_device()
        device_name = torch.cuda.get_device_name(current_device)
        print(f"[CUDA] torch.cuda.device_count(): {device_count}")
        print(f"[CUDA] torch.cuda.current_device(): {current_device}")
        print(f"[CUDA] torch.cuda.get_device_name(): {device_name}")
    else:
        print("[CUDA] CUDA not available. Check drivers/PyTorch install.")
    model_device = getattr(getattr(tracker, "model", None), "device", None)
    if model_device is None:
        model_device = getattr(getattr(tracker.model, "model", None), "device", None)
    print(f"[CUDA] Ultralytics model device: {model_device}")

def main():
    in_path  = 'input_videos/video_cut_3.mp4'
    out_path = 'output_videos/video_promote_heatmap_last.avi'

    cap, fps, w, h = open_video(in_path)
    tracker_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    tracker = Tracker(
        'models/best_ylv8_ep50.pt',
        use_boost=True,
        device=tracker_device,
        reid_weights=str(REID_WEIGHTS_PATH),
        reid_backend="torchreid",
        reid_batch=16
    )
    log_cuda_state(tracker)
    team_assigner = TeamAssigner()
    torso_cropper = None
    try:
        torso_cropper = TorsoCropper(
            pose_model_path=POSE_MODEL_PATH,
            device=tracker_device,
            conf_threshold=0.45,
            imgsz=256,
        )
    except Exception:
        pass
    gsr = GameStateAdapter(out_jsonl="output_videos/game_state.jsonl")
    jersey_ocr = None
    try:
        jersey_ocr = JerseyRecogniser(
            parseq_root=Path("parseq"),
            checkpoint_path=PARSEQ_CKPT_PATH,
            pose_model_path=POSE_MODEL_PATH,
            device='cuda' if tracker_device.startswith('cuda') and torch.cuda.is_available() else 'cpu',
            history_window=30,
            confidence_threshold=0.6,
            vote_min_confidence=0.55,
            vote_min_support=2,
            vote_high_threshold=0.65,
            vote_count_min=4,
            vote_count_margin=2,
        )
    except Exception as exc:
        print(f"[JerseyOCR] Không thể khởi tạo OCR: {exc}")
        return

    jersey_display_map: Dict[str, int] = {}
    active_jersey_ids: Set[int] = set()
    jersey_position_history: Dict[str, Deque[Tuple[int, float, float]]] = {}
    track_jersey_cache: Dict[int, Dict[str, float]] = {}
    jersey_track_cache: Dict[str, Dict[str, int]] = {}

    writer = create_writer(out_path, fps, w, h)

    cm_est = None
    frame_idx = 0
    cam_calib = CamCalib(MODEL_PATH, LINES_FILE, persist_heatmap=True, smoothing_window=5)
    ball_action = BallActionSpot()
    prev_field = None
    heatmap_logger = HeatmapMetadataLogger(HEATMAP_METADATA_PATH)
    CROP_DEBUG_ROOT.mkdir(parents=True, exist_ok=True)
    CAM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    cam_log_file = CAM_LOG_PATH.open("w", encoding="utf-8")
    cam_log_file.write("frame,dx,dy\n")
    last_h_reset_frame = -9999
    calib_low_conf_streak = 0
    pan_active = False
    cumulative_small_pan = 0.0

    for frame in iter_frames(cap):
        # Track 1 frame
        tracks_one = tracker.get_object_tracks([frame], read_from_stub=False, stub_path=None)
        cur_tracks = {k: v[0] for k, v in tracks_one.items()}

        # Team info sẽ được gán sau khi OCR cập nhật jersey

        if jersey_ocr is not None and cur_tracks['players']:
            player_entries = list(cur_tracks['players'].items())
            jersey_candidates: Dict[str, list] = {}
            pending_players = []
            for display_id, info in player_entries:
                bbox = info.get('bbox')
                source_tid = info.get('source_track_id')
                decision = None
                center = None
                if bbox:
                    x1, y1, x2, y2 = map(int, bbox)
                    center = ((x1 + x2) * 0.5, (y1 + y2) * 0.5)
                    attempt = (OCR_FRAME_STRIDE <= 1) or (frame_idx % OCR_FRAME_STRIDE == 0) or ('jersey_number' not in info)
                    if attempt:
                        reading = jersey_ocr.read_number(frame, (x1, y1, x2, y2), frame_idx=frame_idx)
                        decision = jersey_ocr.confirm_number(display_id, reading, frame_idx=frame_idx)
                    else:
                        decision = jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                else:
                    decision = jersey_ocr.confirm_number(display_id, None, frame_idx=frame_idx)
                info.pop('jersey_number', None)
                info.pop('jersey_confidence', None)

                resolved = None
                if (
                    decision
                    and decision.is_confirmed
                    and decision.text
                    and (decision.mean_confidence or 0.0) >= JERSEY_CACHE_MIN_CONFIDENCE
                ):
                    resolved = {
                        "jersey": decision.text,
                        "confidence": decision.mean_confidence,
                        "consensus": decision.consensus,
                        "votes": decision.votes,
                        "source": "ocr",
                    }
                elif source_tid is not None:
                    cached = track_jersey_cache.get(source_tid)
                    if cached:
                        age = frame_idx - cached.get("frame", -1)
                        jersey = cached.get("jersey")
                        owner = jersey_track_cache.get(jersey) if jersey else None
                        owner_active = (
                            owner
                            and owner.get("track_id") is not None
                            and owner["track_id"] != source_tid
                            and frame_idx - owner.get("frame", -1) <= JERSEY_CACHE_TTL_FRAMES
                        )
                        if (
                            jersey
                            and age <= JERSEY_CACHE_TTL_FRAMES
                            and not owner_active
                        ):
                            resolved = {
                                "jersey": jersey,
                                "confidence": cached.get("confidence", 0.5),
                                "consensus": cached.get("consensus", 0.0),
                                "votes": cached.get("votes", 0),
                                "source": "cache",
                            }

                if resolved:
                    jersey = resolved["jersey"]
                    jersey_candidates.setdefault(jersey, []).append({
                        "info": info,
                        "confidence": resolved["confidence"],
                        "consensus": resolved.get("consensus", 0.0),
                        "votes": resolved.get("votes", 0),
                        "center": center,
                        "source_tid": source_tid,
                        "original_id": display_id,
                        "jersey": jersey,
                    })
                    if source_tid is not None:
                        track_jersey_cache[source_tid] = {
                            "jersey": jersey,
                            "frame": frame_idx,
                            "confidence": resolved["confidence"],
                            "consensus": resolved.get("consensus", 0.0),
                            "votes": resolved.get("votes", 0),
                        }
                        jersey_track_cache[jersey] = {
                            "track_id": source_tid,
                            "frame": frame_idx,
                        }
                else:
                    pending_players.append((info, source_tid, display_id))

            selected_candidates = []
            for jersey, candidates in jersey_candidates.items():
                history = jersey_position_history.get(jersey)
                expected_pos = None
                loyalty_display = jersey_display_map.get(jersey)
                if history:
                    last_frame, last_cx, last_cy = history[-1]
                    if frame_idx - last_frame <= 20:
                        expected_pos = (last_cx, last_cy)
                        if len(history) >= 2:
                            prev_frame, prev_cx, prev_cy = history[-2]
                            frame_delta = max(1, last_frame - prev_frame)
                            vx = (last_cx - prev_cx) / frame_delta
                            vy = (last_cy - prev_cy) / frame_delta
                            delta_frames = max(0, frame_idx - last_frame)
                            expected_pos = (last_cx + vx * delta_frames, last_cy + vy * delta_frames)

                def candidate_key(candidate):
                    center = candidate['center']
                    loyalty_rank = 0 if loyalty_display == candidate['original_id'] else 1
                    if center is None:
                        distance = 1e9
                    else:
                        if expected_pos is not None:
                            distance = math.hypot(center[0] - expected_pos[0], center[1] - expected_pos[1])
                        elif history:
                            _, last_cx, last_cy = history[-1]
                            distance = math.hypot(center[0] - last_cx, center[1] - last_cy)
                        else:
                            distance = 1e9
                    return (
                        loyalty_rank,
                        distance,
                        -candidate['consensus'],
                        -candidate['votes'],
                        -candidate['confidence'],
                    )

                best_candidate = min(candidates, key=candidate_key)
                best_info = best_candidate['info']
                best_info['jersey_number'] = best_candidate['jersey']
                best_info['jersey_confidence'] = best_candidate['confidence']
                selected_candidates.append(best_candidate)
                for other in candidates:
                    if other is best_candidate:
                        continue
                    pending_players.append((other['info'], other['source_tid'], other['original_id']))

            jersey_confirmations = [
                (cand['jersey'], cand['info'], cand['source_tid'], cand['original_id'])
                for cand in selected_candidates
            ]

            assigned_players = {}
            new_player_id_map = {}
            used_ids: Set[int] = set()
            if tracker.max_player_ids:
                all_ids = list(range(1, tracker.max_player_ids + 1))
            else:
                all_ids = sorted({pid for pid, _ in player_entries})
            reserved_ids = set(jersey_display_map.values())
            display_to_jersey_local = {display_id: jersey for jersey, display_id in jersey_display_map.items()}

            for jersey, info, source_tid, original_id in jersey_confirmations:
                if jersey not in jersey_display_map:
                    available_for_new = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                    if not available_for_new:
                        continue
                    assigned_display = available_for_new[0]
                    jersey_display_map[jersey] = assigned_display
                    reserved_ids.add(assigned_display)
                else:
                    assigned_display = jersey_display_map[jersey]
                if assigned_display in assigned_players:
                    alternatives = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
                    if alternatives:
                        assigned_display = alternatives[0]
                        jersey_display_map[jersey] = assigned_display
                        reserved_ids.add(assigned_display)
                    else:
                        continue
                previous_jersey = display_to_jersey_local.get(assigned_display)
                if previous_jersey and previous_jersey != jersey:
                    jersey_display_map.pop(previous_jersey, None)
                    display_to_jersey_local.pop(assigned_display, None)
                display_to_jersey_local[assigned_display] = jersey
                used_ids.add(assigned_display)
                info['display_id'] = assigned_display
                if source_tid is not None:
                    new_player_id_map[source_tid] = assigned_display
                assigned_players[assigned_display] = info

            temp_pool = [i for i in all_ids if i not in reserved_ids and i not in used_ids]
            temp_pool.sort(reverse=True)
            for info, source_tid, original_id in pending_players:
                if original_id is not None and original_id not in used_ids:
                    assigned_display = original_id
                elif temp_pool:
                    assigned_display = temp_pool.pop(0)
                else:
                    remaining = [i for i in all_ids if i not in used_ids]
                    if not remaining:
                        continue
                    assigned_display = remaining[0]
                used_ids.add(assigned_display)
                info['display_id'] = assigned_display
                if source_tid is not None:
                    new_player_id_map[source_tid] = assigned_display
                assigned_players[assigned_display] = info

            cur_tracks['players'] = dict(sorted(assigned_players.items()))
            active_ids = set(cur_tracks['players'].keys())
            stale_ids = active_jersey_ids - active_ids
            for stale_id in stale_ids:
                jersey_ocr.reset_history(stale_id)
            active_jersey_ids = active_ids

            tracker.sync_display_assignments(jersey_display_map, new_player_id_map)

            active_source_tracks = {
                info.get("source_track_id")
                for info in cur_tracks['players'].values()
                if info.get("source_track_id") is not None
            }
            for track_id in list(track_jersey_cache.keys()):
                meta = track_jersey_cache.get(track_id) or {}
                last_frame = meta.get("frame", -1)
                if (
                    track_id not in active_source_tracks
                    and frame_idx - last_frame > JERSEY_CACHE_TTL_FRAMES
                ):
                    track_jersey_cache.pop(track_id, None)
            for jersey_key in list(jersey_track_cache.keys()):
                jersey_meta = jersey_track_cache.get(jersey_key) or {}
                if frame_idx - jersey_meta.get("frame", -1) > JERSEY_CACHE_TTL_FRAMES:
                    jersey_track_cache.pop(jersey_key, None)

            for display_id, info in cur_tracks['players'].items():
                jersey = info.get('jersey_number')
                bbox = info.get('bbox')
                if jersey and bbox:
                    x1, y1, x2, y2 = map(float, bbox)
                    center_x = 0.5 * (x1 + x2)
                    center_y = 0.5 * (y1 + y2)
                    history = jersey_position_history.get(jersey)
                    if history is None:
                        history = deque(maxlen=20)
                        jersey_position_history[jersey] = history
                    history.append((frame_idx, center_x, center_y))
        else:
            cur_tracks['players'] = dict(sorted(cur_tracks['players'].items()))

        # Gán team dựa trên display_id nếu đã có mapping
        for pid, info in cur_tracks['players'].items():
            key = f"pid_{int(pid):02d}"
            team = team_assigner.assign_team_by_key(key)
            if team is not None:
                info['team'] = team
                info['team_color'] = team_assigner.get_team_color(team)

        if (
            jersey_ocr is not None
            and torso_cropper is not None
            and cur_tracks['players']
            and (
                team_assigner.last_fit_frame < 0
                or frame_idx - team_assigner.last_fit_frame >= TEAM_REFIT_INTERVAL
            )
        ):
            refit_dump_dir = CROP_DEBUG_ROOT / f"frame_{frame_idx:06d}"
            refit_dump_dir.mkdir(parents=True, exist_ok=True)
            samples = []
            for pid, info in cur_tracks['players'].items():
                bbox = info.get('bbox')
                if bbox is None:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                patch = torso_cropper.crop(frame, (x1, y1, x2, y2))
                if patch is None:
                    continue
                jersey = info.get('jersey_number') or "NA"
                dump_name = f"id{int(pid):02d}_jersey_{jersey}_{len(samples):02d}.png"
                cv2.imwrite(str(refit_dump_dir / dump_name), patch)
                color_vec = TeamAssigner._color_from_patch(patch)
                if color_vec is not None:
                    key = f"pid_{int(pid):02d}"
                    samples.append((key, color_vec))
            if samples:
                team_assigner.add_samples(samples)
                if team_assigner.fit():
                    team_assigner.last_fit_frame = frame_idx

        feets = []
        colors = []
        heatmap_labels = []
        heatmap_metadata = []
        for pid, info in cur_tracks['players'].items():
            team = info.get('team')
            color = info.get('team_color', (0, 0, 255))
            if team is None:
                key = f"pid_{int(pid):02d}"
                team = team_assigner.assign_team_by_key(key)
                if team is not None:
                    info['team'] = team
            if color is None or (isinstance(color, (list, tuple, np.ndarray)) and len(color) == 0):
                info['team_color'] = team_assigner.get_team_color(team)
                color = info['team_color']
            x1, y1, x2, y2 = info["bbox"]
            x1_n, y1_n, x2_n, y2_n = x1 / w, y1 / h, x2 / w, y2 / h
            feet_point = cam_calib.calibrate_player_feet((x1_n, y1_n, x2_n, y2_n))
            feets.append(feet_point)
            colors.append(color)
            label = pid
            heatmap_labels.append(label)
            heatmap_metadata.append({
                "display_id": int(pid),
                "source_track_id": int(info.get("source_track_id")) if info.get("source_track_id") is not None else None,
                "team": int(info.get("team", 0)) if info.get("team") is not None else None,
                "jersey": info.get("jersey_number"),
            })

        # Draw
        drawn = tracker.draw_annotations_frame(frame, cur_tracks)
        
        # camera movement
        if cm_est is None:                            
            cm_est = CameraMovementEstimator(frame)  
            dx, dy = 0.0, 0.0                        
        else:
            dx, dy = cm_est.update(frame)
        cam_log_file.write(f"{frame_idx},{dx:.3f},{dy:.3f}\n")
        
        drawn = cm_est.draw_overlay(drawn, dx, dy)  

        should_reset_h = False
        movement_speed = math.hypot(dx, dy)
        if movement_speed > CAM_RESET_SPEED_THRESHOLD:
            pan_active = True
        elif pan_active and movement_speed < CAM_PAN_SETTLE_THRESHOLD:
            if (frame_idx - last_h_reset_frame) > CAM_RESET_COOLDOWN:
                should_reset_h = True
                pan_active = False

        if abs(dx) > CAM_SMALL_PAN_STEP:
            cumulative_small_pan += abs(dx)
        else:
            cumulative_small_pan = max(0.0, cumulative_small_pan - abs(dx))
        if (
            cumulative_small_pan >= CAM_SMALL_PAN_ACCUM_THRESHOLD
            and (frame_idx - last_h_reset_frame) > CAM_RESET_COOLDOWN
        ):
            should_reset_h = True
            cumulative_small_pan = 0.0
        if cam_calib.last_keypoint_count > 0 and cam_calib.last_keypoint_count < KEYPOINT_MIN_VALID:
            calib_low_conf_streak += 1
        else:
            calib_low_conf_streak = 0
        if (
            calib_low_conf_streak >= KEYPOINT_LOW_STREAK
            and (frame_idx - last_h_reset_frame) > CAM_RESET_COOLDOWN
        ):
            should_reset_h = True
        if should_reset_h:
            reason_low = calib_low_conf_streak
            cam_calib.reset()
            last_h_reset_frame = frame_idx
            calib_low_conf_streak = 0
            cumulative_small_pan = 0.0
            print(f"[CamCalib] Force reset at frame {frame_idx} (dx={dx:.2f}, low_conf={reason_low})")

        # Field Markings
        cam_calib(drawn)
        if feets and colors:
            prev_field, heatmap_points = cam_calib.draw(
                drawn,
                colors,
                feets,
                labels=heatmap_labels,
                metadata=heatmap_metadata
            )
        else:
            heatmap_points = []
        heatmap_logger.log(frame_idx, heatmap_points)
        if prev_field is not None:
            drawn = blend(drawn, prev_field, scale=0.5, alpha=0.4)

        # Mapping overlay
        display_to_jersey = tracker.get_display_to_jersey()
        display_to_track = tracker.get_display_to_track()
        if display_to_jersey:
            mapping_entries = sorted(display_to_jersey.items())
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            line_height = 24
            padding_x, padding_y = 12, 12
            text_lines = ["Mapping (ID↔Jersey)"]
            for display_id, jersey in mapping_entries:
                text_lines.append(f"ID {display_id:02d} ↔ #{jersey}")
            text_sizes = [cv2.getTextSize(line, font, font_scale, thickness)[0] for line in text_lines]
            box_width = (max(size[0] for size in text_sizes) if text_sizes else 0) + padding_x * 2
            box_height = line_height * len(text_lines) + padding_y * 2
            top_left = (w - box_width - 10, 10)
            bottom_right = (w - 10, 10 + box_height)
            cv2.rectangle(drawn, top_left, bottom_right, (0, 0, 0), -1)
            for idx, line in enumerate(text_lines):
                origin = (top_left[0] + padding_x, top_left[1] + padding_y + idx * line_height)
                cv2.putText(drawn, line, origin, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
            
        # Action Spotting
        drawn = ball_action.visualize_frame(drawn, frame_idx)

        gsr.emit(frame_idx, cur_tracks)
        write_frame(writer, drawn)
        frame_idx += 1
        
        if frame_idx >= fps * 300: 
            break

    close_video(cap)
    close_writer(writer)
    heatmap_logger.close()
    cam_log_file.close()
    print(f"[DONE] Wrote: {out_path}")


if __name__ == '__main__':
    main()

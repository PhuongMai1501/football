# FootballP Tracker

Hệ thống theo dõi cầu thủ bóng đá 2D: lấy video đầu vào, chạy YOLOv8 + BoT-SORT để theo dõi người chơi, gán màu/đội, đọc số áo, hiệu chỉnh sân (HRNet) và phủ heatmap top-view + các overlay (camera movement, action spotting). README này tổng hợp cách cài đặt và mô tả flow hiện tại của pipeline.

---

## 1. Kiến trúc tổng quan

```
Video → YOLOv8 + BoT-SORT → Tracker (bbox & IDs)
        ↓
     TeamAssigner (crop áo + KMeans màu)
        ↓
Jersey OCR (PARSeq + pose crop) → map số áo ↔ display ID
        ↓
CamCalib (HRNet keypoints → homography H) → Heatmap 2D + Field overlay
        ↓
Camera Movement Estimator (dx, dy) + reset H khi lia
        ↓
BallAction overlay + GameState JSONL
```

Tất cả được điều phối trong `main.py`. Các module chính:
- `trackers/`: YOLOv8 inference, BoT-SORT, vẽ bbox/ellipse.
- `team_assigner/`: crop thân áo từ pose, gom màu bằng KMeans định kỳ.
- `ocr/`: PARSeq đọc số áo, lock display ID với jersey number.
- `FieldMarkings/`: HRNet calibrator + heatmap top view.
- `camera_movement_estimator/`: optical flow để đo dx/dy.
- `BallAction/`: overlay highlight PASS/DRIVE từ `.npz`.

---

## 2. Yêu cầu môi trường

- Python 3.10–3.11 khớp với `pyproject.toml`.
- GPU NVIDIA + CUDA 11.8 (nếu muốn chạy YOLO/HRNet trên GPU). CPU vẫn chạy được nhưng chậm.
- FFmpeg (để OpenCV đọc ghi video).
- (Khuyến nghị) [uv](https://github.com/astral-sh/uv) để đồng bộ nhanh dựa trên `uv.lock`.

Kiểm tra CUDA:
```bash
nvidia-smi
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```

---

## 3. Cài đặt phụ thuộc

### 3.1. Dùng `uv`
```bash
pip install uv                # hoặc curl script từ dự án uv
uv pip sync                   # đọc danh sách từ uv.lock

# Tạo virtualenv (tùy chọn)
uv venv .venv
source .venv/bin/activate
uv pip sync
```

### 3.2. Dùng `pip` thuần
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r BoT-SORT/requirements.txt   # nếu cần train/finetune tracker
pip install -e .
```

> Lưu ý: `pyproject.toml` tham chiếu `torch/torchvision` từ index CUDA 11.8. Nếu không có GPU, hãy cài bản CPU hoặc sửa lại URL.

---

## 4. Chuẩn bị dữ liệu & trọng số

- `models/best_ylv8_ep50.pt`: checkpoint YOLOv8 detection/tracking.
- `models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth`: HRNet keypoints cho sân.
- `models/yolov8m-pose.pt`: YOLO pose phục vụ crop số áo.
- `parseq/outputs/.../checkpoints/*.ckpt`: trọng số PARSeq OCR (`PARSEQ_CKPT_PATH` trong `main.py`).
- `BallAction/models/.../test_raw_predictions.npz`: đầu vào cho overlay hành động.
- Video đầu vào đặt trong `input_videos/`. Thư mục đầu ra `output_videos/` sẽ chứa video xử lý và log JSONL.

---

## 5. Chạy pipeline

1. Đảm bảo mọi trọng số + video đã đặt đúng vị trí.
2. Mở `main.py` chỉnh `in_path`, `out_path` nếu cần.
3. Chạy:
   ```bash
   uv run python main.py   # hoặc uv run main.py
   ```
4. Kết quả:
   - Video overlay ở `output_videos/...`.
   - `output_videos/game_state.jsonl`: log trạng thái từng frame.
   - `output_videos/camera_movement.log`: log dx/dy để debug reset homography.
   - Nếu cần, heatmap metadata (tọa độ sân) có trong `output_videos/heatmap_points.jsonl`.

Mặc định `main.py` xử lý tối đa 300 giây (giới hạn `fps * 300`). Điều chỉnh tùy nhu cầu.

---

## 6. Flow chi tiết & các điểm cần chú ý

### 6.1. Tracking & gán ID
- `Tracker` dùng YOLOv8 + BoT-SORT để theo dõi player/ref/ball.  
- Nếu `use_boost=True`, tối đa 22 display ID (1–22). BoT-SORT `source_track_id` vẫn được lưu để debug.  
- `tracker.draw_annotations_frame` vẽ ellipse + display_id trên frame.

### 6.2. Jersey OCR & team mapping
- `JerseyRecogniser` crop vùng áo bằng pose, đọc số áo bằng PARSeq và giữ cache để tránh nhảy ID khi OCR mất tín hiệu.  
- Số áo ổn định được map vào `display_id`, đồng thời logging mapping ở góc phải video.
- `TeamAssigner` không còn lấy màu mỗi frame; thay vào đó, cứ 300 frame gom crop áo (pose + bbox) rồi chạy KMeans k=2 để gán team cho từng display_id. Các patch dùng để phân cụm được lưu trong `output_videos/croped_output/frame_xxxxxx/`.

### 6.3. Heatmap & reset homography
- `CamCalib` chạy HRNet + CameraCreator để suy ra homography H.  
- Có smoothing nhẹ nhưng tránh chấp nhận H quá xấu (det < 1e-6).  
- Heatmap 2D (top-view) dùng persistence layer 0.5 để giảm nhấp nháy.  
- Cơ chế reset:
  - Nếu vận tốc lia `sqrt(dx^2+dy^2)` vượt 90 và sau đó giảm còn <5 (đã qua 30 frame kể từ lần reset trước) → reset H.
  - Nếu số keypoint HRNet đủ tin cậy <20 liên tục 4 frame → reset.
  - Nếu các chuyển động nhỏ liên tục (|dx| > 5) cộng dồn đến 120 pixel → reset để tránh lệch chậm.
- Mọi lần reset đều log `[CamCalib] Force reset ...`, ngay khi reset `Cumulative small pan` cũng được xóa.

### 6.4. Camera movement & log
- `CameraMovementEstimator` dựa trên LK optical flow ở hai mép ảnh.  
- `cm_est.draw_overlay` hiển thị dx/dy trong video.  
- Log `frame,dx,dy` giúp phân tích threshold lia sau mỗi lần chạy.

### 6.5. Action spotting & export
- `BallAction/Test_Visual.py` đọc `.npz` có sẵn để vẽ overlay PASS/DRIVE.  
- `GameStateAdapter` viết JSONL gồm bbox + thông tin team/jersey/source_track_id để hệ thống khác xử lý.

---

## 7. Các script hỗ trợ / huấn luyện

- **YOLO training:** `training/` chứa notebook + script tham khảo. Sau khi huấn luyện xong, copy checkpoint vào `models/`.
- **Field markings HRNet/line:** xem README trong `FieldMarkings/`. Sử dụng Argus + Hydra config.
- **Ball action training:** `BallAction/scripts/ball_action/train.py`. Module inference cho overlay chỉ cần `.npz`.
- **BoT-SORT training:** `BoT-SORT/tools/` nếu muốn tinh chỉnh ReID.

---

## 8. Ghi chú thêm

- `.gitignore` đã loại bỏ toàn bộ file media và trọng số để tránh đẩy nhầm lên repo.  
- Nếu cần bật/tắt persistence, điều chỉnh `persist_heatmap` khi khởi tạo `CamCalib` trong `main.py`.  
- Một số đường dẫn (trọng số, video) đang hard-code; nên dùng biến môi trường hoặc CLI arguments nếu muốn linh hoạt.

---

## 9. Liên hệ / hỗ trợ

Nếu gặp lỗi liên quan đến HRNet/FieldMarkings:
- Kiểm tra log `[CamCalib] frame keypoints>=...` để biết HRNet có detect đủ line không.
- Theo dõi `det(H)`/`trace(H)` để xác định homography có bị lật/hỏng.

Đối với vấn đề OCR/team mapping:
- Bật `JerseyRecogniser` debug (`debug=True`) để lưu crop áo và đọc log.
- Kiểm tra `output_videos/croped_output/` xem patch áo có đúng vị trí hay không.

Hy vọng README giúp bạn hoặc đồng đội setup nhanh và hiểu rõ pipeline hiện tại!

# Hướng dẫn thiết lập & vận hành

Tài liệu này tóm tắt kiến trúc chính của dự án, cách chuẩn bị môi trường, chạy `main.py` và ghi nhận các mô-đun đã/ chưa tận dụng CUDA để tiện tối ưu sau này.

## 1. Tổng quan nhanh
- Pipeline chính (`main.py`) nhận video trong `input_videos/`, dùng YOLOv8 + BoT-SORT/ByteTrack để theo dõi người chơi, phân cụm màu áo (`team_assigner/`), hiệu chỉnh homography sân (`FieldMarkings/`), ước lượng chuyển động camera (`camera_movement_estimator/`) rồi phủ heatmap/ảnh phụ lên khung hình đầu ra.
- `BallAction/` tải sẵn dự đoán hành động (PASS/DRIVE) từ tệp `.npz` và vẽ biểu đồ để người dùng rà soát highlight.
- `GameStateAdapter` ghi lại bounding-box (player/referee/ball) thành JSONL để tích hợp với hệ thống khác.
- Dự án sử dụng `uv` để khóa môi trường (xem `pyproject.toml`, `uv.lock`) và vẫn duy trì `requirements.txt` tối giản cho trường hợp cần `pip`.

## 2. Chuẩn bị môi trường

### 2.1. Yêu cầu hệ thống
- Python >= 3.10; khuyến nghị dùng 3.10–3.11 trùng với `uv.lock`.
- GPU NVIDIA với driver + CUDA 11.8 nếu muốn chạy các mô hình Torch (YOLO, HRNet, BallAction) trên GPU. Nếu chỉ cần CPU, vẫn chạy được nhưng chậm.
- FFmpeg/codec đầy đủ để OpenCV đọc/ghi `.mp4`/`.avi`. Trên Ubuntu có thể cài `sudo apt install ffmpeg`.
- (Tùy chọn) `uv` CLI để quản lý phụ thuộc nhanh hơn pip.

### 2.2. Thiết lập bằng `uv`
```bash
# 1. Cài uv (một lần)
pip install uv

# 2. Đồng bộ phụ thuộc dựa trên uv.lock
uv pip sync

# 3. (Khuyến nghị) Tạo virtualenv nội bộ
uv venv .venv
source .venv/bin/activate
uv pip sync
```
`pyproject.toml` đã trỏ `torch/torchvision` về index `https://download.pytorch.org/whl/cu118`, vì vậy bạn cần CUDA 11.8 runtime tương thích (driver >= 520).

### 2.3. Thiết lập bằng `pip` thông thường
Nếu không dùng `uv`, có thể:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r BoT-SORT/requirements.txt  # để train/finetune BoT-SORT
pip install -e .                          # để cài project như package tùy chọn
```
Lưu ý: `pynvvideocodec` yêu cầu toolkit NVIDIA Video Codec SDK; nếu không cần giải mã tăng tốc, có thể bỏ gói này khỏi `pyproject.toml`.

### 2.4. Kiểm tra CUDA sẵn sàng
```bash
nvidia-smi
python - <<'PY'
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
PY
```
Nếu muốn chạy các mô-đun Torch trên CPU, chỉnh `DEVICE` trong từng file (ví dụ `FieldMarkings/run.py`) về `'cpu'` và truyền `device='cpu'` khi khởi tạo `YOLO`.

## 3. Chuẩn bị dữ liệu & trọng số
- `models/best_ylv8_ep50.pt`: trọng số YOLOv8 phục vụ detection/tracking. Có thể thay bằng checkpoint khác miễn tương thích Ultralytics.
- `models/HRNet_57_hrnet48x2_57_003/evalai-018-0.536880.pth`: trọng số HRNet keypoint cho chuẩn sân. Sửa biến `MODEL_PATH` trong `FieldMarkings/run.py` nếu đặt ở nơi khác.
- `BallAction/models/ball_action/test_raw_predictions.npz`: file xác suất PASS/DRIVE cho từng frame dùng bởi `BallAction/Test_Visual.py`. Muốn cập nhật, chạy `BallAction/scripts/ball_action/predict.py`.
- Video đầu vào đặt trong `input_videos/` (mặc định `main.py` đọc `input_videos/video_cut_3_v2.mp4`). Thư mục đầu ra `output_videos/` phải tồn tại hoặc được tạo trước.
- `parseq/outputs/parseq/03_11_2025/checkpoints/*.ckpt`: trọng số PARSeq OCR (chỉnh lại biến `PARSEQ_CKPT_PATH` trong `main.py` nếu bạn huấn luyện ở nơi khác). Model pose YOLOv8 để crop số áo mặc định ở `models/yolov8m-pose.pt`.

## 4. Chạy pipeline chính
1. Chuẩn bị đầy đủ các mô hình ở mục 3 và đảm bảo `input_videos/` có file nguồn.
2. Chỉnh lại đường dẫn `in_path`, `out_path` trong `main.py` nếu cần.
3. Chạy:
   ```bash
   uv run python main.py
   # hoặc
   uv run main.py
   ```
4. Kết quả:
   - Video có overlay lưu tại `output_videos/...`.
   - Game-state từng khung hình ghi vào `output_videos/game_state.jsonl`.
   - Logger in `[DONE] Wrote: ...` khi hoàn tất hoặc sau tối đa ~300 giây hình (giới hạn `fps * 300` trong code).

## 5. Thành phần chính & cách tuỳ chỉnh
- **Tracking (`trackers/tracker.py`)**  
  - Dùng `ultralytics.YOLO` để suy luận. Bật BoT-SORT bằng `Tracker(..., use_boost=True)` (mặc định trong `main.py`).  
  - Để ép chạy CPU: `self.model = YOLO(model_path).to('cpu')` và đặt `args.device="cpu"` trong cấu hình BoT-SORT.
- **Team clustering (`team_assigner/team_assigner.py`)**  
  - Cắt vùng thân áo, chạy `KMeans` (CPU) để gán màu đội. Có thể điều chỉnh tỷ lệ cắt (20–80%) hoặc số team (mặc định 2) nếu cần xử lý nhiều đội hơn.
- **Camera movement (`camera_movement_estimator/camera_movement_estimator.py`)**  
  - Thuật toán LK optical flow trên hai mép khung hình. Tham số `minimum_distance`, `mask`, `lk_params` có thể tinh chỉnh cho video rung mạnh.
- **Field markings (`FieldMarkings/run.py`)**  
  - `CamCalib` tải HRNet bằng Argus loader và chạy trên `DEVICE='cuda:0'`. Với CPU, đổi sang `'cpu'` và cân nhắc tắt `torch.cuda.amp` trong `src/models/hrnet/metamodel.py`.
  - `blend` trộn sân đen với frame hiện tại; `scale` và `alpha` điều chỉnh kích thước/độ trong suốt.
- **Ball actions overlay (`BallAction/Test_Visual.py`)**  
  - Sử dụng dự đoán đã lưu, không cần GPU. Nếu muốn chạy end-to-end, dùng script `BallAction/scripts/action/predict.py` với tham số `--device cuda:0`.
- **Game state export (`gsr_adapter.py`)**  
  - Có thể mở rộng JSONL bằng cách ghi thêm metadata vào `players_out`/`refs_out`. File hiện tại ghi `cx, cy, w, h, team`.
- **Jersey OCR (`ocr/jersey_recognizer.py` + `parseq/`)**  
  - `JerseyRecogniser` dùng PARSeq để đọc số áo, kết hợp với YOLOv8 pose để crop vùng giáp ngực. Kết quả được dùng để khóa `display_id` theo số áo và hiển thị mapping ngay trên video. Đảm bảo cài đặt phụ thuộc trong thư mục `parseq/requirements` và cập nhật đường dẫn checkpoint (`PARSEQ_CKPT_PATH`) cùng pose model nếu bạn thay đổi trọng số.

## 6. Huấn luyện & script bổ sung
- **YOLOv5/v8 training**  
  - Notebook `training/football_training_yolo_v5l_cl.ipynb` mô tả pipeline huấn luyện. Sau khi hoàn tất, copy trọng số vào `models/` và cập nhật đường dẫn trong `main.py`.
  - Script nhanh để kiểm tra mô hình: `python yolo_inference.py` (đầu ra vào `runs/detect/` của Ultralytics).
- **FieldMarkings HRNet/Line models**  
  - Cấu hình ở `FieldMarkings/src/models/hrnet/train_config.yaml` và `.../line/train_config.yaml`; đều đặt `device: cuda:0`.  
  - Dùng `python FieldMarkings/src/train_hrnet.py --config ...` (tham khảo README trong thư mục đó).
- **BallAction**  
  - Dựng dataset SoCCerNet → chỉnh trong `BallAction/src/constants.py`.  
  - Train: `python BallAction/scripts/ball_action/train.py --config ...` (mặc định dùng AMP + CUDA, xem `BallAction/src/argus_models.py`).
- **BoT-SORT**  
  - Nếu cần cải thiện re-identification, cài thêm yêu cầu trong `BoT-SORT/requirements.txt` và chạy các tool ở `BoT-SORT/tools/`.

## 7. Tổng hợp hỗ trợ CUDA
Để xác định mô-đun nào đã chạm tới CUDA, chạy `rg -n "cuda" -g"*"` tại thư mục gốc. Các phát hiện chính:

| Thành phần | File chính | Trạng thái CUDA | Ghi chú |
| --- | --- | --- | --- |
| YOLOv8 Tracking | `trackers/tracker.py` | Có (Ultralytics tự chọn GPU; BoT-SORT hard-code `device="cuda"`) | Đổi sang CPU bằng cách truyền `device="cpu"` khi khởi tạo YOLO & cấu hình BoT-SORT. |
| Field calibration (HRNet) | `FieldMarkings/run.py`, `src/models/hrnet/metamodel.py` | Có (mặc định `DEVICE='cuda:0'`, dùng `torch.cuda.amp`) | Để chạy CPU cần sửa `DEVICE`, tắt AMP, chú ý tốc độ chậm hơn nhiều. |
| Field line model | `FieldMarkings/src/models/line/metamodel.py`, `train_config.yaml` | Có | Cả huấn luyện & inference dùng CUDA + AMP. |
| Ball action training/inference | `BallAction/src/argus_models.py`, `scripts/ball_action/predict.py` | Có | `BallAction/Test_Visual.py` riêng chỉ đọc `.npz`, không cần GPU. |
| BoT-SORT training infra | `BoT-SORT/yolov7/*`, `BoT-SORT/yolox/*` | Có | Các script training yêu cầu đa GPU; inference trong pipeline chỉ cần 1 GPU. |
| Boosted tracks helper | `booststracks.py` | Có (ưu tiên CUDA nếu khả dụng) | Có fallback CPU (`torch.cuda.is_available()`), nhưng nên kiểm tra hiệu năng. |
| Camera movement, team assigner, GSR adapter | `camera_movement_estimator/`, `team_assigner/`, `gsr_adapter.py` | Chưa | Chạy hoàn toàn trên CPU, chiếm ít tài nguyên → không cần tối ưu CUDA. |

Khi tối ưu tiếp theo, ưu tiên:
1. Cho phép chọn GPU/CPU qua tham số CLI thay vì hard-code (`FieldMarkings/run.py`, `Tracker`).
2. Gom logic AMP/CUDA vào một tiện ích chung để tránh lặp lại trong HRNet & line model.
3. Kiểm tra lại `BallAction` để tránh load CUDA khi chỉ cần visualizer.

---

Bất kỳ thay đổi nào sau này (ví dụ cập nhật trọng số, mở rộng số đội, thay đổi định dạng game-state) đều nên cập nhật lại tài liệu này để đội khác dễ tiếp quản.

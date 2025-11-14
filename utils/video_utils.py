# utils/video_utils.py
import cv2

def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h

def iter_frames(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        yield frame

def create_writer(path, fps, width, height, fourcc_str="XVID"):
    fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open writer for {path}")
    return writer

def write_frame(writer, frame):
    writer.write(frame)

def close_video(cap):
    if cap is not None:
        cap.release()

def close_writer(writer):
    if writer is not None:
        writer.release()

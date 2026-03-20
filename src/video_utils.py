import os
import cv2


def read_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps is None or fps <= 0:
        fps = 30.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append(frame)

        if max_frames is not None and len(frames) >= max_frames:
            break

    cap.release()
    return frames, fps


def write_video(frames, path, fps=30):
    if not frames:
        raise ValueError("No frames provided to write_video().")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    h, w = frames[0].shape[:2]
    is_color = len(frames[0].shape) == 3

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), is_color)

    for frame in frames:
        out.write(frame)

    out.release()
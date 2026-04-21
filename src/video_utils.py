import cv2

def read_video(path, max_frames=None):
    cap = cv2.VideoCapture(path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()
    return frames

def write_video(frames, path, fps=30, is_color=False):
    if not is_color:
        frames = [cv2.cvtColor(f, cv2.COLOR_GRAY2BGR) if f.ndim == 2 else f for f in frames]
    else:
        # ordered_dither returns RGB, convert to BGR for OpenCV
        frames = [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) if f.ndim == 3 else f for f in frames]
    h, w = frames[0].shape[:2]
    print(f"[write_video] Writing {len(frames)} frames, shape={frames[0].shape}, path={path}")

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    if not out.isOpened():
        print("[write_video] avc1 failed, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(path, fourcc, fps, (w, h), True)

    for f in frames:
        out.write(f)

    out.release()
import os
import cv2
import numpy as np

from optical_flow import compute_flow

INPUT_VIDEO = "data/videos"
OUTPUT_VIDEO = "outputs/videos"
VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# -----------------------------
# Dot settings
# -----------------------------
GRID_SIZE = 10
RADIUS_ALPHA = 0.75
R_MIN = 1.0
R_MAX = 7.0
DOT_COLOR = 0           # black dots
BG_COLOR = 255          # white background

# -----------------------------
# Tracking settings
# -----------------------------
FLOW_SCALE = 0.5
FLOW_MAX_STEP = 20.0
BLUR_KSIZE = 5

# -----------------------------
# Dot birth/death
# -----------------------------
MASK_THRESHOLD = 20         # ignore nearly-white background when seeding
NEW_DOT_DIST = 8.0
MIN_DOT_RADIUS = 0.6

# -----------------------------
# "Depth" mode
# -----------------------------
# Choose one:
#   "intensity" -> darker = closer = bigger dots
#   "vertical"  -> lower in image = closer = bigger dots
#   "hybrid"    -> mix of both
DEPTH_MODE = "hybrid"

# hybrid weights
INTENSITY_WEIGHT = 0.6
VERTICAL_WEIGHT = 0.4

# -----------------------------
# Output
# -----------------------------
SAVE_COMPARISON = True


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    k = BLUR_KSIZE if BLUR_KSIZE % 2 == 1 else BLUR_KSIZE + 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    return gray


def resize_flow(flow: np.ndarray, target_shape) -> np.ndarray:
    target_h, target_w = target_shape
    src_h, src_w = flow.shape[:2]

    if (src_h, src_w) == (target_h, target_w):
        return flow.astype(np.float32)

    scale_x = target_w / src_w
    scale_y = target_h / src_h

    flow_rs = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    flow_rs = flow_rs.astype(np.float32)
    flow_rs[..., 0] *= scale_x
    flow_rs[..., 1] *= scale_y
    return flow_rs


def make_depth_like_map(gray: np.ndarray) -> np.ndarray:
    """
    Returns a depth-like map in [0,1]:
      0 = near
      1 = far

    This is not true depth unless you replace it later with a real depth estimator.
    """
    h, w = gray.shape
    gray_f = gray.astype(np.float32) / 255.0

    # intensity-based "depth"
    # darker -> near -> depth small
    depth_intensity = gray_f.copy()

    # vertical "depth"
    # lower in image -> near -> depth small
    y = np.linspace(0.0, 1.0, h, dtype=np.float32).reshape(h, 1)
    depth_vertical = 1.0 - y
    depth_vertical = np.repeat(depth_vertical, w, axis=1)

    if DEPTH_MODE == "intensity":
        depth = depth_intensity
    elif DEPTH_MODE == "vertical":
        depth = depth_vertical
    else:
        depth = INTENSITY_WEIGHT * depth_intensity + VERTICAL_WEIGHT * depth_vertical

    depth = np.clip(depth, 0.0, 1.0)
    return depth.astype(np.float32)


def radius_from_depth(depth_vals: np.ndarray) -> np.ndarray:
    """
    depth 0 = near -> large radius
    depth 1 = far  -> small radius
    """
    return (R_MIN + (1.0 - depth_vals) * (R_MAX - R_MIN)).astype(np.float32)


def initialize_dots(gray: np.ndarray, depth_map: np.ndarray):
    h, w = gray.shape
    xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
    ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

    dot_x = []
    dot_y = []
    dot_r = []

    for y in ys:
        for x in xs:
            # skip very bright background-like regions
            if gray[int(y), int(x)] > 255 - MASK_THRESHOLD:
                continue

            d = depth_map[int(y), int(x)]
            r = radius_from_depth(np.array([d], dtype=np.float32))[0]

            if r >= MIN_DOT_RADIUS:
                dot_x.append(float(x))
                dot_y.append(float(y))
                dot_r.append(float(r))

    return {
        "x": np.array(dot_x, dtype=np.float32),
        "y": np.array(dot_y, dtype=np.float32),
        "r": np.array(dot_r, dtype=np.float32),
    }


def advect_dots(dot_state, flow: np.ndarray, gray: np.ndarray):
    if dot_state is None or len(dot_state["x"]) == 0:
        return {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

    h, w = gray.shape
    x = dot_state["x"]
    y = dot_state["y"]
    r = dot_state["r"]

    xi = np.clip(np.round(x).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(y).astype(np.int32), 0, h - 1)

    dx = flow[yi, xi, 0]
    dy = flow[yi, xi, 1]

    mag = np.sqrt(dx * dx + dy * dy)
    keep = mag < FLOW_MAX_STEP

    x_new = x[keep] + dx[keep]
    y_new = y[keep] + dy[keep]
    r_new = r[keep]

    xi2 = np.round(x_new).astype(np.int32)
    yi2 = np.round(y_new).astype(np.int32)

    inside = (
        (xi2 >= 0) & (xi2 < w) &
        (yi2 >= 0) & (yi2 < h)
    )

    x_new = x_new[inside]
    y_new = y_new[inside]
    r_new = r_new[inside]
    xi2 = xi2[inside]
    yi2 = yi2[inside]

    # kill dots that move into nearly-white empty regions
    valid = gray[yi2, xi2] < 255 - MASK_THRESHOLD

    return {
        "x": x_new[valid].astype(np.float32),
        "y": y_new[valid].astype(np.float32),
        "r": r_new[valid].astype(np.float32),
    }


def update_dot_radii(dot_state, depth_map: np.ndarray):
    if dot_state is None or len(dot_state["x"]) == 0:
        return dot_state

    h, w = depth_map.shape
    xi = np.clip(np.round(dot_state["x"]).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(dot_state["y"]).astype(np.int32), 0, h - 1)

    d = depth_map[yi, xi]
    target_r = radius_from_depth(d)

    dot_state["r"] = RADIUS_ALPHA * dot_state["r"] + (1.0 - RADIUS_ALPHA) * target_r
    return dot_state


def seed_new_dots(dot_state, gray: np.ndarray, depth_map: np.ndarray):
    h, w = gray.shape
    xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
    ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

    if dot_state is None:
        dot_state = {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

    existing_x = dot_state["x"]
    existing_y = dot_state["y"]

    new_x = []
    new_y = []
    new_r = []

    for y in ys:
        for x in xs:
            if gray[int(y), int(x)] > 255 - MASK_THRESHOLD:
                continue

            if len(existing_x) > 0:
                d2 = (existing_x - x) ** 2 + (existing_y - y) ** 2
                if np.min(d2) < (NEW_DOT_DIST ** 2):
                    continue

            d = depth_map[int(y), int(x)]
            r = radius_from_depth(np.array([d], dtype=np.float32))[0]

            if r >= MIN_DOT_RADIUS:
                new_x.append(float(x))
                new_y.append(float(y))
                new_r.append(float(r))

    if len(new_x) == 0:
        return dot_state

    dot_state["x"] = np.concatenate([dot_state["x"], np.array(new_x, dtype=np.float32)])
    dot_state["y"] = np.concatenate([dot_state["y"], np.array(new_y, dtype=np.float32)])
    dot_state["r"] = np.concatenate([dot_state["r"], np.array(new_r, dtype=np.float32)])
    return dot_state


def render_dots(frame_shape, dot_state):
    h, w = frame_shape[:2]
    out = np.full((h, w), BG_COLOR, dtype=np.uint8)

    if dot_state is None or len(dot_state["x"]) == 0:
        return out

    for x, y, r in zip(dot_state["x"], dot_state["y"], dot_state["r"]):
        gx = int(round(x))
        gy = int(round(y))
        rr = int(round(max(0.0, r)))

        if rr > 0 and 0 <= gx < w and 0 <= gy < h:
            cv2.circle(out, (gx, gy), rr, DOT_COLOR, -1)

    return out


def add_label(img: np.ndarray, text: str) -> np.ndarray:
    if len(img.shape) == 2:
        out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        out = img.copy()

    cv2.rectangle(out, (10, 10), (330, 52), (255, 255, 255), -1)
    cv2.putText(out, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def build_comparison(original_bgr: np.ndarray, stable: np.ndarray) -> np.ndarray:
    if len(stable.shape) == 2:
        stable_bgr = cv2.cvtColor(stable, cv2.COLOR_GRAY2BGR)
    else:
        stable_bgr = stable.copy()

    left = add_label(original_bgr, "Original")
    right = add_label(stable_bgr, "Tracked Dots (size from depth-like map)")
    return np.hstack([left, right])


def process_single_video(input_path: str, output_dir: str):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    stem, ext = os.path.splitext(os.path.basename(input_path))
    stable_path = os.path.join(output_dir, f"{stem}_depthdots{ext}")
    comp_path = os.path.join(output_dir, f"{stem}_depthdots_comparison{ext}")

    stable_writer = None
    comp_writer = None

    prev_gray = None
    dot_state = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_gray(gray_raw)
        depth_map = make_depth_like_map(gray)

        h, w = gray.shape

        if prev_gray is None:
            dot_state = initialize_dots(gray, depth_map)
        else:
            prev_flow_gray = prev_gray
            curr_flow_gray = gray

            if FLOW_SCALE != 1.0:
                prev_flow_gray = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)
                curr_flow_gray = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)

            flow = compute_flow(prev_flow_gray, curr_flow_gray)
            flow_full = resize_flow(flow, (h, w))

            dot_state = advect_dots(dot_state, flow_full, gray)
            dot_state = update_dot_radii(dot_state, depth_map)
            dot_state = seed_new_dots(dot_state, gray, depth_map)

        stable = render_dots(frame.shape, dot_state)

        if stable_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            stable_writer = cv2.VideoWriter(stable_path, fourcc, fps, (w, h), False)

        stable_writer.write(stable)

        if SAVE_COMPARISON:
            comparison = build_comparison(frame, stable)

            if comp_writer is None:
                ch, cw = comparison.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                comp_writer = cv2.VideoWriter(comp_path, fourcc, fps, (cw, ch), True)

            comp_writer.write(comparison)

        prev_gray = gray
        frame_idx += 1

        if frame_idx % 25 == 0:
            print(f"  processed {frame_idx} frames for {stem}...")

    cap.release()
    if stable_writer is not None:
        stable_writer.release()
    if comp_writer is not None:
        comp_writer.release()

    print(f"Saved tracked depth-dots: {stable_path}")
    if SAVE_COMPARISON:
        print(f"Saved comparison: {comp_path}")


def main():
    os.makedirs(OUTPUT_VIDEO, exist_ok=True)

    filenames = [
        f for f in os.listdir(INPUT_VIDEO)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{INPUT_VIDEO}'.")
        return

    for filename in filenames:
        input_path = os.path.join(INPUT_VIDEO, filename)
        print(f"Processing {input_path}...")
        process_single_video(input_path, OUTPUT_VIDEO)


if __name__ == "__main__":
    main()
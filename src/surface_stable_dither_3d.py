# import os
# import cv2
# import math
# import numpy as np
# import torch

# from optical_flow import compute_flow

# INPUT_VIDEO = "data/videos"
# OUTPUT_VIDEO = "outputs/videos"
# VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# # -----------------------------
# # General settings
# # -----------------------------
# SAVE_COMPARISON = True
# FLOW_SCALE = 0.5
# BLUR_KSIZE = 5

# # -----------------------------
# # Dot settings
# # -----------------------------
# GRID_SIZE = 12
# RADIUS_ALPHA = 0.70
# DOT_COLOR = 0
# BG_COLOR = 255

# # Depth -> radius mapping
# R_MIN = 1.0
# R_MAX = 8.0

# # Dot filtering
# MIN_DOT_RADIUS = 0.8
# NEW_DOT_DIST = 10.0
# FLOW_MAX_STEP = 25.0
# BRIGHTNESS_MASK_THRESHOLD = 245   # avoid spawning on nearly white regions

# # -----------------------------
# # Depth model settings
# # -----------------------------
# # model_type: "DPT_Large", "DPT_Hybrid", "MiDaS_small"
# MIDAS_MODEL_TYPE = "DPT_Hybrid"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Depth smoothing
# DEPTH_BLUR_KSIZE = 5

# # Optional depth reuse: estimate depth every N frames and interpolate in between
# DEPTH_EVERY_N_FRAMES = 1

# # -----------------------------
# # Preprocessing
# # -----------------------------
# CONTRAST_ALPHA = 1.05
# CONTRAST_BETA = 0


# def preprocess_gray(gray: np.ndarray) -> np.ndarray:
#     gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)
#     k = BLUR_KSIZE if BLUR_KSIZE % 2 == 1 else BLUR_KSIZE + 1
#     if k > 1:
#         gray = cv2.GaussianBlur(gray, (k, k), 0)
#     return gray


# def resize_flow(flow: np.ndarray, target_shape) -> np.ndarray:
#     target_h, target_w = target_shape
#     src_h, src_w = flow.shape[:2]

#     if (src_h, src_w) == (target_h, target_w):
#         return flow.astype(np.float32)

#     scale_x = target_w / src_w
#     scale_y = target_h / src_h

#     flow_rs = cv2.resize(flow, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
#     flow_rs = flow_rs.astype(np.float32)
#     flow_rs[..., 0] *= scale_x
#     flow_rs[..., 1] *= scale_y
#     return flow_rs


# class MiDaSDepthEstimator:
#     def __init__(self, model_type="DPT_Hybrid", device="cpu"):
#         self.device = device
#         print(f"Loading MiDaS model: {model_type} on {device}")

#         self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
#         self.model.to(device)
#         self.model.eval()

#         transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
#         if model_type in ["DPT_Large", "DPT_Hybrid"]:
#             self.transform = transforms.dpt_transform
#         else:
#             self.transform = transforms.small_transform

#     @torch.no_grad()
#     def predict(self, bgr: np.ndarray) -> np.ndarray:
#         """
#         Returns normalized depth in [0,1]:
#           0 = near
#           1 = far
#         """
#         rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         input_batch = self.transform(rgb).to(self.device)

#         prediction = self.model(input_batch)
#         prediction = torch.nn.functional.interpolate(
#             prediction.unsqueeze(1),
#             size=rgb.shape[:2],
#             mode="bicubic",
#             align_corners=False,
#         ).squeeze()

#         depth = prediction.cpu().numpy().astype(np.float32)

#         # MiDaS output is inverse-like depth: larger often means nearer
#         # Normalize to [0,1], then flip so 0=near, 1=far
#         depth -= depth.min()
#         denom = max(1e-6, float(depth.max()))
#         depth /= denom
#         depth = 1.0 - depth

#         if DEPTH_BLUR_KSIZE > 1:
#             k = DEPTH_BLUR_KSIZE if DEPTH_BLUR_KSIZE % 2 == 1 else DEPTH_BLUR_KSIZE + 1
#             depth = cv2.GaussianBlur(depth, (k, k), 0)

#         depth = np.clip(depth, 0.0, 1.0)
#         return depth.astype(np.float32)


# def radius_from_depth(depth_vals: np.ndarray) -> np.ndarray:
#     """
#     depth 0 = near -> bigger dots
#     depth 1 = far  -> smaller dots
#     """
#     r = R_MIN + (1.0 - depth_vals) * (R_MAX - R_MIN)
#     return r.astype(np.float32)


# def initialize_dots(gray: np.ndarray, depth_map: np.ndarray):
#     h, w = gray.shape
#     xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
#     ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

#     dot_x = []
#     dot_y = []
#     dot_r = []

#     for y in ys:
#         for x in xs:
#             if gray[int(y), int(x)] >= BRIGHTNESS_MASK_THRESHOLD:
#                 continue

#             d = depth_map[int(y), int(x)]
#             r = radius_from_depth(np.array([d], dtype=np.float32))[0]

#             if r >= MIN_DOT_RADIUS:
#                 dot_x.append(float(x))
#                 dot_y.append(float(y))
#                 dot_r.append(float(r))

#     return {
#         "x": np.array(dot_x, dtype=np.float32),
#         "y": np.array(dot_y, dtype=np.float32),
#         "r": np.array(dot_r, dtype=np.float32),
#     }


# def advect_dots(dot_state, flow: np.ndarray, gray: np.ndarray):
#     if dot_state is None or len(dot_state["x"]) == 0:
#         return {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

#     h, w = gray.shape
#     x = dot_state["x"]
#     y = dot_state["y"]
#     r = dot_state["r"]

#     xi = np.clip(np.round(x).astype(np.int32), 0, w - 1)
#     yi = np.clip(np.round(y).astype(np.int32), 0, h - 1)

#     dx = flow[yi, xi, 0]
#     dy = flow[yi, xi, 1]

#     mag = np.sqrt(dx * dx + dy * dy)
#     keep = mag < FLOW_MAX_STEP

#     x_new = x[keep] + dx[keep]
#     y_new = y[keep] + dy[keep]
#     r_new = r[keep]

#     xi2 = np.round(x_new).astype(np.int32)
#     yi2 = np.round(y_new).astype(np.int32)

#     inside = (
#         (xi2 >= 0) & (xi2 < w) &
#         (yi2 >= 0) & (yi2 < h)
#     )

#     x_new = x_new[inside]
#     y_new = y_new[inside]
#     r_new = r_new[inside]
#     xi2 = xi2[inside]
#     yi2 = yi2[inside]

#     valid = gray[yi2, xi2] < BRIGHTNESS_MASK_THRESHOLD

#     return {
#         "x": x_new[valid].astype(np.float32),
#         "y": y_new[valid].astype(np.float32),
#         "r": r_new[valid].astype(np.float32),
#     }


# def update_dot_radii(dot_state, depth_map: np.ndarray):
#     if dot_state is None or len(dot_state["x"]) == 0:
#         return dot_state

#     h, w = depth_map.shape
#     xi = np.clip(np.round(dot_state["x"]).astype(np.int32), 0, w - 1)
#     yi = np.clip(np.round(dot_state["y"]).astype(np.int32), 0, h - 1)

#     d = depth_map[yi, xi]
#     target_r = radius_from_depth(d)

#     dot_state["r"] = RADIUS_ALPHA * dot_state["r"] + (1.0 - RADIUS_ALPHA) * target_r
#     return dot_state


# def seed_new_dots(dot_state, gray: np.ndarray, depth_map: np.ndarray):
#     h, w = gray.shape
#     xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
#     ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

#     if dot_state is None:
#         dot_state = {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

#     existing_x = dot_state["x"]
#     existing_y = dot_state["y"]

#     new_x = []
#     new_y = []
#     new_r = []

#     for y in ys:
#         for x in xs:
#             if gray[int(y), int(x)] >= BRIGHTNESS_MASK_THRESHOLD:
#                 continue

#             if len(existing_x) > 0:
#                 d2 = (existing_x - x) ** 2 + (existing_y - y) ** 2
#                 if np.min(d2) < (NEW_DOT_DIST ** 2):
#                     continue

#             d = depth_map[int(y), int(x)]
#             r = radius_from_depth(np.array([d], dtype=np.float32))[0]

#             if r >= MIN_DOT_RADIUS:
#                 new_x.append(float(x))
#                 new_y.append(float(y))
#                 new_r.append(float(r))

#     if len(new_x) == 0:
#         return dot_state

#     dot_state["x"] = np.concatenate([dot_state["x"], np.array(new_x, dtype=np.float32)])
#     dot_state["y"] = np.concatenate([dot_state["y"], np.array(new_y, dtype=np.float32)])
#     dot_state["r"] = np.concatenate([dot_state["r"], np.array(new_r, dtype=np.float32)])
#     return dot_state


# def render_dots(frame_shape, dot_state):
#     h, w = frame_shape[:2]
#     out = np.full((h, w), BG_COLOR, dtype=np.uint8)

#     if dot_state is None or len(dot_state["x"]) == 0:
#         return out

#     # draw farther dots first, nearer dots later
#     order = np.argsort(dot_state["r"])
#     xs = dot_state["x"][order]
#     ys = dot_state["y"][order]
#     rs = dot_state["r"][order]

#     for x, y, r in zip(xs, ys, rs):
#         gx = int(round(x))
#         gy = int(round(y))
#         rr = int(round(max(0.0, r)))

#         if rr > 0 and 0 <= gx < w and 0 <= gy < h:
#             cv2.circle(out, (gx, gy), rr, DOT_COLOR, -1)

#     return out


# def add_label(img: np.ndarray, text: str) -> np.ndarray:
#     if len(img.shape) == 2:
#         out = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     else:
#         out = img.copy()

#     cv2.rectangle(out, (10, 10), (430, 52), (255, 255, 255), -1)
#     cv2.putText(out, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
#     return out


# def build_comparison(original_bgr: np.ndarray, stable: np.ndarray) -> np.ndarray:
#     stable_bgr = cv2.cvtColor(stable, cv2.COLOR_GRAY2BGR)
#     left = add_label(original_bgr, "Original")
#     right = add_label(stable_bgr, "3D-ish tracked dots (depth-sized)")
#     return np.hstack([left, right])


# def process_single_video(input_path: str, output_dir: str, depth_estimator: MiDaSDepthEstimator):
#     cap = cv2.VideoCapture(input_path)
#     if not cap.isOpened():
#         print(f"Could not open {input_path}")
#         return

#     fps = cap.get(cv2.CAP_PROP_FPS)
#     if not fps or fps <= 1e-3:
#         fps = 30.0

#     stem, ext = os.path.splitext(os.path.basename(input_path))
#     stable_path = os.path.join(output_dir, f"{stem}_3d_dots{ext}")
#     comp_path = os.path.join(output_dir, f"{stem}_3d_dots_comparison{ext}")

#     stable_writer = None
#     comp_writer = None

#     prev_gray = None
#     dot_state = None
#     current_depth = None

#     frame_idx = 0
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         gray = preprocess_gray(gray_raw)
#         h, w = gray.shape

#         if (frame_idx % DEPTH_EVERY_N_FRAMES == 0) or (current_depth is None):
#             current_depth = depth_estimator.predict(frame)

#         if prev_gray is None:
#             dot_state = initialize_dots(gray, current_depth)
#         else:
#             prev_flow_gray = prev_gray
#             curr_flow_gray = gray

#             if FLOW_SCALE != 1.0:
#                 prev_flow_gray = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)
#                 curr_flow_gray = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)

#             flow = compute_flow(prev_flow_gray, curr_flow_gray)
#             flow_full = resize_flow(flow, (h, w))

#             dot_state = advect_dots(dot_state, flow_full, gray)
#             dot_state = update_dot_radii(dot_state, current_depth)
#             dot_state = seed_new_dots(dot_state, gray, current_depth)

#         stable = render_dots(frame.shape, dot_state)

#         if stable_writer is None:
#             fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#             stable_writer = cv2.VideoWriter(stable_path, fourcc, fps, (w, h), False)

#         stable_writer.write(stable)

#         if SAVE_COMPARISON:
#             comparison = build_comparison(frame, stable)
#             if comp_writer is None:
#                 ch, cw = comparison.shape[:2]
#                 fourcc = cv2.VideoWriter_fourcc(*"mp4v")
#                 comp_writer = cv2.VideoWriter(comp_path, fourcc, fps, (cw, ch), True)

#             comp_writer.write(comparison)

#         prev_gray = gray
#         frame_idx += 1

#         if frame_idx % 10 == 0:
#             print(f"  processed {frame_idx} frames for {stem}...")

#     cap.release()
#     if stable_writer is not None:
#         stable_writer.release()
#     if comp_writer is not None:
#         comp_writer.release()

#     print(f"Saved 3D-ish tracked dots: {stable_path}")
#     if SAVE_COMPARISON:
#         print(f"Saved comparison: {comp_path}")


# def main():
#     os.makedirs(OUTPUT_VIDEO, exist_ok=True)

#     filenames = [
#         f for f in os.listdir(INPUT_VIDEO)
#         if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
#     ]
#     if not filenames:
#         print(f"No supported videos found in '{INPUT_VIDEO}'.")
#         return

#     depth_estimator = MiDaSDepthEstimator(
#         model_type=MIDAS_MODEL_TYPE,
#         device=DEVICE
#     )

#     for filename in filenames:
#         input_path = os.path.join(INPUT_VIDEO, filename)
#         print(f"Processing {input_path}...")
#         process_single_video(input_path, OUTPUT_VIDEO, depth_estimator)


# if __name__ == "__main__":
#     main()
    



import os
import cv2
import numpy as np
import torch

from optical_flow import compute_flow

INPUT_VIDEO = "data/videos"
OUTPUT_VIDEO = "outputs/videos"
VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

# -----------------------------
# General settings
# -----------------------------
SAVE_COMPARISON = True
FLOW_SCALE = 0.5
BLUR_KSIZE = 5

# -----------------------------
# Dot settings
# -----------------------------
GRID_SIZE = 8
RADIUS_ALPHA = 0.70
DOT_COLOR = 0
BG_COLOR = 255

# Depth -> radius mapping
R_MIN = 1.0
R_MAX = 6.0

# Dot filtering
MIN_DOT_RADIUS = 0.8
NEW_DOT_DIST = 6.0
FLOW_MAX_STEP = 18.0
BRIGHTNESS_MASK_THRESHOLD = 245

# -----------------------------
# Shape detection settings
# -----------------------------
SHAPE_MODE = "auto"
# options:
#   "auto"      -> detect largest geometric shape
#   "circle"    -> prefer circle
#   "polygon"   -> prefer triangle/rectangle/other polygon
#   "threshold" -> simple dark-shape threshold mask

MIN_SHAPE_AREA = 800
CANNY_1 = 50
CANNY_2 = 150

# -----------------------------
# Depth model settings
# -----------------------------
MIDAS_MODEL_TYPE = "DPT_Hybrid"   # or "MiDaS_small" for speed
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_BLUR_KSIZE = 5
DEPTH_EVERY_N_FRAMES = 1

# -----------------------------
# Preprocessing
# -----------------------------
CONTRAST_ALPHA = 1.05
CONTRAST_BETA = 0


def preprocess_gray(gray: np.ndarray) -> np.ndarray:
    gray = cv2.convertScaleAbs(gray, alpha=CONTRAST_ALPHA, beta=CONTRAST_BETA)
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


class MiDaSDepthEstimator:
    def __init__(self, model_type="DPT_Hybrid", device="cpu"):
        self.device = device
        print(f"Loading MiDaS model: {model_type} on {device}")

        self.model = torch.hub.load("intel-isl/MiDaS", model_type, trust_repo=True)
        self.model.to(device)
        self.model.eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = transforms.dpt_transform
        else:
            self.transform = transforms.small_transform

    @torch.no_grad()
    def predict(self, bgr: np.ndarray) -> np.ndarray:
        """
        Returns normalized depth in [0,1]:
          0 = near
          1 = far
        """
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        input_batch = self.transform(rgb).to(self.device)

        prediction = self.model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth = prediction.cpu().numpy().astype(np.float32)

        depth -= depth.min()
        denom = max(1e-6, float(depth.max()))
        depth /= denom

        # flip so 0=near, 1=far
        depth = 1.0 - depth

        if DEPTH_BLUR_KSIZE > 1:
            k = DEPTH_BLUR_KSIZE if DEPTH_BLUR_KSIZE % 2 == 1 else DEPTH_BLUR_KSIZE + 1
            depth = cv2.GaussianBlur(depth, (k, k), 0)

        return np.clip(depth, 0.0, 1.0).astype(np.float32)


def radius_from_depth(depth_vals: np.ndarray) -> np.ndarray:
    """
    depth 0 = near -> bigger dots
    depth 1 = far  -> smaller dots
    """
    r = R_MIN + (1.0 - depth_vals) * (R_MAX - R_MIN)
    return r.astype(np.float32)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    return mask


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return np.zeros_like(mask)

    best_idx = -1
    best_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > best_area and area >= MIN_SHAPE_AREA:
            best_area = area
            best_idx = i

    out = np.zeros_like(mask)
    if best_idx != -1:
        out[labels == best_idx] = 255
    return out


def detect_threshold_shape(gray: np.ndarray) -> np.ndarray:
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = clean_mask(mask)
    return largest_component(mask)


def detect_circle_mask(gray: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(gray, dtype=np.uint8)

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=40,
        param1=100,
        param2=20,
        minRadius=10,
        maxRadius=min(gray.shape[:2]) // 3
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        best = max(circles, key=lambda c: c[2])
        x, y, r = best
        cv2.circle(mask, (x, y), r, 255, -1)

    return clean_mask(mask)


def detect_polygon_mask(gray: np.ndarray) -> np.ndarray:
    edges = cv2.Canny(gray, CANNY_1, CANNY_2)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(gray, dtype=np.uint8)
    best = None
    best_area = 0

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_SHAPE_AREA:
            continue

        approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)
        # prefer geometric-ish polygons
        if 3 <= len(approx) <= 8 and area > best_area:
            best_area = area
            best = approx

    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, -1)

    return clean_mask(mask)


def detect_shape_mask(gray: np.ndarray) -> np.ndarray:
    if SHAPE_MODE == "circle":
        mask = detect_circle_mask(gray)
        if np.count_nonzero(mask) > 0:
            return mask
        return detect_threshold_shape(gray)

    if SHAPE_MODE == "polygon":
        mask = detect_polygon_mask(gray)
        if np.count_nonzero(mask) > 0:
            return mask
        return detect_threshold_shape(gray)

    if SHAPE_MODE == "threshold":
        return detect_threshold_shape(gray)

    # auto: try circle, then polygon, then generic threshold
    mask = detect_circle_mask(gray)
    if np.count_nonzero(mask) > 0:
        return mask

    mask = detect_polygon_mask(gray)
    if np.count_nonzero(mask) > 0:
        return mask

    return detect_threshold_shape(gray)


def initialize_dots(depth_map: np.ndarray, object_mask: np.ndarray):
    h, w = object_mask.shape
    xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
    ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

    dot_x, dot_y, dot_r = [], [], []

    for y in ys:
        for x in xs:
            if object_mask[int(y), int(x)] == 0:
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


def advect_dots(dot_state, flow: np.ndarray, object_mask: np.ndarray):
    if dot_state is None or len(dot_state["x"]) == 0:
        return {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

    h, w = object_mask.shape
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

    valid = object_mask[yi2, xi2] > 0

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


def seed_new_dots(dot_state, depth_map: np.ndarray, object_mask: np.ndarray):
    h, w = object_mask.shape
    xs = np.arange(GRID_SIZE // 2, w, GRID_SIZE)
    ys = np.arange(GRID_SIZE // 2, h, GRID_SIZE)

    if dot_state is None:
        dot_state = {"x": np.array([], np.float32), "y": np.array([], np.float32), "r": np.array([], np.float32)}

    existing_x = dot_state["x"]
    existing_y = dot_state["y"]

    new_x, new_y, new_r = [], [], []

    for y in ys:
        for x in xs:
            if object_mask[int(y), int(x)] == 0:
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

    # draw smaller/farther first, bigger/nearer later
    order = np.argsort(dot_state["r"])
    xs = dot_state["x"][order]
    ys = dot_state["y"][order]
    rs = dot_state["r"][order]

    for x, y, r in zip(xs, ys, rs):
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

    cv2.rectangle(out, (10, 10), (470, 52), (255, 255, 255), -1)
    cv2.putText(out, text, (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def draw_mask_overlay(frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = frame.copy()
    if len(overlay.shape) == 2:
        overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    green = np.zeros_like(overlay)
    green[..., 1] = 180
    m = mask > 0
    overlay[m] = cv2.addWeighted(overlay, 0.55, green, 0.45, 0)[m]
    return overlay


def build_comparison(original_bgr: np.ndarray, stable: np.ndarray, mask: np.ndarray) -> np.ndarray:
    stable_bgr = cv2.cvtColor(stable, cv2.COLOR_GRAY2BGR)
    masked = draw_mask_overlay(original_bgr, mask)

    left = add_label(original_bgr, "Original")
    mid = add_label(masked, "Detected Shape Mask")
    right = add_label(stable_bgr, "3D-ish tracked dots on shape")

    return np.hstack([left, mid, right])


def process_single_video(input_path: str, output_dir: str, depth_estimator: MiDaSDepthEstimator):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"Could not open {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1e-3:
        fps = 30.0

    stem, ext = os.path.splitext(os.path.basename(input_path))
    stable_path = os.path.join(output_dir, f"{stem}_3d_dots{ext}")
    comp_path = os.path.join(output_dir, f"{stem}_3d_dots_comparison{ext}")

    stable_writer = None
    comp_writer = None

    prev_gray = None
    dot_state = None
    current_depth = None

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = preprocess_gray(gray_raw)
        h, w = gray.shape

        object_mask = detect_shape_mask(gray)

        if (frame_idx % DEPTH_EVERY_N_FRAMES == 0) or (current_depth is None):
            current_depth = depth_estimator.predict(frame)

        if prev_gray is None:
            dot_state = initialize_dots(current_depth, object_mask)
        else:
            prev_flow_gray = prev_gray
            curr_flow_gray = gray

            if FLOW_SCALE != 1.0:
                prev_flow_gray = cv2.resize(prev_gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)
                curr_flow_gray = cv2.resize(gray, None, fx=FLOW_SCALE, fy=FLOW_SCALE, interpolation=cv2.INTER_AREA)

            flow = compute_flow(prev_flow_gray, curr_flow_gray)
            flow_full = resize_flow(flow, (h, w))

            dot_state = advect_dots(dot_state, flow_full, object_mask)
            dot_state = update_dot_radii(dot_state, current_depth)
            dot_state = seed_new_dots(dot_state, current_depth, object_mask)

        stable = render_dots(frame.shape, dot_state)

        if stable_writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            stable_writer = cv2.VideoWriter(stable_path, fourcc, fps, (w, h), False)

        stable_writer.write(stable)

        if SAVE_COMPARISON:
            comparison = build_comparison(frame, stable, object_mask)

            if comp_writer is None:
                ch, cw = comparison.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                comp_writer = cv2.VideoWriter(comp_path, fourcc, fps, (cw, ch), True)

            comp_writer.write(comparison)

        prev_gray = gray
        frame_idx += 1

        if frame_idx % 10 == 0:
            print(f"  processed {frame_idx} frames for {stem}...")

    cap.release()
    if stable_writer is not None:
        stable_writer.release()
    if comp_writer is not None:
        comp_writer.release()

    print(f"Saved 3D-ish tracked dots: {stable_path}")
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

    depth_estimator = MiDaSDepthEstimator(
        model_type=MIDAS_MODEL_TYPE,
        device=DEVICE
    )

    for filename in filenames:
        input_path = os.path.join(INPUT_VIDEO, filename)
        print(f"Processing {input_path}...")
        process_single_video(input_path, OUTPUT_VIDEO, depth_estimator)


if __name__ == "__main__":
    main()    
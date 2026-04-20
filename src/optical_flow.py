import cv2
import numpy as np

# Dense optical flow (existing option)
def compute_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(
        prev,
        curr,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0
    )
    return flow

# Dense warp helper
def warp_image(img, flow, border_value=255):
    h, w = flow.shape[:2]

    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x - flow[..., 0]).astype(np.float32)
    map_y = (grid_y - flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        img,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped

# Detect good feature points to track in a grayscale image
def detect_features(gray, max_corners=200, quality_level=0.01, min_distance=10, block_size=7):
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        blockSize=block_size
    )
    return points

# Track points from prev_gray to curr_gray using Lucas-Kanade
def track_features_lk(prev_gray, curr_gray, prev_pts):
    if prev_pts is None or len(prev_pts) == 0:
        return None, None, None

    curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        curr_gray,
        prev_pts,
        None,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            30,
            0.01
        )
    )
    return curr_pts, status, err

# Filter matched point pairs using LK status
def filter_valid_tracks(prev_pts, curr_pts, status):
    if prev_pts is None or curr_pts is None or status is None:
        return None, None

    status = status.reshape(-1).astype(bool)
    prev_valid = prev_pts[status]
    curr_valid = curr_pts[status]

    if len(prev_valid) == 0 or len(curr_valid) == 0:
        return None, None

    return prev_valid, curr_valid

# Estimate global affine transform from tracked points
def estimate_affine_from_tracks(prev_pts, curr_pts):
    if prev_pts is None or curr_pts is None:
        return None

    if len(prev_pts) < 3 or len(curr_pts) < 3:
        return None

    M, inliers = cv2.estimateAffinePartial2D(
        prev_pts,
        curr_pts,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0
    )
    return M

# Warp image using a 2x3 affine matrix
def warp_image_affine(img, M, output_shape, border_value=255):
    h, w = output_shape
    warped = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value
    )
    return warped
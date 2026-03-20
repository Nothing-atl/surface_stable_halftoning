import cv2
import numpy as np

DEFAULT_FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}


def compute_flow(prev_gray, curr_gray, params=None):
    """
    Dense Farneback optical flow.
    prev_gray and curr_gray must both be single-channel uint8 images.
    """
    p = DEFAULT_FLOW_PARAMS.copy()
    if params is not None:
        p.update(params)

    flow = cv2.calcOpticalFlowFarneback(
        prev_gray,
        curr_gray,
        None,
        p["pyr_scale"],
        p["levels"],
        p["winsize"],
        p["iterations"],
        p["poly_n"],
        p["poly_sigma"],
        p["flags"],
    )
    return flow


def sample_flow(flow, xs, ys):
    """
    Sample flow vectors at float point coordinates using nearest-neighbor lookup.
    Returns dx, dy arrays with the same shape as xs and ys.
    """
    h, w = flow.shape[:2]
    ix = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)

    dx = flow[iy, ix, 0]
    dy = flow[iy, ix, 1]
    return dx, dy
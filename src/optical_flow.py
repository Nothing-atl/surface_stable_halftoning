import cv2
import numpy as np

# Compute dense optical flow between two grayscale frames using Farneback.
# Returns a flow field of shape (H, W, 2), where:
# flow[..., 0] is horizontal displacement
# flow[..., 1] is vertical displacement
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

# Warp an image using a dense optical flow field.
# The flow is assumed to describe motion from prev -> curr.
# This function remaps the previous output into the current frame coordinates.
def warp_image(img, flow):
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
        borderValue=255
    )
    return warped
import cv2
import numpy as np

# Compute dense optical flow between two grayscale frames
def compute_flow(prev, curr):
    flow = cv2.calcOpticalFlowFarneback(
        prev, curr, None,
        0.5, 3, 21, 3, 5, 1.2, 0
    )
    return flow

# def compute_flow(prev, curr):
#     prev_u8 = prev.astype(np.uint8)
#     curr_u8 = curr.astype(np.uint8)

#     dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
#     dis.setUseSpatialPropagation(True)
#     flow = dis.calc(prev_u8, curr_u8, None)

#     return flow.astype(np.float32)
# Warp an image using optical flow.
# If img is larger than flow resolution, resize the flow first.
def warp_image(img, flow):
    ih, iw = img.shape[:2]
    fh, fw = flow.shape[:2]

    flow_resized = flow.copy()

    if (ih, iw) != (fh, fw):
        scale_x = iw / fw
        scale_y = ih / fh

        flow_resized = cv2.resize(flow, (iw, ih), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= scale_x
        flow_resized[..., 1] *= scale_y

    grid_x, grid_y = np.meshgrid(np.arange(iw), np.arange(ih))

    # remap uses destination -> source coordinates,
    # so use minus flow to sample from the previous frame
    map_x = (grid_x - flow_resized[..., 0]).astype(np.float32)
    map_y = (grid_y - flow_resized[..., 1]).astype(np.float32)

    warped = cv2.remap(
        img.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )
    return warped
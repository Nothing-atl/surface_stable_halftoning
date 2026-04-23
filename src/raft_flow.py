import cv2
import torch
import numpy as np
from torchvision.models.optical_flow import raft_small, Raft_Small_Weights

# Cap the largest image dimension sent into RAFT
RAFT_MAX_SIZE = 384

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

_weights = Raft_Small_Weights.DEFAULT
_transforms = _weights.transforms()
_model = raft_small(weights=_weights, progress=True).to(DEVICE)
_model.eval()


def _to_rgb(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


def _resize_for_raft(prev_bgr, curr_bgr, max_size=RAFT_MAX_SIZE):
    h, w = prev_bgr.shape[:2]
    scale = min(1.0, max_size / max(h, w))

    if scale < 1.0:
        new_w = max(8, int(round(w * scale)))
        new_h = max(8, int(round(h * scale)))

        # RAFT works more nicely if dimensions are divisible by 8
        new_w = (new_w // 8) * 8
        new_h = (new_h // 8) * 8

        prev_small = cv2.resize(prev_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        curr_small = cv2.resize(curr_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        prev_small = prev_bgr
        curr_small = curr_bgr
        new_h, new_w = h, w

    return prev_small, curr_small, (h, w), (new_h, new_w)


def _prepare_pair(prev_bgr, curr_bgr):
    prev_rgb = _to_rgb(prev_bgr)
    curr_rgb = _to_rgb(curr_bgr)

    prev_t = torch.from_numpy(prev_rgb).permute(2, 0, 1).float() / 255.0
    curr_t = torch.from_numpy(curr_rgb).permute(2, 0, 1).float() / 255.0

    prev_t = prev_t.unsqueeze(0)
    curr_t = curr_t.unsqueeze(0)

    prev_t, curr_t = _transforms(prev_t, curr_t)

    return prev_t.to(DEVICE), curr_t.to(DEVICE)


def compute_raft_flow(prev_bgr, curr_bgr):
    """
    Returns dense optical flow as numpy array of shape (H, W, 2)
    in the original frame resolution.
    """
    prev_small, curr_small, (orig_h, orig_w), (small_h, small_w) = _resize_for_raft(prev_bgr, curr_bgr)

    image1, image2 = _prepare_pair(prev_small, curr_small)

    with torch.no_grad():
        flow_predictions = _model(image1, image2)

    flow = flow_predictions[-1][0].detach().cpu().permute(1, 2, 0).numpy()

    # Resize flow back to original resolution if needed
    if (small_h, small_w) != (orig_h, orig_w):
        flow_resized = cv2.resize(flow, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        flow_resized[..., 0] *= orig_w / small_w
        flow_resized[..., 1] *= orig_h / small_h
        flow = flow_resized

    return flow.astype(np.float32)


def warp_image_with_flow(img, flow, border_value=255):
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
        borderValue=border_value,
    )
    return warped


def flow_to_rgb(flow):
    fx = flow[..., 0]
    fy = flow[..., 1]

    mag, ang = cv2.cartToPolar(fx, fy, angleInDegrees=True)

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = (ang / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
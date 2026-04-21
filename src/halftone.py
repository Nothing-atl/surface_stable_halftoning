import numpy as np
import cv2
import torch
from moge.model.v2 import MoGeModel

# Configuration
CELL_SIZE = 8
DOWNSAMPLE = 4 
MOGE_MAX_SIZE = 256 # Max resolution fed to MoGe
N_BUCKETS = 1024 # Number of discrete ellipse shapes
MAX_STRETCH = 3.0 # Max stretch ratio of ellipse axes

NORMAL_BLEND_ALPHA = 0.5 # Blend weight for new normal map vs previous on keyframes
SCENE_CHANGE_THRESHOLD = 30 # Mean abs diff threshold for scene change detection

BAYER_4 = (1/16.0) * np.array([
    [0,8,2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
])

# Load model once at module level
device = torch.device(
    "cuda" if torch.backends.mps.is_available()
    else "cpu"
)
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
model.eval()

# Precomputed cell offsets, reused for every kernel build
_cy, _cx = CELL_SIZE // 2, CELL_SIZE // 2
_ky, _kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE]
_dx, _dy = np.broadcast_arrays((_kx - _cx).astype(np.float32), (_ky - _cy).astype(np.float32))
OFFSETS = np.stack([_dx.ravel(), _dy.ravel()], axis=1)  # (CELL_SIZE^2, 2)
RADIUS_SQ = (CELL_SIZE * 0.48) ** 2


def create_normal_map(frame): # Run MoGe-2 on frame, return (Height, Width, 3) normal map in camera space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    scale = MOGE_MAX_SIZE / max(h, w) # downscale frame to reduce inference time
    if scale < 1.0:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    tensor = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.infer(tensor)
    normal_map = output["normal"].squeeze(0).cpu().numpy() # (H, W, 3), in camera space
    mask = output["mask"].squeeze(0).cpu().numpy()
    return normal_map, mask


def is_scene_change(prev_gray, curr_gray, threshold=SCENE_CHANGE_THRESHOLD): # Detect abrupt scene changes by mean absolute difference
    diff = cv2.absdiff(curr_gray, prev_gray)
    return diff.mean() > threshold


def build_kernel_for_normal(nx, ny, nz): # Build a circle stamp of size CELL_SIZE * CELL_SIZE
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    nx, ny, nz = (nx, ny, nz) / norm if norm > 1e-6 else (0.0, 0.0, 1.0)

    # Project image-plane basis vectors onto the tangent plane: t = v - (v·n)n
    # M maps image-space 2D offsets to surface-projected 2D offsets
    # A point (dx, dy) is inside the ellipse if M_inv @ [dx, dy] is inside the unit circle
    M = np.array([[1 - nx*nx, -nx*ny], [-ny*nx,  1 - ny*ny]], dtype=np.float32)
    det = M[0,0]*M[1,1] - M[0,1]*M[1,0]
    M_inv = (np.array([[M[1,1], -M[0,1]], [-M[1,0], M[0,0]]], dtype=np.float32) / det
             if abs(det) > 1e-6 else np.eye(2, dtype=np.float32))

    # Clamp stretch via SVD to avoid degenerate line-like ellipses
    U, s, Vt = np.linalg.svd(M_inv)
    M_inv = (U * np.clip(s, 1.0 / MAX_STRETCH, MAX_STRETCH)) @ Vt

    # Back-project all cell offsets through M_inv, check which fall inside the unit circle
    bp = OFFSETS @ M_inv.T
    return (bp[:, 0]**2 + bp[:, 1]**2 <= RADIUS_SQ).reshape(CELL_SIZE, CELL_SIZE)


def compute_all_kernels(normal_small, mask_small): # Compute one ellipse kernel per cell using quantized normal buckets
    sh, sw, _ = normal_small.shape

    # Normalize normals
    nx, ny, nz = normal_small[..., 0], normal_small[..., 1], normal_small[..., 2]
    norm = np.where((n := np.sqrt(nx**2 + ny**2 + nz**2)) < 1e-6, 1.0, n)
    nx, ny, nz = nx/norm, ny/norm, nz/norm

    # Quantize nx, ny to bucket grid
    # Nearby normals snap to the same bucket, reusing same kernel
    bins = int(np.sqrt(N_BUCKETS))
    qx = np.clip(((nx + 1) / 2 * bins).astype(int), 0, bins - 1)
    qy = np.clip(((ny + 1) / 2 * bins).astype(int), 0, bins - 1)
    bucket_ids = np.where(mask_small, qy * bins + qx, N_BUCKETS)  # N_BUCKETS = invalid/face-on

    # Build one kernel per unique bucket
    unique_buckets = np.unique(bucket_ids)
    kernel_cache = {}
    for bid in unique_buckets:
        if bid == N_BUCKETS:
            kernel_cache[bid] = build_kernel_for_normal(0.0, 0.0, 1.0)
        else:
            ys, xs = np.where(bucket_ids == bid)
            mid = len(ys) // 2
            kernel_cache[bid] = build_kernel_for_normal(nx[ys[mid], xs[mid]], ny[ys[mid], xs[mid]], nz[ys[mid], xs[mid]])

    # Vectorized lookup: map bucket_ids → kernel_table index → kernel
    bid_to_idx = np.zeros(N_BUCKETS + 2, dtype=int)
    kernel_table = np.zeros((len(unique_buckets), CELL_SIZE, CELL_SIZE), dtype=bool)
    for i, bid in enumerate(unique_buckets):
        bid_to_idx[bid] = i
        kernel_table[i] = kernel_cache[bid]

    return kernel_table[bid_to_idx[bucket_ids.ravel()]].reshape(sh, sw, CELL_SIZE, CELL_SIZE)


def ordered_dither(gray, frame, normal_map=None, valid_mask=None, prev_normal=None, prev_gray=None, color=False):
    if normal_map is None:
        normal_map, valid_mask = create_normal_map(frame)

    # Blend normal map with previous keyframe normal to avoid abrupt kernel changes
    if prev_normal is not None:
        normal_map = cv2.addWeighted(normal_map, NORMAL_BLEND_ALPHA, prev_normal, 1.0 - NORMAL_BLEND_ALPHA, 0)

    h, w = gray.shape
    small_gray = cv2.resize(gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA) # Downsample the image
    sh, sw = small_gray.shape

    # Blend grayscale with previous frame to reduce dot flickering, skip on scene changes
    if prev_gray is not None and not is_scene_change(prev_gray, gray):
        prev_small = cv2.resize(prev_gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA)
        small_gray = cv2.addWeighted(small_gray, 0.8, prev_small, 0.2, 0)

    tiled = np.tile(BAYER_4, (sh // 4 + 1, sw // 4 + 1))[:sh, :sw] # Repeat the Bayer matrix so it covers the whole image. The matrix repeats every 4x4 pixels

    # Get normal map and downsample to match small
    normal_small = cv2.resize(normal_map, (sw, sh), interpolation=cv2.INTER_NEAREST)
    mask_small = cv2.resize(valid_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)
    kernels = compute_all_kernels(normal_small, mask_small)

    if color:
        # Dither each RGB channel independently with the same ellipse kernels
        rgb_small = cv2.resize(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
            (sw, sh), interpolation=cv2.INTER_AREA
        ).astype(np.float32) / 255.0

        out = np.full((sh, sw, CELL_SIZE, CELL_SIZE, 3), 255, dtype=np.uint8)
        for c in range(3):
            on_mask = rgb_small[..., c] < tiled # Compare each pixel intensity with the threshold matrix
            out[..., c] = np.where(on_mask[:, :, None, None] & kernels, 0, 255)

        # Transpose to (sh, CELL_SIZE, sw, CELL_SIZE, 3) then reshape to final image
        return out.transpose(0, 2, 1, 3, 4).reshape(sh * CELL_SIZE, sw * CELL_SIZE, 3)
    else:
        img = small_gray.astype(np.float32) / 255.0
        on_mask = img < tiled
        out = np.full((sh, sw, CELL_SIZE, CELL_SIZE), 255, dtype=np.uint8)
        out = np.where(on_mask[:, :, None, None] & kernels, 0, out)
        return out.transpose(0, 2, 1, 3).reshape(sh * CELL_SIZE, sw * CELL_SIZE)
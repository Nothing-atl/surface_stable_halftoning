import numpy as np
import cv2
import torch
from moge.model.v2 import MoGeModel

CELL_SIZE = 8 # Expands each pixel by CELL_SIZExCELL_SIZE to fit circle
DOWNSAMPLE = 4

BAYER_4 = (1/16.0) * np.array([
    [0,8,2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
])

# Load model once at module level
device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cpu"
)
model = MoGeModel.from_pretrained("Ruicheng/moge-2-vitl-normal").to(device)
model.eval()

def create_normal_map(frame): # Run MoGe-2 on frame, return (Height, Width, 3) normal map in camera space
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = torch.tensor(rgb / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model.infer(tensor)

    normal_map = output["normal"].squeeze(0).cpu().numpy() # (H,W,3), in camera space
    mask = output["mask"].squeeze(0).cpu().numpy()
    return normal_map, mask

def normal_to_ellipse_kernel(nx, ny, nz):
    n = np.array([nx, ny, nz])
    norm = np.linalg.norm(n)
    n = n / norm

    # Project image-plane basis vectors onto the tangent plane: t = v - (v·n)n
    ex = np.array([1.0, 0.0, 0.0])
    ey = np.array([0.0, 1.0, 0.0])
    tx = ex - np.dot(ex, n) * n  # tangent along x
    ty = ey - np.dot(ey, n) * n  # tangent along y

    # M maps image-space 2D offsets to surface-projected 2D offsets
    M = np.array([[tx[0], ty[0]], [tx[1], ty[1]]], dtype=np.float32)

    # A point (dx, dy) is inside the ellipse if M_inv at [dx, dy] is inside the unit circle
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        M_inv = np.eye(2, dtype=np.float32)

    # Build a circle stamp of size CELL_SIZE * CELL_SIZE
    cy, cx = CELL_SIZE // 2, CELL_SIZE // 2
    ky, kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE] # Two 1D arrays for rows and cols
    dy = (ky - cy).astype(np.float32) # (cell_size, 1)
    dx = (kx - cx).astype(np.float32) # (1, cell_size)
    offsets = np.stack(np.broadcast_arrays(dx, dy), axis=-1)

    # Back-project: (H, W, 2) @ M_inv.T
    bp = offsets @ M_inv.T
    dist_sq = bp[..., 0]**2 + bp[..., 1]**2
    radius = CELL_SIZE * 0.48

    return dist_sq <= radius**2

def ordered_dither(gray, frame):
    h, w = gray.shape
    small = cv2.resize(gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA) # Downsample the image
    sh, sw = small.shape

    img = small.astype(np.float32) / 255.0
    tiled = np.tile(BAYER_4, (sh//4 + 1, sw//4 + 1)) # Repeat the Bayer matrix so it covers the whole image. The matrix repeats every 4x4 pixels
    tiled = tiled[:sh, :sw] # Crop the tiled matrix so it matches the exact image size
    on_mask = img < tiled # Compare each pixel intensity with the threshold matrix.
    
    # Get normal map and downsample to match small
    normal_map, valid_mask = create_normal_map(frame)
    normal_small = cv2.resize(normal_map, (sw, sh), interpolation=cv2.INTER_NEAREST)  # (sh, sw, 3)
    mask_small = cv2.resize(valid_mask.astype(np.uint8), (sw, sh), interpolation=cv2.INTER_NEAREST).astype(bool)

    out = np.full((sh * CELL_SIZE, sw * CELL_SIZE), 255, dtype=np.uint8)

    for cy in range(sh):
        for cx in range(sw):
            if not on_mask[cy, cx]:
                continue # no dot to stamp
            if mask_small[cy, cx]: # Use normal map if confident
                nx, ny, nz = normal_small[cy, cx]
            else: # else, surface faces the camera
                nx, ny, nz = 0.0, 0.0, 1.0

            # Build the ellipse stamp for this cell's angle
            kernel = normal_to_ellipse_kernel(nx, ny, nz)
            # Stamp onto canvas at cell's position
            y0, x0 = cy * CELL_SIZE, cx * CELL_SIZE
            out[y0:y0 + CELL_SIZE, x0:x0 + CELL_SIZE][kernel] = 0

    return out
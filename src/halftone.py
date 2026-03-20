import numpy as np
import cv2

# Legacy parameters used by the old ordered dithering pipeline
CELL_SIZE = 8
DOWNSAMPLE = 4

# New parameters for the persistent dot-field pipeline
DEFAULT_GRID_SIZE = 8
DEFAULT_BLUR_KSIZE = 5
DEFAULT_MAX_RADIUS_RATIO = 0.45

BAYER_4 = (1 / 16.0) * np.array([
    [0, 8, 2, 10],
    [12, 4, 14, 6],
    [3, 11, 1, 9],
    [15, 7, 13, 5]
], dtype=np.float32)


def ordered_dither(gray):
    """
    Legacy ordered dithering pipeline kept intact for continuity with the
    existing report and prior outputs.
    """
    h, w = gray.shape
    small = cv2.resize(
        gray,
        (w // DOWNSAMPLE, h // DOWNSAMPLE),
        interpolation=cv2.INTER_AREA
    )

    img = small.astype(np.float32) / 255.0
    sh, sw = img.shape

    tiled = np.tile(BAYER_4, (sh // 4 + 1, sw // 4 + 1))
    tiled = tiled[:sh, :sw]

    on_mask = img < tiled

    cy, cx = CELL_SIZE // 2, CELL_SIZE // 2
    ky, kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE]
    kernel = (np.sqrt((ky - cy) ** 2 + (kx - cx) ** 2) <= CELL_SIZE * 0.45)

    expanded = np.repeat(np.repeat(on_mask, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    kernel_tiled = np.tile(kernel, (sh, sw))

    out = np.where(expanded & kernel_tiled, 0, 255).astype(np.uint8)
    return out


def preprocess_gray(gray, blur_ksize=DEFAULT_BLUR_KSIZE):
    """
    Light smoothing before sampling intensities. This reduces tiny frame-to-frame
    intensity fluctuations that would otherwise make radii twitch.
    """
    if blur_ksize is None or blur_ksize <= 1:
        return gray

    if blur_ksize % 2 == 0:
        blur_ksize += 1

    return cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)


def radius_from_intensity(intensities, grid_size=DEFAULT_GRID_SIZE, max_radius_ratio=DEFAULT_MAX_RADIUS_RATIO):
    """
    Map intensity to dot radius.
    Darker pixel -> larger dot.
    Brighter pixel -> smaller dot.

    max_radius_ratio = 0.45 means the circle stays just under half the cell width,
    so neighboring dots usually do not fully merge into a solid fill.
    """
    max_radius = grid_size * max_radius_ratio
    darkness = 1.0 - (intensities.astype(np.float32) / 255.0)
    return darkness * max_radius


def sample_gray_at_points(gray, xs, ys):
    """
    Fast nearest-neighbor sampling for point locations.
    xs, ys are float arrays in image coordinates.
    """
    h, w = gray.shape
    ix = np.clip(np.rint(xs).astype(np.int32), 0, w - 1)
    iy = np.clip(np.rint(ys).astype(np.int32), 0, h - 1)
    return gray[iy, ix]


def generate_frame_dots(
    gray,
    grid_size=DEFAULT_GRID_SIZE,
    blur_ksize=DEFAULT_BLUR_KSIZE,
    max_radius_ratio=DEFAULT_MAX_RADIUS_RATIO
):
    """
    Create one dot per grid cell at the cell center.
    This is the new baseline representation.

    We keep the centers on a regular screen-space grid here.
    The stabilized version will start from this same representation but then
    advect the centers through time.
    """
    smooth = preprocess_gray(gray, blur_ksize=blur_ksize)
    h, w = smooth.shape

    xs = np.arange(grid_size / 2, w, grid_size, dtype=np.float32)
    ys = np.arange(grid_size / 2, h, grid_size, dtype=np.float32)

    grid_x, grid_y = np.meshgrid(xs, ys)

    flat_x = grid_x.reshape(-1)
    flat_y = grid_y.reshape(-1)

    intensities = sample_gray_at_points(smooth, flat_x, flat_y)
    radii = radius_from_intensity(
        intensities,
        grid_size=grid_size,
        max_radius_ratio=max_radius_ratio
    )

    return {
        "x": flat_x.copy(),
        "y": flat_y.copy(),
        "home_x": flat_x.copy(),
        "home_y": flat_y.copy(),
        "r": radii.astype(np.float32)
    }


def clone_dot_state(state):
    return {
        "x": state["x"].copy(),
        "y": state["y"].copy(),
        "home_x": state["home_x"].copy(),
        "home_y": state["home_y"].copy(),
        "r": state["r"].copy()
    }


def render_dots(state, frame_shape, background=255):
    """
    Render black circles on a white background.
    frame_shape is expected to be (h, w) from a grayscale frame.
    """
    h, w = frame_shape
    canvas = np.full((h, w), background, dtype=np.uint8)

    xs = state["x"]
    ys = state["y"]
    rs = state["r"]

    for x, y, r in zip(xs, ys, rs):
        radius = int(round(float(r)))
        if radius <= 0:
            continue

        cx = int(round(float(x)))
        cy = int(round(float(y)))

        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(canvas, (cx, cy), radius, 0, thickness=-1, lineType=cv2.LINE_AA)

    return canvas
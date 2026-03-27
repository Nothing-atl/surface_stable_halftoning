import numpy as np
import cv2

CELL_SIZE = 8 # Expands each pixel by CELL_SIZExCELL_SIZE to fit circle
DOWNSAMPLE = 4

BAYER_4 = (1/16.0) * np.array([
    [0,8,2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
])

# def ordered_dither(gray):
#     h, w = gray.shape
#     small = cv2.resize(gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA) # Downsample the image

#     img = small.astype(np.float32) / 255.0
#     sh, sw = img.shape
#     tiled = np.tile(BAYER_4, (sh//4 + 1, sw//4 + 1)) # Repeat the Bayer matrix so it covers the whole image. The matrix repeats every 4x4 pixels
#     tiled = tiled[:sh, :sw] # Crop the tiled matrix so it matches the exact image size
#     on_mask = img < tiled # Compare each pixel intensity with the threshold matrix.
    
#     # Build a circle stamp of size CELL_SIZE * CELL_SIZE
#     cy, cx = CELL_SIZE // 2, CELL_SIZE // 2
#     ky, kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE] # Two 1D arrays for rows and cols
#     kernel = (np.sqrt((ky - cy)**2 + (kx - cx)**2) <= CELL_SIZE * 0.45) # Place pixels for circle if distance <= radius to create circle

#     expanded = np.repeat(np.repeat(on_mask, CELL_SIZE, axis=0), CELL_SIZE, axis=1) # Scale on_mask to output resolution
#     kernel_tiled = np.tile(kernel, (sh, sw)) # Repeat the circle stamp across every cell

#     out = np.where(expanded & kernel_tiled, 0, 255).astype(np.uint8) # Black where cell is stamped and inside circle
#     return out


def ordered_dither(gray):
    h, w = gray.shape

    # Slight contrast boost
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)

    small = cv2.resize(gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA)

    img = small.astype(np.float32) / 255.0

    # Add a little noise so the pattern looks less perfectly regular
    img = small.astype(np.float32) / 255.0
    img = np.clip(img, 0.0, 1.0)

    sh, sw = img.shape
    tiled = np.tile(BAYER_4, (sh//4 + 1, sw//4 + 1))
    tiled = tiled[:sh, :sw]
    on_mask = img < tiled

    cy, cx = CELL_SIZE // 2, CELL_SIZE // 2
    ky, kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE]
    kernel = (np.sqrt((ky - cy)**2 + (kx - cx)**2) <= CELL_SIZE * 0.45)

    expanded = np.repeat(np.repeat(on_mask, CELL_SIZE, axis=0), CELL_SIZE, axis=1)
    kernel_tiled = np.tile(kernel, (sh, sw))

    out = np.where(expanded & kernel_tiled, 0, 255).astype(np.uint8)
    return out

def make_threshold_map(shape, cell_size=8):
    """
    Make a repeated Bayer threshold map at full output resolution.
    This is the pattern we will transport through time.
    """
    h, w = shape

    bayer = np.array([
        [0,  8,  2, 10],
        [12, 4, 14,  6],
        [3, 11,  1,  9],
        [15, 7, 13,  5]
    ], dtype=np.float32) / 16.0

    tiled = np.tile(
        bayer,
        (int(np.ceil(h / 4)), int(np.ceil(w / 4)))
    )[:h, :w]

    # expand each threshold cell
    threshold = cv2.resize(
        tiled,
        (w * cell_size, h * cell_size),
        interpolation=cv2.INTER_NEAREST
    )

    return threshold


def dots_from_tone_and_threshold(gray, threshold_map, cell_size=8):
    small = cv2.resize(
        gray,
        (gray.shape[1] // 4, gray.shape[0] // 4),
        interpolation=cv2.INTER_AREA
    ).astype(np.float32) / 255.0

    tone_big = cv2.resize(
        small,
        (small.shape[1] * cell_size, small.shape[0] * cell_size),
        interpolation=cv2.INTER_NEAREST
    )

    on_mask = tone_big < threshold_map

    cy, cx = cell_size // 2, cell_size // 2
    ky, kx = np.ogrid[0:cell_size, 0:cell_size]
    kernel = (np.sqrt((ky - cy) ** 2 + (kx - cx) ** 2) <= cell_size * 0.45)

    sh, sw = small.shape
    kernel_tiled = np.tile(kernel, (sh, sw))

    out = np.where(on_mask & kernel_tiled, 0, 255).astype(np.uint8)
    return out
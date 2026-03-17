import numpy as np
import cv2

CELL_SIZE = 16 # Expands each pixel by CELL_SIZExCELL_SIZE to fit circle
DOWNSAMPLE = 16

BAYER_4 = (1/16.0) * np.array([
    [0,8,2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
])

def ordered_dither(gray):
    h, w = gray.shape
    small = cv2.resize(gray, (w // DOWNSAMPLE, h // DOWNSAMPLE), interpolation=cv2.INTER_AREA) # Downsample the image

    img = small.astype(np.float32) / 255.0
    sh, sw = img.shape
    tiled = np.tile(BAYER_4, (sh//4 + 1, sw//4 + 1)) # Repeat the Bayer matrix so it covers the whole image. The matrix repeats every 4x4 pixels
    tiled = tiled[:sh, :sw] # Crop the tiled matrix so it matches the exact image size
    on_mask = img < tiled # Compare each pixel intensity with the threshold matrix.
    
    # Build a circle stamp of size CELL_SIZE * CELL_SIZE
    cy, cx = CELL_SIZE // 2, CELL_SIZE // 2
    ky, kx = np.ogrid[0:CELL_SIZE, 0:CELL_SIZE] # Two 1D arrays for rows and cols
    kernel = (np.sqrt((ky - cy)**2 + (kx - cx)**2) <= CELL_SIZE * 0.45) # Place pixels for circle if distance <= radius to create circle

    expanded = np.repeat(np.repeat(on_mask, CELL_SIZE, axis=0), CELL_SIZE, axis=1) # Scale on_mask to output resolution
    kernel_tiled = np.tile(kernel, (sh, sw)) # Repeat the circle stamp across every cell

    out = np.where(expanded & kernel_tiled, 0, 255).astype(np.uint8) # Black where cell is stamped and inside circle
    return out
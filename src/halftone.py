import numpy as np

BAYER_4 = (1/16.0) * np.array([
    [0, 8, 2,10],
    [12,4,14,6],
    [3,11,1,9],
    [15,7,13,5]
])

def ordered_dither(gray):
    img = gray.astype(np.float32) / 255.0
    h, w = img.shape
    tiled = np.tile(BAYER_4, (h//4 + 1, w//4 + 1))# Repeat the Bayer matrix so it covers the whole image.The matrix repeats every 4x4 pixels
    tiled = tiled[:h, :w] # Crop the tiled matrix so it matches the exact image size
    out = (img > tiled) * 255.  # Compare each pixel intensity with the threshold matrix.If pixel > threshold → white (255). If pixel ≤ threshold → black (0)
    return out.astype(np.uint8)
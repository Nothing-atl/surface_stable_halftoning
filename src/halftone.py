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
    tiled = np.tile(BAYER_4, (h//4 + 1, w//4 + 1))
    tiled = tiled[:h, :w]
    out = (img > tiled) * 255
    return out.astype(np.uint8)
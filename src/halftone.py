import numpy as np
import cv2

CELL_SIZE = 4

def build_dot_grid(gray: np.ndarray) -> list[dict]:
    h, w = gray.shape
    img = gray.astype(np.float32) / 255.0
    dots = []

    for y in range(0, h, CELL_SIZE):
        for x in range(0, w, CELL_SIZE):
            cell = img[y:y+CELL_SIZE, x:x+CELL_SIZE]
            intensity = cell.mean()
            max_radius = CELL_SIZE // 2
            radius = max_radius * (1.0 - intensity)

            cx = x + CELL_SIZE // 2
            cy = y + CELL_SIZE // 2

            dots.append({
                "cx": cx,
                "cy": cy,
                "radius": radius,
                "intensity": intensity,
            })

    return dots


def render_dots(dots: list[dict], h: int, w: int) -> np.ndarray:
    canvas = np.ones((h, w), dtype=np.uint8) * 255
    for dot in dots:
        r = int(round(dot["radius"]))
        if r > 0:
            cv2.circle(canvas, (int(dot["cx"]), int(dot["cy"])), r, 0, -1)
    return canvas
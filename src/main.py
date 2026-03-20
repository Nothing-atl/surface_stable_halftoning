import os
import cv2
import numpy as np

from video_utils import read_video, write_video
from stabilized_halftone import render_baseline_sequence, render_stabilized_sequence

INPUT_VIDEO_DIR = "data/videos"

OUTPUT_VIDEO_DIR = "outputs/videos"
OUTPUT_FRAME_DIR = "outputs/frames"
OUTPUT_DIAGNOSTIC_DIR = "outputs/diagnostics"

VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

GRID_SIZE = 8
BLUR_KSIZE = 5
MAX_RADIUS_RATIO = 0.45

RADIUS_ALPHA = 0.6
DRIFT_LIMIT_CELLS = 2.0
SNAP_STRENGTH = 0.10
BLEND_ALPHA = 0.7

FLOW_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 21,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}

MAX_FRAMES = 60  # Set to None for full videos


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_frame_sequence(frames, output_dir):
    ensure_dir(output_dir)
    for i, frame in enumerate(frames):
        path = os.path.join(output_dir, f"frame_{i + 1:04d}.png")
        cv2.imwrite(path, frame)


def save_sampled_frame_sequence(frames, output_dir, step=10):
    """
    Save every `step`-th frame using 1-based human-readable numbering.

    Example:
    if frames has 60 items and step=10, this saves:
    frame_0001.png
    frame_0011.png
    frame_0021.png
    frame_0031.png
    frame_0041.png
    frame_0051.png
    """
    ensure_dir(output_dir)

    for i, frame in enumerate(frames):
        if i % step == 0:
            path = os.path.join(output_dir, f"frame_{i + 1:04d}.png")
            cv2.imwrite(path, frame)


def amplified_instability(rendered_frames, gray_frames, alpha=6.0):
    """
    Compare the temporal difference in the rendered result against the temporal
    difference in the source grayscale frames.

    instability = | |R_t - R_{t+1}| - |G_t - G_{t+1}| |
    """
    if len(rendered_frames) < 2 or len(gray_frames) < 2:
        return None, None, None

    rendered_diff = cv2.absdiff(rendered_frames[0], rendered_frames[1])
    gray_diff = cv2.absdiff(gray_frames[0], gray_frames[1])

    if gray_diff.shape != rendered_diff.shape:
        gray_diff = cv2.resize(gray_diff, (rendered_diff.shape[1], rendered_diff.shape[0]))

    instability = cv2.absdiff(rendered_diff, gray_diff)
    instability_amp = cv2.convertScaleAbs(instability, alpha=alpha)

    return rendered_diff, gray_diff, instability_amp


def save_diagnostics(video_stem, baseline_frames, stabilized_frames, gray_frames):
    diag_dir = os.path.join(OUTPUT_DIAGNOSTIC_DIR, video_stem)
    ensure_dir(diag_dir)

    b_diff, gray_diff, b_instability = amplified_instability(baseline_frames, gray_frames)
    s_diff, _, s_instability = amplified_instability(stabilized_frames, gray_frames)

    if b_diff is None:
        return

    cv2.imwrite(os.path.join(diag_dir, "ground_truth_diff.png"), gray_diff)
    cv2.imwrite(os.path.join(diag_dir, "baseline_diff.png"), b_diff)
    cv2.imwrite(os.path.join(diag_dir, "baseline_instability.png"), b_instability)
    cv2.imwrite(os.path.join(diag_dir, "stabilized_diff.png"), s_diff)
    cv2.imwrite(os.path.join(diag_dir, "stabilized_instability.png"), s_instability)

    baseline_mean = float(np.mean(b_instability))
    stabilized_mean = float(np.mean(s_instability))
    improvement = baseline_mean - stabilized_mean

    with open(os.path.join(diag_dir, "metrics.txt"), "w", encoding="utf-8") as f:
        f.write(f"Baseline mean instability: {baseline_mean:.4f}\n")
        f.write(f"Stabilized mean instability: {stabilized_mean:.4f}\n")
        f.write(f"Mean instability reduction: {improvement:.4f}\n")


def process_video_file(input_path):
    filename = os.path.basename(input_path)
    stem, _ = os.path.splitext(filename)

    frames, fps = read_video(input_path, max_frames=MAX_FRAMES)

    baseline_frames, gray_frames = render_baseline_sequence(
        frames,
        grid_size=GRID_SIZE,
        blur_ksize=BLUR_KSIZE,
        max_radius_ratio=MAX_RADIUS_RATIO,
    )

    stabilized_frames, _ = render_stabilized_sequence(
    frames,
    grid_size=GRID_SIZE,
    blur_ksize=BLUR_KSIZE,
    max_radius_ratio=MAX_RADIUS_RATIO,
    radius_alpha=RADIUS_ALPHA,
    drift_limit_cells=DRIFT_LIMIT_CELLS,
    snap_strength=SNAP_STRENGTH,
    flow_params=FLOW_PARAMS,
    blend_alpha=BLEND_ALPHA,
    )

    baseline_video_path = os.path.join(OUTPUT_VIDEO_DIR, "baseline", f"{stem}.mp4")
    stabilized_video_path = os.path.join(OUTPUT_VIDEO_DIR, "stabilized", f"{stem}.mp4")

    write_video(baseline_frames, baseline_video_path, fps=fps)
    write_video(stabilized_frames, stabilized_video_path, fps=fps)

    # # Save all generated frames
    # save_frame_sequence(
    #     baseline_frames,
    #     os.path.join(OUTPUT_FRAME_DIR, "baseline", stem, "all_frames")
    # )
    # save_frame_sequence(
    #     stabilized_frames,
    #     os.path.join(OUTPUT_FRAME_DIR, "stabilized", stem, "all_frames")
    # )

    # Save report-friendly sampled frames every 10 frames
    save_sampled_frame_sequence(
        baseline_frames,
        os.path.join(OUTPUT_FRAME_DIR, "baseline", stem, "report_frames"),
        step=10
    )
    save_sampled_frame_sequence(
        stabilized_frames,
        os.path.join(OUTPUT_FRAME_DIR, "stabilized", stem, "report_frames"),
        step=10
    )

    save_diagnostics(stem, baseline_frames, stabilized_frames, gray_frames)

    print(f"Finished processing: {filename}")
    print(f"  Baseline video:   {baseline_video_path}")
    print(f"  Stabilized video: {stabilized_video_path}")


def main():
    ensure_dir(OUTPUT_VIDEO_DIR)
    ensure_dir(OUTPUT_FRAME_DIR)
    ensure_dir(OUTPUT_DIAGNOSTIC_DIR)

    filenames = [
        f for f in os.listdir(INPUT_VIDEO_DIR)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]

    if not filenames:
        print(f"No supported videos found in '{INPUT_VIDEO_DIR}'.")
        return

    for filename in filenames:
        process_video_file(os.path.join(INPUT_VIDEO_DIR, filename))


if __name__ == "__main__":
    main()
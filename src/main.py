import cv2
import os
import numpy as np
from video_utils import read_video, write_video
from halftone import ordered_dither, create_normal_map
from raft_flow import compute_raft_flow, flow_to_rgb, warp_image_with_flow

NORMAL_KEYFRAME_INTERVAL = 5

INPUT_VIDEO = "data/videos"
OUTPUT_VIDEO = "outputs/videos"
OUTPUT_FRAME = "outputs/frames"

INPUT_IMAGE = "data/images"
OUTPUT_IMAGE = "outputs/images"

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
VID_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}

def process_video(input_dir, output_dir):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        frames = read_video(input_path)
        output_frames = []
        cached_normal = None
        cached_mask = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if i % NORMAL_KEYFRAME_INTERVAL == 0:
                cached_normal, cached_mask = create_normal_map(frame)
            dither = ordered_dither(gray, frame, cached_normal, cached_mask)
            output_frames.append(dither)
        write_video(output_frames, output_path)
        print(f"Baseline halftone video saved: {output_path}")

def process_video_raft_warp_debug(input_dir, output_dir, max_frames=5):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)

        stem, ext = os.path.splitext(filename)
        output_filename = f"{stem}_raft_warp_debug{ext}"
        output_path = os.path.join(output_dir, output_filename)

        frames = read_video(input_path, max_frames=max_frames)
        if len(frames) < 2:
            print(f"Not enough frames in '{input_path}' for RAFT warp debug.")
            continue

        debug_frames = []

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

            print(f"[RAFT WARP] Computing flow for frame pair {i}/{len(frames)-1} in {filename}")
            flow = compute_raft_flow(prev_frame, curr_frame)

            warped_prev_gray = warp_image_with_flow(prev_gray, flow, border_value=255)

            raw_diff = cv2.absdiff(curr_gray, prev_gray)
            warped_diff = cv2.absdiff(curr_gray, warped_prev_gray)

            # Add labels to each tile
            prev_vis = cv2.cvtColor(prev_gray, cv2.COLOR_GRAY2BGR)
            curr_vis = cv2.cvtColor(curr_gray, cv2.COLOR_GRAY2BGR)
            warped_vis = cv2.cvtColor(warped_prev_gray, cv2.COLOR_GRAY2BGR)
            diff_vis = cv2.cvtColor(warped_diff, cv2.COLOR_GRAY2BGR)

            cv2.putText(prev_vis, "prev", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(curr_vis, "curr", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(warped_vis, "warped prev", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.putText(diff_vis, "abs(curr - warped)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            top_row = np.hstack([prev_vis, curr_vis])
            bottom_row = np.hstack([warped_vis, diff_vis])
            grid = np.vstack([top_row, bottom_row])

            debug_frames.append(grid)

            # Print simple numeric check too
            raw_error = raw_diff.mean()
            warped_error = warped_diff.mean()
            print(
                f"[RAFT WARP] Pair {i}: "
                f"raw mean abs diff = {raw_error:.2f}, "
                f"warped mean abs diff = {warped_error:.2f}"
            )

        write_video(debug_frames, output_path, is_color=True)
        print(f"RAFT warp debug video saved: {output_path}")


def process_video_raft_flow_vis(input_dir, output_dir, max_frames=10):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)

        stem, ext = os.path.splitext(filename)
        output_filename = f"{stem}_raft_flow_vis{ext}"
        output_path = os.path.join(output_dir, output_filename)

        frames = read_video(input_path, max_frames=max_frames)
        if len(frames) < 2:
            print(f"Not enough frames in '{input_path}' for RAFT flow visualization.")
            continue

        vis_frames = []

        # First frame placeholder
        vis_frames.append(frames[0])

        for i in range(1, len(frames)):
            prev_frame = frames[i - 1]
            curr_frame = frames[i]

            print(f"[RAFT VIS] Computing flow for frame pair {i}/{len(frames)-1} in {filename}")
            flow = compute_raft_flow(prev_frame, curr_frame)
            flow_vis = flow_to_rgb(flow)
            vis_frames.append(flow_vis)

        write_video(vis_frames, output_path, is_color=True)
        print(f"RAFT flow visualization video saved: {output_path}")

def process_video_raft_halftone_stabilized(input_dir, output_dir, alpha=0.7, max_frames=5):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)

        stem, ext = os.path.splitext(filename)
        alpha_tag = str(alpha).replace(".", "")
        output_filename = f"{stem}_raft_halftone_stabilized_a{alpha_tag}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        frames = read_video(input_path, max_frames=max_frames)
        if not frames:
            print(f"No frames read from '{input_path}'.")
            continue

        output_frames = []
        cached_normal = None
        cached_mask = None
        prev_halftone = None
        prev_frame = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % NORMAL_KEYFRAME_INTERVAL == 0:
                cached_normal, cached_mask = create_normal_map(frame)

            print(f"[RAFT HALFTONE] Halftoning frame {i + 1}/{len(frames)} in {filename}")
            curr_halftone = ordered_dither(
                gray=gray,
                frame=frame,
                normal_map=cached_normal,
                valid_mask=cached_mask
            )

            if prev_halftone is None or prev_frame is None:
                stabilized = curr_halftone
            else:
                print(f"[RAFT HALFTONE] Computing flow for frame {i + 1}/{len(frames)} in {filename}")
                flow = compute_raft_flow(prev_frame, frame)

                # Resize flow to halftone resolution
                h_out, w_out = curr_halftone.shape
                h_in, w_in = gray.shape

                flow_resized = cv2.resize(flow, (w_out, h_out), interpolation=cv2.INTER_LINEAR)
                flow_resized[..., 0] *= w_out / w_in
                flow_resized[..., 1] *= h_out / h_in

                warped_prev_halftone = warp_image_with_flow(prev_halftone, flow_resized, border_value=255)

                blended = cv2.addWeighted(
                    curr_halftone,
                    alpha,
                    warped_prev_halftone,
                    1.0 - alpha,
                    0
                )

                _, stabilized = cv2.threshold(blended, 127, 255, cv2.THRESH_BINARY)

            output_frames.append(stabilized)

            prev_halftone = curr_halftone
            prev_frame = frame

        write_video(output_frames, output_path, is_color=False)
        print(f"RAFT halftone stabilized video saved: {output_path}")

def process_video_raft_gray_stabilized(input_dir, output_dir, alpha=0.8, max_frames=5):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)

        stem, ext = os.path.splitext(filename)
        alpha_tag = str(alpha).replace(".", "")
        output_filename = f"{stem}_raft_gray_stabilized_a{alpha_tag}{ext}"
        output_path = os.path.join(output_dir, output_filename)

        frames = read_video(input_path, max_frames=max_frames)

        output_frames = []
        prev_gray = None
        prev_frame = None

        cached_normal = None
        cached_mask = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % NORMAL_KEYFRAME_INTERVAL == 0:
                cached_normal, cached_mask = create_normal_map(frame)

            if prev_gray is None:
                stabilized_gray = gray
            else:
                flow = compute_raft_flow(prev_frame, frame)

                warped_prev_gray = warp_image_with_flow(
                    prev_gray,
                    flow,
                    border_value=255
                )

                stabilized_gray = cv2.addWeighted(
                    gray,
                    alpha,
                    warped_prev_gray,
                    1.0 - alpha,
                    0
                )
                stabilized_gray = cv2.GaussianBlur(stabilized_gray, (3, 3), 0)

            dither = ordered_dither(
                gray=stabilized_gray,
                frame=frame,
                normal_map=cached_normal,
                valid_mask=cached_mask
            )

            output_frames.append(dither)

            prev_gray = gray
            prev_frame = frame

        write_video(output_frames, output_path, is_color=False)
        print(f"RAFT gray stabilized video saved: {output_path}")

def process_frames_raft_gray_stabilized(input_dir, output_dir, n=5, alpha=0.8):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        stem = os.path.splitext(filename)[0]
        alpha_tag = str(alpha).replace(".", "")
        output_folder = os.path.join(output_dir, f"{stem}_raft_gray_stabilized_a{alpha_tag}")
        os.makedirs(output_folder, exist_ok=True)

        frames = read_video(input_path, max_frames=n)
        if not frames:
            print(f"No frames read from '{input_path}'.")
            continue

        prev_gray = None
        prev_frame = None
        cached_normal = None
        cached_mask = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % NORMAL_KEYFRAME_INTERVAL == 0:
                cached_normal, cached_mask = create_normal_map(frame)

            if prev_gray is None:
                stabilized_gray = gray
            else:
                flow = compute_raft_flow(prev_frame, frame)
                warped_prev_gray = warp_image_with_flow(prev_gray, flow, border_value=255)

                stabilized_gray = cv2.addWeighted(
                    gray,
                    alpha,
                    warped_prev_gray,
                    1.0 - alpha,
                    0
                )
                stabilized_gray = cv2.GaussianBlur(stabilized_gray, (3, 3), 0)

            dither = ordered_dither(
                gray=stabilized_gray,
                frame=frame,
                normal_map=cached_normal,
                valid_mask=cached_mask
            )

            output_frame = os.path.join(output_folder, f"frame_{i:04d}.png")
            cv2.imwrite(output_frame, dither)

            prev_gray = gray
            prev_frame = frame

        print(f"Saved {n} RAFT gray stabilized frames to: {output_folder}")

def process_frames(input_dir, output_dir, n=20):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in VID_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported videos found in '{input_dir}'.")
        return

    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        stem = os.path.splitext(filename)[0]
        output_folder = os.path.join(output_dir, stem)
        os.makedirs(output_folder, exist_ok=True)

        frames = read_video(input_path, max_frames=n)
        cached_normal = None
        cached_mask = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if i % NORMAL_KEYFRAME_INTERVAL == 0:
                cached_normal, cached_mask = create_normal_map(frame)

            dither = ordered_dither(
                gray=gray,
                frame=frame,
                normal_map=cached_normal,
                valid_mask=cached_mask
            )

            output_frame = os.path.join(output_folder, f"frame_{i:04d}.png")
            cv2.imwrite(output_frame, dither)

        print(f"Saved {n} halftone frames to: {output_folder}")

def process_image(input_dir, output_dir):
    filenames = [
        f for f in os.listdir(input_dir)
        if os.path.splitext(f)[1].lower() in IMG_EXTENSIONS
    ]
    if not filenames:
        print(f"No supported images found in '{input_dir}'.")
        return
    
    for filename in filenames:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        img = cv2.imread(input_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dither = ordered_dither(gray, img)
        cv2.imwrite(output_path, dither)
        print(f"Baseline halftone image saved: {output_path}")

def process_diff():
    img1 = cv2.imread(r"outputs\frames\clouds\frame_0000.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(r"outputs\frames\clouds\frame_0001.png", cv2.IMREAD_GRAYSCALE)
    orig1 = cv2.imread(r"data\videos\grayscale\frame_0000.png", cv2.IMREAD_GRAYSCALE)
    orig2 = cv2.imread(r"data\videos\grayscale\frame_0001.png", cv2.IMREAD_GRAYSCALE)
    orig1 = cv2.resize(orig1, (img1.shape[1], img1.shape[0]))
    orig2 = cv2.resize(orig2, (img2.shape[1], img2.shape[0]))

    dither_diff = cv2.absdiff(img1, img2)
    ground_truth_diff = cv2.absdiff(orig1, orig2)

    instability = cv2.absdiff(dither_diff, ground_truth_diff)
    instability_amplified = cv2.convertScaleAbs(instability, alpha=10)

    cv2.imwrite(r"outputs\frames\clouds\test1.png", dither_diff)
    cv2.imwrite(r"outputs\frames\clouds\test2.png", ground_truth_diff)
    cv2.imwrite(r"outputs\frames\clouds\diff.png", instability_amplified)

def main():
    process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    process_video_raft_gray_stabilized(INPUT_VIDEO, OUTPUT_VIDEO, alpha=0.6, max_frames=None)

if __name__ == "__main__":
    main()
import cv2
import os
from video_utils import read_video, write_video
from halftone import ordered_dither
from optical_flow import compute_flow, warp_image

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

        # to process all frames:
        # frames = read_video(input_path)

        # to process some frames for testing:
        frames = read_video(input_path, max_frames=60)

        output_frames = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dither = ordered_dither(gray, frame)
            output_frames.append(dither)
        write_video(output_frames, output_path)
        print(f"Baseline halftone video saved: {output_path}")

def process_video_stabilized(input_dir, output_dir, alpha=0.7):
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
        output_filename = f"{stem}_stabilized{ext}"
        output_path = os.path.join(output_dir, output_filename)

        # to process all frames:
        # frames = read_video(input_path)

        # to process some frames for testing:
        frames = read_video(input_path, max_frames=60)

        if not frames:
            print(f"No frames read from '{input_path}'.")
            continue

        output_frames = []

        prev_gray = None
        prev_dither = None

        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dither = ordered_dither(gray, frame)

            if prev_gray is None or prev_dither is None:
                stabilized = dither
            else:
                flow = compute_flow(prev_gray, gray)

                dither_h, dither_w = dither.shape[:2]
                gray_h, gray_w = gray.shape[:2]

                if (dither_h, dither_w) != (gray_h, gray_w):
                    flow_for_dither = cv2.resize(
                        flow,
                        (dither_w, dither_h),
                        interpolation=cv2.INTER_LINEAR
                    )

                    scale_x = dither_w / gray_w
                    scale_y = dither_h / gray_h
                    flow_for_dither[..., 0] *= scale_x
                    flow_for_dither[..., 1] *= scale_y
                else:
                    flow_for_dither = flow

                warped_prev = warp_image(prev_dither, flow_for_dither)

                stabilized = cv2.addWeighted(
                    dither,
                    alpha,
                    warped_prev,
                    1.0 - alpha,
                    0
                )

            output_frames.append(stabilized)

            prev_gray = gray
            prev_dither = stabilized

            print(f"Processed frame {i + 1}/{len(frames)} for {filename}")

        write_video(output_frames, output_path)
        print(f"Stabilized halftone video saved: {output_path}")

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
        for i, frame in enumerate(frames):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dither = ordered_dither(gray, frame)
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
    process_video_stabilized(INPUT_VIDEO, OUTPUT_VIDEO)

if __name__ == "__main__":
    main()
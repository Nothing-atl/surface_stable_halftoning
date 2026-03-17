import cv2
import os
from video_utils import read_video, write_video
from halftone import ordered_dither

INPUT_VIDEO = "data/videos"
OUTPUT_VIDEO = "outputs/videos"

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
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            dither = ordered_dither(gray)
            output_frames.append(dither)
        write_video(output_frames, output_path)
        print(f"Baseline halftone video saved: {output_path}")

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
        dither = ordered_dither(gray)
        cv2.imwrite(output_path, dither)
        print(f"Baseline halftone image saved: {output_path}")

def main():
    # process_video(INPUT_VIDEO, OUTPUT_VIDEO)
    process_image(INPUT_IMAGE, OUTPUT_IMAGE)

if __name__ == "__main__":
    main()
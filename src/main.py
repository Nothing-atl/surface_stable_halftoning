import cv2
from video_utils import read_video, write_video
from halftone import build_dot_grid, render_dots

INPUT_VIDEO = "data/people_walking.mp4"
OUTPUT_VIDEO = "outputs/baseline_output.mp4"


def main():
    frames = read_video(INPUT_VIDEO)
    output_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dots = build_dot_grid(gray)
        output = render_dots(dots, *gray.shape)
        output_frames.append(output)
    write_video(output_frames, OUTPUT_VIDEO)
    print("Baseline halftone video saved!...")

if __name__ == "__main__":
    main()
import argparse
import csv
import math
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from inference_utils import predict, visualize
from model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", default="outputs/best_csrnet.pth")
    parser.add_argument("--output", default="outputs/video_result.mp4")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise OSError(f"Video tidak bisa dibuka: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError("FPS video tidak valid.")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    progress = tqdm(total=int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), desc="Video")
    index = 0
    try:
        with open(output.with_suffix(".csv"), "w", newline="") as file:
            table = csv.writer(file)
            table.writerow(["frame", "time_seconds", "estimated_count"])
            while True:
                success, frame = capture.read()
                if not success:
                    break
                if writer is None:
                    height, width = frame.shape[:2]
                    if height % 2 or width % 2:
                        raise ValueError("Video MP4 membutuhkan lebar dan tinggi genap.")
                    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                    if not writer.isOpened():
                        raise OSError(f"Video writer gagal: {output}")
                count, density = predict(model, frame, device)
                writer.write(visualize(frame, count, density))
                table.writerow([index, index / fps, count])
                index += 1
                progress.update(1)
        if index == 0:
            raise ValueError("Tidak ada frame yang bisa dibaca.")
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        progress.close()
    print(f"Output: {output} | Frames: {index}")


if __name__ == "__main__":
    main()

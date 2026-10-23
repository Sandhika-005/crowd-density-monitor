import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from inference_utils import predict, visualize
from model import load_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--checkpoint", default="outputs/best_csrnet.pth")
    parser.add_argument("--output", default="outputs/photo_result.jpg")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, device)
    frame = cv2.imread(args.image)
    if frame is None:
        raise FileNotFoundError(args.image)
    count, density = predict(model, frame, device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), visualize(frame, count, density)):
        raise OSError(f"Gagal menyimpan gambar: {output}")
    np.save(output.with_suffix(".npy"), density)
    output.with_suffix(".json").write_text(json.dumps(dict(image=args.image, estimated_count=count), indent=2))
    print(f"Estimated count: {count:.2f} | Output: {output}")


if __name__ == "__main__":
    main()

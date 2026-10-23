import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import cKDTree
from tqdm import tqdm

from dataset import read_config


def density_map(height, width, points, mode="adaptive", fixed_sigma=4.0):
    h, w = height // 8, width // 8
    if min(h, w) < 1:
        raise ValueError("Ukuran gambar minimal 8 x 8.")
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(points).all():
        raise ValueError("Koordinat anotasi tidak valid.")
    scaled = points / 8.0
    density = np.zeros((h, w), dtype=np.float64)
    if mode == "adaptive" and len(points) > 1:
        distances, _ = cKDTree(scaled).query(scaled, k=min(4, len(points)))
        sigmas = np.maximum(0.3 * distances[:, 1:].mean(axis=1), 1.0)
    else:
        sigmas = np.full(len(points), 1.0 if mode == "adaptive" else fixed_sigma / 8.0)
    for (x, y), sigma in zip(scaled, sigmas):
        x, y = np.clip(x, 0, w - 1), np.clip(y, 0, h - 1)
        radius = max(1, int(np.ceil(3 * sigma)))
        cx, cy = int(round(x)), int(round(y))
        x1, x2 = max(0, cx - radius), min(w, cx + radius + 1)
        y1, y2 = max(0, cy - radius), min(h, cy + radius + 1)
        xx, yy = np.meshgrid(np.arange(x1, x2), np.arange(y1, y2))
        kernel = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
        density[y1:y2, x1:x2] += kernel / kernel.sum()
    density = density.astype(np.float32)
    if not np.isclose(density.sum(dtype=np.float64), len(points), atol=1e-3, rtol=0):
        raise ValueError("Density map tidak mempertahankan jumlah anotasi.")
    return density


def process(root, tag, config):
    paths = sorted((Path(root) / "images").glob("*.jpg"))
    if not paths:
        raise FileNotFoundError(f"Gambar tidak ditemukan: {root}")
    destination = Path(config["processed_dir"]) / "density" / tag
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for path in tqdm(paths, desc=tag):
        label = Path(root) / "ground-truth" / f"GT_{path.stem}.mat"
        points = loadmat(label)["image_info"][0, 0][0, 0][0]
        with Image.open(path) as image:
            width, height = image.size
        density = density_map(height, width, points, config["density_mode"], config["fixed_sigma"])
        density_path = destination / f"{path.stem}.npy"
        np.save(density_path, density)
        rows.append(dict(filename=path.name, image_path=str(path.resolve()), density_path=str(density_path.resolve()), real_count=len(points)))
    return rows


def write_manifest(path, rows):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["filename", "image_path", "density_path", "real_count"])
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_config(args.config)
    if not 0 < config["validation_ratio"] < 1 or config["fixed_sigma"] <= 0:
        raise ValueError("validation_ratio harus 0–1 dan sigma harus positif.")
    train = process(config["train_path"], "train", config)
    indices = np.random.default_rng(config["seed"]).permutation(len(train))
    n_val = max(1, int(len(train) * config["validation_ratio"]))
    if n_val >= len(train):
        raise ValueError("Data training tidak cukup.")
    splits = {"valid": [train[i] for i in indices[:n_val]], "train": [train[i] for i in indices[n_val:]]}
    splits["test"] = process(config["test_path"], "test", config)
    for name, rows in splits.items():
        write_manifest(Path(config["processed_dir"]) / f"{name}.csv", rows)
        print(f"{name}: {len(rows)} gambar")


if __name__ == "__main__":
    main()

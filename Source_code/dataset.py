import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def read_config(path):
    path = Path(path).resolve()
    config = json.loads(path.read_text())
    for key in ("train_path", "test_path", "processed_dir", "output_dir"):
        value = Path(config[key])
        config[key] = str(value if value.is_absolute() else path.parent / value)
    return config


class CrowdDataset(Dataset):
    def __init__(self, manifest):
        with open(manifest, newline="") as file:
            self.rows = list(csv.DictReader(file))
        if not self.rows:
            raise ValueError(f"Dataset kosong: {manifest}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        with Image.open(row["image_path"]) as image:
            image = TRANSFORM(image.convert("RGB"))
        density = torch.from_numpy(np.load(row["density_path"], allow_pickle=False)).unsqueeze(0)
        count = torch.tensor(float(row["real_count"]), dtype=torch.float32)
        return image, density, count, row["filename"]

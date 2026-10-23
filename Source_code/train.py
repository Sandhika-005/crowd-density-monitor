import argparse
import json
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CrowdDataset, read_config
from model import CSRNet


def run_epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_total, absolute, squared, n = 0.0, 0.0, 0.0, 0
    rows = []
    with torch.set_grad_enabled(optimizer is not None):
        for images, density, people, filenames in tqdm(loader, leave=False):
            images, density, people = images.to(device), density.to(device), people.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            prediction = model(images)
            if prediction.shape != density.shape:
                raise ValueError(f"Output {prediction.shape}, target {density.shape}")
            loss = torch.nn.functional.mse_loss(prediction, density)
            if not torch.isfinite(loss):
                raise FloatingPointError("Loss NaN atau Inf.")
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            counts = prediction.detach().sum(dim=(1, 2, 3))
            errors = counts - people
            loss_total += loss.item() * len(images)
            absolute += errors.abs().sum().item()
            squared += errors.square().sum().item()
            n += len(images)
            rows.extend(dict(filename=f, real_count=r, pred_count=p, absolute_error=abs(p - r)) for f, r, p in zip(filenames, people.cpu().tolist(), counts.cpu().tolist()))
    return dict(loss=loss_total / n, mae=absolute / n, rmse=(squared / n) ** 0.5), rows


def plot_history(history, best_epoch, output):
    frame = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, metric in zip(axes, ["loss", "mae", "rmse"]):
        for split in ["train", "val"]:
            ax.plot(frame["epoch"], frame[f"{split}_{metric}"], label=split)
        ax.axvline(best_epoch, color="gray", linestyle="--", label="Best epoch")
        ax.set(xlabel="Epoch", ylabel=metric.upper(), title=metric.upper())
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    config = read_config(args.config)
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    if config["epochs"] < 1:
        raise ValueError("epochs minimal 1.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2))
    loaders = {}
    for split in ["train", "valid", "test"]:
        ds = CrowdDataset(Path(config["processed_dir"]) / f"{split}.csv")
        loaders[split] = DataLoader(ds, batch_size=1, shuffle=split == "train", num_workers=config["num_workers"], pin_memory=device.type == "cuda")
    model = CSRNet(pretrained=config["pretrained"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    best, best_epoch, history = float("inf"), 0, []
    for epoch in range(1, config["epochs"] + 1):
        started = time.time()
        train, _ = run_epoch(model, loaders["train"], device, optimizer)
        valid, _ = run_epoch(model, loaders["valid"], device)
        improved = valid["mae"] < best
        if improved:
            best, best_epoch = valid["mae"], epoch
        state = dict(epoch=epoch, model_state=model.state_dict(), optimizer_state=optimizer.state_dict(), best_val_mae=best, config=config)
        torch.save(state, output / "last_csrnet.pth")
        if improved:
            torch.save(state, output / "best_csrnet.pth")
        history.append(dict(epoch=epoch, **{f"train_{k}": v for k, v in train.items()}, **{f"val_{k}": v for k, v in valid.items()}))
        pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
        print(f"Epoch {epoch} | Train MAE {train['mae']:.2f} | Val MAE {valid['mae']:.2f} | Val RMSE {valid['rmse']:.2f} | {time.time() - started:.1f}s")
    checkpoint = torch.load(output / "best_csrnet.pth", map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state"])
    metrics = {}
    for split in ["valid", "test"]:
        metrics[split], rows = run_epoch(model, loaders[split], device)
        pd.DataFrame(rows).to_csv(output / f"{split}_predictions.csv", index=False)
        print(f"{split.upper()}: {metrics[split]}")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_history(history, best_epoch, output)


if __name__ == "__main__":
    main()

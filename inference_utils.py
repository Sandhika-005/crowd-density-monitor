import cv2
import numpy as np
import torch
from PIL import Image

from dataset import TRANSFORM


@torch.inference_mode()
def predict(model, frame, device):
    if min(frame.shape[:2]) < 8:
        raise ValueError("Ukuran gambar minimal 8 x 8.")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(device)
    density = model(tensor)[0, 0].cpu().numpy()
    if not np.isfinite(density).all():
        raise FloatingPointError("Prediksi NaN atau Inf.")
    return float(density.sum(dtype=np.float64)), density


def visualize(frame, count, density):
    positive = np.maximum(density, 0)
    maximum = positive.max()
    heat = (positive / maximum * 255).astype(np.uint8) if maximum > 0 else np.zeros_like(positive, dtype=np.uint8)
    heat = cv2.resize(heat, (frame.shape[1], frame.shape[0]))
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.65, heat, 0.35, 0)
    cv2.putText(overlay, f"Estimated count: {count:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return overlay

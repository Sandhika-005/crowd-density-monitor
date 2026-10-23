import torch
from torch import nn
from torchvision import models


class CSRNet(nn.Module):
    def __init__(self, pretrained=False):
        super().__init__()
        weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        self.frontend = nn.Sequential(*list(models.vgg16(weights=weights).features.children())[:23])
        layers = []
        for _ in range(6):
            layers.extend([nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True)])
        self.backend = nn.Sequential(*layers)
        self.output = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        return self.output(self.backend(self.frontend(x)))


def load_model(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = CSRNet()
    model.load_state_dict(checkpoint.get("model_state", checkpoint))
    return model.to(device).eval()

import math

import torch
from torch import nn
from torchvision import models

BACKBONES = {
    "tiny": (models.convnext_tiny, models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1, 768),
    "small": (models.convnext_small, models.ConvNeXt_Small_Weights.IMAGENET1K_V1, 768),
    "base": (models.convnext_base, models.ConvNeXt_Base_Weights.IMAGENET1K_V1, 1024),
}
ECA_KERNEL = 3


def eca_kernel_size(channels, gamma=2, b=1):
    k = int(abs(math.log2(channels) / gamma + b / gamma))
    return k if k % 2 else k + 1


class ECA(nn.Module):
    def __init__(self, kernel_size=ECA_KERNEL):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        s = x.mean(dim=(2, 3))
        s = torch.sigmoid(self.conv(s[:, None]))
        return x * s.transpose(1, 2)[..., None]


class ConvNeXtECA(nn.Module):
    def __init__(self, arch="tiny", n_classes=3, pretrained=True):
        super().__init__()
        build, weights, self.n_features = BACKBONES[arch]
        backbone = build(weights=weights if pretrained else None)
        self.stages = backbone.features
        self.eca = ECA()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = backbone.classifier[0]
        self.head = nn.Linear(self.n_features, n_classes)

    def features(self, x):
        return self.norm(self.pool(self.eca(self.stages(x)))).flatten(1)

    def forward(self, x):
        return self.head(self.features(x))

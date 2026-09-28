"""DenseNet-121 for multi-label chest X-ray classification."""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import DenseNet121_Weights, densenet121


class DenseNet121MultiLabel(nn.Module):
    """``features -> ReLU -> global average pool -> Linear(1024, num_classes)``; outputs logits.

    torchvision's DenseNet applies its final ReLU functionally and in place inside
    ``forward``, which gets in the way of Grad-CAM hooks. Here that ReLU is a separate,
    non-in-place module, ``self.relu``: the natural Grad-CAM target layer (plan 3.11). The
    sigmoid is applied only at inference; training uses ``BCEWithLogitsLoss`` on the logits.
    """

    def __init__(self, num_classes: int = 14, pretrained: bool = True, memory_efficient: bool = False):
        super().__init__()
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights, memory_efficient=memory_efficient)
        self.features = backbone.features
        self.relu = nn.ReLU(inplace=False)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(backbone.classifier.in_features, num_classes)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.features(x))
        return self.classifier(torch.flatten(self.pool(x), 1))


def build_model(cfg: dict) -> DenseNet121MultiLabel:
    model_cfg = cfg["model"]
    if model_cfg["name"] != "densenet121":
        raise ValueError(f"Unknown model {model_cfg['name']!r}")
    return DenseNet121MultiLabel(
        num_classes=len(cfg["data"]["classes"]),
        pretrained=model_cfg["pretrained"],
        memory_efficient=model_cfg["memory_efficient"],
    )

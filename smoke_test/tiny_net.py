"""Minimal trigger-shaped network for the toolchain smoke test.

Input (1, 60, 12, 12) NCHW matches the per-time-slice detector tensor of the
2311.04983 reconstruction nets (60 vertical positions treated as channels over
the 12x12 horizontal grid). Only ops that mapped 100% to the Edge TPU in the
old project are used: conv / relu / pooling / fully-connected.
"""

import torch
from torch import nn

INPUT_SHAPE = (1, 60, 12, 12)


class TinyTriggerNet(nn.Module):
    def __init__(self, n_out: int = 8):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(60, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # fixed-kernel global pooling over the 12x12 grid: converts to a
            # native AVERAGE_POOL_2D (AdaptiveAvgPool2d decomposes to
            # TRANSPOSE+SUM, which quantizes less gracefully)
            nn.AvgPool2d(kernel_size=12),
        )
        self.head = nn.Linear(64, n_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x).flatten(1)
        return self.head(z)


def make_model(seed: int = 0) -> TinyTriggerNet:
    torch.manual_seed(seed)
    model = TinyTriggerNet()
    model.eval()
    return model

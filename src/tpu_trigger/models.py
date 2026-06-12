"""Trigger classifier variants for Coral Edge TPU deployment.

Input is (B, 16, 1, T) NCHW: 16 correlated time series as channels, time as
the single spatial (width) axis. Every (1, k) Conv2d is a 1D temporal conv
that mixes all channels. Only Edge-TPU-mappable ops are used: CONV_2D /
DEPTHWISE_CONV_2D, RELU, ADD (residual), AVERAGE_POOL_2D, FULLY_CONNECTED.
T must be static and divisible by 16.
"""

import torch
from torch import nn

N_CH = 16
N_CLASSES = 2


class _Block(nn.Sequential):
    """Conv(1,k) + BN + ReLU; BN folds into the conv at conversion."""

    def __init__(self, cin, cout, k=7, stride=1, dilation=1):
        pad = dilation * (k // 2)
        super().__init__(
            nn.Conv2d(cin, cout, (1, k), stride=(1, stride),
                      padding=(0, pad), dilation=(1, dilation), bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )


class _Residual(nn.Module):
    """x + Block(x): lowers to an ADD op (Edge TPU supported)."""

    def __init__(self, ch, k=5, dilation=1):
        super().__init__()
        self.body = _Block(ch, ch, k=k, dilation=dilation)

    def forward(self, x):
        return x + self.body(x)


def _ds_block(cin, cout, k=7, stride=1):
    """Depthwise (1,k) + pointwise 1x1, MobileNet-style."""
    return nn.Sequential(
        nn.Conv2d(cin, cin, (1, k), stride=(1, stride),
                  padding=(0, k // 2), groups=cin, bias=False),
        nn.BatchNorm2d(cin),
        nn.ReLU(inplace=True),
        nn.Conv2d(cin, cout, 1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class TriggerNet(nn.Module):
    def __init__(self, features, width, t_out, n_classes=N_CLASSES):
        super().__init__()
        self.features = features
        # fixed-kernel global pooling over time -> native AVERAGE_POOL_2D
        self.pool = nn.AvgPool2d((1, t_out))
        self.head = nn.Linear(width, n_classes)

    def forward(self, x):
        z = self.pool(self.features(x)).flatten(1)
        return self.head(z)


def make_model(variant: str, T: int = 256, n_ch: int = N_CH) -> TriggerNet:
    if T % 16 != 0:
        raise ValueError("T must be divisible by 16")
    if variant == "plain":
        f = nn.Sequential(
            _Block(n_ch, 32, k=7, stride=2),
            _Block(32, 64, k=7, stride=2),
            _Block(64, 64, k=5, stride=2),
            _Block(64, 128, k=5, stride=2),
        )
        return TriggerNet(f, 128, T // 16)
    if variant == "dilated":
        f = nn.Sequential(
            _Block(n_ch, 64, k=5, stride=2),
            _Residual(64, k=5, dilation=1),
            _Residual(64, k=5, dilation=2),
            _Residual(64, k=5, dilation=4),
            _Block(64, 96, k=5, stride=2),
            _Residual(96, k=5, dilation=1),
            _Residual(96, k=5, dilation=2),
            _Block(96, 128, k=5, stride=2),
        )
        return TriggerNet(f, 128, T // 8)
    if variant == "depthwise":
        f = nn.Sequential(
            _Block(n_ch, 32, k=7, stride=2),
            _ds_block(32, 64, stride=2),
            _ds_block(64, 96, stride=2),
            _ds_block(96, 128, stride=2),
        )
        return TriggerNet(f, 128, T // 16)
    raise ValueError(f"unknown variant {variant!r}")


VARIANTS = ("plain", "dilated", "depthwise")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

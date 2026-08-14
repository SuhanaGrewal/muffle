"""Phase 1 baseline: a small CNN over LFCC features, treating (freq, time) as a 1-channel
image. Purpose is to validate the full pipeline cheaply on CPU/MPS, not to be
state-of-the-art — Phase 2's frozen-SSL model is where generalization work happens.
"""

from __future__ import annotations

import torch
from torch import nn


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(torch.relu(self.bn(self.conv(x))))


class SpoofCNN(nn.Module):
    """Input: (batch, n_feat, time) LFCC features (from features/lfcc.py).
    Output: (batch, 2) logits — index 0 = bonafide, index 1 = spoof (see LABEL_TO_INT).
    """

    def __init__(self, n_feat: int = 180, hidden_channels: tuple[int, ...] = (16, 32, 64)):
        super().__init__()
        channels = [1, *hidden_channels]
        self.blocks = nn.Sequential(
            *[_ConvBlock(channels[i], channels[i + 1]) for i in range(len(hidden_channels))]
        )
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.classifier = nn.Linear(hidden_channels[-1], 2)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.unsqueeze(1)  # (batch, 1, n_feat, time)
        x = self.blocks(x)
        x = self.pool(x).flatten(1)  # (batch, hidden_channels[-1])
        return self.classifier(x)

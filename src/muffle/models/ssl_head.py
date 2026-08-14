"""Phase 2 model: a small trainable head on top of frozen SSL (wav2vec2/WavLM) hidden
states. Attentive pooling learns which frames matter most (e.g. voiced segments where
synthesis artifacts are most visible) instead of plain mean-pooling every frame equally.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentivePooling(nn.Module):
    """Learns a per-frame attention weight and returns the weighted mean over time."""

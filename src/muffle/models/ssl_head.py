"""Phase 2 model: a small trainable head on top of frozen SSL (wav2vec2/WavLM) hidden
states. Attentive pooling learns which frames matter most (e.g. voiced segments where
synthesis artifacts are most visible) instead of plain mean-pooling every frame equally.
"""

from __future__ import annotations

import torch
from torch import nn


class AttentivePooling(nn.Module):
    """Learns a per-frame attention weight and returns the weighted mean over time."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: (batch, time, hidden_size) -> pooled: (batch, hidden_size)."""
        weights = torch.softmax(self.attn(hidden_states), dim=1)  # (batch, time, 1)
        return (hidden_states * weights).sum(dim=1)


class SSLHeadClassifier(nn.Module):
    """Input: (batch, time, hidden_size) frozen SSL hidden states.
    Output: (batch, 2) logits -- index 0 = bonafide, index 1 = spoof.
    """

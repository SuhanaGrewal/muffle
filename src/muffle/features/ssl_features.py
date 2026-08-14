"""Frozen SSL (wav2vec2 / WavLM) feature extractor. The backbone is never fine-tuned --
full fine-tuning needs GPU budget this project doesn't assume -- only a small classifier
head is trained on top of its frozen hidden states.
"""

from __future__ import annotations

import torch

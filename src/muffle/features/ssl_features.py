"""Frozen SSL (wav2vec2 / WavLM) feature extractor. The backbone is never fine-tuned --
full fine-tuning needs GPU budget this project doesn't assume -- only a small classifier
head is trained on top of its frozen hidden states.
"""

from __future__ import annotations

import torch
from transformers import AutoFeatureExtractor, AutoModel


class SSLFeatureExtractor:
    def __init__(self, model_name: str = "microsoft/wavlm-base-plus", sample_rate: int = 16_000):
        self.sample_rate = sample_rate
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()

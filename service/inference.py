"""Loads a trained checkpoint once (not per-request) and scores raw audio bytes."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from muffle.data.datasets import _resample
from muffle.factory import build_feature_extractor, build_model


class InferenceEngine:
    """Owns one loaded model + extractor; construct once at app startup."""

    def __init__(self, config_path: str | Path, checkpoint_path: str | Path, device: str = "cpu"):
        cfg = yaml.safe_load(Path(config_path).read_text())
        self.device = torch.device(device)

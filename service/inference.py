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

        self.extractor = build_feature_extractor(cfg)
        if hasattr(self.extractor, "to"):
            self.extractor = self.extractor.to(self.device)

        self.model = build_model(cfg).to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.sample_rate = cfg["data"]["sample_rate"]
        self.window_seconds = cfg["data"]["duration_seconds"]
        self.model_version = cfg["model_name"]

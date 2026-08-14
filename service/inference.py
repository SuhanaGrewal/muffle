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

    @torch.no_grad()
    def predict(self, audio_bytes: bytes) -> dict:
        start = time.monotonic()

        waveform, file_sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        if file_sr != self.sample_rate:
            waveform = _resample(waveform, file_sr, self.sample_rate)

        window_len = int(self.sample_rate * self.window_seconds)
        n_windows = max(1, -(-len(waveform) // window_len))  # ceil division

        padded_len = n_windows * window_len
        n_repeats = padded_len // len(waveform) + 1
        padded = np.tile(waveform, n_repeats)[:padded_len]
        windows = padded.reshape(n_windows, window_len)

        batch = torch.from_numpy(windows.copy()).to(self.device)
        features = self.extractor(batch)
        logits = self.model(features)  # (n_windows, 2)

        # Mean-pool per-window scores rather than per-window votes, so one ambiguous
        # window doesn't flip the verdict on an otherwise-confident long clip.
        mean_logits = logits.mean(dim=0)
        score_raw = (mean_logits[0] - mean_logits[1]).item()

        probs = torch.softmax(mean_logits, dim=0)
        bonafide_prob = probs[0].item()
        verdict = "human" if bonafide_prob >= 0.5 else "ai_generated"
        confidence = bonafide_prob if verdict == "human" else 1 - bonafide_prob

        return {
            "verdict": verdict,
            "confidence": confidence,
            "score_raw": score_raw,
            "model_version": self.model_version,
            "processing_time_ms": (time.monotonic() - start) * 1000,
        }

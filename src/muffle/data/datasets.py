"""torch Dataset over a manifest CSV produced by manifests.py."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from torch.utils.data import Dataset

from muffle.augment import rawboost_augment

LABEL_TO_INT = {"bonafide": 0, "spoof": 1}


def _pad_or_trim(waveform: np.ndarray, target_len: int) -> np.ndarray:
    if len(waveform) >= target_len:
        start = 0 if target_len == len(waveform) else np.random.randint(0, len(waveform) - target_len + 1)
        return waveform[start : start + target_len]
    # RawBoost/ASVspoof-baseline-style repeat-padding rather than zero-padding, since
    # zero-padding introduces an artificial silence edge the model can learn to key off.
    n_repeats = target_len // len(waveform) + 1
    return np.tile(waveform, n_repeats)[:target_len]


class AudioManifestDataset(Dataset):
    """Loads (waveform, label, metadata) triples from a manifest CSV.

    Waveforms are resampled to `sample_rate` (if needed) and pad/trimmed to a fixed
    `duration_seconds` window so batches can be collated without ragged-length handling.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        sample_rate: int = 16_000,
        duration_seconds: float = 4.0,
        split: str | None = None,
        augment: bool = False,
    ):
        self.manifest = pd.read_csv(manifest_path)
        if split is not None:
            self.manifest = self.manifest[self.manifest["split"] == split].reset_index(drop=True)
        self.sample_rate = sample_rate
        self.target_len = int(sample_rate * duration_seconds)
        # RawBoost-style augmentation -- train split only (never dev/eval, which must
        # stay a clean, unperturbed measurement of the model's actual behavior).
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        row = self.manifest.iloc[idx]
        path = row["path"]

        # Seek-and-read only the needed window instead of decoding the whole file first --
        # matters a lot for datasets with long clips (e.g. DEEP-VOICE's up-to-10-minute
        # files): reading+resampling the full file per __getitem__ call made training
        # painfully slow, since only a few seconds of it are ever used.
        info = sf.info(path)
        file_sr = info.samplerate
        native_target_len = max(1, int(round(self.target_len * file_sr / self.sample_rate)))

        if info.frames > native_target_len:
            start = np.random.randint(0, info.frames - native_target_len + 1)
            waveform, _ = sf.read(
                path, start=start, frames=native_target_len, dtype="float32", always_2d=False
            )
        else:
            waveform, _ = sf.read(path, dtype="float32", always_2d=False)

        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # downmix to mono

        if file_sr != self.sample_rate:
            waveform = _resample(waveform, file_sr, self.sample_rate)

        waveform = _pad_or_trim(waveform, self.target_len)

        if self.augment:
            waveform = rawboost_augment(waveform)

        return {
            "waveform": torch.from_numpy(waveform.copy()),
            "label": torch.tensor(LABEL_TO_INT[row["label"]], dtype=torch.long),
            "dataset": row["dataset"],
            "attack_id": row["attack_id"] if pd.notna(row["attack_id"]) else None,
            "path": row["path"],
        }


def collate_batch(items: list[dict]) -> dict:
    """Default collate chokes on the `None` attack_id (bonafide clips have none) — stack
    the tensor fields and leave string/None metadata as plain lists.
    """
    return {
        "waveform": torch.stack([item["waveform"] for item in items]),
        "label": torch.stack([item["label"] for item in items]),
        "dataset": [item["dataset"] for item in items],
        "attack_id": [item["attack_id"] for item in items],
        "path": [item["path"] for item in items],
    }


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    import torchaudio

    tensor = torch.from_numpy(waveform).unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()

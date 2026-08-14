"""LFCC (Linear Frequency Cepstral Coefficient) front-end, the classic ASVspoof-baseline
feature: a linear-spaced filterbank + DCT (vs. mel-spaced for MFCC), which preserves
high-frequency detail where synthesis artifacts tend to show up. Static coefficients are
augmented with delta and delta-delta (velocity/acceleration) features, matching the
standard anti-spoofing feature-extraction recipe.
"""

from __future__ import annotations

import torch
import torchaudio


class LFCCExtractor:
    def __init__(
        self,
        sample_rate: int = 16_000,
        n_lfcc: int = 60,
        n_filter: int = 128,
        win_length_ms: float = 20.0,
        hop_length_ms: float = 10.0,
    ):
        n_fft = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)

        self._lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_lfcc=n_lfcc,
            n_filter=n_filter,
            speckwargs={"n_fft": n_fft, "hop_length": hop_length, "win_length": n_fft},
        )

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        """waveform: (batch, samples) or (samples,) -> features: (batch, 3*n_lfcc, time)."""
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        static = self._lfcc(waveform)  # (batch, n_lfcc, time)
        delta = torchaudio.functional.compute_deltas(static)
        delta2 = torchaudio.functional.compute_deltas(delta)

        return torch.cat([static, delta, delta2], dim=1)  # (batch, 3*n_lfcc, time)

    def to(self, device: torch.device) -> "LFCCExtractor":
        # torchaudio's LFCC is an nn.Module (registers a window/DCT matrix as buffers) --
        # without this, its buffers stay on CPU while input waveforms move to MPS/CUDA,
        # and torch.stft errors on the device mismatch.
        self._lfcc = self._lfcc.to(device)
        return self

"""RawBoost-inspired raw-waveform augmentation (Tak et al., ASVspoof2021 baseline),
applied during training only. This is a from-description reimplementation of RawBoost's
three noise families, not a port of the reference code -- the goal is the same mechanism,
not byte-identical output.

Why this, now: every model version this project trained (v1-v5) exhibited the same
failure -- the small trainable head learned to key off incidental recording-condition
artifacts (sample rate/bit-depth history in v2-v4's garystafford data, audio cleanliness
in v5's LibriSpeech data) rather than genuine synthesis cues. RawBoost directly attacks
that: by randomly distorting the channel/noise conditions of every training clip, it
makes those incidental artifacts an unreliable signal, forcing the head to rely on
whatever's left. Reported effect on ASVspoof2021 LA (cross-condition) EER: 9.50% -> 5.31%
with RawNet2 trained on ASVspoof2019 LA only (arxiv 2111.04433).
"""

from __future__ import annotations

import numpy as np


def _linear_convolutive_noise(waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Simulates a random channel/microphone impulse response by convolving with a
    short random FIR filter whose coefficients decay in magnitude.
    """
    n_taps = rng.integers(3, 8)
    taps = rng.normal(0, 1, size=n_taps) * np.exp(-np.arange(n_taps) * rng.uniform(0.3, 1.0))
    taps /= np.abs(taps).sum() + 1e-8
    return np.convolve(waveform, taps, mode="same").astype(np.float32)


def _impulsive_signal_dependent_noise(waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sparse impulsive bursts whose amplitude scales with the local signal -- simulates
    transient noise (clicks, pops) rather than a constant noise floor.
    """
    mask = rng.random(waveform.shape) < 0.005
    bursts = rng.normal(0, 1, size=waveform.shape) * np.abs(waveform) * rng.uniform(2, 6)
    return (waveform + mask * bursts).astype(np.float32)


def _stationary_signal_independent_noise(waveform: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Constant-level background noise (colored via a random low-order IIR-ish smoothing),
    independent of the signal -- simulates ambient/room noise floor.
    """
    target_snr_db = rng.uniform(10, 30)
    signal_power = np.mean(waveform**2) + 1e-8
    noise = rng.normal(0, 1, size=waveform.shape)
    noise = np.convolve(noise, np.ones(3) / 3, mode="same")  # light smoothing -> not pure white
    noise_power = np.mean(noise**2) + 1e-8
    scale = np.sqrt(signal_power / (10 ** (target_snr_db / 10)) / noise_power)
    return (waveform + noise * scale).astype(np.float32)


_AUGMENTATIONS = [
    _linear_convolutive_noise,
    _impulsive_signal_dependent_noise,
    _stationary_signal_independent_noise,
]


def rawboost_augment(waveform: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """Applies a random subset (1-3, order shuffled) of the three RawBoost-style
    augmentations to one waveform. Always returns a new array; peak-normalizes back to
    the original max amplitude afterward so augmentation can't trivially change overall
    loudness into a new shortcut of its own.
    """
    rng = rng or np.random.default_rng()
    original_peak = np.abs(waveform).max() + 1e-8

    chosen = [aug for aug in _AUGMENTATIONS if rng.random() < 0.7]
    if not chosen:
        chosen = [rng.choice(_AUGMENTATIONS)]
    rng.shuffle(chosen)

    augmented = waveform.copy()
    for aug in chosen:
        augmented = aug(augmented, rng)

    new_peak = np.abs(augmented).max() + 1e-8
    return (augmented * (original_peak / new_peak)).astype(np.float32)

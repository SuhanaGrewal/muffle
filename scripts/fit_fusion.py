"""Fits a logistic-regression fusion layer on dev-set CNN+WavLM scores (learns each
model's optimal weight instead of assuming a plain average), then applies it to the
held-out eval subset and reports EER -- fitting and evaluating on the same rows would
be leakage, so this always needs two disjoint score pairs.

    python scripts/fit_fusion.py \\
        --dev-cnn-scores scores/cnn_dev_scores.npz --dev-wavlm-scores scores/wavlm_dev_scores.npz \\
        --eval-cnn-scores scores/cnn_subset_scores.npz --eval-wavlm-scores scores/wavlm_subset_scores.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from muffle.metrics import compute_eer


def normalize(scores: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (scores - mean) / std


def load_pair(cnn_path: Path, wavlm_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cnn = np.load(cnn_path, allow_pickle=True)
    wavlm = np.load(wavlm_path, allow_pickle=True)
    if not np.array_equal(cnn["labels"], wavlm["labels"]) or not np.array_equal(
        cnn["attack_ids"], wavlm["attack_ids"]
    ):
        raise ValueError(f"{cnn_path} and {wavlm_path} aren't row-aligned.")
    return cnn["scores"], wavlm["scores"], cnn["labels"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-cnn-scores", required=True, type=Path)
    parser.add_argument("--dev-wavlm-scores", required=True, type=Path)
    parser.add_argument("--eval-cnn-scores", required=True, type=Path)
    parser.add_argument("--eval-wavlm-scores", required=True, type=Path)
    args = parser.parse_args()

    dev_cnn, dev_wavlm, dev_labels = load_pair(args.dev_cnn_scores, args.dev_wavlm_scores)
    eval_cnn, eval_wavlm, eval_labels = load_pair(args.eval_cnn_scores, args.eval_wavlm_scores)

    # Normalize using dev-set statistics only -- the eval set must never influence fitting.
    cnn_mean, cnn_std = dev_cnn.mean(), dev_cnn.std()
    wavlm_mean, wavlm_std = dev_wavlm.mean(), dev_wavlm.std()

    dev_features = np.column_stack(
        [normalize(dev_cnn, cnn_mean, cnn_std), normalize(dev_wavlm, wavlm_mean, wavlm_std)]
    )
    eval_features = np.column_stack(
        [normalize(eval_cnn, cnn_mean, cnn_std), normalize(eval_wavlm, wavlm_mean, wavlm_std)]
    )

    # bonafide=0, spoof=1 (LABEL_TO_INT); flip so higher score = more bonafide-like,
    # consistent with the raw model scores fed into it.
    fusion = LogisticRegression()
    fusion.fit(dev_features, 1 - dev_labels)
    fused_scores = fusion.decision_function(eval_features)

    eval_bonafide = eval_labels == 0
    eer, threshold = compute_eer(fused_scores[eval_bonafide], fused_scores[~eval_bonafide])

    print(f"Fusion weights -- CNN: {fusion.coef_[0][0]:.3f}, WavLM: {fusion.coef_[0][1]:.3f}, intercept: {fusion.intercept_[0]:.3f}")
    print(f"Fused (logistic regression) EER: {eer * 100:.3f}%  (threshold={threshold:.3f})")


if __name__ == "__main__":
    main()

"""Combines CNN and WavLM scores (produced by score_for_ensemble.py on the same manifest)
into an ensemble EER, alongside each model's standalone EER for comparison.

    python scripts/ensemble_eval.py \\
        --cnn-scores scores/cnn_subset_scores.npz \\
        --wavlm-scores scores/wavlm_subset_scores.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from muffle.metrics import compute_eer


def normalize(scores: np.ndarray) -> np.ndarray:
    return (scores - scores.mean()) / scores.std()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnn-scores", required=True, type=Path)
    parser.add_argument("--wavlm-scores", required=True, type=Path)
    args = parser.parse_args()

    cnn = np.load(args.cnn_scores, allow_pickle=True)
    wavlm = np.load(args.wavlm_scores, allow_pickle=True)

    if not np.array_equal(cnn["labels"], wavlm["labels"]) or not np.array_equal(
        cnn["attack_ids"], wavlm["attack_ids"]
    ):
        raise ValueError(
            "Score files aren't row-aligned -- both must be scored against the exact same manifest."
        )

    bonafide = cnn["labels"] == 0  # LABEL_TO_INT: bonafide=0, spoof=1

    ensemble_scores = (normalize(cnn["scores"]) + normalize(wavlm["scores"])) / 2

    for name, scores in [
        ("CNN", cnn["scores"]),
        ("WavLM", wavlm["scores"]),
        ("Ensemble (normalized average)", ensemble_scores),
    ]:
        eer, threshold = compute_eer(scores[bonafide], scores[~bonafide])
        print(f"{name:30s} EER: {eer * 100:.3f}%  (threshold={threshold:.3f})")


if __name__ == "__main__":
    main()

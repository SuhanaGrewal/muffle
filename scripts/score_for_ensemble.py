"""Scores a trained checkpoint against a manifest and saves raw (scores, labels,
attack_ids) to an .npz file -- the building block for ensembling multiple models'
predictions, which needs each model scored on the exact same rows to combine per-file.

    python scripts/score_for_ensemble.py --config configs/ssl_wavlm_head.yaml \\
        --checkpoint checkpoints/ssl_wavlm_head/best.pt \\
        --manifest data/processed/eval_subset_for_comparison.csv \\
        --out /tmp/wavlm_scores.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from muffle.data.datasets import AudioManifestDataset, collate_batch
from muffle.factory import build_feature_extractor, build_model
from muffle.train import resolve_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(cfg["train"]["device"])

    extractor = build_feature_extractor(cfg)
    if hasattr(extractor, "to"):
        extractor = extractor.to(device)
    model = build_model(cfg).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

"""Evaluation entrypoint: reports EER overall and broken down by attack type, and (with
--cross-dataset-manifest) evaluates a trained checkpoint against a different dataset's
manifest with no fine-tuning, to measure the generalization gap that's the real test of
this kind of detector.

    python -m muffle.evaluate --config configs/baseline_lfcc_cnn.yaml \\
        --checkpoint checkpoints/baseline_lfcc_cnn/best.pt --split eval
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from muffle.data.datasets import AudioManifestDataset, collate_batch
from muffle.features.lfcc import LFCCExtractor
from muffle.metrics import compute_eer
from muffle.models.cnn_baseline import SpoofCNN
from muffle.train import resolve_device


@torch.no_grad()
def score_manifest(model, extractor, manifest_path, cfg, device, split: str):
    ds = AudioManifestDataset(
        manifest_path,
        sample_rate=cfg["data"]["sample_rate"],
        duration_seconds=cfg["data"]["duration_seconds"],
        split=split,
    )
    loader = DataLoader(
        ds, batch_size=cfg["train"]["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_batch
    )

    model.eval()
    scores, labels, attack_ids = [], [], []
    for batch in loader:
        waveforms = batch["waveform"].to(device)
        features = extractor(waveforms)
        logits = model(features)
        batch_scores = (logits[:, 0] - logits[:, 1]).cpu().numpy()
        scores.append(batch_scores)
        labels.append(batch["label"].numpy())
        attack_ids.extend(batch["attack_id"])

    return np.concatenate(scores), np.concatenate(labels), attack_ids


def report_eer(scores, labels, attack_ids, dataset_name: str) -> float:
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]
    eer, _ = compute_eer(bonafide_scores, spoof_scores)
    print(f"\n=== {dataset_name} ===")
    print(f"overall EER: {eer:.4%}  (n_bonafide={len(bonafide_scores)}, n_spoof={len(spoof_scores)})")

    per_attack = {}
    for score, label, attack_id in zip(scores, labels, attack_ids):
        if label == 1 and attack_id is not None:
            per_attack.setdefault(attack_id, []).append(score)
    if per_attack:
        print("per-attack-type EER (each vs. all bonafide):")
        for attack_id in sorted(per_attack):
            attack_scores = np.array(per_attack[attack_id])
            attack_eer, _ = compute_eer(bonafide_scores, attack_scores)
            print(f"  {attack_id}: {attack_eer:.4%} (n={len(attack_scores)})")

    return eer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split", default="eval")
    parser.add_argument(
        "--cross-dataset-manifest",
        action="append",
        default=[],
        help="Additional manifest CSV(s) to evaluate the same checkpoint against, "
        "e.g. --cross-dataset-manifest data/processed/wavefake_manifest.csv. "
        "Repeatable.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    device = resolve_device(cfg["train"]["device"])

    extractor = LFCCExtractor(
        sample_rate=cfg["data"]["sample_rate"],
        n_lfcc=cfg["features"]["n_lfcc"],
        n_filter=cfg["features"]["n_filter"],
        win_length_ms=cfg["features"]["win_length_ms"],
        hop_length_ms=cfg["features"]["hop_length_ms"],
    )
    model = SpoofCNN(
        n_feat=cfg["features"]["n_lfcc"] * 3,
        hidden_channels=tuple(cfg["model"]["hidden_channels"]),
    ).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    scores, labels, attack_ids = score_manifest(
        model, extractor, cfg["data"]["manifest_path"], cfg, device, args.split
    )
    report_eer(scores, labels, attack_ids, dataset_name=f"in-domain ({args.split})")

    for manifest_path in args.cross_dataset_manifest:
        scores, labels, attack_ids = score_manifest(
            model, extractor, manifest_path, cfg, device, split=args.split
        )
        report_eer(scores, labels, attack_ids, dataset_name=f"cross-dataset: {manifest_path}")


if __name__ == "__main__":
    main()

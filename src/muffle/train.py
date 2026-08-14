"""Training entrypoint for the Phase 1 LFCC+CNN baseline.

    python -m muffle.train --config configs/baseline_lfcc_cnn.yaml
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
from muffle.features.lfcc import LFCCExtractor
from muffle.metrics import compute_eer
from muffle.models.cnn_baseline import SpoofCNN


def resolve_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_dataloaders(cfg: dict) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_ds = AudioManifestDataset(
        data_cfg["manifest_path"],
        sample_rate=data_cfg["sample_rate"],
        duration_seconds=data_cfg["duration_seconds"],
        split="train",
    )
    dev_ds = AudioManifestDataset(
        data_cfg["manifest_path"],
        sample_rate=data_cfg["sample_rate"],
        duration_seconds=data_cfg["duration_seconds"],
        split="dev",
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=True,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_batch,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=cfg["train"]["batch_size"],
        shuffle=False,
        num_workers=cfg["train"]["num_workers"],
        collate_fn=collate_batch,
    )
    return train_loader, dev_loader


def train_one_epoch(model, extractor, loader, optimizer, criterion, device) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for batch in tqdm(loader, desc="train", leave=False):
        waveforms = batch["waveform"].to(device)
        labels = batch["label"].to(device)

        features = extractor(waveforms)
        logits = model(features)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate_split(model, extractor, loader, device) -> float:
    """Returns EER on the given split. Countermeasure score = logit(bonafide) - logit(spoof),
    so higher score means more bonafide-like, matching compute_eer's target/nontarget convention.
    """
    model.eval()
    scores, labels = [], []
    for batch in tqdm(loader, desc="eval", leave=False):
        waveforms = batch["waveform"].to(device)
        features = extractor(waveforms)
        logits = model(features)
        batch_scores = (logits[:, 0] - logits[:, 1]).cpu().numpy()
        scores.append(batch_scores)
        labels.append(batch["label"].numpy())

    scores = np.concatenate(scores)
    labels = np.concatenate(labels)
    bonafide_scores = scores[labels == 0]
    spoof_scores = scores[labels == 1]
    eer, _ = compute_eer(bonafide_scores, spoof_scores)
    return eer


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    torch.manual_seed(cfg["train"]["seed"])

    device = resolve_device(cfg["train"]["device"])
    print(f"Using device: {device}")

    train_loader, dev_loader = build_dataloaders(cfg)

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

    class_weights = torch.tensor(cfg["train"]["class_weights"], dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"]
    )

    checkpoint_dir = Path(cfg["train"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    best_eer = float("inf")
    for epoch in range(cfg["train"]["epochs"]):
        train_loss = train_one_epoch(model, extractor, train_loader, optimizer, criterion, device)
        dev_eer = evaluate_split(model, extractor, dev_loader, device)
        print(f"epoch {epoch}: train_loss={train_loss:.4f} dev_eer={dev_eer:.4%}")

        torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_dir / "last.pt")
        if dev_eer < best_eer:
            best_eer = dev_eer
            torch.save({"model_state": model.state_dict(), "config": cfg}, checkpoint_dir / "best.pt")
            print(f"  new best dev_eer={best_eer:.4%}, saved to {checkpoint_dir / 'best.pt'}")

    print(f"Training done. Best dev EER: {best_eer:.4%}")


if __name__ == "__main__":
    main()

"""Build unified train/dev/eval manifests (path, label, dataset, attack_id, split) from each
dataset's own protocol/metadata format, so downstream code (Dataset classes, cross-dataset
evaluation) never has to know per-dataset file formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["path", "label", "dataset", "attack_id", "speaker_id", "split"]

# ASVspoof2019 LA protocol filenames per split, relative to the dataset root.
_ASVSPOOF2019_LA_PROTOCOLS = {
    "train": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}
_ASVSPOOF2019_LA_AUDIO_DIRS = {
    "train": "ASVspoof2019_LA_train/flac",
    "dev": "ASVspoof2019_LA_dev/flac",
    "eval": "ASVspoof2019_LA_eval/flac",
}


def build_asvspoof2019_la_manifest(dataset_root: Path) -> pd.DataFrame:
    """Parse ASVspoof2019 LA's protocol files.

    Each protocol line is whitespace-separated:
        SPEAKER_ID  AUDIO_FILE_NAME  ENVIRONMENT_ID(-)  ATTACK_ID(- or A01..A19)  KEY(bonafide/spoof)
    """
    rows = []
    for split, protocol_rel_path in _ASVSPOOF2019_LA_PROTOCOLS.items():
        protocol_path = dataset_root / protocol_rel_path
        if not protocol_path.exists():
            raise FileNotFoundError(
                f"Missing protocol file for split={split!r}: {protocol_path}\n"
                "Run scripts/download_asvspoof2019.sh and check the extracted layout "
                "matches the comment at the top of that script."
            )

        audio_dir = dataset_root / _ASVSPOOF2019_LA_AUDIO_DIRS[split]

        with protocol_path.open() as f:
            for line in f:
                parts = line.split()
                if len(parts) != 5:
                    continue
                speaker_id, filename, _env_id, attack_id, key = parts
                rows.append(
                    {
                        "path": str(audio_dir / f"{filename}.flac"),
                        "label": key,  # "bonafide" or "spoof"
                        "dataset": "asvspoof2019_la",
                        "attack_id": None if attack_id == "-" else attack_id,
                        "speaker_id": speaker_id,
                        "split": split,
                    }
                )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        raise ValueError(f"Parsed zero rows from protocol files under {dataset_root}")
    return manifest


def build_deep_voice_manifest(dataset_root: Path) -> pd.DataFrame:
    """DEEP-VOICE has no official splits and only 64 files total (8 real, 56 RVC fakes) --
    too few files, and too speaker-imbalanced, for a speaker-disjoint split to mean much.
    Uses a naive deterministic split (last file per class -> eval, second-to-last -> dev,
    rest -> train) since this dataset's role is a cheap real-audio sanity check, not the
    statistically rigorous training run.
    """
    real_files = sorted((dataset_root / "REAL").glob("*.wav"))
    fake_files = sorted((dataset_root / "FAKE").glob("*.wav"))

    if not real_files or not fake_files:
        raise FileNotFoundError(
            f"Expected {dataset_root}/REAL/*.wav and {dataset_root}/FAKE/*.wav -- "
            "run scripts/download_deep_voice.sh and check the extracted layout."
        )

    def split_for(index: int, count: int) -> str:
        if index == count - 1:
            return "eval"
        if index == count - 2:
            return "dev"
        return "train"

    rows = []
    for i, path in enumerate(real_files):
        rows.append(
            {
                "path": str(path),
                "label": "bonafide",
                "dataset": "deep_voice",
                "attack_id": None,
                "speaker_id": path.stem,
                "split": split_for(i, len(real_files)),
            }
        )


_BUILDERS = {
    "asvspoof2019_la": build_asvspoof2019_la_manifest,
    "deep_voice": build_deep_voice_manifest,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(_BUILDERS))
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw"),
        help="Parent directory containing data/raw/<dataset>/",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output CSV path (default: data/processed/<dataset>_manifest.csv)",
    )
    args = parser.parse_args()

    dataset_root = args.data_root / args.dataset
    manifest = _BUILDERS[args.dataset](dataset_root)

    out_path = args.out or Path("data/processed") / f"{args.dataset}_manifest.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    counts = manifest.groupby(["split", "label"]).size()
    print(f"Wrote {len(manifest)} rows to {out_path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()

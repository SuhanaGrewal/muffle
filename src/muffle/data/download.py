"""Verify a dataset's extracted layout before manifests.py tries to parse it.

Downloading itself is a manual, click-through step (see scripts/download_asvspoof2019.sh
and the README's dataset table for why) — this module only checks that what landed on
disk looks like what the rest of the pipeline expects, so a bad/partial extraction fails
fast with a clear message instead of surfacing as a confusing FileNotFoundError deep in
manifest parsing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Rough expected file counts per ASVspoof2019 LA split (train/dev/eval flac counts),
# confirmed against a real extraction (25,380 / 24,986 / 43,638) -- used as a sanity
# check, not an exact match, since it should catch "extraction was truncated" without
# being brittle to minor deltas. Paths include the LA.zip's LA/ wrapper folder.
_ASVSPOOF2019_LA_EXPECTED_MIN_FILES = {
    "LA/ASVspoof2019_LA_train/flac": 20_000,
    "LA/ASVspoof2019_LA_dev/flac": 20_000,
    "LA/ASVspoof2019_LA_eval/flac": 35_000,
}
_ASVSPOOF2019_LA_REQUIRED_PROTOCOLS = [
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
]


def verify_asvspoof2019_la(dataset_root: Path) -> list[str]:
    """Return a list of problems found (empty list == looks good)."""
    problems = []

    for protocol_rel in _ASVSPOOF2019_LA_REQUIRED_PROTOCOLS:
        if not (dataset_root / protocol_rel).exists():
            problems.append(f"missing protocol file: {protocol_rel}")

    for audio_dir_rel, min_count in _ASVSPOOF2019_LA_EXPECTED_MIN_FILES.items():
        audio_dir = dataset_root / audio_dir_rel
        if not audio_dir.is_dir():
            problems.append(f"missing audio directory: {audio_dir_rel}")
            continue
        n_files = sum(1 for _ in audio_dir.glob("*.flac"))
        if n_files < min_count:
            problems.append(
                f"{audio_dir_rel} has only {n_files} .flac files, "
                f"expected at least ~{min_count} — extraction may be incomplete"
            )

    return problems


def verify_deep_voice(dataset_root: Path) -> list[str]:
    """Return a list of problems found (empty list == looks good)."""
    problems = []
    for subdir, min_count in (("REAL", 4), ("FAKE", 30)):
        audio_dir = dataset_root / "KAGGLE" / "AUDIO" / subdir
        if not audio_dir.is_dir():
            problems.append(f"missing audio directory: KAGGLE/AUDIO/{subdir}")
            continue
        n_files = sum(1 for _ in audio_dir.glob("*.wav"))
        if n_files < min_count:
            problems.append(
                f"KAGGLE/AUDIO/{subdir} has only {n_files} .wav files, expected at least ~{min_count}"
            )
    return problems


def verify_in_the_wild(dataset_root: Path) -> list[str]:
    """Return a list of problems found (empty list == looks good). In-the-Wild's zip
    extracts to release_in_the_wild.zip's top-level meta.csv + N.wav files -- unlike the
    other datasets, this is real-world audio never used for training, only as a held-out
    cross-dataset generalization benchmark (see README).
    """
    problems = []
    meta_csv = dataset_root / "release_in_the_wild" / "meta.csv"
    if not meta_csv.exists():
        problems.append(f"missing metadata file: {meta_csv.relative_to(dataset_root.parent.parent)}")
        return problems

    audio_dir = meta_csv.parent
    n_files = sum(1 for _ in audio_dir.glob("*.wav"))
    if n_files < 20_000:
        problems.append(
            f"{audio_dir} has only {n_files} .wav files, expected ~31,779 -- extraction may be incomplete"
        )
    return problems


_VERIFIERS = {
    "asvspoof2019_la": verify_asvspoof2019_la,
    "deep_voice": verify_deep_voice,
    "in_the_wild": verify_in_the_wild,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=sorted(_VERIFIERS))
    parser.add_argument("--data-root", type=Path, default=Path("data/raw"))
    args = parser.parse_args()

    dataset_root = args.data_root / args.dataset
    problems = _VERIFIERS[args.dataset](dataset_root)

    if problems:
        print(f"Problems found under {dataset_root}:")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print(f"{args.dataset} layout looks correct at {dataset_root}")


if __name__ == "__main__":
    main()

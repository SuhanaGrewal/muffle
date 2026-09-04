"""Build unified train/dev/eval manifests (path, label, dataset, attack_id, split) from each
dataset's own protocol/metadata format, so downstream code (Dataset classes, cross-dataset
evaluation) never has to know per-dataset file formats.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

MANIFEST_COLUMNS = ["path", "label", "dataset", "attack_id", "speaker_id", "split"]

# ASVspoof2019 LA protocol filenames per split, relative to the dataset root. The zip
# extracts with an extra LA/ wrapper folder (LA.zip -> LA/ASVspoof2019_LA_train/... ),
# discovered only once a real download landed -- not documented anywhere we could see
# in advance.
_ASVSPOOF2019_LA_PROTOCOLS = {
    "train": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt",
    "dev": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt",
    "eval": "LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt",
}
_ASVSPOOF2019_LA_AUDIO_DIRS = {
    "train": "LA/ASVspoof2019_LA_train/flac",
    "dev": "LA/ASVspoof2019_LA_dev/flac",
    "eval": "LA/ASVspoof2019_LA_eval/flac",
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
    real_files = sorted((dataset_root / "KAGGLE" / "AUDIO" / "REAL").glob("*.wav"))
    fake_files = sorted((dataset_root / "KAGGLE" / "AUDIO" / "FAKE").glob("*.wav"))

    if not real_files or not fake_files:
        raise FileNotFoundError(
            f"Expected {dataset_root}/KAGGLE/AUDIO/REAL/*.wav and .../FAKE/*.wav -- "
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
    for i, path in enumerate(fake_files):
        rows.append(
            {
                "path": str(path),
                "label": "spoof",
                "dataset": "deep_voice",
                "attack_id": "rvc",
                "speaker_id": path.stem,
                "split": split_for(i, len(fake_files)),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


_GARYSTAFFORD_PREFIX_TO_ATTACK_ID = {
    "el": "elevenlabs",
    "po": "amazon_polly",
    "hg": "hexgrad_kokoro",
    "hu": "hume_ai",
    "lv": "luvvoice",
    "sp": "speechify",
}


def build_garystafford_manifest(dataset_root: Path) -> pd.DataFrame:
    """garystafford/deepfake-audio-detection, materialized locally by
    scripts/materialize_garystafford.py. Unlike DEEP-VOICE, this has enough samples
    (933 real, 933 fake) for a real proportional 80/10/10 split rather than a
    last-few-files placeholder.

    Each fake filename is prefixed with which TTS platform made it (e.g. el_0001_...flac
    = ElevenLabs) -- mapped to a real attack_id per platform instead of one generic
    "commercial_tts" label, so subsampling/analysis can tell platforms apart.
    """
    real_files = sorted((dataset_root / "real").glob("*.flac"))
    fake_files = sorted((dataset_root / "fake").glob("*.flac"))

    if not real_files or not fake_files:
        raise FileNotFoundError(
            f"Expected {dataset_root}/real/*.flac and {dataset_root}/fake/*.flac -- "
            "run scripts/materialize_garystafford.py first."
        )

    def split_for(index: int) -> str:
        if index % 10 == 0:
            return "eval"
        if index % 10 == 1:
            return "dev"
        return "train"

    rows = []
    for i, path in enumerate(real_files):
        rows.append(
            {
                "path": str(path),
                "label": "bonafide",
                "dataset": "garystafford",
                "attack_id": None,
                "speaker_id": None,
                "split": split_for(i),
            }
        )
    for i, path in enumerate(fake_files):
        prefix = path.stem.split("_")[0]
        attack_id = _GARYSTAFFORD_PREFIX_TO_ATTACK_ID.get(prefix, "commercial_tts_unknown")
        rows.append(
            {
                "path": str(path),
                "label": "spoof",
                "dataset": "garystafford",
                "attack_id": attack_id,
                "speaker_id": None,
                "split": split_for(i),
            }
        )

    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


def build_in_the_wild_manifest(dataset_root: Path) -> pd.DataFrame:
    """In-the-Wild (Muller et al.) -- real-world deepfakes of public figures. Every row
    is split="eval": this dataset is never trained on, only used as a held-out
    cross-dataset generalization benchmark (see README's "Why this is hard" section).
    """
    release_dir = dataset_root / "release_in_the_wild"
    meta_csv = release_dir / "meta.csv"
    if not meta_csv.exists():
        raise FileNotFoundError(
            f"Missing {meta_csv} -- run scripts/download_in_the_wild.sh and check the extracted layout."
        )

    meta = pd.read_csv(meta_csv)
    label_map = {"bona-fide": "bonafide", "spoof": "spoof"}

    rows = []
    for _, row in meta.iterrows():
        rows.append(
            {
                "path": str(release_dir / row["file"]),
                "label": label_map[row["label"]],
                "dataset": "in_the_wild",
                "attack_id": None,
                "speaker_id": row.get("speaker"),
                "split": "eval",
            }
        )

    manifest = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    if manifest.empty:
        raise ValueError(f"Parsed zero rows from {meta_csv}")
    return manifest


_BUILDERS = {
    "asvspoof2019_la": build_asvspoof2019_la_manifest,
    "deep_voice": build_deep_voice_manifest,
    "garystafford": build_garystafford_manifest,
    "in_the_wild": build_in_the_wild_manifest,
}


def combine_manifests(manifest_paths: list[Path]) -> pd.DataFrame:
    """Concatenate several per-dataset manifests into one training manifest. Each
    input keeps its own `split` column values (already train/dev/eval per-dataset),
    so combining datasets doesn't change any individual dataset's split assignment --
    it just pools them under the same split label for a bigger, more diverse training set.
    """
    frames = [pd.read_csv(p) for p in manifest_paths]
    combined = pd.concat(frames, ignore_index=True)
    return combined[MANIFEST_COLUMNS]


def subsample_manifest(df: pd.DataFrame, max_per_group: int, seed: int = 0) -> pd.DataFrame:
    """Cap each (split, label, dataset) group at `max_per_group` rows via random sampling,
    so a heavier model (e.g. frozen-SSL) can do a quick trial run in a fraction of the
    time -- at the cost of a noisier, less statistically robust result than the full
    dataset.

    Grouping by dataset (not just split/label) matters: pooling across datasets before
    capping lets a huge dataset (ASVspoof2019 LA's ~23k train spoof rows) crowd out a
    small-but-diverse one (garystafford's ~750, spanning 6 different TTS platforms) down
    to almost nothing by pure chance, even though both are capped at the same ceiling.
    Stratifying by dataset means a source under the cap contributes everything it has.
    """
    # Iterate the groupby explicitly rather than .apply(): pandas 2.2+ excludes the
    # grouping columns from what .apply() passes to the callback by default, which
    # silently dropped them here.
    parts = [
        group.sample(n=min(len(group), max_per_group), random_state=seed)
        for _, group in df.groupby(["split", "label", "dataset"])
    ]
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(_BUILDERS), help="Build one dataset's manifest")
    parser.add_argument(
        "--combine",
        nargs="+",
        type=Path,
        metavar="MANIFEST_CSV",
        help="Combine multiple existing manifest CSVs into one (instead of building one)",
    )
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
        help="Output CSV path (default depends on mode, see below)",
    )
    parser.add_argument(
        "--subsample-max-per-group",
        type=int,
        default=None,
        help="Cap each (split, label) group at this many rows, for a quick shorter-window run",
    )
    args = parser.parse_args()

    if args.combine:
        manifest = combine_manifests(args.combine)
        out_path = args.out or Path("data/processed/combined_manifest.csv")
    elif args.dataset:
        dataset_root = args.data_root / args.dataset
        manifest = _BUILDERS[args.dataset](dataset_root)
        out_path = args.out or Path("data/processed") / f"{args.dataset}_manifest.csv"
    else:
        parser.error("pass either --dataset or --combine")

    if args.subsample_max_per_group:
        manifest = subsample_manifest(manifest, args.subsample_max_per_group)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    counts = manifest.groupby(["split", "label"]).size()
    print(f"Wrote {len(manifest)} rows to {out_path}")
    print(counts.to_string())


if __name__ == "__main__":
    main()

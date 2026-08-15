"""Materializes the garystafford/deepfake-audio-detection HF dataset to local files,
so it can go through the same manifest + AudioManifestDataset pipeline as every other
dataset here. Streamed (not bulk-downloaded) from the Hub -- only ever holds one sample
in memory at a time -- but written to disk once so training doesn't re-fetch every epoch.

    python scripts/materialize_garystafford.py
"""

from __future__ import annotations

from pathlib import Path

import soundfile as sf
from datasets import load_dataset

OUT_ROOT = Path("data/raw/garystafford")
LABEL_TO_DIR = {0: "real", 1: "fake"}


def main() -> None:
    for name in LABEL_TO_DIR.values():
        (OUT_ROOT / name).mkdir(parents=True, exist_ok=True)

    ds = load_dataset("garystafford/deepfake-audio-detection", split="train", streaming=True)

    counts = {"real": 0, "fake": 0}
    for i, sample in enumerate(ds):
        label_dir = LABEL_TO_DIR[sample["label"]]
        waveform = sample["audio"].get_all_samples()
        out_path = OUT_ROOT / label_dir / f"{i:05d}.flac"
        sf.write(out_path, waveform.data.numpy().T, waveform.sample_rate)
        counts[label_dir] += 1
        if i % 200 == 0:
            print(f"{i} samples written so far ({counts})")

    print(f"Done. Wrote {counts}")


if __name__ == "__main__":
    main()

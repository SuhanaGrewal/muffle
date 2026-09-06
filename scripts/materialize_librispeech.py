"""Materializes LibriSpeech dev-clean from HuggingFace's mirror (openslr/librispeech_asr)
to local files, since the official OpenSLR direct download was too slow to be practical
(~30-60kB/s observed). Streamed (not bulk-downloaded), written to disk in the same
<speaker>/<chapter>/<speaker>-<chapter>-<utterance>.flac layout the official release
uses, so build_librispeech_manifest (which expects that layout) works unchanged.

    python scripts/materialize_librispeech.py
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import soundfile as sf
from datasets import load_dataset

OUT_ROOT = Path("data/raw/librispeech/LibriSpeech/dev-clean")


def main() -> None:
    ds = load_dataset("openslr/librispeech_asr", "clean", split="validation", streaming=True)

    utterance_counts: dict[tuple[int, int], int] = defaultdict(int)
    n_written = 0

    for sample in ds:
        speaker_id = sample["speaker_id"]
        chapter_id = sample["chapter_id"]
        idx = utterance_counts[(speaker_id, chapter_id)]
        utterance_counts[(speaker_id, chapter_id)] += 1

        out_dir = OUT_ROOT / str(speaker_id) / str(chapter_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{speaker_id}-{chapter_id}-{idx:04d}.flac"

        waveform = sample["audio"].get_all_samples()
        sf.write(out_path, waveform.data.numpy().T, waveform.sample_rate)

        n_written += 1
        if n_written % 200 == 0:
            print(f"{n_written} utterances written so far")

    print(f"Done. Wrote {n_written} utterances across {len(utterance_counts)} (speaker, chapter) pairs")


if __name__ == "__main__":
    main()

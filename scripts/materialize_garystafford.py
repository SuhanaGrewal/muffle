"""Materializes the garystafford/deepfake-audio-detection HF dataset to local files,
preserving each file's original name -- which encodes which TTS platform generated it
(el_=ElevenLabs, po_=Amazon Polly, hg_=Hexgrad Kokoro, hu_=Hume AI, lv_=Luvvoice,
sp_=Speechify; yt_=real YouTube speaker). manifests.py uses this prefix to assign a real
per-platform attack_id instead of one generic "commercial_tts" label for every fake file.
The `datasets` streaming loader used previously only exposes audio+label, not filenames,
so this downloads the raw fake/ and real/ folders directly via huggingface_hub instead.

    python scripts/materialize_garystafford.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import snapshot_download

OUT_ROOT = Path("data/raw/garystafford")


def main() -> None:
    # max_workers kept low -- HF's xet backend rate-limited (429) the default concurrency
    # partway through a 1866-file download.
    local_dir = snapshot_download(
        repo_id="garystafford/deepfake-audio-detection",
        repo_type="dataset",
        allow_patterns=["fake/*", "real/*"],
        local_dir=OUT_ROOT,
        max_workers=4,
    )
    n_fake = sum(1 for _ in (OUT_ROOT / "fake").glob("*.flac"))
    n_real = sum(1 for _ in (OUT_ROOT / "real").glob("*.flac"))
    print(f"Done. {local_dir}: {n_real} real, {n_fake} fake (original filenames preserved)")


if __name__ == "__main__":
    main()

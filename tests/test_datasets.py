import numpy as np
import pandas as pd
import soundfile as sf

from muffle.data.datasets import AudioManifestDataset


def _write_sine_wav(path, seconds: float, sample_rate: int):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    sf.write(path, audio, sample_rate)


def test_dataset_pads_short_clips_and_labels_correctly(tmp_path):
    sample_rate = 16_000
    short_path = tmp_path / "short.wav"
    long_path = tmp_path / "long.wav"
    _write_sine_wav(short_path, seconds=1.0, sample_rate=sample_rate)  # shorter than window
    _write_sine_wav(long_path, seconds=6.0, sample_rate=sample_rate)  # longer than window

    manifest = pd.DataFrame(
        [
            {
                "path": str(short_path),
                "label": "bonafide",
                "dataset": "synthetic",
                "attack_id": None,
                "speaker_id": "spk1",
                "split": "train",
            },
            {
                "path": str(long_path),
                "label": "spoof",
                "dataset": "synthetic",
                "attack_id": "A01",
                "speaker_id": "spk2",
                "split": "train",
            },
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    dataset = AudioManifestDataset(
        manifest_path, sample_rate=sample_rate, duration_seconds=4.0, split="train"
    )
    assert len(dataset) == 2

    short_item = dataset[0]
    long_item = dataset[1]

    target_len = sample_rate * 4
    assert short_item["waveform"].shape[0] == target_len
    assert long_item["waveform"].shape[0] == target_len
    assert short_item["label"].item() == 0  # bonafide
    assert long_item["label"].item() == 1  # spoof
    assert long_item["attack_id"] == "A01"
    assert short_item["attack_id"] is None

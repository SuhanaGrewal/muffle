import io

import numpy as np
import soundfile as sf
import torch
import yaml
from fastapi.testclient import TestClient

from muffle.factory import build_model


def _make_tiny_checkpoint(tmp_path, sample_rate=16_000, duration_seconds=1.0):
    """A freshly-initialized (untrained) checkpoint -- enough to test the service's
    plumbing (request -> response shape), not model accuracy.
    """
    cfg = {
        "model_name": "test_tiny_cnn",
        "model_type": "cnn_baseline",
        "data": {"sample_rate": sample_rate, "duration_seconds": duration_seconds},
        "features": {"n_lfcc": 20, "n_filter": 40, "win_length_ms": 20.0, "hop_length_ms": 10.0},
        "model": {"hidden_channels": [8, 16]},
    }

    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(cfg))

    checkpoint_path = tmp_path / "checkpoint.pt"
    model = build_model(cfg)
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    return config_path, checkpoint_path


def _sine_wav_bytes(seconds=1.0, sample_rate=16_000, freq=220):
    t = np.linspace(0, seconds, int(sample_rate * seconds), endpoint=False)
    audio = (0.1 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf


def test_health_and_detect_endpoints(tmp_path, monkeypatch):
    config_path, checkpoint_path = _make_tiny_checkpoint(tmp_path)
    monkeypatch.setenv("MUFFLE_CONFIG", str(config_path))
    monkeypatch.setenv("MUFFLE_CHECKPOINT", str(checkpoint_path))
    monkeypatch.setenv("MUFFLE_DEVICE", "cpu")

    from service.app import app  # imported after env vars are set, before lifespan startup

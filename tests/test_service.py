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

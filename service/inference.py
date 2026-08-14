"""Loads a trained checkpoint once (not per-request) and scores raw audio bytes."""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

from muffle.data.datasets import _resample
from muffle.factory import build_feature_extractor, build_model

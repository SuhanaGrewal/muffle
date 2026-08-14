"""Builds a feature extractor + model pair from a config's `model_type`, so train.py and
evaluate.py don't need to know which Phase's architecture they're running.
"""

from __future__ import annotations

from muffle.features.lfcc import LFCCExtractor
from muffle.features.ssl_features import SSLFeatureExtractor
from muffle.models.cnn_baseline import SpoofCNN
from muffle.models.ssl_head import SSLHeadClassifier

"""Builds a feature extractor + model pair from a config's `model_type`, so train.py and
evaluate.py don't need to know which Phase's architecture they're running.
"""

from __future__ import annotations

from muffle.features.lfcc import LFCCExtractor
from muffle.features.ssl_features import SSLFeatureExtractor
from muffle.models.cnn_baseline import SpoofCNN
from muffle.models.ssl_head import SSLHeadClassifier


def build_feature_extractor(cfg: dict):
    model_type = cfg["model_type"]

    if model_type == "cnn_baseline":
        return LFCCExtractor(
            sample_rate=cfg["data"]["sample_rate"],
            n_lfcc=cfg["features"]["n_lfcc"],
            n_filter=cfg["features"]["n_filter"],
            win_length_ms=cfg["features"]["win_length_ms"],
            hop_length_ms=cfg["features"]["hop_length_ms"],
        )

    if model_type == "ssl_head":
        return SSLFeatureExtractor(
            model_name=cfg["features"]["ssl_model_name"],
            sample_rate=cfg["data"]["sample_rate"],
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")


def build_model(cfg: dict):
    model_type = cfg["model_type"]

    if model_type == "cnn_baseline":
        return SpoofCNN(
            n_feat=cfg["features"]["n_lfcc"] * 3,
            hidden_channels=tuple(cfg["model"]["hidden_channels"]),
        )

    if model_type == "ssl_head":
        return SSLHeadClassifier(
            hidden_size=cfg["model"]["ssl_hidden_size"],
            mlp_hidden=cfg["model"]["mlp_hidden"],
            dropout=cfg["model"].get("dropout", 0.1),
        )

    raise ValueError(f"Unknown model_type: {model_type!r}")

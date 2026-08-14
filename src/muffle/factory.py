"""Builds a feature extractor + model pair from a config's `model_type`, so train.py and
evaluate.py don't need to know which Phase's architecture they're running.
"""

from __future__ import annotations

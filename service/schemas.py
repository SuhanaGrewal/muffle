"""Pydantic request/response models for the /detect and /health endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DetectResponse(BaseModel):
    verdict: Literal["human", "ai_generated"]
    confidence: float
    score_raw: float
    model_version: str
    processing_time_ms: float

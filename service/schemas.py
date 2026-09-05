"""Pydantic request/response models for the /detect and /health endpoints."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class DetectResponse(BaseModel):
    verdict: Literal["human", "ai_generated", "no_speech_detected"]
    confidence: float
    score_raw: float
    model_version: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_version: str

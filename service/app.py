"""FastAPI app: POST /detect (upload an audio clip), GET /health.

    uvicorn service.app:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from service.inference import InferenceEngine
from service.schemas import DetectResponse, HealthResponse

_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    config_path = os.environ.get("MUFFLE_CONFIG", "configs/baseline_lfcc_cnn.yaml")
    checkpoint_path = os.environ.get(
        "MUFFLE_CHECKPOINT", "checkpoints/baseline_lfcc_cnn/best.pt"
    )
    device = os.environ.get("MUFFLE_DEVICE", "cpu")

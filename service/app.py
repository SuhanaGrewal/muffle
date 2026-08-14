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

    _engine = InferenceEngine(config_path, checkpoint_path, device=device)
    yield
    _engine = None


app = FastAPI(title="muffle", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return HealthResponse(status="ok", model_version=_engine.model_version)


@app.post("/detect", response_model=DetectResponse)
async def detect(file: UploadFile = File(...)) -> DetectResponse:
    if _engine is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    audio_bytes = await file.read()
    try:
        result = _engine.predict(audio_bytes)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"could not process audio: {exc}") from exc

    return DetectResponse(**result)

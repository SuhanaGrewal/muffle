"""FastAPI app: POST /detect (upload an audio clip), GET /health.

    uvicorn service.app:app --reload
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from service.inference import InferenceEngine
from service.schemas import DetectResponse, HealthResponse

_STATIC_DIR = Path(__file__).parent / "static"

_engine: InferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _engine
    # WavLM is the default: cross-dataset eval on In-the-Wild (real-world deepfakes,
    # never trained on) showed the CNN baseline collapses to 55.28% EER (chance level)
    # while WavLM holds at 14.85% -- the frozen-SSL generalization bet actually paid off.
    # See README's Results section.
    config_path = os.environ.get("MUFFLE_CONFIG", "configs/ssl_wavlm_head.yaml")
    checkpoint_path = os.environ.get(
        "MUFFLE_CHECKPOINT", "checkpoints/ssl_wavlm_head/best.pt"
    )
    device = os.environ.get("MUFFLE_DEVICE", "cpu")

    _engine = InferenceEngine(config_path, checkpoint_path, device=device)
    yield
    _engine = None


app = FastAPI(title="muffle", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=_STATIC_DIR), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(_STATIC_DIR / "index.html")


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

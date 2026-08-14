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

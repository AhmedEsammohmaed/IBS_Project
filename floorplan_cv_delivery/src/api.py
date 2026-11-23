"""Drop-in FastAPI router; does not access persistence or building records."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from starlette.concurrency import run_in_threadpool

from .inference import predict_floorplan

router = APIRouter(prefix="/floorplan-cv", tags=["floorplan-cv"])
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".pdf"}


@router.post("/predict")
async def predict_endpoint(
    file: UploadFile = File(...),
    dpi: int = Query(200, ge=36, le=1200),
    device: str = Query("auto", pattern="^(auto|cpu|cuda)$"),
    include_ocr: bool = Query(False),
    accept_noncommercial_license: bool = Query(False),
):
    """Return unreviewed pixel-space predictions for one raster file or PDF."""
    from pathlib import PurePosixPath
    suffix = PurePosixPath(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Supported uploads are PNG, JPG, JPEG, and PDF")
    content = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 25 MiB limit")
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    try:
        return await run_in_threadpool(
            predict_floorplan, content,
            {"dpi": dpi, "device": device, "include_ocr": include_ocr,
             "accept_noncommercial_license": accept_noncommercial_license},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

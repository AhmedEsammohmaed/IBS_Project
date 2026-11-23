"""Lazy model loading for the published multi-class floor-plan checkpoint."""

from __future__ import annotations

from functools import lru_cache

MODEL_ID = "swedishtech/RAQEEB-CCTV"
MODEL_REVISION = "d0f2b17"
WEIGHTS_FILE = "raqeeb-cctv-v1.0.0.pt"
WEIGHTS_SHA256 = "9788ae90e35715b1012bbe1bd1b56daa636cb3a1429b0b90ba4b8148a1ff0bcf"
MODEL_VERSION = "1.0.0"
MODEL_LICENSE = "CC-BY-NC-4.0"


@lru_cache(maxsize=2)
def load_model(device: str = "auto", model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    """Load weights lazily and verify the downloaded artifact before use.

    This model is non-commercially licensed. Callers must confirm they have the
    rights required for their intended use before selecting this backend.
    """
    try:
        import hashlib
        import torch
        from huggingface_hub import hf_hub_download
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Install the YOLO inference dependencies before inference") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable in this PyTorch runtime")
    weights_path = hf_hub_download(
        repo_id=model_id, filename=WEIGHTS_FILE, revision=revision,
    )
    digest = hashlib.sha256()
    with open(weights_path, "rb") as weights_file:
        for chunk in iter(lambda: weights_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WEIGHTS_SHA256:
        raise RuntimeError("Downloaded model checksum does not match the pinned model artifact")
    return YOLO(weights_path), device

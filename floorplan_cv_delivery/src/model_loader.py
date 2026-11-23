"""Lazy Hugging Face model loading with explicit CPU/CUDA selection."""

from __future__ import annotations

from functools import lru_cache

MODEL_ID = "phungpx/mask2former-swin-base-cubicassa"
MODEL_REVISION = "main"


@lru_cache(maxsize=2)
def load_model(device: str = "auto", model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    try:
        import torch
        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation
    except ImportError as exc:
        raise RuntimeError("Install floorplan-cv-delivery dependencies before inference") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("device must be 'auto', 'cpu', or 'cuda'")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but CUDA is unavailable in this PyTorch runtime")
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(model_id, revision=revision)
    model.to(device)
    model.eval()
    return processor, model, device

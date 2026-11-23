"""Optional OCR adapter. Requires the pytesseract package and Tesseract binary."""

from __future__ import annotations

from typing import Any

from PIL import Image


def recognize_text(image: Image.Image, source_image_id: str, page_index: int | None = None) -> list[dict[str, Any]]:
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError("OCR requires optional dependency pytesseract and the Tesseract executable") from exc
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    results = []
    for i, raw in enumerate(data["text"]):
        text = raw.strip()
        if not text:
            continue
        try:
            confidence = max(0.0, float(data["conf"][i]) / 100)
        except (ValueError, TypeError):
            confidence = 0.0
        results.append({
            "id": f"ocr-{i + 1:05d}", "text": text, "confidence": confidence,
            "bbox": [int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])],
            "source_image_id": source_image_id, "page_index": page_index,
            "associated_room_candidate": None,
        })
    return results

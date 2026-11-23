"""Optional OCR adapter. Requires the pytesseract package and Tesseract binary."""

from __future__ import annotations

import re
from typing import Any

from PIL import Image

DIMENSION_PATTERN = re.compile(
    r"(?P<value>\d+(?:[.,]\d+)?)\s*(?P<unit>mm|cm|m|ft|feet|in|inch|inches|')(?=\s|$|[.,])",
    re.IGNORECASE,
)
SCALE_PATTERN = re.compile(r"\b(?:scale\s*)?1\s*:\s*\d{2,5}\b", re.IGNORECASE)


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
        dimension_match = DIMENSION_PATTERN.search(text)
        is_scale = bool(SCALE_PATTERN.search(text))
        kind = "scale_reference_candidate" if is_scale else "dimension_text_candidate" if dimension_match else "room_label_candidate"
        results.append({
            "id": f"ocr-{i + 1:05d}", "text": text, "confidence": confidence,
            "bbox": [int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])],
            "source_image_id": source_image_id, "page_index": page_index,
            "associated_room_candidate": None,
            "text_kind": kind,
            "dimension_value_candidate": float(dimension_match.group("value").replace(",", ".")) if dimension_match else None,
            "dimension_unit_candidate": dimension_match.group("unit").lower() if dimension_match else None,
            "engineering_value_verified": False,
        })
    return results

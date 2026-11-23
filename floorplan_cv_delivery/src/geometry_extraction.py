"""Coordinate conventions and geometry helpers for pixel-space predictions."""

from __future__ import annotations

from typing import Any


def bbox_from_points(points: list[list[int]]) -> list[int]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    left, top = min(xs), min(ys)
    return [left, top, max(xs) - left, max(ys) - top]


def detection(element_id: str, class_name: str, class_id: int, confidence: float,
              polygons: list[list[list[int]]], source_image_id: str, page_index: int | None,
              tile_id: int) -> dict[str, Any]:
    all_points = [point for polygon in polygons for point in polygon]
    return {
        "id": element_id,
        "class": class_name,
        "class_id": class_id,
        "confidence": round(float(confidence), 6),
        "geometry_type": "polygon" if polygons else "mask",
        "geometry": {"polygons": [{"points": polygon} for polygon in polygons]},
        "bbox": bbox_from_points(all_points) if all_points else None,
        "source_image_id": source_image_id,
        "page_index": page_index,
        "tile_id": tile_id,
        "uncertainty": {"confidence_is_model_score": True, "human_review_required": True},
    }

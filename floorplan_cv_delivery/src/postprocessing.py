"""Prediction filtering, mask polygonization, duplicate suppression, and JSON shape."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def mask_to_polygons(mask: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> list[list[list[int]]]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[list[int]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < 4:
            continue
        epsilon = max(0.5, 0.002 * cv2.arcLength(contour, True))
        points = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(points) >= 3:
            polygons.append([[int(x + offset_x), int(y + offset_y)] for x, y in points])
    return polygons


def box_iou(a: list[int], b: list[int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def merge_tile_duplicates(items: list[dict[str, Any]], iou_threshold: float = 0.65) -> list[dict[str, Any]]:
    """Greedy same-class box NMS, retaining an audit trail for every merge."""
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: row["confidence"], reverse=True):
        duplicate = next((old for old in kept if item["class_id"] == old["class_id"]
                          and box_iou(item["bbox"], old["bbox"]) >= iou_threshold), None)
        if duplicate is not None:
            duplicate.setdefault("merged_duplicates", []).append({
                "confidence": item["confidence"], "bbox": item["bbox"], "tile_id": item.get("tile_id"),
            })
            continue
        kept.append(item)
    return kept

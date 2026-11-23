"""Independent multi-class floor plan inference (no database or RF logic)."""

from __future__ import annotations

import hashlib
from typing import Any

import cv2
import numpy as np

from .geometry_extraction import detection
from .model_loader import (
    MODEL_ID, MODEL_LICENSE, MODEL_REVISION, MODEL_VERSION, load_model,
)
from .postprocessing import box_iou, mask_to_polygons, merge_tile_duplicates
from .preprocessing import PageImage, Tile, load_pages, make_tiles

DEFAULT_THRESHOLDS = {
    "wall": 0.2773, "door": 0.3033, "window": 0.3493,
    "open_passage": 0.1812, "column": 0.0921, "stair": 0.4084,
    "elevator": 0.7057, "escalator": 0.6837, "exit_marker": 0.3293,
    "gate": 0.5, "space_region": 0.3794,
}
CONFIDENCE_FLOOR = 0.05
REVIEW_FLOOR = 0.10
UNVALIDATED_CLASSES = {"escalator", "gate"}


def _normalize_class(name: str) -> str:
    return "room" if name == "space_region" else name


def _predict_tile(tile: Tile, tile_id: int, model, device: str,
                  threshold: float, imgsz: int) -> list[dict[str, Any]]:
    result = model.predict(
        source=tile.image, imgsz=imgsz, conf=threshold, device=device,
        max_det=1500, verbose=False, save=False,
    )[0]
    rows: list[dict[str, Any]] = []
    if result.boxes is None:
        return rows
    names = result.names
    boxes = result.boxes.xyxy.detach().cpu().numpy()
    class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
    scores = result.boxes.conf.detach().cpu().numpy()
    masks = result.masks.xy if result.masks is not None else [None] * len(boxes)
    for box, class_id, score, polygon in zip(boxes, class_ids, scores, masks):
        class_name = str(names[int(class_id)])
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        bbox = [x1 + tile.x, y1 + tile.y, max(0, x2 - x1), max(0, y2 - y1)]
        polygons: list[list[list[int]]] = []
        if polygon is not None and len(polygon) >= 3:
            points = np.asarray(polygon).round().astype(int)
            points[:, 0] = np.clip(points[:, 0], 0, tile.width - 1) + tile.x
            points[:, 1] = np.clip(points[:, 1], 0, tile.height - 1) + tile.y
            polygons = [points.tolist()]
        rows.append({
            "class_id": int(class_id), "class_name": class_name,
            "confidence": float(score), "bbox": bbox,
            "polygons": polygons, "tile_id": tile_id,
        })
    return rows


def _point_in_polygon(point_x: float, point_y: float, polygon: list[list[int]]) -> bool:
    inside = False
    j = len(polygon) - 1
    for i, (xi, yi) in enumerate(polygon):
        xj, yj = polygon[j]
        if (yi > point_y) != (yj > point_y):
            crossing_x = (xj - xi) * (point_y - yi) / ((yj - yi) or 1e-12) + xi
            if point_x < crossing_x:
                inside = not inside
        j = i
    return inside


def _associate_page_geometry(detections: list[dict[str, Any]]) -> None:
    walls = [d for d in detections if d["class"] == "wall"]
    for row in detections:
        if row["class"] == "window":
            rx, ry, rw, rh = row["bbox"]
            candidates = []
            for wall in walls:
                wx, wy, ww, wh = wall["bbox"]
                gap_x = max(0, wx - (rx + rw), rx - (wx + ww))
                gap_y = max(0, wy - (ry + rh), ry - (wy + wh))
                gap = float(np.hypot(gap_x, gap_y))
                if box_iou(row["bbox"], wall["bbox"]) > 0 or gap <= 64:
                    candidates.append((gap, wall["id"]))
            row["nearby_wall_ids"] = [wall_id for _, wall_id in sorted(candidates)[:3]]
            row["association_is_heuristic"] = True
        if row["class"] == "door":
            points = [p for poly in row["geometry"]["polygons"] for p in poly["points"]]
            if len(points) >= 3:
                rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
                row["orientation_degrees"] = round(float(rect[2]), 2)
                row["orientation_is_approximate"] = True
                row["opening_direction"] = None
        if row["class"] == "column":
            _, _, width, height = row["bbox"]
            row["dimensions_px"] = {"width": width, "height": height}
            ratio = min(width, height) / max(width, height, 1)
            row["shape_candidate"] = "square_or_round" if ratio >= 0.85 else "rectangular"
            row["shape_classification_is_heuristic"] = True
        if row["class"] == "wall":
            points = [p for poly in row["geometry"]["polygons"] for p in poly["points"]]
            if len(points) >= 3:
                rect = cv2.minAreaRect(np.asarray(points, dtype=np.float32))
                a, b = rect[1]
                long_side, short_side = max(a, b), min(a, b)
                if short_side >= 2 and long_side / max(short_side, 1) >= 5:
                    row["estimated_thickness_px_candidate"] = round(float(short_side), 1)
                    row["thickness_estimation_method"] = "minor dimension of minimum-area rotated rectangle"
                    row["thickness_estimate_requires_review"] = True
    for wall in walls:
        wx, wy, ww, wh = wall["bbox"]
        adjacent = []
        for other in walls:
            if other["id"] == wall["id"]:
                continue
            ox, oy, ow, oh = other["bbox"]
            gap_x = max(0, ox - (wx + ww), wx - (ox + ow))
            gap_y = max(0, oy - (wy + wh), wy - (oy + oh))
            gap = float(np.hypot(gap_x, gap_y))
            if gap <= 8:
                adjacent.append(other["id"])
        wall["continuity_candidate_ids"] = adjacent
        wall["continuity_is_heuristic"] = True


def _candidate_dimension_lines(image, source_image_id: str, page_index: int | None) -> list[dict[str, Any]]:
    gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80,
                            minLineLength=max(30, min(image.size) // 80), maxLineGap=8)
    if lines is None:
        return []
    output = []
    for index, line in enumerate(lines[:, 0]):
        x1, y1, x2, y2 = [int(v) for v in line]
        length = float(np.hypot(x2 - x1, y2 - y1))
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if length < 30 or min(abs(angle), abs(abs(angle) - 180), abs(abs(angle) - 90)) > 5:
            continue
        output.append({
            "id": f"line-candidate-{index + 1:05d}", "class": "dimension_line_candidate",
            "confidence": None, "geometry_type": "line",
            "geometry": {"points": [[x1, y1], [x2, y2]]},
            "source_image_id": source_image_id, "page_index": page_index,
            "uncertainty": {"heuristic_candidate": True, "human_review_required": True},
        })
    return output


def _attach_ocr_text(items: list[dict[str, Any]], detections: list[dict[str, Any]]) -> None:
    for item in items:
        x, y, width, height = item["bbox"]
        center = (x + width / 2, y + height / 2)
        candidates = []
        for room in detections:
            if room["class"] != "room":
                continue
            if any(_point_in_polygon(*center, polygon["points"]) for polygon in room["geometry"]["polygons"]):
                candidates.append(room["id"])
        item["associated_room_candidate"] = candidates[0] if candidates else None


def predict_floorplan(image: Any, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict building element candidates from an image or PDF.

    Options: device ('auto'|'cpu'|'cuda'), dpi, confidence_threshold (minimum
    model score), tile_size, tile_overlap, imgsz, include_ocr, include_line_candidates,
    model_id/model_revision, and accept_noncommercial_license=True. Coordinates are
    in original EXIF-oriented pixels, origin top-left, x right and y down; bboxes
    are [x,y,width,height] and polygon vertices are [x,y].
    """
    opts = options or {}
    if not opts.get("accept_noncommercial_license", False):
        raise ValueError(
            "The configured checkpoint is CC BY-NC 4.0. Confirm non-commercial use "
            "or obtain separate permission before setting accept_noncommercial_license=True."
        )
    threshold = float(opts.get("confidence_threshold", CONFIDENCE_FLOOR))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    pages = load_pages(image, int(opts.get("dpi", 200)))
    model_id = str(opts.get("model_id", MODEL_ID))
    revision = str(opts.get("model_revision", MODEL_REVISION))
    model, device = load_model(str(opts.get("device", "auto")), model_id, revision)
    source_hash = hashlib.sha256()
    for page in pages:
        source_hash.update(page.image.tobytes())
    source_image_id = source_hash.hexdigest()[:16]
    all_detections: list[dict[str, Any]] = []
    ocr_results: list[dict[str, Any]] = []
    line_candidates: list[dict[str, Any]] = []
    thresholds = {**DEFAULT_THRESHOLDS, **opts.get("class_thresholds", {})}
    for page in pages:
        tiles = make_tiles(page.image, int(opts.get("tile_size", 1536)), int(opts.get("tile_overlap", 192)))
        rows: list[dict[str, Any]] = []
        for tile_id, tile in enumerate(tiles):
            rows.extend(_predict_tile(tile, tile_id, model, device, threshold, int(opts.get("imgsz", 1024))))
        rows = merge_tile_duplicates(rows, float(opts.get("duplicate_iou_threshold", 0.65)))
        page_detections: list[dict[str, Any]] = []
        for row in rows:
            class_name = _normalize_class(row["class_name"])
            operating_threshold = float(thresholds.get(row["class_name"], threshold))
            item = detection(
                f"prediction-{len(all_detections) + len(page_detections) + 1:06d}",
                class_name, row["class_id"], row["confidence"], row["polygons"],
                source_image_id, page.page_index, row["tile_id"],
            )
            item["source_class"] = row["class_name"]
            item["bbox"] = row["bbox"]
            item["model_version"] = MODEL_VERSION
            item["review_status"] = "review_candidate" if row["confidence"] >= REVIEW_FLOOR else "low_confidence_candidate"
            item["below_validation_threshold"] = row["confidence"] < operating_threshold
            item["operating_threshold"] = operating_threshold
            item["class_evaluation_status"] = "unvalidated" if row["class_name"] in UNVALIDATED_CLASSES else "upstream_tested"
            item["merged_duplicates"] = row.get("merged_duplicates", [])
            page_detections.append(item)
        _associate_page_geometry(page_detections)
        all_detections.extend(page_detections)
        if opts.get("include_ocr", False):
            from .ocr import recognize_text
            page_ocr = recognize_text(page.image, source_image_id, page.page_index)
            _attach_ocr_text(page_ocr, page_detections)
            ocr_results.extend(page_ocr)
        if opts.get("include_line_candidates", False):
            line_candidates.extend(_candidate_dimension_lines(page.image, source_image_id, page.page_index))
    return {
        "model_version": MODEL_VERSION,
        "model": {"id": model_id, "revision": revision, "device": device, "license": MODEL_LICENSE},
        "source": {
            "image_id": source_image_id,
            "pages": [{"page_index": p.page_index, "image_width": p.source_width,
                       "image_height": p.source_height, "source_kind": p.source_kind,
                       "pdf_page_width_points": p.pdf_page_width_points,
                       "pdf_page_height_points": p.pdf_page_height_points} for p in pages],
            "coordinate_system": "image_pixel; origin top-left; x right; y down; bbox [x,y,width,height]",
        },
        "detections": all_detections,
        "ocr": ocr_results if opts.get("include_ocr", False) else None,
        "dimension_line_candidates": line_candidates if opts.get("include_line_candidates", False) else None,
        "review_status": "unreviewed_predictions",
        "engineering_scale_calibrated": False,
        "limitations": [
            "Predictions are suggestions and require human engineering review.",
            "The upstream checkpoint is CC BY-NC 4.0; it is not cleared for commercial deployment.",
            "Wall thickness, wall continuity, and door opening direction are not inferred.",
            "Room polygons are model space-region candidates, not proof that a valid room exists.",
            "Window-wall links and OCR-room links are geometric heuristics only.",
            "Dimension lines are image-processing candidates and require human confirmation.",
        ],
    }

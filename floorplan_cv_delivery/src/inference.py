"""Independent floor plan prediction interface (no database or RF logic)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from .geometry_extraction import detection
from .model_loader import MODEL_ID, MODEL_REVISION, load_model
from .postprocessing import mask_to_polygons, merge_tile_duplicates
from .preprocessing import load_pages, make_tiles

MODEL_VERSION = "1.0.0"


def _predict_tile(tile, tile_id: int, processor, model, device: str, threshold: float) -> list[dict[str, Any]]:
    import torch
    inputs = processor(images=tile.image, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    result = processor.post_process_instance_segmentation(
        outputs, target_sizes=[(tile.height, tile.width)], threshold=threshold
    )[0]
    segmentation = result["segmentation"].detach().cpu().numpy()
    classes = model.config.id2label
    rows: list[dict[str, Any]] = []
    for segment in result.get("segments_info", []):
        class_id = int(segment["label_id"])
        mask = segmentation == int(segment["id"])
        if not mask.any():
            continue
        polygons = mask_to_polygons(mask, tile.x, tile.y)
        all_x = np.where(mask)[1] + tile.x
        all_y = np.where(mask)[0] + tile.y
        rows.append({
            "class_id": class_id,
            "class_name": classes.get(class_id, str(class_id)),
            "confidence": float(segment["score"]),
            "bbox": [int(all_x.min()), int(all_y.min()), int(all_x.max() - all_x.min() + 1), int(all_y.max() - all_y.min() + 1)],
            "polygons": polygons,
            "tile_id": tile_id,
        })
    return rows


def predict_floorplan(image: Any, options: dict[str, Any] | None = None) -> dict[str, Any]:
    """Predict wall instances from image bytes, a path, file-like object, or PIL image.

    Options: device ('auto'|'cpu'|'cuda'), dpi, confidence_threshold, tile_size,
    tile_overlap, model_id, model_revision, and include_ocr (optional).
    Coordinates use the original, EXIF-oriented raster: origin top-left, x right,
    y down; bboxes are [x, y, width, height], polygon vertices are [x, y].
    """
    opts = options or {}
    threshold = float(opts.get("confidence_threshold", 0.35))
    if not 0 <= threshold <= 1:
        raise ValueError("confidence_threshold must be between 0 and 1")
    pages = load_pages(image, int(opts.get("dpi", 200)))
    processor, model, device = load_model(
        str(opts.get("device", "auto")), str(opts.get("model_id", MODEL_ID)),
        str(opts.get("model_revision", MODEL_REVISION)),
    )
    detections: list[dict[str, Any]] = []
    ocr_results: list[dict[str, Any]] = []
    source_hash = hashlib.sha256()
    for page in pages:
        source_hash.update(page.image.tobytes())
    source_image_id = source_hash.hexdigest()[:16]
    for page in pages:
        tile_size, overlap = int(opts.get("tile_size", 1024)), int(opts.get("tile_overlap", 128))
        tiles = make_tiles(page.image, tile_size, overlap)
        rows = []
        for tile_id, tile in enumerate(tiles):
            rows.extend(_predict_tile(tile, tile_id, processor, model, device, threshold))
        rows = merge_tile_duplicates(rows, float(opts.get("duplicate_iou_threshold", 0.65)))
        page_detections = []
        for index, row in enumerate(rows, 1):
            page_detections.append(detection(
                f"prediction-{len(detections) + index:06d}", row["class_name"], row["class_id"],
                row["confidence"], row["polygons"], source_image_id, page.page_index, row["tile_id"],
            ))
        detections.extend(page_detections)
        if opts.get("include_ocr", False):
            from .ocr import recognize_text
            ocr_results.extend(recognize_text(page.image, source_image_id, page.page_index))
    return {
        "model_version": MODEL_VERSION,
        "model": {"id": str(opts.get("model_id", MODEL_ID)), "revision": str(opts.get("model_revision", MODEL_REVISION)), "device": device},
        "source": {
            "image_id": source_image_id,
            "pages": [{"page_index": page.page_index, "image_width": page.source_width,
                       "image_height": page.source_height, "source_kind": page.source_kind,
                       "pdf_page_width_points": page.pdf_page_width_points,
                       "pdf_page_height_points": page.pdf_page_height_points} for page in pages],
            "coordinate_system": "image_pixel; origin top-left; x right; y down; bbox [x,y,width,height]",
        },
        "detections": detections,
        "ocr": ocr_results if opts.get("include_ocr", False) else None,
        "review_status": "unreviewed_predictions",
        "limitations": ["Predictions are suggestions and require human engineering review.",
                        "The configured pretrained checkpoint is documented as wall-only; no door, window, column, or room model is bundled."],
    }

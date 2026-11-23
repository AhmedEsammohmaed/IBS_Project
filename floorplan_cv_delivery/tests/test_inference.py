"""Offline checks for deterministic preprocessing and geometry utilities."""

from io import BytesIO

import numpy as np
from PIL import Image

from floorplan_cv_delivery.src.geometry_extraction import bbox_from_points
from floorplan_cv_delivery.src.postprocessing import box_iou, mask_to_polygons, merge_tile_duplicates
from floorplan_cv_delivery.src.preprocessing import load_pages, make_tiles
from floorplan_cv_delivery.src.inference import _normalize_class, _point_in_polygon, predict_floorplan
from floorplan_cv_delivery.src.ocr import DIMENSION_PATTERN, SCALE_PATTERN


def test_image_load_and_tiles_cover_edges():
    image = Image.new("RGB", (1800, 1200), "white")
    pages = load_pages(image)
    tiles = make_tiles(pages[0].image, tile_size=1024, overlap=128)
    assert pages[0].source_width == 1800
    assert len(tiles) == 4
    assert max(tile.x + tile.width for tile in tiles) == 1800
    assert max(tile.y + tile.height for tile in tiles) == 1200


def test_png_bytes_load():
    payload = BytesIO()
    Image.new("RGB", (32, 24), "white").save(payload, format="PNG")
    page = load_pages(payload.getvalue())[0]
    assert (page.image.width, page.image.height) == (32, 24)


def test_polygon_coordinates_and_bbox():
    mask = np.zeros((20, 30), dtype=np.uint8)
    mask[4:10, 6:15] = 1
    polygons = mask_to_polygons(mask, offset_x=100, offset_y=200)
    assert polygons
    box = bbox_from_points(polygons[0])
    assert box[0] >= 100 and box[1] >= 200


def test_duplicate_tile_candidates_are_suppressed():
    rows = [
        {"class_id": 0, "confidence": 0.8, "bbox": [10, 10, 40, 40]},
        {"class_id": 0, "confidence": 0.9, "bbox": [11, 11, 40, 40]},
        {"class_id": 1, "confidence": 0.7, "bbox": [11, 11, 40, 40]},
    ]
    assert box_iou(rows[0]["bbox"], rows[1]["bbox"]) > 0.65
    merged = merge_tile_duplicates(rows)
    assert len(merged) == 2
    assert merged[0]["merged_duplicates"][0]["tile_id"] is None


def test_room_mapping_and_polygon_point_membership():
    polygon = [[0, 0], [20, 0], [20, 20], [0, 20]]
    assert _normalize_class("space_region") == "room"
    assert _point_in_polygon(10, 10, polygon)
    assert not _point_in_polygon(25, 10, polygon)


def test_ocr_candidate_patterns_do_not_verify_dimensions():
    match = DIMENSION_PATTERN.search("120 cm")
    assert match and match.group("value") == "120" and match.group("unit") == "cm"
    assert SCALE_PATTERN.search("Scale 1:100")


def test_noncommercial_license_gate_precedes_model_loading():
    try:
        predict_floorplan(Image.new("RGB", (20, 20)), {})
    except ValueError as exc:
        assert "CC BY-NC 4.0" in str(exc)
    else:
        raise AssertionError("noncommercial acknowledgment gate should reject by default")

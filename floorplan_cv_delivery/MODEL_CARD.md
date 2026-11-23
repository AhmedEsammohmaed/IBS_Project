# Model card: RAQEEB-CCTV floor-plan segmentation adapter

## Summary

This package wraps RAQEEB-CCTV v1.0.0, a YOLO11s instance-segmentation checkpoint. It returns candidate masks/polygons for 11 floor-plan classes and provides optional OCR and geometric heuristics. It does not train a model and does not modify building records.

## Intended use

Non-commercial, human-reviewed initial parsing of raster floor plans. Every prediction is unreviewed. The planner/backend remains responsible for engineer review, scale calibration, coordinate transformation, persistence, floor assignment, geometry validation, and RF planning.

## Model details and license

- Model/version: `swedishtech/RAQEEB-CCTV`, v1.0.0, revision `d0f2b17`.
- Architecture: YOLO11s-seg, 10.1 million parameters, 1024 pixel model input.
- Framework: Ultralytics and PyTorch.
- Classes: wall, door, window, open passage, column, stair, elevator, escalator, exit marker, gate, and room/space region. See `configs/class_mapping.json`.
- Weights: 20.6 MiB checkpoint, downloaded lazily, SHA-256 verified, not copied into this repository.
- License: CC BY-NC 4.0; commercial use requires separate permission. The model card notes training across multiple public data sources, which may carry additional terms. See `ATTRIBUTION.md`.

## Upstream held-out evaluation

The upstream card reports a 1,682-image held-out set and per-class precision, recall, F1, mask AP50, and mask AP50:95 in `evaluation/metrics.json`. It uses a partial-label protocol. Escalator has only four held-out instances and gate has none. Column and wall segmentation scores are weaker than door/room categories. These results are not independent verification of this wrapper or the deployment planner.

## OCR and heuristic outputs

Optional Tesseract OCR preserves recognized strings, OCR confidence, boxes, room-label/dimension/scale candidates, and possible room association. OCR-to-room association, window-to-wall association, door orientation, wall thickness, column shape, and Hough line candidates are heuristics and must be reviewed. Dimension text is not treated as a verified physical measurement. Door opening direction is not inferred.

## Limitations

No planner-domain held-out dataset or OCR ground truth was provided. This delivery has no independently measured CPU/GPU latency, peak memory, OCR CER/WER, confusion matrix, or deployment false-positive/false-negative cases. GPU support is an optional code path and has not been exercised. This environment could not install PyTorch due to insufficient disk space, so model inference and ONNX export were not run. High-resolution tiling can distort room/space segmentation; the upstream card recommends whole-plan inference for room regions. Scale, physical dimensions, wall continuity, engineering coordinate systems, and engineering readiness are outside the model's authority.

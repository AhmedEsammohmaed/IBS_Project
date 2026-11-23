# Model card: Floor plan wall segmentation component

## Summary

This package wraps the publicly available `phungpx/mask2former-swin-base-cubicassa` checkpoint and converts its wall instance masks into polygons and pixel-space structured output. It does not train a model and does not include or redistribute model weights.

## Intended use

Use as an initial, human-reviewed aid for converting raster floor plans into candidate wall geometry for a downstream building-model workflow. Predictions are not approved engineering geometry, and this component never calibrates scale, assigns engineering coordinates, edits records, or computes RF properties.

## Model details

- Architecture: Mask2Former, Swin-Base backbone.
- Framework: PyTorch and Hugging Face Transformers.
- Task/classes: instance segmentation; upstream model card lists wall as its only class.
- Weights: downloaded from Hugging Face when first loaded; not bundled.
- License: upstream page says `other`; inspect model and dataset terms before production use or redistribution.
- Upstream model card reports validation mask AP 0.263 (IoU 0.50:0.95), AP50 0.620, AP75 0.179. This is not a planner-specific evaluation.

## Limitations and risks

No door, window, column, room, room-label, or dimension model is bundled. Optional Tesseract OCR emits text boxes but does not associate text with rooms or validate dimensions. Wall thickness, continuity, centerlines, material, and RF attenuation are not predicted. Thin lines, low contrast, scan artifacts, non-CubiCasa drawing conventions, and tile-boundary duplicates can affect output. Confidence is a model score, not calibrated probability. The default confidence threshold is an operational starting point, not evaluation-derived.

All geometry is returned in raster pixel coordinates. PDF pages are rasterized independently at configurable DPI; the raster dimensions and PDF point dimensions are included. No physical scale or engineering frame is inferred. A qualified reviewer must inspect every prediction before use.

## Evaluation

No held-out project dataset was supplied. See `evaluation/REPORT.md` and `evaluation/metrics.json`; project-specific metrics and latency remain unavailable.

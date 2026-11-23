# Floor plan CV delivery

Independent inference component for the Private 5G Deployment Planner. It returns editable, unreviewed pixel-space predictions and does not access a database, approve geometry, calibrate engineering scale, change building records, or calculate RF properties.

## Model and license gate

The default checkpoint is RAQEEB-CCTV v1.0.0, a YOLO11s instance-segmentation model. Its model card reports 11 classes covering walls, doors, windows, open passages, columns, stairs, elevators, escalators, exits, gates, and room/space regions. It is distributed under **CC BY-NC 4.0**. This package will refuse inference unless the caller sets `accept_noncommercial_license=True`. That flag confirms an operator's declaration; it does not grant commercial rights. Obtain a separate license from the model owner before any commercial deployment. See `MODEL_CARD.md` and `ATTRIBUTION.md`.

The model card reports per-class held-out results in `evaluation/metrics.json`. These are upstream-reported results, not a validation of this integration or of the planner's drawings. The escalator class has only four test instances and gate has none; both are marked unvalidated. No project-specific labelled evaluation set was supplied.

## Install

Python 3.11–3.14 is recommended. For CPU, install:

```bash
python -m pip install -r requirements.txt
```

For CUDA, first install a PyTorch/torchvision build that matches the target CUDA runtime using the official [PyTorch selector](https://pytorch.org/get-started/locally/), then install the remaining requirements. `device="auto"` uses CUDA when available and otherwise CPU. Explicit CUDA selection fails clearly when unsupported. PDF processing uses PyMuPDF (no Poppler needed). OCR also needs the Tesseract 5 executable installed separately.

Weights are downloaded lazily from Hugging Face revision `d0f2b17`, SHA-256 checked, and cached by Hugging Face. The upstream checkpoint is about 20.6 MiB (21.6 MB). Expect roughly 1–2 GB of host RAM for CPU inference; actual peak use depends on page dimensions and concurrent requests. Limit concurrent inference in the API worker.

## Independent interface

```python
from floorplan_cv_delivery.src import predict_floorplan

result = predict_floorplan("plan.pdf", {
    "device": "auto",                    # auto, cpu, cuda
    "dpi": 200,                           # PDF rendering DPI, 36–1200
    "include_ocr": True,
    "include_line_candidates": True,
    "accept_noncommercial_license": True # only after rights review
})
```

Input may be a PNG/JPG/JPEG/PDF path, bytes, file-like object, or PIL image. Every PDF page is rendered at `dpi / 72` points-to-pixels and retains its zero-based page index, raster dimensions, and PDF page dimensions in points. EXIF orientation is applied to images. No page content is cropped. High-resolution images use overlapping 1536 px tiles with 192 px overlap by default; polygon points and bounding boxes are translated back into the full oriented raster coordinate system. For each duplicate merged by same-class box IoU NMS, the retained prediction records the duplicate tile IDs and scores in `merged_duplicates`.

The Ultralytics model accepts a 1024 px inference size by default. Tiling helps retain small objects in high-resolution plans, but the upstream model card warns that tile-based room segmentation may create false tile-shaped regions. Room candidates therefore require careful review. Lower-score candidates down to 0.05 are retained and labeled for review; per-class operating thresholds are metadata, not silent exclusion rules.

The output includes element ID, class, confidence, polygon, bounding box, source image ID, PDF page index, tile ID, model version, candidate status, and uncertainty flags. Door orientation, wall thickness, column shape, window-to-wall links, OCR-to-room links, and dimension-line detection are marked heuristic candidates. Door opening direction is not inferred. OCR returns text, confidence, pixel box, room-label/dimension/scale candidate type, and a possible containing room. No OCR text is treated as calibrated physical measurement.

See `examples/predictions/sample_schema.json`. It is a schema illustration, not a real prediction. A project input sample was not supplied.

## FastAPI integration

The repository did not contain the FastAPI application mentioned in the handover specification. This package includes a drop-in router in `src/api.py`:

```python
from fastapi import FastAPI
from floorplan_cv_delivery.src.api import router as floorplan_cv_router

app = FastAPI()
app.include_router(floorplan_cv_router)
```

The router accepts one PNG/JPG/JPEG/PDF up to 25 MiB at `POST /floorplan-cv/predict`, runs inference in a worker thread, and maps invalid inputs/model failures to HTTP errors. Add your authentication, rate/concurrency controls, storage, reviewer workflow, and application-specific error handling in the planner. It does not persist or approve predictions.

## ONNX and checkpoint artifacts

The verified upstream PyTorch checkpoint is fetched at runtime. The package currently does **not** include generated ONNX files or claim ONNX validation. Use `python -m pip install -r requirements-export.txt` and `python tools/export_onnx.py --output-dir models` to export; run it on a system with sufficient free disk. Keep generated artifacts and their hashes with your own release after checking license terms. A combined segmentation model produces both boxes and masks; separate detection and segmentation ONNX files are not intrinsically needed. See `models/README.md` and `checkpoints/README.md`.

Measure a target plan on the target hardware with `python tools/benchmark.py plan.pdf --device cpu --accept-noncommercial-license`. The report records one cold-run latency and sampled memory; repeat after warmup for stable service benchmarks and record the machine/runtime details.

## Evaluation status and gaps

`evaluation/metrics.json` captures the upstream model card's held-out test split (1,682 plans) per class. OCR metrics, confusion matrix, hard-example images, CPU latency, GPU latency, and peak memory measured for this package remain unavailable. This environment could not install PyTorch because the system drive ran out of free space, so I could not execute checkpoint inference or export ONNX here. Do not treat upstream benchmark numbers as deployment benchmarks.

The delivery also does not provide an engineer-reviewed test set from your deployment domain, a commercial model license, or labelled room-name/dimension OCR ground truth. These are required to call the handover production-ready. The model does not guarantee wall continuity, wall thickness, or door opening direction; non-residential drawings and unusual scanning conventions may reduce accuracy.

# Floor plan CV delivery

Independent, read-only inference component intended for integration into the Private 5G Deployment Planner. It currently provides wall instance segmentation only, using a pretrained Hugging Face model. It does not access a database or perform RF calculations.

## Install

Python 3.10–3.12 is recommended. Install a CPU or CUDA-compatible PyTorch build for your platform, then:

```bash
python -m pip install -r requirements.txt
```

PDF support uses PyMuPDF and does not require Poppler. Optional OCR requires `pip install pytesseract` and a separately installed Tesseract executable.

## Independent interface

```python
from floorplan_cv_delivery.src import predict_floorplan

result = predict_floorplan("plan.pdf", {
    "device": "auto",                 # auto, cpu, cuda
    "dpi": 200,                        # PDF rasterization, 36–1200
    "confidence_threshold": 0.35,
    "tile_size": 1024,
    "tile_overlap": 128,
    "include_ocr": False,
})
```

The input may be a PNG/JPG/JPEG/PDF path, bytes, file-like object, or PIL image. Every PDF page is processed and retains a zero-based page index. PDF rasterization uses `dpi / 72` points-to-pixels scaling, preserves page aspect ratio, and does not crop content. Large images are covered by overlapping tiles; returned polygon coordinates are translated back into the complete original oriented raster coordinate system. Duplicate candidates are suppressed using same-class bounding-box IoU NMS.

The return object includes `model_version`, model/device information, source page dimensions, detections, optional OCR text boxes, and an explicit unreviewed status. See `examples/predictions/sample_schema.json` for a schema illustration (not a real prediction).

## Integration boundary

Call `predict_floorplan(image, options)` from a FastAPI endpoint or background worker. Keep file size/page count limits, authentication, persistence, engineering-coordinate calibration, reviewer approval, and RF planning in the application/backend. This package never changes building records. Inference errors are raised as clear `ValueError` or `RuntimeError` exceptions; translate them into the API's error contract at the integration layer.

## Runtime and artifacts

Model loading is lazy and cached per device/model/revision. `auto` selects CUDA when `torch.cuda.is_available()` and otherwise CPU. Explicit CUDA selection fails clearly when unavailable. GPU execution has not been tested in this environment and is not claimed as verified. CPU execution code is provided, but a clean-environment inference run and benchmark remain outstanding. Model weights are not committed; first inference downloads them from Hugging Face. The upstream revision is currently `main`; pin a commit SHA and verify license/checksums before production deployment.

## Scope and handover gaps

The selected upstream checkpoint documents **one class: wall**. It cannot satisfy doors, windows, columns, rooms, OCR room association, or dimensions by itself. `src/ocr.py` is an optional generic Tesseract adapter; OCR accuracy is not evaluated. No suitable training data, project-held-out annotations, trained checkpoints, ONNX exports, or measured runtime benchmarks were included with the request. Accordingly this repository does not claim those artifacts or metrics. See `configs/`, `MODEL_CARD.md`, and `evaluation/REPORT.md`.

To complete the full specification, the project owner must provide an appropriately licensed labeled dataset and evaluation split, agree on class definitions and backend schema, train/evaluate the missing class models, validate CPU/GPU deployment on target hardware, and resolve upstream model/dataset licensing. The current package is a runnable integration starting point, not a fully accepted all-class model handover.

The evaluation report is also available as `evaluation/evaluation_report.pdf`.

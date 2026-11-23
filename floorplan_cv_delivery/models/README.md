# Exported models

No ONNX file is committed. The upstream checkpoint combines box detection and instance masks, so one segmentation ONNX generally exposes both outputs. It must be exported and verified with the target ONNX Runtime execution provider before release.

Run `python tools/export_onnx.py --output-dir models` in an environment with Ultralytics, PyTorch, ONNX, ONNX Runtime, and enough free disk. The model is licensed CC BY-NC 4.0; review that license before distributing exports.

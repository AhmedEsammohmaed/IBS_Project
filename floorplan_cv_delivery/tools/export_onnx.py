"""Export the pinned joint box+mask checkpoint and run a CPU ONNX smoke check."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan_cv_delivery.src.model_loader import MODEL_ID, MODEL_REVISION, WEIGHTS_FILE, WEIGHTS_SHA256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("models"))
    parser.add_argument("--imgsz", type=int, default=1024)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(hf_hub_download(MODEL_ID, WEIGHTS_FILE, revision=MODEL_REVISION))
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256:
        raise SystemExit("Checkpoint checksum mismatch")
    model = YOLO(str(checkpoint))
    exported = Path(model.export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=False, opset=17, device="cpu"))
    destination = args.output_dir / "floorplan_segmentation.onnx"
    shutil.copy2(exported, destination)
    # Reload through ONNX Runtime/Ultralytics and execute a blank CPU input.
    check = YOLO(str(destination), task="segment")
    check.predict(source=Image.new("RGB", (640, 480), "white"), imgsz=args.imgsz, device="cpu", verbose=False)
    report = {
        "onnx_file": destination.name,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "exporter": "Ultralytics", "opset": 17, "input_size": args.imgsz,
        "validation": "CPU inference smoke check completed",
        "execution_provider": "CPUExecutionProvider (verify in target runtime)",
        "detection_and_segmentation_outputs": "joint model output",
    }
    (args.output_dir / "onnx_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

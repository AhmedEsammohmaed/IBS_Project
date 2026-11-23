"""Measure end-to-end inference time and sampled process/GPU memory for one plan."""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from floorplan_cv_delivery.src import predict_floorplan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--output", type=Path, default=Path("benchmark.json"))
    parser.add_argument("--accept-noncommercial-license", action="store_true")
    args = parser.parse_args()
    if not args.accept_noncommercial_license:
        parser.error("review the CC BY-NC license, then pass --accept-noncommercial-license")
    process = psutil.Process()
    peak_rss = [process.memory_info().rss]
    stop = threading.Event()

    def sample_memory() -> None:
        while not stop.is_set():
            try:
                peak_rss[0] = max(peak_rss[0], process.memory_info().rss)
            except psutil.Error:
                pass
            stop.wait(0.05)

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    start = time.perf_counter()
    result = predict_floorplan(args.image, {
        "device": args.device, "dpi": args.dpi,
        "accept_noncommercial_license": True,
    })
    elapsed_ms = (time.perf_counter() - start) * 1000
    stop.set()
    sampler.join()
    try:
        import torch
        peak_gpu_bytes = int(torch.cuda.max_memory_allocated()) if args.device == "cuda" else None
    except (ImportError, RuntimeError):
        peak_gpu_bytes = None
    report = {
        "image": str(args.image), "device": args.device,
        "elapsed_ms_including_model_load": round(elapsed_ms, 2),
        "sampled_peak_process_rss_bytes": peak_rss[0],
        "peak_cuda_allocated_bytes": peak_gpu_bytes,
        "input_pages": len(result["source"]["pages"]),
        "input_pixels": [p["image_width"] * p["image_height"] for p in result["source"]["pages"]],
        "prediction_count": len(result["detections"]),
        "model_version": result["model_version"],
        "warmup_runs": 0,
        "note": "One cold-run measurement; repeat after warmup and report hardware/runtime before comparing.",
    }
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

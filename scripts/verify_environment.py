"""Run a local CUDA inference smoke check; this is NOT an OK/NG accuracy test."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for key, subdir in {"HF_HOME": "huggingface", "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ[key] = str(ROOT / ".cache" / subdir)
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"
sys.stdout.reconfigure(encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, help="Optional local RGB test image")
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from rfdetr import RFDETRSmall

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Check GPU driver and CUDA-enabled torch installation.")
    weights = ROOT / "models" / "rf-detr-small.pth"
    manifest = json.loads((ROOT / "models" / "manifest.json").read_text(encoding="utf-8"))
    with weights.open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if sha != manifest["sha256"]:
        raise RuntimeError("Checkpoint SHA256 differs from the recorded official download.")
    artifact_dir = ROOT / "artifacts"
    artifact_dir.mkdir(exist_ok=True)
    if args.image:
        frame = Image.open(args.image).convert("RGB")
        input_kind = "user_supplied_image"
    else:
        frame = Image.new("RGB", (1280, 720), "#e4e7eb")
        draw = ImageDraw.Draw(frame)
        draw.rectangle((180, 170, 420, 420), fill="#426fa8")
        draw.ellipse((650, 220, 930, 500), fill="#c47145")
        input_kind = "synthetic_shapes_smoke_test_not_accuracy_evaluation"
    frame.save(artifact_dir / "smoke-input.png")
    print("Loading verified official Small checkpoint on CUDA...", flush=True)
    started = time.perf_counter()
    model = RFDETRSmall(pretrain_weights=str(weights), device="cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for _ in range(2):
            model.predict(frame, threshold=0.0)
        torch.cuda.synchronize()
        # RF-DETR moves parameters to the requested device lazily at first predict.
        actual_device = str(next(model.model.model.parameters()).device)
        if not actual_device.startswith("cuda"):
            raise RuntimeError(f"Model is unexpectedly on {actual_device}")
        timings = []
        for _ in range(5):
            started = time.perf_counter()
            detections = model.predict(frame, threshold=0.0)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - started) * 1000)
    if len(detections) == 0:
        raise RuntimeError("No raw candidate outputs at threshold zero.")
    if not np.isfinite(detections.xyxy).all() or not np.isfinite(detections.confidence).all():
        raise RuntimeError("Non-finite inference output.")
    report = {
        "status": "passed", "time_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "CUDA model loading and numerical inference only; no defect accuracy or robot validation.",
        "input_kind": input_kind, "input_size": list(frame.size),
        "os": platform.platform(), "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "cuda_runtime": torch.version.cuda, "model_device": actual_device,
        "torch_arch_list": torch.cuda.get_arch_list(),
        "packages": {name: metadata.version(name) for name in ["torch", "torchvision", "rfdetr", "transformers", "supervision"]},
        "weights_sha256": sha, "model_resolution": model.model_config.resolution,
        "load_seconds": round(load_seconds, 3), "warmup_runs": 2, "measured_runs": 5,
        "prediction_ms": [round(x, 3) for x in timings],
        "median_prediction_ms": round(statistics.median(timings), 3),
        "peak_allocated_vram_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "raw_candidates": len(detections),
        "detections_above_0_5": int((detections.confidence >= 0.5).sum()),
        "finite_outputs": True,
        "timing_note": "Local unoptimized single-image predict call with CUDA synchronization; not a production benchmark.",
    }
    (artifact_dir / "environment-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)

if __name__ == "__main__":
    main()

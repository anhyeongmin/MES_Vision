"""Hardware-independent contract tests and optional real RF-DETR CUDA bridge smoke."""
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key, subdir in {"HF_HOME": "huggingface", "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ[key] = str(ROOT / ".cache" / subdir)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []

    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def gpu_check(destination):
    import torch
    from PIL import Image, ImageDraw
    from mes_vision.inputs import ImageSource
    from mes_vision.inspection import InspectionPipeline
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDetector
    path = destination / "gpu.json"
    write(path, {"status": "running"})
    backend = None
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        image = Image.new("RGB", (1280, 720), "#eeeeee")
        draw = ImageDraw.Draw(image)
        draw.rectangle((160, 100, 420, 460), fill="#406faa")
        draw.ellipse((800, 200, 1050, 500), fill="#bb7744")
        input_path = destination / "gpu-synthetic.png"
        image.save(input_path)
        with ImageSource(input_path) as source:
            frame = source.read().frame
        manifest = json.loads((ROOT / "models/manifest.json").read_text(encoding="utf-8"))
        backend = RFDETRBackend(ROOT / "models/rf-detr-small.pth", manifest["sha256"], threshold=0.0, training_scope="coco_general")
        started = time.perf_counter()
        backend.load()
        detector = RFDETRDetector(backend)
        batch = detector.detect(frame)
        torch.cuda.synchronize()
        actual_device = str(next(backend._network.model.model.parameters()).device)
        assert actual_device.startswith("cuda"), actual_device
        assert len(batch.detections) > 0
        assert batch.image_size == (1280, 720)
        backend.threshold = .5  # Smoke-only candidate threshold, not a product criterion.
        report = InspectionPipeline(detector).run(frame)
        assert report.execution_status != "ERROR", report.issues
        assert report.final_decision is None and not report.robot_commands_enabled
        write(destination / "gpu-pipeline.json", report.to_dict())
        write(path, {"status": "passed", "checked_at_utc": datetime.now(timezone.utc).isoformat(),
                     "scope": "actual GPU adapter execution on synthetic image; no product defect accuracy",
                     "gpu": torch.cuda.get_device_name(0), "actual_device": actual_device,
                     "model": batch.model.__dict__ if hasattr(batch.model, "__dict__") else report.to_dict()["detector"],
                     "original_size": batch.image_size, "coordinate_space": batch.coordinate_space,
                     "raw_candidates_at_threshold_0": len(batch.detections),
                     "objects_at_smoke_threshold_0_5": len(report.objects),
                     "load_and_checks_seconds": time.perf_counter() - started,
                     "product_accuracy_tested": False, "camera_or_robot_used": False})
    except Exception as exc:
        write(path, {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise
    finally:
        if backend:
            backend.close()


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    destination = ROOT / "artifacts/inspection-check"
    destination.mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(suite)
    report = {"checked_at_utc": datetime.now(timezone.utc).isoformat(), "passed": result.wasSuccessful(),
              "tests_run": result.testsRun, "passed_tests": result.passed,
              "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures + result.errors],
              "gpu_check_requested": args.gpu, "gpu_check_status": "not_run",
              "product_accuracy_tested": False}
    write(destination / "report.json", report)
    if not result.wasSuccessful():
        return 1
    if args.gpu:
        try:
            gpu_check(destination)
            report["gpu_check_status"] = "passed"
        except Exception:
            report["gpu_check_status"] = "failed"
            report["passed"] = False
            write(destination / "report.json", report)
            raise
        write(destination / "report.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Regression tests, with optional actual training/resume/evaluation/export verification."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.config import JobConfig
from mes_vision.training.data import read_json, write_json, require
from mes_vision.training.fixtures import make_fixture


def verify_exports(destination):
    from mes_vision.inputs import ImageSource
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDetector, RFDETRDefectInspector
    from mes_vision.inspection.geometry import extract_crop
    from mes_vision.inspection import Box
    from mes_vision.training.jobs import verified_artifact
    results = []
    for name, scope in (("objects-resumed", "product_objects"), ("defects-trained", "product_defects")):
        run = destination / name
        manifest, model_meta = read_json(run / "run.json"), read_json(run / "model.json")
        weights = verified_artifact(run, manifest["checkpoints"]["inference"])
        backend = RFDETRBackend(weights, manifest["checkpoints"]["inference"]["sha256"], threshold=0.0, training_scope=scope,
                               class_names=tuple(c["name"] for c in model_meta["categories"]))
        try:
            backend.load()
            with ImageSource(Path(manifest["config"]["dataset_dir"]) / "test/synthetic-0.png") as source:
                frame = source.read().frame
            predictions = backend.predict_rgb(frame.rgb)
            require(predictions and all(0 <= p.class_id < len(model_meta["categories"]) for p in predictions), "export label range mismatch")
            if scope == "product_objects":
                require(RFDETRDetector(backend).detect(frame).image_size == (512, 512), "export object adapter dimensions mismatch")
            else:
                codes = {c["label_index"]: c["name"] for c in model_meta["categories"]}
                check = RFDETRDefectInspector(backend, codes).inspect(extract_crop(frame, "synthetic-object", Box(0, 0, 512, 512)))
                require(all(x.defect_code in codes.values() for x in check.findings), "defect label mapping mismatch")
            results.append({"run": name, "passed": True, "synthetic": True, "candidate_count": len(predictions),
                            "excluded_reserved_slots": backend.excluded_reserved_slots,
                            "classes": model_meta["categories"], "sha256": manifest["checkpoints"]["inference"]["sha256"]})
        finally:
            backend.close()
    write_json(destination / "export-check.json", {"passed": True, "results": results, "product_accuracy_tested": False})


class RecordedResult(unittest.TextTestResult):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.passed = []
    def addSuccess(self, test):
        super().addSuccess(test)
        self.passed.append(test.id())


def gpu_suite(destination):
    manifest = read_json(ROOT / "models/manifest.json")
    for role, name, epochs, precision in [("object_detector", "objects", 2, "32-true"), ("known_defect_detector", "defects", 1, "bf16-mixed")]:
        dataset = make_fixture(destination / f"{name}-data", role)
        config = JobConfig(role, str(dataset), str(ROOT / "models/rf-detr-small.pth"), manifest["sha256"],
                           epochs=epochs, batch_size=1, grad_accum_steps=1, precision=precision, synthetic=True)
        write_json(destination / f"{name}.json", config.snapshot())
    def run(name, arguments, script="training.py"):
        with (destination / f"{name}.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *[str(v) for v in arguments]],
                                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        require(completed.returncode == 0, f"{name} failed; see {destination / (name+'.log')}")
    run("first", ["train", "--config", destination / "objects.json", "--output", destination / "objects-first", "--stop-after-epoch", "1"])
    run("resume", ["train", "--config", destination / "objects.json", "--output", destination / "objects-resumed", "--resume-from", destination / "objects-first"])
    run("defects", ["train", "--config", destination / "defects.json", "--output", destination / "defects-trained"])
    run("test", ["evaluate", "--run", destination / "objects-resumed", "--output", destination / "objects-test", "--split", "test"])
    run("valid", ["evaluate", "--run", destination / "defects-trained", "--output", destination / "defects-valid", "--split", "valid"])
    run("exports", ["--exports-only", destination], "verify_training.py")
    return summarize_session(destination)


def summarize_session(destination):
    from mes_vision.training.jobs import verified_artifact
    for name in ("objects-first", "objects-resumed", "defects-trained"):
        run = destination / name
        manifest = read_json(run / "run.json")
        for entry in manifest["checkpoints"].values():
            verified_artifact(run, entry)
    require(read_json(destination / "objects-test/evaluation.json")["status"] == "COMPLETED", "test failed")
    require(read_json(destination / "defects-valid/evaluation.json")["status"] == "COMPLETED", "validation failed")
    require(read_json(destination / "export-check.json")["passed"], "export check failed")
    first = read_json(destination / "objects-first/run.json")
    resumed = read_json(destination / "objects-resumed/run.json")
    require(first["status"] == "STOPPED" and resumed["status"] == "COMPLETED", "stop/resume status mismatch")
    require(first["audit"]["global_step"] == resumed["audit"]["at_train_start"]["global_step"] < resumed["audit"]["global_step"], "training steps did not continue")
    return {"passed": True, "path": str(destination), "first_global_step": first["audit"]["global_step"],
            "resumed_global_step": resumed["audit"]["global_step"], "restored_optimizer_states": resumed["resume"]["optimizer_state_entries"],
            "optimizer_moments_and_scheduler_verified": True, "object_training_precision": "32-true", "defect_training_precision": "bf16-mixed",
            "validation_test_export_checks": "passed", "product_accuracy_tested": False}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--exports-only", type=Path)
    parser.add_argument("--existing-session", type=Path, help="Verify existing training artifacts and rerun their export inference")
    args = parser.parse_args()
    for key, folder in {"HF_HOME": "huggingface", "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib"}.items():
        os.environ[key] = str(ROOT / ".cache" / folder)
    os.environ["HF_HUB_OFFLINE"] = "1"
    if args.exports_only:
        verify_exports(args.exports_only)
        return 0
    root = ROOT / "artifacts/training-check"
    root.mkdir(parents=True, exist_ok=True)
    result = unittest.TextTestRunner(verbosity=2, resultclass=RecordedResult).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    report = {"passed": result.wasSuccessful(), "tests_run": result.testsRun, "passed_tests": result.passed,
              "failures": [{"test": str(test), "detail": detail} for test, detail in result.failures + result.errors],
              "gpu": {"status": "not_run"}, "product_accuracy_tested": False}
    write_json(root / "report.json", report)
    if not result.wasSuccessful():
        return 1
    if args.gpu or args.existing_session:
        require(not (args.gpu and args.existing_session), "choose --gpu or --existing-session")
        destination = args.existing_session.resolve() if args.existing_session else root / ("run-" + uuid4().hex[:8])
        if not args.existing_session:
            destination.mkdir()
        report["gpu"] = {"status": "running", "path": str(destination)}
        write_json(root / "report.json", report)
        try:
            if args.existing_session:
                verify_exports(destination)
                report["gpu"] = summarize_session(destination)
                report["gpu"]["existing_session_reverified"] = True
            else:
                report["gpu"] = gpu_suite(destination)
        except Exception as exc:
            report["passed"] = False
            report["gpu"] = {"status": "failed", "path": str(destination), "error": str(exc)}
            raise
        finally:
            write_json(root / "report.json", report)
    print(json.dumps({"passed": report["passed"], "tests_run": report["tests_run"], "gpu": report["gpu"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

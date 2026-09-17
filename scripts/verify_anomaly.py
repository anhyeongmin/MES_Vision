"""Regression tests and optional real CUDA DINO normal registration/reload checks."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
from mes_vision.training.data import read_json, write_json, require


def gpu_check(output):
    import numpy as np
    import torch
    from PIL import Image, ImageDraw
    from mes_vision.anomaly.features import DinoFeatures
    from mes_vision.anomaly.bank import Bank, build_bank, BankConfig
    from mes_vision.anomaly.scoring import AnomalyEngine, Criteria, AnomalyInspector, save_score
    from mes_vision.data_management.fixtures import make_fixture
    from mes_vision.data_management.export import export_collection
    from mes_vision.inputs import ImageSource
    from mes_vision.inspection import Box, Detection, InspectionPipeline, Mode, CheckStatus
    from mes_vision.inspection.adapters import MockDetector
    require(torch.cuda.is_available(), "CUDA required")
    torch.set_num_threads(4)
    collection = make_fixture(output / "fixture")
    normal = output / "normal-export"
    report = export_collection(collection, normal, "normal_bank")
    original_path = normal / report["items"][0]["file"]
    with Image.open(original_path) as image:
        rgb = np.array(image)
        altered = image.copy()
    draw = ImageDraw.Draw(altered)
    draw.rectangle((30, 35, 69, 94), fill=(5, 5, 5))
    altered_path = output / "synthetic-altered.png"
    altered.save(altered_path)
    changed = np.array(altered)
    config = output / "build-config.json"
    write_json(config, {"image_size": 448, "bank": asdict(BankConfig(16384, 512, 32, 42))})
    def cli(name, arguments):
        with (output / (name + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run([sys.executable, str(ROOT / "scripts/anomaly.py"), *[str(v) for v in arguments]],
                                    cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=180)
        require(result.returncode == 0, f"CLI {name} failed; inspect log")
    bank_path = output / "bank"
    cli("build", ["build", "--normal-export", normal, "--output", bank_path, "--config", config, "--synthetic"])
    cli("validate-bank", ["validate-bank", "--bank", bank_path, "--product-id", "fixture-part", "--synthetic"])
    cli("inspect", ["inspect", "--bank", bank_path, "--product-id", "fixture-part", "--image", altered_path,
                    "--output", output / "cli-altered", "--synthetic"])
    require(read_json(output / "cli-altered/result.json")["status"] == "UNCERTAIN", "unconfigured criteria must not pass/fail")
    extractor = DinoFeatures(ROOT)
    engine = AnomalyEngine(bank_path, extractor, product_id="fixture-part", allow_synthetic=True)
    torch.cuda.reset_peak_memory_stats()
    reference, ref_grid, ref_nearest = engine.score(rgb)
    altered_result, altered_grid, altered_nearest = engine.score(changed)
    require(np.isfinite(altered_grid).all() and abs(altered_result["raw_score"]-reference["raw_score"]) > 1e-4,
            "synthetic perturbation did not change DINO score")
    save_score(output / "reference-score", rgb, reference, ref_grid, ref_nearest)
    save_score(output / "altered-score", changed, altered_result, altered_grid, altered_nearest)
    timings = []
    for _ in range(3):
        torch.cuda.synchronize()
        started = time.perf_counter()
        repeated, repeated_grid, _ = engine.score(changed)
        torch.cuda.synchronize()
        timings.append((time.perf_counter()-started)*1000)
        np.testing.assert_allclose(repeated_grid, altered_grid, atol=1e-6, rtol=1e-6)
    peak_memory = torch.cuda.max_memory_allocated()/2**20
    extractor.close()
    fresh = DinoFeatures(ROOT)
    reloaded = AnomalyEngine(bank_path, fresh, product_id="fixture-part", allow_synthetic=True)
    reloaded_result, reload_grid, _ = reloaded.score(changed)
    np.testing.assert_allclose(reload_grid, altered_grid, atol=1e-6, rtol=1e-6)
    # Exercise spatial regions with an explicitly unvalidated synthetic display threshold.
    display_threshold = float(np.quantile(reload_grid, .85))
    criteria = Criteria("synthetic-display-only", reloaded.bank.digest, "fixture-part", 0, 2,
                        max(display_threshold, 1e-6), validated=False, kind="synthetic")
    inspector = AnomalyInspector(AnomalyEngine(bank_path, fresh, product_id="fixture-part", criteria=criteria, allow_synthetic=True),
                                evidence_dir=output / "object-evidence")
    whole = np.zeros((changed.shape[0]+40, changed.shape[1]*2+70, 3), np.uint8)
    whole[20:20+rgb.shape[0], 20:20+rgb.shape[1]] = rgb
    second_x = 50+rgb.shape[1]
    whole[20:20+rgb.shape[0], second_x:second_x+rgb.shape[1]] = changed
    whole_path = output / "two-objects.png"
    Image.fromarray(whole).save(whole_path)
    with ImageSource(whole_path) as source:
        frame = source.read().frame
    detector = MockDetector((Detection(Box(20, 20, 20+rgb.shape[1], 20+rgb.shape[0]), .99, 0, "fixture-part"),
                            Detection(Box(second_x, 20, second_x+rgb.shape[1], 20+rgb.shape[0]), .99, 0, "fixture-part")))
    pipeline = InspectionPipeline(detector, (inspector,), mode=Mode.SIMULATION, product_id="fixture-part")
    combined = pipeline.run(frame)
    checks = [next(c for c in obj.checks if c.check_id == "anomaly") for obj in combined.objects]
    require(len(checks) == 2 and all(c.status == CheckStatus.UNCERTAIN for c in checks), "actual anomaly adapter pipeline failure")
    require(checks[1].findings and all(f.original_box is not None for f in checks[1].findings), "anomaly regions not mapped to original frame")
    require(all(f.defect_code is None for c in checks for f in c.findings), "unvalidated regions cannot claim NG_UNKNOWN")
    write_json(output / "pipeline.json", combined.to_dict())
    fresh.close()
    return {"passed": True, "device": torch.cuda.get_device_name(), "synthetic": True,
            "bank_patches": reloaded.bank.meta["bank_patches"], "bank_digest": reloaded.bank.digest,
            "reference_score": reference["raw_score"], "altered_score": altered_result["raw_score"],
            "grid_shape": list(altered_grid.shape), "repeat_and_fresh_reload_equal": True,
            "cli_build_validate_inspect_passed": True, "two_object_pipeline_passed": True,
            "warm_score_ms": timings, "median_warm_score_ms": float(np.median(timings)),
            "peak_torch_allocated_mib": peak_memory, "product_accuracy_tested": False,
            "product_thresholds_set": False, "detector_in_integration_test": "explicit mocked boxes; DINO features and distances real CUDA"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    output = ROOT / "artifacts/anomaly-check" / ("run-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    report = {"passed": False, "path": str(output), "tests_run": result.testsRun, "errors": len(result.errors),
              "failures": len(result.failures), "gpu": None, "product_accuracy_tested": False}
    try:
        require(result.wasSuccessful(), "regression failure; inspect tests.log")
        if args.gpu:
            report["gpu"] = gpu_check(output)
        report["passed"] = True
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "report.json", report)
    write_json(output.parent / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

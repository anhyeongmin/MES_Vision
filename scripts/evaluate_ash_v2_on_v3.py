"""Evaluate the frozen v2 defect model on v3 crops without altering either run."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, "reconfigure"):
        stream.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ash_reinforcement import BATCH, OLD, FINGERPRINT
from mes_vision.training.config import JobConfig
from mes_vision.training.data import read_json, write_json, require, validate_dataset, sha256
from mes_vision.training.jobs import build_components, preflight, verified_artifact, utc


def main():
    require(read_json(BATCH / "review/summary.json")["status"] == "COMPLETED", "threshold/test comparison must finish first")
    require(read_json(BATCH / "detail-defects-test/evaluation.json")["status"] == "COMPLETED", "reinforced native test incomplete")
    old_run = OLD / "detail-defects"
    old, new = read_json(old_run / "run.json"), read_json(BATCH / "detail-defects/run.json")
    config = JobConfig(**new["config"])
    data = preflight(config)
    require(data["fingerprint"] == FINGERPRINT, "v3 dataset changed")
    require(data["categories"] == read_json(old_run / "model.json")["categories"], "class mapping differs")
    weights = verified_artifact(old_run, old["checkpoints"]["inference"])
    output = BATCH / "baseline-on-v3-test"
    output.mkdir(exist_ok=False)
    report = {"status": "RUNNING", "source_run": str(old_run), "split": "test", "synthetic": True,
        "production_ready": False, "cross_dataset_evaluation": True,
        "scope": "frozen v2 model on v3 test; known CAD and new rendering conditions only",
        "weights_sha256": sha256(weights), "source_training_dataset": old["dataset_fingerprint"],
        "dataset_fingerprint": data["fingerprint"], "precision": config.precision,
        "source_script_sha256": sha256(Path(__file__)), "started_at_utc": utc()}
    write_json(output / "evaluation.json", report)
    try:
        module, datamodule, trainer = build_components(config, data, output / "engine", weights=weights, training=False)
        metrics = trainer.test(module, datamodule=datamodule, verbose=False)
        require(metrics and metrics[0], "no evaluation metrics")
        require(validate_dataset(config.dataset_dir)["fingerprint"] == FINGERPRINT, "dataset changed during evaluation")
        report.update(status="COMPLETED", metrics=metrics)
    except BaseException as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_at_utc"] = utc()
        write_json(output / "evaluation.json", report)
    print("FROZEN BASELINE ON V3 TEST COMPLETED")


if __name__ == "__main__":
    main()

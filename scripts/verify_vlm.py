"""Regression/UI checks and optional actual Qwen generation + foreground priority."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.data import read_json, write_json, require, sha256
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.worker import AnalysisWorker
from mes_vision.vlm.gpu import GpuCoordinator


def wait_until(condition, timeout):
    deadline = time.monotonic()+timeout
    while time.monotonic() < deadline:
        if condition(): return
        time.sleep(.05)
    raise TimeoutError("verification wait timed out")


def gpu_check(output):
    import torch
    from mes_vision.inspection import InspectionPipeline
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDetector
    require(torch.cuda.is_available(), "CUDA unavailable")
    data = make_vlm_fixture(output / "gpu-fixture", enabled=True)
    queue = data["queue"]
    first = data["job_id"]
    second = queue.enqueue(data["snapshot"], data["result"].objects[0].object_id, asdict(GenerationConfig()))
    before = sha256(data["snapshot"] / "inspection.json")
    worker = AnalysisWorker(queue, ROOT, allow_synthetic=True)
    worker.run(max_jobs=2)
    jobs = [queue.get(first), queue.get(second)]
    require(all(j["state"] == "COMPLETED" for j in jobs), "actual Qwen structured response failed; inspect queue jobs")
    require(sha256(data["snapshot"] / "inspection.json") == before, "VLM changed original inspection")
    require(all(j["result"]["advisory_only"] for j in jobs), "analysis cannot own final decisions")
    manifest = read_json(ROOT / "models/manifest.json")
    detector_backend = RFDETRBackend(ROOT / "models/rf-detr-small.pth", manifest["sha256"], threshold=.5, training_scope="coco_general")
    gate = GpuCoordinator(queue.root / "gpu-coordination")
    detector_backend.load()
    pipeline = InspectionPipeline(RFDETRDetector(detector_backend), product_id="fixture-part", gpu_coordinator=gate)
    pipeline.run(data["frame"])
    baseline = pipeline.run(data["frame"])
    third = queue.enqueue(data["snapshot"], data["result"].objects[1].object_id, asdict(GenerationConfig(max_new_tokens=129)))
    priority_worker = AnalysisWorker(queue, ROOT, allow_synthetic=True)
    errors = []
    def run():
        try: priority_worker.run(max_jobs=1)
        except Exception as exc: errors.append(str(exc))
    thread = threading.Thread(target=run)
    thread.start()
    try:
        wait_until(lambda: queue.runtime() is not None and queue.runtime()["phase"] == "ANALYZING", 150)
        time.sleep(.5)
        started = time.perf_counter()
        priority_result = pipeline.run(data["frame"])
        total_ms = (time.perf_counter()-started)*1000
        priority_worker.stop_event.set()
        thread.join(12)
        require(not thread.is_alive() and not errors, "priority worker did not stop cleanly")
        require(queue.get(third)["state"] == "PENDING", "foreground did not defer active VLM work")
        require(queue.runtime()["model_pid"] is None, "VLM child remained after foreground priority")
        require(priority_result.execution_status != "ERROR", "RF-DETR failed under coordinated GPU use")
        write_json(output / "foreground-result.json", priority_result.to_dict())
    finally:
        priority_worker.stop_event.set()
        thread.join(12)
        detector_backend.close()
        queue.set_enabled(False)
    result = {"passed": True, "device": torch.cuda.get_device_name(), "synthetic": True, "queue": str(queue.root),
              "structured_responses": [{"job_id": j["id"], "analysis": j["result"]["analysis"], "metrics": j["result"]["metrics"]} for j in jobs],
              "model": jobs[0]["result"]["model"], "inspection_unchanged": True,
              "foreground_baseline_ms": baseline.elapsed_ms, "foreground_wait_for_vlm_stop_ms": priority_result.config["gpu_wait_ms"],
              "foreground_inference_ms_after_stop": priority_result.elapsed_ms, "foreground_total_wall_ms": total_ms,
              "foreground_priority_passed": True, "product_accuracy_tested": False, "laptop_tested": False,
              "timing_scope": "One actual interruption on RTX3090; not a latency guarantee or parallel GPU throughput benchmark"}
    write_json(output / "gpu.json", result)
    return result


def orphan_check(output):
    """Windows process handles verify that forced supervisor death kills its child."""
    if os.name != "nt": return {"tested": False, "reason": "Windows host test"}
    data = make_vlm_fixture(output / "orphan-fixture", enabled=True)
    queue = data["queue"]
    with (output / "orphan-worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen([sys.executable, str(ROOT / "scripts/vlm.py"), "worker", "--queue", str(queue.root),
            "--synthetic", "--simulate", "--mock-behavior", "slow", "--max-jobs", "1"], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        handle = None
        try:
            wait_until(lambda: queue.runtime() is not None and queue.runtime()["phase"] == "ANALYZING", 15)
            import ctypes
            from ctypes import wintypes
            kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
            kernel.OpenProcess.restype = wintypes.HANDLE
            kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
            kernel.WaitForSingleObject.restype = wintypes.DWORD
            kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            handle = kernel.OpenProcess(0x00100000, False, queue.runtime()["model_pid"])
            require(bool(handle), "could not observe model process")
            process.kill()
            process.wait(timeout=5)
            require(kernel.WaitForSingleObject(handle, 5000) == 0, "orphan model process survived supervisor death")
            AnalysisWorker(queue, ROOT, backend="mock", allow_synthetic=True).run(max_jobs=1)
            require(queue.get(data["job_id"])["state"] == "COMPLETED", "interrupted request did not recover")
            return {"tested": True, "child_exited": True, "request_recovered": True, "backend": "explicit mock"}
        finally:
            if process.poll() is None: process.kill(); process.wait(timeout=5)
            if handle: kernel.CloseHandle(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    output = ROOT / "artifacts/vlm-check" / ("run-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    with (output / "tests.log").open("w", encoding="utf-8") as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
    report = {"passed": False, "path": str(output), "tests_run": result.testsRun, "errors": len(result.errors),
              "failures": len(result.failures), "gpu": None, "product_accuracy_tested": False}
    try:
        require(result.wasSuccessful(), "regression failure; inspect tests.log")
        report["orphan_recovery"] = orphan_check(output)
        with (output / "ui.log").open("w", encoding="utf-8") as log:
            ui = subprocess.run([sys.executable, str(ROOT / "scripts/verify_vlm_ui.py"), "--output", str(output / "ui")],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=60, env=dict(os.environ, QT_QPA_PLATFORM="offscreen"))
        require(ui.returncode == 0, "UI verification failed; inspect ui.log")
        report["ui"] = read_json(output / "ui/report.json")
        if args.gpu: report["gpu"] = gpu_check(output)
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

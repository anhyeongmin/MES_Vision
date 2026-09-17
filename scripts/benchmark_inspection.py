"""Resident GPU workload and optional actual background Qwen contention measurement."""
from pathlib import Path
import argparse
import csv
from dataclasses import asdict
import json
import sys
import threading
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--counts", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--bank-size", type=int, default=2048)
    parser.add_argument("--vlm", action="store_true")
    args = parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    from filelock import FileLock
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.anomaly.features import DinoFeatures
    from mes_vision.anomaly.scoring import nearest_neighbors
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import require, write_json, sha256
    from mes_vision.validation.metrics import distribution
    from mes_vision.vlm.fixtures import make_vlm_fixture
    from mes_vision.vlm.worker import AnalysisWorker
    from mes_vision.vlm.backend import GenerationConfig
    require(torch.cuda.is_available(), "CUDA required")
    require(5 <= args.samples <= 10000 and all(1 <= n <= 100 for n in args.counts)
        and len(set(args.counts)) == len(args.counts) and 1 <= args.bank_size <= 20000, "Invalid benchmark workload")
    args.output.mkdir(parents=True, exist_ok=False)
    config = JobConfig.load(args.project_root / "configs/training/objects.json")
    backends = [RFDETRBackend(config.initial_weights, config.initial_sha256, threshold=.4, training_scope="coco_general") for _ in range(2)]
    features = None; worker = None; thread = None; errors = []
    report = {"schema_version": 1, "status": "RUNNING", "gpu": torch.cuda.get_device_name(),
        "product_accuracy_validated": False, "laptop_tested": False, "workload_kind": "synthetic RGB and repeated prepared crops, generic COCO weights",
        "scope": "One full-frame detector call + N defect calls + N DINO/nearest-neighbor calls + PNG/JSON writes; no camera/Qt/tracking/robot",
        "bank_vectors": args.bank_size, "bank_kind": "seeded normalized random vectors", "counts": args.counts,
        "weights_sha256": config.initial_sha256, "inference_profiles": [b.inference_profile for b in backends],
        "phases": {}, "vlm": {"requested": args.vlm}, "acceptance": "NOT_ASSESSED"}
    from mes_vision.vlm.gpu import GpuCoordinator
    gate = GpuCoordinator(args.project_root / "artifacts/operation/vlm/gpu-coordination")
    lease = FileLock(str(gate.root / "resident.lock"), timeout=0)
    def measure(fn):
        began = time.perf_counter(); value = fn(); torch.cuda.synchronize()
        return value, (time.perf_counter()-began)*1000
    rng = np.random.default_rng(17)
    rgb = rng.integers(70, 190, (480, 640, 3), dtype=np.uint8)
    crop = rgb[100:300, 200:400].copy()
    memory = rng.normal(size=(args.bank_size, 384)).astype(np.float32)
    memory /= np.linalg.norm(memory, axis=1, keepdims=True)
    rows = []
    def cycle(count, phase):
        began = time.perf_counter()
        row = {"phase": phase, "count": count, "defect_ms": 0., "features_ms": 0., "distance_ms": 0.}
        _, row["detect_ms"] = measure(lambda: backends[0].predict_rgb(rgb))
        for _ in range(count):
            _, ms = measure(lambda: backends[1].predict_rgb(crop)); row["defect_ms"] += ms
            patches, ms = measure(lambda: features.extract(crop)); row["features_ms"] += ms
            _, ms = measure(lambda: nearest_neighbors(patches.reshape(-1, 384), memory, device="cuda")); row["distance_ms"] += ms
        store_start = time.perf_counter()
        target = args.output / "workload-snapshots" / f"{len(rows):06d}"
        target.mkdir(parents=True)
        Image.fromarray(rgb).save(target / "frame.png")
        for i in range(count): Image.fromarray(crop).save(target / f"crop-{i}.png")
        write_json(target / "measurement.json", row)
        row["save_ms"] = (time.perf_counter()-store_start)*1000
        row["total_ms"] = (time.perf_counter()-began)*1000
        free, total = torch.cuda.mem_get_info(); row["gpu_free_gib"] = free/1024**3
        rows.append(row)
        return row
    def summary(phase):
        selected = [r for r in rows if r["phase"] == phase]
        return {str(n): {k: distribution([r[k] for r in selected if r["count"] == n]) for k in
            ("detect_ms", "defect_ms", "features_ms", "distance_ms", "save_ms", "total_ms")} for n in args.counts}
    try:
        lease.acquire()
        with gate.foreground(timeout=60):
            began = time.perf_counter()
            for b in backends: b.load()
            features = DinoFeatures(args.project_root); features.load()
            for b in backends: b.predict_rgb(rgb)
            patches = features.extract(crop)
            nearest_neighbors(patches.reshape(-1, 384), memory, device="cuda"); torch.cuda.synchronize()
            report["load_and_warmup_seconds"] = time.perf_counter()-began
            identities = [id(b._network) for b in backends]+[id(features.model)]
            for n in args.counts:
                for _ in range(args.samples): cycle(n, "VLM_OFF")
                print(f"VLM_OFF count={n} measured", flush=True)
            report["phases"]["VLM_OFF"] = summary("VLM_OFF")
        if args.vlm:
            free, total = torch.cuda.mem_get_info()
            require(free >= 12*1024**3 and total >= 20*1024**3, "Resident policy defers VLM on this GPU; parallel run not attempted")
            fixture = make_vlm_fixture(args.output / "vlm-fixture", enabled=True)
            queue = fixture["queue"]
            # Worker and workload share the same lease policy, but the benchmark queue is isolated from operations.
            worker = AnalysisWorker(queue, args.project_root, allow_synthetic=True, idle_seconds=0)
            worker.gpu = gate
            write_json(gate.root / "resident.json", {"active": True, "allow_vlm": True})
            original_hash = sha256(fixture["snapshot"] / "inspection.json")
            jobs = [fixture["job_id"]]
            for tokens in (129, 130):
                jobs.append(queue.enqueue(fixture["snapshot"], fixture["result"].objects[1].object_id, asdict(GenerationConfig(max_new_tokens=tokens))))
            def run_worker():
                try: worker.run(max_jobs=len(jobs))
                except Exception as exc: errors.append(str(exc))
            thread = threading.Thread(target=run_worker, daemon=True); thread.start()
            deadline = time.monotonic()+240; index = 0
            while thread.is_alive() and time.monotonic() < deadline:
                state_before = queue.runtime() or {}
                if state_before.get("phase") in {"ANALYZING", "MODEL_LOADING"}:
                    n = args.counts[index % len(args.counts)]; index += 1
                    row = cycle(n, "VLM_ON_BOUNDARY")
                    state_after = queue.runtime() or {}
                    # Only same-job intervals fully inside ANALYZING are included in contention summaries.
                    if state_after.get("phase") == state_before.get("phase") and state_before.get("job_id") == state_after.get("job_id"):
                        row["phase"] = "VLM_ON" if state_before["phase"] == "ANALYZING" else "VLM_LOADING"
                else: time.sleep(.05)
            require(not thread.is_alive() and not errors, "VLM worker timeout/failure: "+str(errors))
            outcomes = [queue.get(identity) for identity in jobs]
            report["vlm"].update(jobs=[{"id": j["id"], "state": j["state"], "error": j["error"], "result": j["result"]} for j in outcomes],
                inspection_unchanged=sha256(fixture["snapshot"] / "inspection.json") == original_hash)
            report["phases"]["VLM_ON"] = summary("VLM_ON")
            report["phases"]["VLM_LOADING"] = summary("VLM_LOADING")
            require(all(j["state"] == "COMPLETED" and j["result"]["advisory_only"] for j in outcomes), "Actual VLM output failed; inspect job results")
            require(report["vlm"]["inspection_unchanged"], "VLM changed baseline evidence")
            require(all(report["phases"]["VLM_ON"][str(n)]["total_ms"]["n"] >= 5 for n in args.counts), "Insufficient overlapping samples; increase VLM workload")
        require(identities == [id(b._network) for b in backends]+[id(features.model)], "Resident models were replaced")
        report.update(status="COMPLETED", reloaded_between_frames=False)
    except Exception as exc:
        report.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if worker: worker.stop_event.set()
        if thread: thread.join(15)
        if thread and thread.is_alive():
            report.update(status="FAILED", error="VLM worker cleanup timeout")
        for b in backends: b.close()
        if features: features.close()
        if lease.is_locked:
            write_json(gate.root / "resident.json", {"active": False, "allow_vlm": True}); lease.release()
        report["samples"] = rows
        write_json(args.output / "report.json", report)
        if rows:
            with (args.output / "samples.csv").open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "phases", "vlm")}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

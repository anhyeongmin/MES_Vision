"""Wall-clock continuous software test with explicitly injected camera/model outputs."""
from pathlib import Path
import argparse
import json
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tests")]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from test_operation import OperationTests, detection, frame
    from mes_vision.training.data import require, write_json
    from mes_vision.validation.metrics import distribution
    import psutil
    require(10 <= args.seconds <= 86400, "Duration must be 10..86400 seconds")
    args.output.mkdir(parents=True, exist_ok=False)
    fixture = OperationTests(); fixture.setUp()
    engine = fixture.engine; models = fixture.models
    sequence = 0; timings = []; cycles = []; rss = []; publications = []
    report = {"passed": False, "synthetic_injected_outputs": True, "product_accuracy_validated": False,
        "actual_camera_robot_gpu_inference_tested": False, "requested_seconds": args.seconds,
        "scope": "20 Hz offered frames in one process: tracker, selective inspections, asynchronous snapshots, SQLite publication, pause/recheck/removal"}
    generation = 1
    def enable(value):
        nonlocal generation
        generation += 1; engine.set_enabled(value, generation)
    def count():
        with fixture.store.connect() as db: return db.execute("SELECT COUNT(*) FROM inspections").fetchone()[0]
    def drive(seconds, detections):
        nonlocal sequence
        models.detections = detections
        end = time.monotonic()+seconds
        while time.monotonic() < end:
            start = time.perf_counter(); f = frame(sequence); sequence += 1
            batch, assigned = engine.detect(f, time.monotonic())
            saved = engine.finish_save()
            if saved: publications.extend(saved["links"])
            if engine.saving is None:
                pending = engine.inspect(f, batch, assigned)
                if pending: engine.begin_save(pending)
            timings.append((time.perf_counter()-start)*1000)
            time.sleep(max(0, .05-(time.perf_counter()-start)))
        if engine.saving:
            engine.saving[2].result(timeout=5)
            saved = engine.finish_save()
            if saved: publications.extend(saved["links"])
    try:
        engine.prepare()
        started = time.monotonic()
        objects = (detection(), detection(x=180), detection(x=340))
        while time.monotonic()-started < args.seconds:
            before = count(); models.defect = False
            drive(1.1, objects); require(count() == before+3, "Three stable new objects were not published exactly once")
            drive(.5, objects); require(count() == before+3, "Stationary object produced duplicate inspection")
            enable(False); drive(.5, objects); require(count() == before+3, "Pause published a new inspection")
            enable(True); drive(.5, objects); require(count() == before+3, "Resume duplicated unchanged objects")
            target = next(t for t in engine.tracker.tracks.values() if not t.missing)
            models.defect = True; engine.tracker.recheck(target.id, time.monotonic())
            drive(1.1, objects); require(count() == before+4, "Selective recheck did not produce exactly one revision")
            require(target.result["decision"] == "NG", "Injected defect did not update the rechecked verdict")
            drive(1.2, ()); require(not any(t.result for t in engine.tracker.tracks.values()), "Removed objects kept actionable verdicts")
            cycles.append({"cycle": len(cycles)+1, "new_objects": 3, "rechecks": 1, "inspection_rows": count(), "elapsed_seconds": time.monotonic()-started})
            rss.append(psutil.Process().memory_info().rss/1024**2)
            write_json(args.output / "progress.json", {"cycles": cycles, "frames": sequence})
        require(models.loads == 1, "Models were reloaded during continuous input")
        require(len({(p["track_id"], p["revision"]) for p in publications}) == len(publications), "Duplicate published track revision")
        with fixture.store.connect() as db:
            require(db.execute("PRAGMA quick_check").fetchone()[0] == "ok", "Database integrity check failed")
        report.update(passed=True, elapsed_seconds=time.monotonic()-started, offered_frames=sequence, cycles=cycles,
            model_load_calls=models.loads, published_revisions=len(publications), loop_ms=distribution(timings),
            rss_mib_after_each_cycle=rss, rss_growth_after_first_cycle_mib=rss[-1]-rss[0],
            memory_leak_acceptance="NOT_ASSESSED: bounded observations only", operation_counts=fixture.store.counts(fixture.session))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        fixture.tearDown()
        report["models_closed"] = models.closed
        write_json(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("cycles", "rss_mib_after_each_cycle")}, ensure_ascii=False), flush=True)


if __name__ == "__main__": main()

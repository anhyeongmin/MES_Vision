"""Optional VLM queue, native results viewer, and isolated background worker."""
import argparse
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.training.data import read_json


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"): stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("init", "enable", "disable", "status", "enqueue", "retry", "cancel", "worker", "ui"):
        command = sub.add_parser(name)
        command.add_argument("--queue", type=Path, required=True)
        if name == "init": command.add_argument("--capacity", type=int, default=100)
        if name == "enqueue":
            command.add_argument("--snapshot", type=Path, required=True)
            command.add_argument("--object-id", required=True)
            command.add_argument("--config", type=Path, default=ROOT / "configs/vlm/generation.json")
        if name in {"retry", "cancel"}: command.add_argument("--job-id", required=True)
        if name in {"worker", "ui"}:
            command.add_argument("--synthetic", action="store_true")
            command.add_argument("--simulate", action="store_true", help="explicit mock backend; requires --synthetic")
        if name == "worker":
            command.add_argument("--max-jobs", type=int)
            command.add_argument("--idle-exit-seconds", type=float)
            command.add_argument("--mock-behavior", choices=["ok", "invalid", "slow", "crash", "delay"], default="ok")
        if name == "ui": command.add_argument("--read-only-worker", action="store_true", help="do not start a worker in this window")
    fixture = sub.add_parser("make-fixture")
    fixture.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "make-fixture":
            from mes_vision.vlm.fixtures import make_vlm_fixture
            data = make_vlm_fixture(args.output)
            print(json.dumps({"queue": str(data["queue"].root), "job_id": data["job_id"], "vlm_enabled": False, "synthetic": True}))
            return 0
        queue = AnalysisQueue(args.queue, capacity=args.capacity if args.command == "init" else None)
        if args.command in {"enable", "disable"}: queue.set_enabled(args.command == "enable")
        elif args.command == "enqueue":
            identity = queue.enqueue(args.snapshot, args.object_id, read_json(args.config))
            print(json.dumps(queue.get(identity), ensure_ascii=False, indent=2))
            return 0
        elif args.command == "retry": queue.retry(args.job_id)
        elif args.command == "cancel": queue.cancel(args.job_id)
        elif args.command == "worker":
            from mes_vision.vlm.worker import AnalysisWorker
            worker = AnalysisWorker(queue, ROOT, backend="mock" if args.simulate else "qwen", allow_synthetic=args.synthetic, mock_behavior=args.mock_behavior)
            worker.run(max_jobs=args.max_jobs, idle_exit_seconds=args.idle_exit_seconds)
        elif args.command == "ui":
            from PySide6.QtWidgets import QApplication
            from mes_vision.vlm.viewer import AnalysisViewer
            app = QApplication(sys.argv[:1])
            window = AnalysisViewer(queue, ROOT, run_worker=not args.read_only_worker, allow_synthetic=args.synthetic,
                                    backend="mock" if args.simulate else "qwen")
            window.show()
            return app.exec()
        print(json.dumps(queue.summary(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

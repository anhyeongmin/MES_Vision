"""Full-scene held-out evaluation, kept outside the operator's live inspection UI."""
from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    for name in ("collection", "output"): parser.add_argument("--"+name, type=Path, required=True)
    config = parser.add_mutually_exclusive_group(required=True)
    config.add_argument("--product", type=Path)
    config.add_argument("--session-id", help="Use the frozen product/equipment from an existing operating session")
    parser.add_argument("--equipment", type=Path)
    parser.add_argument("--runtime", type=Path, help="Operating folder; defaults to project artifacts/operation")
    parser.add_argument("--split", choices=("test", "challenge"), action="append")
    parser.add_argument("--iou", type=float, default=.5)
    args = parser.parse_args()
    from mes_vision.training.data import read_json
    from mes_vision.validation.replay import ReplayRunner, session_configuration
    from mes_vision.vlm.gpu import GpuCoordinator
    runtime = args.runtime or args.project_root / "artifacts/operation"
    if args.session_id:
        if args.equipment: parser.error("--equipment cannot override a frozen session")
        product, equipment = session_configuration(runtime, args.session_id)
    else:
        if not args.equipment: parser.error("--product requires --equipment")
        product, equipment = read_json(args.product), read_json(args.equipment)
    # Offline evaluation belongs on an idle machine; this lock coordinates with the chosen runtime.
    from filelock import FileLock
    gate = GpuCoordinator(runtime / "vlm/gpu-coordination")
    with FileLock(str(gate.root / "resident.lock"), timeout=0), gate.foreground(timeout=60):
        report = ReplayRunner(args.project_root, product, equipment).run(
            args.collection, args.output, splits=tuple(args.split or ("test", "challenge")), threshold=args.iou)
    print(f"{report['frames']} frames, {report['missed']} missed, {report['failed_frames']} errors; {args.output / 'report.json'}")
    return 1 if report["failed_frames"] else 0


if __name__ == "__main__": raise SystemExit(main())

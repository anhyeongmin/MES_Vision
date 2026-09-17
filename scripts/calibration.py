"""Fit, inspect, accept and use a saved planar calibration; never operate a robot."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.calibration import fit, load, spec_from_dict, GeometryContext
from mes_vision.training.data import write_json, read_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo"); demo.add_argument("--output", type=Path, required=True)
    for name in ("fit", "accept", "inspect", "map", "ui"):
        command = sub.add_parser(name)
        command.add_argument("--input", type=Path, required=True)
        if name in {"fit", "accept"}: command.add_argument("--output", type=Path, required=True)
        if name == "accept": command.add_argument("--reference", required=True)
        if name == "map":
            command.add_argument("--context", type=Path, required=True)
            command.add_argument("--pixel", type=float, nargs=2, required=True)
            command.add_argument("--plane-z-mm", type=float, required=True)
            command.add_argument("--kind", choices=["real", "synthetic"], required=True)
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from mes_vision.calibration.fixtures import make_spec
            args.output.mkdir(parents=True, exist_ok=False)
            spec = make_spec()
            write_json(args.output / "points.json", asdict(spec))
            write_json(args.output / "context.json", asdict(spec.context))
            calibration = fit(spec)
            calibration.save(args.output / "draft.json")
            calibration = calibration.accept("SYNTHETIC-VALIDATION-ONLY")
            calibration.save(args.output / "accepted.json")
        elif args.command == "ui":
            from PySide6.QtWidgets import QApplication
            from mes_vision.calibration.viewer import CalibrationViewer
            app = QApplication(sys.argv[:1])
            window = CalibrationViewer(read_json(args.input)); window.show()
            return app.exec()
        elif args.command == "fit":
            calibration = fit(spec_from_dict(read_json(args.input))); calibration.save(args.output)
        else:
            calibration = load(args.input)
            if args.command == "accept": calibration = calibration.accept(args.reference); calibration.save(args.output)
            if args.command == "map":
                data = read_json(args.context)
                context = GeometryContext(**dict(data, image_size=tuple(data["image_size"]), transformations=tuple(data.get("transformations", ()))))
                xy = calibration.map_xy(args.pixel, context, plane_z_mm=args.plane_z_mm, kind=args.kind)
                print(json.dumps({"source_pixel": args.pixel, "robot_xy_mm": xy, "plane_z_mm": args.plane_z_mm,
                                  "calibration_identity": calibration.identity, "kind": args.kind, "robot_commands_enabled": False}))
                return 0
        print(json.dumps({"status": calibration.data["status"], "metrics": calibration.data["metrics"], "failures": calibration.data["failures"],
                          "identity": calibration.identity, "robot_commands_enabled": False}, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())

"""Evaluate saved inspection evidence or create an explicitly synthetic decision demo."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.decision import apply_policy, load_policy
from mes_vision.decision.io import run_from_dict, frame_evidence_from_dict
from mes_vision.training.data import read_json, require, write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo")
    demo.add_argument("--output", type=Path, required=True)
    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--policy", type=Path, default=ROOT / "configs/decision/unconfigured.json")
    evaluate.add_argument("--frame-evidence", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--simulate", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "demo":
            from PIL import Image, ImageDraw
            from mes_vision.decision.fixtures import make_case
            from mes_vision.vlm.snapshots import save_snapshot
            from mes_vision.vlm.queue import AnalysisQueue
            from mes_vision.vlm.backend import GenerationConfig
            data = make_case(mixed=True)
            args.output.mkdir(parents=True, exist_ok=False)
            for name, value in (("raw", data["raw"].to_dict()), ("policy", asdict(data["policy"])),
                                ("frame-evidence", asdict(data["evidence"])), ("result", data["result"].to_dict())):
                write_json(args.output / f"{name}.json", value)
            snapshot = args.output / "snapshot"
            save_snapshot(snapshot, data["frame"], data["result"], kind="synthetic", criteria={"version": data["policy"].version, "description": "SYNTHETIC TEST ONLY"})
            queue = AnalysisQueue(args.output / "queue")
            for obj in data["result"].objects: queue.enqueue(snapshot, obj.object_id, asdict(GenerationConfig()))
            canvas = Image.new("RGB", (480, 240), "#eef2f7")
            canvas.paste(Image.fromarray(data["frame"].rgb), (0, 30))
            draw = ImageDraw.Draw(canvas)
            draw.text((10, 7), "SYNTHETIC ONLY / NO ROBOT ACTIONS", fill="#172030")
            for obj in data["result"].objects:
                b = obj.effective_box
                color = {"OK": "#18734b", "NG": "#b52c26", "REVIEW": "#9c6400"}[obj.final_decision]
                draw.rectangle((b.x1, b.y1+30, b.x2, b.y2+30), outline=color, width=3)
                draw.text((b.x1, 215), obj.final_decision, fill=color)
            canvas.save(args.output / "decisions.png")
            result = data["result"]
        else:
            policy = load_policy(args.policy)
            require(policy.kind != "synthetic" or args.simulate, "synthetic policy requires --simulate")
            source = run_from_dict(read_json(args.input))
            evidence = frame_evidence_from_dict(read_json(args.frame_evidence)) if args.frame_evidence else None
            result = apply_policy(source, policy, evidence)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(result.to_dict(), stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps({"decision": result.final_decision, "status": result.decision_status,
                          "counts": result.decision_details["counts"], "robot_commands_enabled": False, "output": str(args.output.resolve())}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())

"""Preview and run an explicit virtual Dobot workflow. Hardware transport is absent."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.robot import RobotController, load_profile
from mes_vision.robot.fixtures import make_fixture, prepare, drive
from mes_vision.robot.planning import STAGES
from mes_vision.training.data import write_json


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("configuration")
    inspect.add_argument("--profile", type=Path, default=ROOT / "configs/robot/unconfigured.json")
    for name in ("preview", "simulate"):
        command = sub.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--decision", choices=["OK", "NG", "REVIEW"], default="OK")
        command.add_argument("--allow-review", action="store_true", help="explicit synthetic quarantine route")
        if name == "simulate":
            command.add_argument("--fault-stage", choices=STAGES)
            command.add_argument("--fault", choices=["reject", "timeout", "disconnect", "failure", "unknown", "unverified", "wrong_id", "wrong_pose"])
            command.add_argument("--stop-stage", choices=STAGES)
    args = parser.parse_args()
    try:
        if args.command == "configuration":
            print(json.dumps({"profile": asdict(load_profile(args.profile)), "hardware_adapter": "NOT_IMPLEMENTED", "real_motion_enabled": False}, ensure_ascii=False, indent=2))
            return 0
        faults = {}
        if args.command == "simulate":
            if bool(args.fault_stage) != bool(args.fault): raise ValueError("fault and fault-stage must be supplied together")
            if args.fault_stage: faults[args.fault_stage] = args.fault
        data = make_fixture(decision=args.decision, allow_review=args.allow_review, faults=faults)
        plan = prepare(data)
        args.output.mkdir(parents=True, exist_ok=False)
        write_json(args.output / "inspection.json", data["inspection"].to_dict())
        write_json(args.output / "plan.json", asdict(plan))
        report = {"synthetic": True, "real_commands_sent": 0, "hardware_tested": False,
                  "object_id": plan.object_id, "decision": plan.decision, "plan_digest": plan.digest}
        if args.command == "simulate":
            with RobotController(args.output / "journal", data["adapter"]) as controller:
                controller.start(plan, data["scene"], now=data["now"])
                report["state"] = drive(controller, data, stop_stage=args.stop_stage)
                report["cycle_verified"] = controller.state == "RECAPTURE"
                report["events"] = controller.journal.events()
                report["commands"] = [asdict(c) for c in data["adapter"].commands]
                report["stop_requests"] = data["adapter"].stop_calls
        else:
            report.update(state="PREVIEW_ONLY", cycle_verified=False, commands=[])
        write_json(args.output / "report.json", report)
        print(json.dumps({k: v for k, v in report.items() if k not in {"commands", "events"}}, ensure_ascii=False, indent=2))
        return 0  # Fault scenarios are intentional simulation results; state is explicit in the report.
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__": raise SystemExit(main())

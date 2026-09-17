"""Regression, CLI scenarios and real process restart test; entirely simulated robots."""
import argparse
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
from mes_vision.training.data import write_json, read_json, require
from mes_vision.robot import RobotController
from mes_vision.robot.fixtures import make_fixture, prepare


def crash_child(directory):
    data = make_fixture()
    with RobotController(directory / "journal", data["adapter"]) as controller:
        controller.start(prepare(data), data["scene"], now=1000)
        controller.tick(data["scene"], now=1000)
        write_json(directory / "ready.json", {"state": controller.state, "commands_submitted": len(data["adapter"].commands)})
        time.sleep(30)  # Test-only process is killed by the parent after ready.json appears.


def verify_restart(directory):
    directory.mkdir()
    with (directory / "child.log").open("w", encoding="utf-8") as stream:
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--crash-child", str(directory)], cwd=ROOT,
                    stdout=stream, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            deadline = time.monotonic()+15
            while not (directory / "ready.json").exists():
                require(child.poll() is None and time.monotonic() < deadline, "crash child did not reach first accepted command")
                time.sleep(.05)
            child.kill(); child.wait(timeout=5)
            data = make_fixture()
            with RobotController(directory / "journal", data["adapter"]) as restarted:
                require(restarted.state == "RECOVERY", "restart silently resumed")
                restarted.tick(data["scene"], now=1000)
                require(not data["adapter"].commands, "restart resent a command")
            return {"passed": True, "forced_process_exit_tested": True, "state": "RECOVERY", "commands_resent": 0, "hardware_tested": False}
        finally:
            if child.poll() is None: child.kill(); child.wait(timeout=5)


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--crash-child", type=Path)
    args = parser.parse_args()
    if args.crash_child: crash_child(args.crash_child); return 0
    output = ROOT / "artifacts/robot-check" / ("run-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {"passed": False, "path": str(output), "hardware_tested": False, "real_commands_sent": 0}
    try:
        with (output / "tests.log").open("w", encoding="utf-8") as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
        report.update(tests_run=tests.testsRun, errors=len(tests.errors), failures=len(tests.failures))
        require(tests.wasSuccessful(), "regression failed; inspect tests.log")
        scenarios = [
            ("ok", [], "RECAPTURE", 0), ("ng", ["--decision", "NG"], "RECAPTURE", 0),
            ("review_blocked", ["--decision", "REVIEW"], None, 1),
            ("review_quarantine", ["--decision", "REVIEW", "--allow-review"], "RECAPTURE", 0),
            ("pick_unverified", ["--fault-stage", "VERIFY_PICK", "--fault", "unverified"], "FAULT", 0),
            ("place_unverified", ["--fault-stage", "VERIFY_PLACE", "--fault", "unverified"], "FAULT", 0),
            ("timeout", ["--fault-stage", "MOVE_PLACE", "--fault", "timeout"], "FAULT", 0),
            ("disconnect", ["--fault-stage", "LIFT_PICK", "--fault", "disconnect"], "FAULT", 0),
            ("stop", ["--stop-stage", "LIFT_PICK"], "STOPPED", 0),
        ]
        results = []
        for name, options, expected, exit_code in scenarios:
            with (output / f"{name}.log").open("w", encoding="utf-8") as log:
                proc = subprocess.run([sys.executable, "scripts/robot.py", "simulate", "--output", str(output / name), *options],
                        cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=30)
            require(proc.returncode == exit_code, "unexpected scenario exit: " + name)
            if expected:
                outcome = read_json(output / name / "report.json")
                require(outcome["state"] == expected and outcome["real_commands_sent"] == 0, "unexpected scenario result: " + name)
                require(outcome["cycle_verified"] == (expected == "RECAPTURE"), "acceptance was treated as success")
            else: require(not (output / name).exists(), "blocked plan created a motion journal")
            results.append({"scenario": name, "state": expected or "BLOCKED", "passed": True})
        report["scenarios"] = results
        report["restart"] = verify_restart(output / "forced-restart")
        report["passed"] = True
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"
        (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "report.json", report)
    write_json(output.parent / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())

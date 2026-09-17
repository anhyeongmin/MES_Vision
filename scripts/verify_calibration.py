"""Full regression, CLI replay, Qt point editing and calibrated robot simulation."""
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.data import read_json, write_json, require
from mes_vision.calibration import load, spec_from_dict, map_target


def verify_ui(output, spec):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    from mes_vision.calibration.viewer import CalibrationViewer
    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"
    if font_path.is_file():
        font = QFontDatabase.addApplicationFont(str(font_path)); app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 10))
    window = CalibrationViewer(spec); errors = []; window.error = lambda e: errors.append(str(e))
    checks = []
    def check(name, condition): require(condition, name); checks.append(name)
    window.show(); app.processEvents()
    check("mapping_initially_disabled", not window.map_button.isEnabled())
    window.fit_button.click(); app.processEvents()
    check("fit_and_holdout_metrics_visible", "독립 확인점 최대 오차" in window.summary.toPlainText() and not errors)
    check("fit_does_not_activate_mapping", window.accept_button.isEnabled() and not window.map_button.isEnabled())
    window.accept_reference("SYNTHETIC-UI-VALIDATION"); app.processEvents()
    check("accepted_mapping_displays_mm", "로봇 X =" in window.mapping.text() and "mm" in window.mapping.text())
    window.pixel_x.setValue(0); app.processEvents()
    check("outside_region_blocked", "차단" in window.mapping.text())
    window.pixel_x.setValue(75); window.pixel_y.setValue(90); app.processEvents()
    window.save_to(output / "ui-calibration.json")
    check("saved_settings_reload", load(output / "ui-calibration.json").identity == window.calibration.identity)
    old = window.table.item(0, 5).text()
    window.table.item(0, 5).setText(str(float(old)+10)); app.processEvents()
    check("edit_invalidates_mapping_and_save", window.calibration is None and not window.map_button.isEnabled() and not window.save_button.isEnabled())
    window.fit_button.click(); app.processEvents()
    check("measurement_error_blocks_acceptance", window.calibration is not None and bool(window.calibration.data["failures"]) and not window.accept_button.isEnabled())
    window.table.item(0, 5).setText(old); window.fit_button.click(); window.accept_reference("SYNTHETIC-UI-VALIDATION"); app.processEvents()
    screenshot = output / "calibration-viewer.png"
    check("screenshot_written", window.grab().save(str(screenshot)))
    check("no_unexpected_errors", not errors)
    window.close(); app.processEvents()
    return {"passed": True, "checks": checks, "screenshot": str(screenshot), "mode": "Qt offscreen; synthetic coordinates"}


def verify_robot(output, calibration):
    from mes_vision.robot import build_plan, RobotController
    from mes_vision.robot.fixtures import make_fixture, drive
    data = make_fixture()
    spec = spec_from_dict(calibration.data["specification"])
    target = map_target(calibration, data["inspection"], data["object_id"], (75., 90.), spec.context,
                        plane_z_mm=20., r_deg=0, grasp_policy_version=data["profile"].grasp_policy_version)
    data["profile"] = replace(data["profile"], calibration_version=calibration.identity)
    data["scene"] = replace(data["scene"], calibration_version=calibration.identity)
    plan = build_plan(data["inspection"], data["object_id"], target, data["profile"], data["scene"], now=data["now"])
    with RobotController(output / "robot-journal", data["adapter"]) as controller:
        controller.start(plan, data["scene"], now=data["now"])
        require(drive(controller, data) == "RECAPTURE", "calibrated simulation failed")
        result = {"passed": True, "state": controller.state, "mapped_target": asdict(target), "real_commands_sent": 0,
                  "plan": asdict(plan), "events": controller.journal.events()}
    write_json(output / "robot.json", result)
    return {k: v for k, v in result.items() if k not in {"events", "plan"}}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    output = ROOT / "artifacts/calibration-check" / ("run-" + uuid4().hex[:8]); output.mkdir(parents=True)
    report = {"passed": False, "path": str(output), "hardware_tested": False, "real_commands_sent": 0}
    try:
        with (output / "tests.log").open("w", encoding="utf-8") as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
        report.update(tests_run=tests.testsRun, errors=len(tests.errors), failures=len(tests.failures))
        require(tests.wasSuccessful(), "regression failure; inspect tests.log")
        with (output / "demo.log").open("w", encoding="utf-8") as stream:
            proc = subprocess.run([sys.executable, "scripts/calibration.py", "demo", "--output", str(output / "demo")], cwd=ROOT,
                                  stdout=stream, stderr=subprocess.STDOUT, timeout=30)
        require(proc.returncode == 0, "calibration CLI demo failed")
        calibration = load(output / "demo/accepted.json")
        spec = spec_from_dict(calibration.data["specification"])
        actual = calibration.map_xy((75., 90.), spec.context, plane_z_mm=20., kind="synthetic")
        with (output / "map.json").open("w", encoding="utf-8") as stream:
            proc = subprocess.run([sys.executable, "scripts/calibration.py", "map", "--input", str(output / "demo/accepted.json"),
                 "--context", str(output / "demo/context.json"), "--pixel", "75", "90", "--plane-z-mm", "20", "--kind", "synthetic"],
                 cwd=ROOT, stdout=stream, stderr=subprocess.PIPE, timeout=30)
        require(proc.returncode == 0 and tuple(read_json(output / "map.json")["robot_xy_mm"]) == actual, "mapping replay failed")
        report["fit"] = {"passed": True, "metrics": calibration.data["metrics"], "identity": calibration.identity,
                         "mapped_example_xy_mm": actual, "raw_pixel": [75, 90], "synthetic": True}
        report["ui"] = verify_ui(output, read_json(output / "demo/points.json"))
        report["robot_integration"] = verify_robot(output, calibration)
        report["passed"] = True
    except Exception as exc:
        import traceback
        report["error"] = f"{type(exc).__name__}: {exc}"; (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(output / "report.json", report); write_json(output.parent / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__": raise SystemExit(main())

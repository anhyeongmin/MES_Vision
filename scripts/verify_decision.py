"""Full CPU regression, saved decision replay and native Qt integration."""
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.data import read_json, write_json, require, sha256


def child(arguments, log):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run([sys.executable, *arguments], cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, QT_QPA_PLATFORM="offscreen", PYTHONPATH=str(ROOT / "src")), timeout=60)
    require(completed.returncode == 0, "subprocess failed: " + str(log))


def verify_ui(directory):
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    from mes_vision.vlm.queue import AnalysisQueue
    from mes_vision.vlm.viewer import AnalysisViewer
    app = QApplication.instance() or QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"
    if font_path.is_file():
        font = QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 10))
    queue = AnalysisQueue(directory / "demo/queue")
    viewer = AnalysisViewer(queue, ROOT, run_worker=False)
    viewer.show(); app.processEvents()
    checks = []
    def check(name, condition):
        require(condition, name); checks.append(name)
    check("vlm_default_off", viewer.toggle.text() == "VLM OFF")
    check("three_objects_visible", viewer.table.rowCount() == 3)
    check("three_distinct_verdicts_visible", {viewer.table.item(i, 1).text() for i in range(3)} == {"정상", "불량", "보류"})
    for suffix, expected in (("0001", "최종 판정: 정상"), ("0002", "최종 판정: 불량"), ("0003", "최종 판정: 판정 보류")):
        index = next(i for i, r in enumerate(viewer.rows) if r["object_id"].endswith(suffix))
        viewer.table.selectRow(index); app.processEvents()
        check("decision_text_" + suffix, expected in viewer.reasons.toPlainText())
        check("simulation_labeled_" + suffix, "모의 판정" in viewer.reasons.toPlainText())
        if suffix == "0002":
            check("multiple_defects_and_error_visible", all(s in viewer.reasons.toPlainText() for s in ("NG03", "NG06", "검사 오류", "필수 검사 완료: 아니오")))
        if suffix == "0003": check("uncertainty_explained", "검사 결과 불확실" in viewer.reasons.toPlainText())
    index = next(i for i, r in enumerate(viewer.rows) if r["object_id"].endswith("0002"))
    viewer.table.selectRow(index); app.processEvents()
    before = viewer.reasons.toPlainText()
    viewer.toggle.click(); app.processEvents()
    viewer.toggle.click(); app.processEvents()
    check("toggle_does_not_change_verdict_or_reason", viewer.reasons.toPlainText() == before and not queue.enabled())
    check("vlm_remains_separate", "OFF로 미요청" in viewer.analysis.toPlainText())
    screenshot = directory / "decision-viewer.png"
    check("screenshot_saved", viewer.grab().save(str(screenshot)))
    viewer.close(); app.processEvents()
    return {"passed": True, "checks": checks, "screenshot": str(screenshot), "mode": "Qt offscreen events; no GPU inference"}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    output = ROOT / "artifacts/decision-check" / ("run-" + uuid4().hex[:8])
    output.mkdir(parents=True)
    report = {"passed": False, "path": str(output), "product_accuracy_tested": False, "robot_commands_sent": False}
    try:
        with (output / "tests.log").open("w", encoding="utf-8") as stream:
            tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.discover(str(ROOT / "tests"), pattern="test_*.py"))
        report.update(tests_run=tests.testsRun, errors=len(tests.errors), failures=len(tests.failures))
        require(tests.wasSuccessful(), "regression failed; inspect tests.log")
        child(["scripts/decision.py", "demo", "--output", str(output / "demo")], output / "demo.log")
        before = sha256(output / "demo/raw.json")
        child(["scripts/decision.py", "evaluate", "--input", str(output / "demo/raw.json"), "--policy", str(output / "demo/policy.json"),
               "--frame-evidence", str(output / "demo/frame-evidence.json"), "--simulate", "--output", str(output / "replayed.json")], output / "replay.log")
        replay = read_json(output / "replayed.json")
        require(replay == read_json(output / "demo/result.json") and sha256(output / "demo/raw.json") == before, "saved replay changed source or decision")
        child(["scripts/decision.py", "evaluate", "--input", str(output / "demo/raw.json"), "--output", str(output / "unconfigured.json")], output / "unconfigured.log")
        require(read_json(output / "unconfigured.json")["final_decision"] == "REVIEW", "unconfigured policy must hold")
        report["saved_replay"] = {"passed": True, "input_unchanged": True, "counts": replay["decision_details"]["counts"], "unconfigured_review": True}
        report["ui"] = verify_ui(output)
        child(["scripts/verify_vlm_ui.py", "--output", str(output / "vlm-ui-regression")], output / "vlm-ui.log")
        report["vlm_ui_regression"] = read_json(output / "vlm-ui-regression/report.json")
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

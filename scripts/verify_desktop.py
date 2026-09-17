"""Exercise the real desktop widgets and child process; all fixtures are synthetic."""
import argparse
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", action="store_true", help="Also exercise RF-DETR through the desktop child on a synthetic image file")
    args = parser.parse_args(); output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtWidgets import QApplication
    from PySide6.QtGui import QFont, QFontDatabase
    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest
    from mes_vision.desktop.window import DesktopWindow
    from mes_vision.training.data import write_json
    app = QApplication(sys.argv[:1])
    font = QFontDatabase.addApplicationFont(str(Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"))
    if QFontDatabase.applicationFontFamilies(font): app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 10))
    window = DesktopWindow(ROOT, output / "runtime", vlm_backend="mock"); window.show()
    checks = []; report = {"passed": False, "synthetic": True, "checks": checks, "production_ready": False}
    def check(condition, label):
        if not condition: raise AssertionError(label + ": " + str(window.last_error))
        checks.append(label)
    def wait(predicate, timeout=20000):
        deadline = time.monotonic() + timeout/1000
        while not predicate() and time.monotonic() < deadline: app.processEvents(); QTest.qWait(20)
        check(predicate(), "wait " + str(len(checks)))
    def click(widget): QTest.mouseClick(widget, Qt.LeftButton); app.processEvents()
    try:
        check(not window.queue.enabled() and window.pages.count() == 5, "five pages and VLM default OFF")
        click(window.start_button); wait(lambda: window.process is None)
        check(window.current is not None and window.objects.rowCount() == 3, "inspection child publishes three saved objects")
        check(window.last_error is None, "no inspection or queue registration error")
        first_run = window.current["run_id"]
        check([o["final_decision"] for o in window.current["objects"]] == ["OK", "NG", "REVIEW"], "three different basic decisions")
        check(len(window.queue.list()) == 3 and all(r["state"] == "SKIPPED_DISABLED" for r in window.queue.list()), "OFF still preserves original reasons")
        window.objects.selectRow(1); app.processEvents()
        check("NG03" in window.reasons.toPlainText() and "NG06" in window.reasons.toPlainText(), "NG reasons visible before VLM")
        window.grab().save(str(output / "inspection.png"))
        click(window.robot_button); check(window.robot_active, "selected NG starts calibrated simulated plan")
        wait(lambda: not window.robot_active)
        check(window.controller.state == "RECAPTURE" and not window.robot_button.isEnabled(), "verified simulated placement requires new frame")
        click(window.vlm_toggle); check(window.queue.enabled(), "global VLM ON")
        check(all(r["state"] == "SKIPPED_DISABLED" for r in window.queue.list()), "ON does not silently retry old jobs")
        click(window.start_button); wait(lambda: window.process is None)
        check(window.current["run_id"] != first_run, "new run identity")
        wait(lambda: sum(r["state"] == "COMPLETED" for r in window.queue.list()) == 3, timeout=30000)
        check([o["final_decision"] for o in window.current["objects"]] == ["OK", "NG", "REVIEW"], "background VLM leaves basic verdict unchanged")
        window.objects.selectRow(2); app.processEvents(); check(not window.robot_button.isEnabled(), "review object cannot move")
        window.objects.selectRow(0); click(window.robot_button); click(window.robot_stop)
        check(window.controller.state == "STOPPED", "user stops simulated robot")
        window.recover_robot(); check(window.controller.state == "RECAPTURE" and not window.robot_button.isEnabled(), "recovery discards old scene")
        window.objects.selectRow(1); window.open_selected_vlm(); app.processEvents()
        check(window.queue.get(window.vlm_view.selected_key)["object_id"] == window.selected_object()["object_id"], "VLM page opens the selected object from the correct run")
        window.grab().save(str(output / "vlm.png"))
        click(window.vlm_toggle); check(not window.queue.enabled(), "global VLM OFF")
        window.navigation.setCurrentRow(1); window.product.setText("saved-test-product"); window.width_mm.setValue(330.); click(window.save_settings_button)
        check(window.store.settings()["product_id"] == "saved-test-product", "product settings persist")
        window.navigation.setCurrentRow(3); window.refresh_history(); window.history.selectRow(0); window.open_history()
        check(window.history_only and not window.robot_button.isEnabled(), "history cannot authorize robot")
        window.navigation.setCurrentRow(3); app.processEvents(); window.grab().save(str(output / "history.png"))
        window.navigation.setCurrentRow(0)
        before = len(window.store.list()); click(window.start_button); click(window.stop_button); wait(lambda: window.process is None)
        check(len(window.store.list()) == before and not window.robot_button.isEnabled(), "cancelled child cannot publish or authorize motion")
        window.mode.setCurrentIndex(1); window.image_path.setText(str(output / "missing.png")); click(window.start_button)
        check(window.process is None and "사진" in window.last_error, "missing file never falls back to synthetic")
        window.mode.setCurrentIndex(0); click(window.start_button); wait(lambda: window.process is None)
        window.objects.selectRow(1); app.processEvents(); window.grab().save(str(output / "inspection.png"))
        window.close(); wait(lambda: not window.isVisible())
        check(window.worker is None or not window.worker.isRunning(), "window closes without a running worker")
        reopened = DesktopWindow(ROOT, output / "runtime", run_vlm_worker=False); reopened.show(); app.processEvents()
        check(not reopened.queue.enabled() and reopened.settings["product_id"] == "saved-test-product" and len(reopened.history_rows) == before+1, "restart restores OFF settings and history")
        if args.gpu:
            reopened.mode.setCurrentIndex(1); reopened.image_path.setText(str(window.current_path / "frame.png"))
            click(reopened.start_button)
            heartbeat = 0; deadline = time.monotonic()+180
            while reopened.process is not None and time.monotonic() < deadline:
                app.processEvents(); QTest.qWait(50); heartbeat += 1
            check(reopened.process is None and reopened.current is not None and reopened.last_error is None,
                "actual GPU child publishes a model_file result: " + str(reopened.last_error))
            check(reopened.current["mode"] == "model_file" and reopened.current["detector"]["kind"] == "model"
                and reopened.current["final_decision"] == "REVIEW" and not reopened.robot_button.isEnabled(),
                "general model cannot approve product quality or robot movement")
            from mes_vision.training.data import read_json
            report["gpu"] = dict(read_json(reopened.current_path / "desktop-timing.json"),
                ui_event_cycles=heartbeat, input_is_synthetic_image=True, objects=len(reopened.current["objects"]))
            reopened.grab().save(str(output / "file-model.png"))
        reopened.close(); report["passed"] = True
    except Exception as exc:
        import traceback
        report["error"] = str(exc); (output / "error.log").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        window.close()
        deadline = time.monotonic()+15
        while window.isVisible() and time.monotonic() < deadline: app.processEvents(); QTest.qWait(20)
        write_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    multiprocessing.freeze_support()
    raise SystemExit(main())

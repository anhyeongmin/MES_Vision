"""Qt widget tests for optional analysis and immediate basic inspection reasons."""
import argparse
from dataclasses import asdict
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.viewer import AnalysisViewer
from mes_vision.training.data import require, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    fixture = make_vlm_fixture(output / "fixture")
    queue, identity = fixture["queue"], fixture["job_id"]
    checks = []
    def check(name, value):
        require(value, name)
        checks.append(name)
    app = QApplication([])
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"
    if font_path.is_file():
        font = QFontDatabase.addApplicationFont(str(font_path))
        app.setFont(QFont(QFontDatabase.applicationFontFamilies(font)[0], 10))
    window = AnalysisViewer(queue, ROOT, allow_synthetic=True, backend="mock")
    errors = []
    window.error = lambda error: errors.append(str(error))
    window.worker_thread.worker.mock_behavior = "delay"
    window.show()
    app.processEvents()
    def wait_for(condition, timeout=12):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if condition(): return
            QTest.qWait(20)
        raise AssertionError("Qt/worker wait timed out")
    try:
        check("default_off_visible", window.toggle.text() == "VLM OFF" and not queue.enabled())
        check("ng_reason_visible_before_vlm", "NG03" in window.reasons.toPlainText() and "균열" in window.reasons.toPlainText())
        check("no_model_loaded_while_off", queue.get(identity)["attempt"] == 0)
        QTest.mouseClick(window.toggle, Qt.LeftButton)
        check("toggle_on_persisted", AnalysisQueue(queue.root).enabled())
        check("on_does_not_silently_retry_old_job", queue.get(identity)["state"] == "SKIPPED_DISABLED")
        QTest.mouseClick(window.retry, Qt.LeftButton)
        wait_for(lambda: queue.get(identity)["state"] == "COMPLETED")
        window.refresh()
        check("analysis_added_to_separate_panel", "합성 시험용" in window.analysis.toPlainText())
        check("base_reason_preserved_after_analysis", "NG03" in window.reasons.toPlainText())
        window.worker_thread.worker.mock_behavior = "slow"
        wait_for(lambda: queue.runtime()["phase"] == "WARM_IDLE")
        QTest.mouseClick(window.toggle, Qt.LeftButton)
        wait_for(lambda: queue.runtime()["model_pid"] is None)
        check("off_releases_warm_model", not queue.enabled())
        QTest.mouseClick(window.toggle, Qt.LeftButton)
        pending = queue.enqueue(fixture["snapshot"], fixture["result"].objects[1].object_id, asdict(GenerationConfig(max_new_tokens=127)))
        wait_for(lambda: queue.runtime()["phase"] == "ANALYZING")
        QTest.mouseClick(window.toggle, Qt.LeftButton)
        wait_for(lambda: queue.get(pending)["state"] == "CANCELLED" and queue.runtime()["model_pid"] is None)
        check("off_stops_active_analysis", not queue.enabled() and queue.get(pending)["state"] == "CANCELLED")
        check("completed_history_survives_off", queue.get(identity)["state"] == "COMPLETED")
        window.selected_key = identity
        window.refresh()
        app.processEvents()
        screenshot = output / "vlm-viewer.png"
        check("screenshot_written", window.grab().save(str(screenshot)))
        check("no_unexpected_dialog_errors", not errors)
        window.close()
        wait_for(lambda: not window.isVisible())
        check("close_stops_owned_worker", not window.worker_thread.isRunning())
        write_json(output / "report.json", {"passed": True, "checks": checks, "mode": "Qt offscreen events with explicit mock backend",
                                           "synthetic": True, "screenshot": str(screenshot)})
        print(f"Qt checks passed: {len(checks)}")
    finally:
        if window.worker_thread.isRunning():
            window.worker_thread.stop_event.set()
            window.worker_thread.wait(10000)


if __name__ == "__main__":
    main()

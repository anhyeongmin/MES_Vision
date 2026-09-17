"""Render and exercise the actual Qt widgets offscreen with synthetic data."""
import argparse
import os
from pathlib import Path
import sys
import time
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from PySide6.QtCore import Qt, QPointF
from PySide6.QtTest import QTest
from PySide6.QtGui import QFontDatabase, QFont
from PySide6.QtWidgets import QApplication, QInputDialog
from mes_vision.data_management.fixtures import make_fixture
from mes_vision.data_management.collection import Collection
from mes_vision.data_management.export import export_collection
from mes_vision.data_management.labeler import Labeler
from mes_vision.training.data import require, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    checks = []
    def check(name, condition):
        require(condition, name)
        checks.append(name)
    app = QApplication([])
    # Offscreen Qt has no Windows font discovery; use the locally installed font.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts/malgun.ttf"
    if font_path.is_file():
        font_id = QFontDatabase.addApplicationFont(str(font_path))
        families = QFontDatabase.applicationFontFamilies(font_id)
        require(bool(families), "Korean preview font could not be loaded")
        app.setFont(QFont(families[0], 10))
    collection = make_fixture(root / "fixture")
    window = Labeler(collection.root)
    errors = []
    window.error = lambda message: errors.append(str(message))
    window.show()
    app.processEvents()
    window.canvas.fitInView(window.canvas.sceneRect(), Qt.KeepAspectRatio)
    app.processEvents()
    check("opens_original_image_with_two_objects", len(window.draft["objects"]) == 2 and not window.dirty)
    window.objects.setCurrentRow(0)
    window.note.setPlainText("편집 즉시 임시 내용에 반영")
    window.objects.setCurrentRow(1)
    check("switching_object_preserves_unsaved_text", window.draft["objects"][0]["note"] == "편집 즉시 임시 내용에 반영" and window.dirty)
    check("save_without_apply_button", window.save() and Collection(collection.root).record(window.draft["id"])["objects"][0]["note"] == "편집 즉시 임시 내용에 반영")
    check("edit_revokes_review", not window.draft["reviewed"])
    window.reviewer.setText("UI_SYNTHETIC_REVIEWER")
    window.review()
    check("explicit_review_updates_caption", window.draft["reviewed"] and "검토 완료" in window.captures.item(0).text())
    window.objects.setCurrentRow(0)
    previous = list(window.selected_object()["bbox"])
    window.mode.setCurrentIndex(3)
    a = window.canvas.mapFromScene(QPointF(18, 22))
    b = window.canvas.mapFromScene(QPointF(123, 157))
    QTest.mousePress(window.canvas.viewport(), Qt.LeftButton, pos=a)
    QTest.mouseMove(window.canvas.viewport(), b)
    QTest.mouseRelease(window.canvas.viewport(), Qt.LeftButton, pos=b)
    changed = window.selected_object()["bbox"]
    check("mouse_drag_maps_to_original_pixels", all(abs(actual-wanted) < 1 for actual, wanted in zip(changed, [18, 22, 123, 157])))
    window.undo()
    check("undo_restores_original_box", window.selected_object()["bbox"] == previous)
    window.mode.setCurrentIndex(1)
    a = window.canvas.mapFromScene(QPointF(30, 172))
    b = window.canvas.mapFromScene(QPointF(90, 195))
    with patch.object(QInputDialog, "getText", return_value=("UI-ONLY-OBJECT", True)):
        QTest.mousePress(window.canvas.viewport(), Qt.LeftButton, pos=a)
        QTest.mouseMove(window.canvas.viewport(), b)
        QTest.mouseRelease(window.canvas.viewport(), Qt.LeftButton, pos=b)
    check("mouse_draw_adds_identified_object", len(window.draft["objects"]) == 3 and window.selected_object()["specimen_id"] == "UI-ONLY-OBJECT")
    window.delete_object()
    check("delete_object_and_save", len(window.draft["objects"]) == 2 and window.save())
    window.review()
    window.objects.setCurrentRow(0)
    window.specimen.setText("")
    check("invalid_text_blocks_save_without_losing_draft", not window.save() and window.specimen.text() == "" and window.dirty)
    check("invalid_save_reported_to_user", len(errors) == 1)
    errors.clear()
    window.undo()
    check("undo_recovers_invalid_edit", window.specimen.text() == "train-normal" and window.save())
    window.review()
    results = []
    window.run_work(lambda: export_collection(window.collection, root / "normal-export", "normal_bank"), results.append)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and (window.worker.isRunning() or not window.central.isEnabled()):
        app.processEvents()
        QTest.qWait(10)
    app.processEvents()
    check("worker_export_keeps_event_loop_responsive", bool(results) and results[0]["status"] == "COMPLETE" and window.central.isEnabled())
    window.mode.setCurrentIndex(0)
    window.objects.setCurrentRow(1)
    app.processEvents()
    screenshot = root / "labeler.png"
    check("screenshot_written", window.grab().save(str(screenshot)))
    check("no_unexpected_dialog_errors", not errors)
    window.close()
    write_json(root / "report.json", {"passed": True, "mode": "Qt offscreen widget events", "checks": checks,
        "screenshot": str(screenshot), "synthetic": True, "native_monitor_interaction_tested": False})
    print(f"Qt checks passed: {len(checks)}; {screenshot}")


if __name__ == "__main__":
    main()

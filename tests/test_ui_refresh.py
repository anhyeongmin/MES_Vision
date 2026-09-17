"""UI refresh under slow reads, stale responses, navigation and queue changes."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from PySide6.QtCore import QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QTableWidget
from mes_vision.ui_refresh import update_table
from mes_vision.operation.window import DesktopWindow
from mes_vision.vlm.viewer import AnalysisViewer
from mes_vision.vlm.fixtures import make_vlm_fixture

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


def wait_until(predicate, timeout=5):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        APP.processEvents()
        time.sleep(.005)
    APP.processEvents()
    if not predicate(): raise AssertionError("UI work did not finish")


class TableRefreshTests(unittest.TestCase):
    def test_unchanged_rows_emit_no_cell_changes_and_keep_scroll_and_selection(self):
        table = QTableWidget(0, 2)
        table.resize(300, 200); table.show(); APP.processEvents()
        rows = [(str(i), ((str(i), str(i)), ("pending", "pending"))) for i in range(200)]
        update_table(table, rows, selected_key="100")
        table.verticalScrollBar().setValue(60)
        item = table.item(100, 0); scroll = table.verticalScrollBar().value()
        changes = QSignalSpy(table.model().dataChanged)
        for _ in range(10): update_table(table, rows, selected_key="100")
        self.assertEqual(changes.count(), 0)
        self.assertIs(table.item(100, 0), item)
        self.assertEqual(table.currentRow(), 100)
        self.assertEqual(table.verticalScrollBar().value(), scroll)
        rows[100] = ("100", (("100", "100"), ("completed", "completed")))
        update_table(table, rows, selected_key="100")
        self.assertEqual(changes.count(), 2)  # text and tooltip of one cell only
        self.assertIs(table.item(100, 0), item)
        table.close()

    def test_insert_remove_and_empty_preserve_identity_not_old_row_number(self):
        table = QTableWidget(0, 1)
        rows = [(key, ((key, key),)) for key in ("a", "b", "c")]
        update_table(table, rows, selected_key="b")
        rows.insert(0, ("new", (("new", "new"),)))
        update_table(table, rows, selected_key="b")
        self.assertEqual(table.currentRow(), 2)
        update_table(table, rows, selected_key="c")
        self.assertEqual(table.currentRow(), 3)
        update_table(table, [], selected_key="c")
        self.assertEqual(table.currentRow(), -1)
        self.assertEqual(table.rowCount(), 0)


class OperatorRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.w = DesktopWindow(ROOT, self.temp.name, start_worker=False)
        self.w.timer.stop(); self.w.analysis.timer.stop()

    def tearDown(self):
        self.w.robot = self.w.camera = self.w.engine = None
        self.w.closing = True
        self.w.vlm_read.invalidate(); self.w.analysis.stop_refresh()
        wait_until(lambda: not self.w.vlm_read.busy and not self.w.analysis.refresh_read.busy and not self.w.readiness_read.busy and not self.w.tasks)
        self.w.close(); APP.processEvents(); self.temp.cleanup()

    def test_25_ticks_do_not_repeat_database_reads_before_one_second(self):
        with patch.object(self.w.queue, "enabled", wraps=self.w.queue.enabled) as read:
            self.w.tick(); wait_until(lambda: not self.w.vlm_read.busy)
            base = self.w.vlm_probe_at - 1
            with patch("mes_vision.operation.window.time.monotonic", return_value=base + .5):
                for _ in range(25): self.w.tick()
            self.assertEqual(read.call_count, 1)
            self.w.vlm_probe_at = 0
            self.w.tick(); wait_until(lambda: not self.w.vlm_read.busy)
            self.assertEqual(read.call_count, 2)

    def test_slow_database_does_not_block_events_or_robot_watchdog(self):
        entered, release = threading.Event(), threading.Event()
        threads = []
        def slow():
            threads.append(threading.get_ident()); entered.set(); release.wait(5); return False
        with patch.object(self.w.queue, "enabled", slow):
            try:
                self.w.tick(); wait_until(entered.is_set)
                for _ in range(20): self.w.tick()
                self.assertEqual(len(threads), 1)
                self.assertNotEqual(threads[0], threading.get_ident())
                delivered = []; QTimer.singleShot(0, lambda: delivered.append(True))
                wait_until(lambda: bool(delivered))
                stopped = []
                self.w.robot = SimpleNamespace(stop_motion=lambda: stopped.append(True))
                self.w.robot_seen_at = time.monotonic() - 6
                self.w.running = True; self.w.safe_tick()
                self.assertTrue(stopped); self.assertFalse(self.w.running)
            finally:
                release.set(); wait_until(lambda: not self.w.vlm_read.busy)

    def test_old_on_read_cannot_reverse_operator_off(self):
        entered, release = threading.Event(), threading.Event()
        enabled = self.w.queue.enabled
        ui_thread = threading.get_ident()
        def old():
            if threading.get_ident() == ui_thread: return enabled()
            entered.set(); release.wait(5); return True
        self.w.toggle_vlm(True)
        with patch.object(self.w.queue, "enabled", old):
            try:
                self.w.vlm_probe_at = 0; self.w.tick(); wait_until(entered.is_set)
                self.w.analysis.toggle_vlm(False)
                self.assertFalse(self.w.vlm.isChecked())
            finally:
                release.set(); wait_until(lambda: not self.w.vlm_read.busy)
        self.assertFalse(self.w.vlm.isChecked()); self.assertFalse(self.w.queue.enabled())

    def test_closing_waits_for_read_and_discards_late_result(self):
        entered, release = threading.Event(), threading.Event()
        def old(): entered.set(); release.wait(5); return True
        self.w.show(); APP.processEvents()
        with patch.object(self.w.queue, "enabled", old):
            try:
                self.w.tick(); wait_until(entered.is_set)
                self.assertFalse(self.w.close()); self.assertTrue(self.w.closing)
            finally:
                release.set(); wait_until(lambda: not self.w.vlm_read.busy)
        wait_until(lambda: not self.w.readiness_read.busy)
        self.w.close(); self.assertFalse(self.w.isVisible()); self.assertFalse(self.w.vlm.isChecked())


class ViewerRefreshTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = make_vlm_fixture(Path(self.temp.name) / "fixture")
        self.queue = self.fixture["queue"]
        self.w = AnalysisViewer(self.queue, ROOT, run_worker=False)
        self.w.timer.stop()

    def tearDown(self):
        self.w.stop_refresh()
        wait_until(lambda: not self.w.refresh_read.busy)
        self.w.close(); APP.processEvents(); self.temp.cleanup()

    def test_hidden_page_does_not_query_and_show_fetches_current_state(self):
        with patch.object(self.queue, "list", wraps=self.queue.list) as read:
            for _ in range(10): self.w.poll_refresh()
            self.assertEqual(read.call_count, 0)
            self.queue.set_enabled(True)
            self.w.show(); wait_until(lambda: not self.w.refresh_read.busy)
            self.assertEqual(read.call_count, 1); self.assertTrue(self.w.toggle.isChecked())
            self.w.hide(); self.w.poll_refresh(); self.assertEqual(read.call_count, 1)

    def test_unchanged_queue_does_not_rebuild_table_or_reload_evidence(self):
        changes = QSignalSpy(self.w.table.model().dataChanged)
        with patch("mes_vision.vlm.viewer.load_snapshot", side_effect=AssertionError("unnecessary disk read")) as read:
            for _ in range(5): self.w.refresh()
            self.assertEqual(changes.count(), 0); self.assertEqual(read.call_count, 0)
        self.assertIn("균열", self.w.reasons.toPlainText())

    def test_hide_during_read_discards_old_rows_and_show_reloads(self):
        self.w.show(); wait_until(lambda: not self.w.refresh_read.busy)
        before = self.w.toggle.isChecked()
        entered, release = threading.Event(), threading.Event()
        original = self.w.read_refresh
        def slow():
            data = original(); entered.set(); release.wait(5); return data
        with patch.object(self.w, "read_refresh", slow):
            try:
                self.w.poll_refresh(); wait_until(entered.is_set)
                self.w.hide(); self.queue.set_enabled(True)
            finally:
                release.set(); wait_until(lambda: not self.w.refresh_read.busy)
        self.assertEqual(self.w.toggle.isChecked(), before)
        self.w.show(); wait_until(lambda: not self.w.refresh_read.busy)
        self.assertTrue(self.w.toggle.isChecked())

    def test_empty_queue_clears_previously_selected_evidence(self):
        self.assertTrue(self.w.reasons.toPlainText())
        with self.queue.connect() as db: db.execute("DELETE FROM jobs")
        self.w.refresh()
        self.assertEqual(self.w.table.rowCount(), 0)
        self.assertIsNone(self.w.selected_key)
        self.assertFalse(self.w.reasons.toPlainText()); self.assertFalse(self.w.analysis.toPlainText())

    def test_new_result_updates_analysis_without_changing_basic_evidence(self):
        reasons = self.w.reasons.toPlainText()
        key = self.w.selected_key
        self.queue.set_enabled(True); self.queue.retry(key)
        job = self.queue.claim("ui-test")
        result = {"analysis": {"observation": "A visible crack.", "needs_review": True}, "limitations": []}
        self.queue.finish(key, job["token"], "COMPLETED", result=result)
        self.w.show(); wait_until(lambda: not self.w.refresh_read.busy)
        self.assertEqual(self.w.selected_key, key)
        self.assertIn("A visible crack.", self.w.analysis.toPlainText())
        self.assertEqual(self.w.table.item(0, 3).text(), "A visible crack.")
        self.assertEqual(self.w.reasons.toPlainText(), reasons)


if __name__ == "__main__": unittest.main()

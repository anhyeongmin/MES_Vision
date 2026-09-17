import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QInputDialog,QFileDialog
import test_operation
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.storage import StorageService
from mes_vision.operation.storage_dialog import StorageDialog
from mes_vision.vlm.viewer import WorkerThread

ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])


def wait_until(predicate,seconds=8):
    end=time.monotonic()+seconds
    while not predicate() and time.monotonic()<end: APP.processEvents(); time.sleep(.005)
    APP.processEvents()
    if not predicate(): raise AssertionError("Qt work did not finish")


class StorageUiTests(unittest.TestCase):
    def setUp(self):
        self.f=test_operation.OperationTests(); self.f.setUp(); self.f.models.defect=True
        self.f.models.detections=(test_operation.detection(),test_operation.detection(x=210))
        self.f.engine.commit(self.f.inspect()); self.f.engine.close(); self.f.store.end_session(self.f.session)
        self.external=tempfile.TemporaryDirectory(); self.out=Path(self.external.name)
        self.w=DesktopWindow(ROOT,self.f.root,start_worker=False); self.w.timer.stop(); self.w.analysis.timer.stop(); self.w.show(); APP.processEvents()
        self.dialog=None

    def tearDown(self):
        if self.dialog:
            wait_until(lambda:not self.dialog.busy); self.dialog.close(); self.dialog.deleteLater()
        wait_until(lambda:not self.w.tasks and not self.w.readiness_read.busy)
        if self.w.worker: self.w.worker.stop_event.set(); wait_until(lambda:not self.w.worker.isRunning())
        self.w.closing=True; self.w.close(); APP.processEvents(); self.f.tearDown(); self.external.cleanup()

    def test_filters_and_clear_unknown_review_shortcut(self):
        self.assertEqual(self.w.history_result["summary"]["inspections"],2)
        self.w.history_filter.setCurrentIndex(1); self.w.search_history(); wait_until(lambda:not self.w.history_busy)
        self.assertEqual(self.w.history_table.rowCount(),0)
        self.w.find_unreviewed(); wait_until(lambda:not self.w.history_busy)
        self.assertEqual(self.w.history_table.rowCount(),2); self.assertEqual(self.w.history_search.decision,"ATTENTION")

    def test_late_query_cannot_replace_new_filter(self):
        started=threading.Event(); release=threading.Event(); original=StorageService.search; calls=[]
        def slow(service,*args,**kwargs):
            calls.append(1)
            if len(calls)==1: started.set(); release.wait(3)
            return original(service,*args,**kwargs)
        with patch.object(StorageService,"search",slow):
            self.w.search_history(); wait_until(started.is_set)
            self.w.history_filter.setCurrentIndex(1); self.w.search_history(); release.set()
            wait_until(lambda:len(calls)>=2 and not self.w.history_busy)
        self.assertEqual(self.w.history_result["summary"]["inspections"],0)

    def test_record_reason_and_manual_hold_share_same_evidence(self):
        row=self.w.history_rows[0]; self.w.show_record(row["object_id"])
        self.assertIn("균열",self.w.history_details.toPlainText())
        with patch.object(QInputDialog,"getText",return_value=("고객 검토",True)): self.w.hold_record()
        wait_until(lambda:not self.w.history_busy)
        self.assertTrue(all(r["hold_note"]=="고객 검토" for r in self.w.history_rows))
        self.assertFalse(self.w.saved_canvas.pixmap.isNull())

    def test_export_button_writes_complete_search(self):
        destination=self.out/"report"
        with patch.object(QFileDialog,"getSaveFileName",return_value=(str(destination),"")): self.w.export_history()
        wait_until(lambda:not self.w.tasks and not self.w.readiness_read.busy)
        self.assertTrue((destination/"inspections.csv").exists()); self.assertIn("2건",self.w.status.text())

    def test_storage_busy_guard_and_restore_result(self):
        self.dialog=d=StorageDialog(self.w.store,ROOT,self.w); d.show()
        released=threading.Event(); d.run(lambda:released.wait(3)); self.assertTrue(d.busy); d.reject(); self.assertTrue(d.isVisible())
        released.set(); wait_until(lambda:not d.busy)
        d.run(lambda:d.service.backup(self.out/"backup")); wait_until(lambda:not d.busy)
        self.assertTrue((self.out/"backup/backup.json").exists())
        d.run(lambda:d.service.restore(self.out/"backup",self.out/"restored"),d.restore_done); wait_until(lambda:not d.busy)
        self.assertTrue(d.open_restored.isEnabled()); self.assertIn("VLM은 OFF",d.message.text())

    def test_worker_releases_before_maintenance_and_restarts_off(self):
        self.w.worker=WorkerThread(self.w.queue,ROOT,allow_synthetic=False,backend="qwen"); self.w.worker.start()
        visited=[]
        def visit(dialog):
            with dialog.service.exclusive(idle=True): visited.append(True)
            return 0
        with patch.object(StorageDialog,"exec",visit):
            self.w.open_storage(); wait_until(lambda:bool(visited) and not self.w.maintenance_waiting)
        self.assertFalse(self.w.queue.enabled()); self.w.timer.stop(); self.w.analysis.timer.stop()

    def test_selected_record_resolves_new_archive_path_for_vlm(self):
        from mes_vision.operation.maintenance import Maintenance
        row=self.w.history_rows[0]; self.w.show_record(row["object_id"]); service=Maintenance(self.w.store)
        service.save_policy(dict(service.policy(),archive_directory=str(self.out/"archive")))
        with self.w.store.connect() as db: db.execute("UPDATE inspections SET created=?",(time.time()-900*86400,))
        service.archive(service.preview_archive()); self.w.queue.set_enabled(True)
        self.w.request_analysis(); self.assertNotEqual(self.w.record[0]["path"],row["path"])
        self.assertEqual(self.w.queue.list()[0]["snapshot_path"],self.w.record[0]["path"]); self.w.queue.set_enabled(False)


if __name__=="__main__": unittest.main()

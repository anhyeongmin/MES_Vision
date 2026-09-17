"""Manual dialog lifetime, evidence binding and GPU scheduling regressions."""
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from PySide6.QtCore import QTimer, QProcess
from PySide6.QtWidgets import QApplication
from mes_vision.station.manual_dialog import ManualInspectionDialog
from mes_vision.station.photo_inspection import ResidentPhotos
from mes_vision.training.data import write_json
from test_photo_inspection import Registry, ROOT

APP = QApplication.instance() or QApplication([])


class ManualVlmTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.camera = Mock(); self.camera.disposed=False
        self.camera.stopping.is_set.return_value=False
        self.camera.get_latest.return_value=(None, None)
        self.dialog = ManualInspectionDialog(ROOT,self.path,{'driver':'uvc'},camera=self.camera)
        self.dialog.timer.stop()
        self.addCleanup(self.dialog.deleteLater)

    def test_x_actually_hides_and_finishes_without_stopping_borrowed_camera(self):
        d=self.dialog; done=Mock(); d.finished.connect(done); d.show()
        APP.processEvents(); self.assertTrue(d.isVisible())
        d.close(); APP.processEvents()
        self.assertFalse(d.isVisible()); done.assert_called_once()
        self.camera.stopping.set.assert_not_called()
        d.close(); APP.processEvents(); done.assert_called_once()

    def test_modal_x_returns_and_own_camera_shutdown_is_idempotent(self):
        d=self.dialog; d.owns_camera=True
        def close():
            d.close(); d.close(); self.camera.disposed=True; d.tick()
        QTimer.singleShot(0,close)
        self.assertEqual(d.exec(),0)
        self.camera.stopping.set.assert_called_once()

    def test_close_waits_for_resident_exit(self):
        d=self.dialog; d.show(); d.server=Mock(); server=d.server
        d.close(); d.close(); self.assertTrue(d.isVisible())
        server.kill.assert_called_once(); d.finished(0,0); d.tick()
        self.assertFalse(d.isVisible())

    def test_actual_process_exit_reaps_server_and_closes_dialog(self):
        import sys
        d=self.dialog; d.show(); d.start_server()
        # start_server wiring is the regression target. Stop before loading CUDA.
        d.close()
        deadline=time.monotonic()+10
        while d.isVisible() and time.monotonic()<deadline:
            APP.processEvents(); time.sleep(.01)
        self.assertFalse(d.isVisible())
        self.assertIsNone(d.server); self.assertTrue(d.close_ready)

    def test_optional_error_cannot_cancel_pending_capture(self):
        d=self.dialog; d.pending={'role':'overview'}
        d.vlm_guard(lambda: (_ for _ in ()).throw(ValueError('bad explanation')))
        self.assertEqual(d.pending,{'role':'overview'})
        self.assertIn('bad explanation',d.vlm_text.toPlainText())

    def bind(self, key, digest):
        self.dialog.details[key]=dict(associated=True,snapshot_digest=digest,
            vlm_object_id='obj-'+key,vlm_job='job-'+key)

    def test_selection_binds_explanation_and_does_not_change_baseline(self):
        d=self.dialog; q=Mock(); q.enabled.return_value=True; d.vlm_queue=q
        self.bind('1','a'); self.bind('2','b')
        q.get.side_effect=lambda jid:dict(snapshot_digest='a' if jid=='job-1' else 'b',
            object_id='obj-1' if jid=='job-1' else 'obj-2',state='COMPLETED',
            result={'analysis':{'observation':jid}})
        d.reasons.setPlainText('baseline REVIEW'); d.selected='1'; d.refresh_vlm()
        self.assertIn('job-1',d.vlm_text.toPlainText())
        d.selected='2'; d.refresh_vlm(); self.assertIn('job-2',d.vlm_text.toPlainText())
        self.assertNotIn('job-1',d.vlm_text.toPlainText())
        self.assertEqual(d.reasons.toPlainText(),'baseline REVIEW')
        q.get.return_value={}; q.get.side_effect=lambda _:dict(snapshot_digest='wrong',object_id='obj-2')
        with self.assertRaisesRegex(ValueError,'identity mismatch'): d.refresh_vlm()

    def test_off_failure_and_reset_preserve_base_results(self):
        d=self.dialog; q=Mock(); d.vlm_queue=q; q.enabled.return_value=False
        d.reasons.setPlainText('NG05'); d.refresh_vlm()
        self.assertIn('OFF',d.vlm_text.toPlainText()); self.assertEqual(d.reasons.toPlainText(),'NG05')
        d.vlm_failed('error'); self.assertEqual(d.reasons.toPlainText(),'NG05')
        d.vlm_jobs={'mine'}; d.reset_cycle(); q.cancel.assert_called_once_with('mine')

    def test_background_resident_yields_gpu_but_retains_models(self):
        bundle=dict(models=dict(overview={},objects={},defects={}))
        path=self.path/'bundle.json'; write_json(path,bundle)
        with patch('mes_vision.station.photo_inspection.load_bundle',return_value=bundle), \
             patch('torch.cuda.is_available',return_value=True), \
             patch('torch.cuda.mem_get_info',return_value=(16*1024**3,24*1024**3)):
            registry=Registry()
            with ResidentPhotos(path,ROOT,self.path,registry,allow_background=True) as resident:
                self.assertTrue(resident.allow_vlm)
                self.assertFalse(resident.gpu.resident_blocks_vlm())
                self.assertFalse(resident.gpu.foreground_requested())
                with resident.gpu.background_lock(): pass
                self.assertEqual(registry.closed,[])
            self.assertEqual(len(registry.closed),3)


if __name__=='__main__': unittest.main()

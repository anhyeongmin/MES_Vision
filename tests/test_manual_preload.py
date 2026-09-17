import tempfile,time,unittest
from pathlib import Path
from unittest.mock import Mock
from PySide6.QtWidgets import QApplication
from mes_vision.station.manual_dialog import ManualInspectionDialog
ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])

class ManualPreloadTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  camera=Mock();camera.disposed=False;camera.stopping.is_set.return_value=False
  self.d=ManualInspectionDialog(ROOT,Path(self.tmp.name),{'driver':'uvc'},camera=camera)
  self.d.timer.stop();self.addCleanup(self.d.deleteLater)
  self.d.server=Mock();self.d.server_bundle=self.d.bundle
  self.thread=Mock();self.d.vlm_worker=self.thread
  self.d.ensure_vlm_worker=Mock(return_value=self.thread)
 def test_prepare_loads_vlm_without_a_photo_when_enabled(self):
  self.d.vlm_queue.set_enabled(True);self.d.prepare_models()
  self.thread.worker.request_preload.assert_called_once_with(self.d.preload_owner)
  self.assertTrue(self.d.vlm_preload_wanted);self.assertEqual(self.d.vlm_queue.list(),[])
 def test_off_prepare_does_not_load_vlm(self):
  self.d.prepare_models();self.d.ensure_vlm_worker.assert_not_called()
 def test_release_button_releases_preload_and_vision(self):
  self.d.vlm_queue.set_enabled(True);self.d.prepare_models();self.d.release_models()
  self.thread.worker.release_preload.assert_called_once_with(self.d.preload_owner)
  self.d.server.kill.assert_called_once();self.assertFalse(self.d.vlm_preload_wanted)
 def test_only_fresh_ready_state_is_shown(self):
  self.d.vlm_queue.set_enabled(True)
  self.d.vlm_queue.runtime(dict(model_ready=True,phase='WARM_IDLE',updated=time.time()))
  self.d.refresh_vlm();self.assertIn('준비 완료',self.d.vlm_model_status.text())
  self.d.vlm_queue.runtime(dict(model_ready=True,phase='WARM_IDLE',updated=0))
  self.d.refresh_vlm();self.assertIn('미로드',self.d.vlm_model_status.text())

if __name__=='__main__':unittest.main()

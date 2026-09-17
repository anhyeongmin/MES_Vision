import tempfile,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from mes_vision.station.manual_dialog import ManualInspectionDialog
from mes_vision.vlm.region_backend import QUESTIONS
ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])

class ManualCapabilitiesTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  camera=Mock();camera.disposed=False;camera.stopping.is_set.return_value=False
  self.d=ManualInspectionDialog(ROOT,Path(self.tmp.name),{'driver':'uvc'},camera=camera)
  self.d.timer.stop();self.addCleanup(self.d.deleteLater)
  self.d.vlm_queue=Mock();self.d.vlm_queue.enabled.return_value=True
  self.d.vlm_queue.enqueue.return_value='queued';self.d.vlm_worker=Mock();self.d.vlm_worker.isRunning.return_value=True
  self.d.refresh_vlm=Mock();self.d.vlm_capabilities=Mock(return_value=set(QUESTIONS))
 def test_every_class_can_enqueue_including_no_defect(self):
  for code in QUESTIONS:
   obj={'object_id':'one','checks':[{'findings':[] if code=='OK' else [{'defect_code':code}]}]}
   result={'objects':[obj]}
   self.d.details={'1':dict(associated=True,result=result,snapshot='snapshot',snapshot_digest='digest')}
   with patch('mes_vision.station.manual_dialog.load_snapshot',return_value=({},result,'digest')):
    self.d.request_vlm('1')
   self.assertEqual(self.d.details['1']['vlm_job'],'queued')
  self.assertEqual(self.d.vlm_queue.enqueue.call_count,7)
 def test_off_and_unassociated_do_not_enqueue(self):
  self.d.vlm_queue.enabled.return_value=False;self.d.request_vlm('1')
  self.d.vlm_queue.enabled.return_value=True
  self.d.details={'1':dict(associated=False)};self.d.request_vlm('1')
  self.d.vlm_queue.enqueue.assert_not_called()
 def test_legacy_model_does_not_accept_untrained_capability(self):
  self.d.vlm_capabilities.return_value={'NG04','NG05'}
  result={'objects':[{'object_id':'one','checks':[{'findings':[{'defect_code':'NG06'}]}]}]}
  self.d.details={'1':dict(associated=True,result=result,snapshot='snapshot',snapshot_digest='digest')}
  with patch('mes_vision.station.manual_dialog.load_snapshot',return_value=({},result,'digest')):
   self.d.request_vlm('1')
  self.d.vlm_queue.enqueue.assert_not_called()

if __name__=='__main__':unittest.main()

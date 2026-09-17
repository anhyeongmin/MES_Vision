import tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch,Mock
from filelock import FileLock
from mes_vision.vlm.worker import AnalysisWorker
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.training.data import write_json
ROOT=Path(__file__).resolve().parents[1]

class PreloadTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.q=AnalysisQueue(Path(self.tmp.name)/'queue');self.q.set_enabled(True)
  self.w=AnalysisWorker(self.q,ROOT,backend='mock',allow_synthetic=True,idle_seconds=0,keep_alive_seconds=.15)
  self.errors=[];self.thread=None
 def tearDown(self):
  self.w.stop_event.set()
  if self.thread:self.thread.join(8);self.assertFalse(self.thread.is_alive())
  self.assertFalse(self.errors);self.tmp.cleanup()
 def start(self):
  def run():
   try:self.w.run()
   except Exception as e:self.errors.append(repr(e))
  self.thread=threading.Thread(target=run);self.thread.start()
 def wait(self,fn,timeout=8):
  end=time.monotonic()+timeout
  while time.monotonic()<end:
   if fn():return
   if self.errors:self.fail(str(self.errors))
   time.sleep(.02)
  self.fail(str(self.q.runtime()))
 def ready(self):return (self.q.runtime() or {}).get('phase')=='WARM_IDLE'
 def test_no_photo_needed_and_hold_until_last_owner_releases(self):
  self.w.request_preload('a');self.w.request_preload('b');self.start();self.wait(self.ready)
  pid=self.q.runtime()['model_pid'];time.sleep(.4)
  self.assertEqual(self.q.runtime()['model_pid'],pid);self.assertEqual(self.q.list(),[])
  self.w.release_preload('a');time.sleep(.2);self.assertTrue(self.w.preload_requested.is_set())
  self.w.release_preload('b');self.wait(lambda:not (self.q.runtime() or {}).get('model_ready'))
  self.assertEqual(self.q.list(),[])
 def test_disabled_clears_intent_and_does_not_reload_on_its_own(self):
  self.w.request_preload();self.start();self.wait(self.ready)
  self.q.set_enabled(False);self.wait(lambda:(self.q.runtime() or {}).get('phase')=='DISABLED')
  self.assertFalse(self.w.preload_requested.is_set());self.q.set_enabled(True);time.sleep(.3)
  self.assertIsNone(self.q.runtime()['model_pid'])
 def test_preload_failure_is_reported_once_without_fake_inspection(self):
  with patch('mes_vision.vlm.worker.ModelProcess',side_effect=RuntimeError('load broken')) as create:
   self.w.request_preload();self.start();self.wait(lambda:(self.q.runtime() or {}).get('preload_error'))
   time.sleep(.2);self.assertEqual(create.call_count,1);self.assertEqual(self.q.list(),[])
   self.w.stop_event.set();self.thread.join(8)
 def test_sharing_resident_keeps_pid_across_foreground(self):
  with FileLock(str(self.w.gpu.root/'resident.lock'),timeout=0):
   write_json(self.w.gpu.root/'resident.json',{'active':True,'allow_vlm':True})
   self.w.request_preload();self.start();self.wait(self.ready);pid=self.q.runtime()['model_pid']
   with self.w.gpu.foreground(timeout=5):
    self.wait(lambda:self.q.runtime()['phase']=='WAITING_FOREGROUND')
    self.assertEqual(self.q.runtime()['model_pid'],pid)
   self.wait(self.ready);self.assertEqual(self.q.runtime()['model_pid'],pid)
   self.assertEqual(self.q.runtime()['model_load_count'],1)
 def test_unapproved_foreground_releases_weights(self):
  self.w.request_preload();self.start();self.wait(self.ready)
  with self.w.gpu.foreground(timeout=5):
   self.wait(lambda:self.q.runtime()['phase']=='WAITING_FOREGROUND')
   self.assertIsNone(self.q.runtime()['model_pid'])
 def test_memory_reply_is_drained_only_for_approved_retention(self):
  self.w.preload_requested.set();self.w.waiting_idle_memory=True
  child=Mock();child.connection.poll.return_value=True
  child.connection.recv.return_value={'type':'idle_memory'};self.w.child=child
  try:
   with patch.object(self.w,'interruption',return_value='FOREGROUND_PRIORITY'),patch.object(self.w.gpu,'resident_allows_retention',return_value=True):
    self.assertEqual(self.w.wait_message(None,.1),{'type':'idle_memory'})
   with patch.object(self.w,'interruption',return_value='FOREGROUND_PRIORITY'),patch.object(self.w.gpu,'resident_allows_retention',return_value=False):
    with self.assertRaisesRegex(Exception,'FOREGROUND_PRIORITY'):self.w.wait_message(None,.1)
   child.connection.recv.assert_called_once()
  finally:self.w.child=None

if __name__=='__main__':unittest.main()

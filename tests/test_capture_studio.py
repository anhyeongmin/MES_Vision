import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
from copy import deepcopy
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget
from PySide6.QtCore import QTimer
from mes_vision.operation.capture_controls import validate_controls
from mes_vision.operation.training_recorder import TrainingRecorder
from mes_vision.operation.catalog import default_equipment,OperationStore,new_product
from mes_vision.operation.capture_dialog import CaptureDialog
from mes_vision.station.process_camera import ProcessCameraPort
from mes_vision.operation.settings_ui import assign_acquisition_revision

APP=QApplication.instance() or QApplication([])
RANGES={'exposure':dict(min=-13,max=-1,step=1,default=-6,manual=True,auto=True),
        'gain':dict(min=0,max=100,step=1,default=0,manual=True,auto=False)}

class CaptureTests(unittest.TestCase):
    def test_bounds_and_unsupported_controls(self):
        for value in (-13,-6,-1): validate_controls(dict(auto_exposure=False,exposure=value,gain=0),RANGES)
        for value in (-14,-.5,-6.5,float('nan')):
            with self.assertRaises(ValueError): validate_controls(dict(auto_exposure=False,exposure=value,gain=0),RANGES)
        with self.assertRaises(ValueError): validate_controls(dict(auto_exposure=False,exposure=-6,gain=0),{})

    def test_mp4_roundtrip_and_metadata_without_burned_annotations(self):
        import av
        with tempfile.TemporaryDirectory() as root:
            settings=dict(width=64,height=48,fps=30,auto_exposure=False,exposure=-6,gain=0)
            recorder=TrainingRecorder(root,dict(label='NG04',product_id='part',purpose='evaluation'),settings)
            for i in range(8):
                recorder.add(NS(rgb=np.full((48,64,3),100+i,np.uint8),width=64,height=48,sequence=i,frame_id=str(i),media_time_seconds=None))
                time.sleep(.005)
            folder=Path(recorder.close()); manifest=json.loads((folder/'recording.json').read_text())
            self.assertEqual(manifest['status'],'COMPLETED'); self.assertEqual(manifest['frames'],8)
            self.assertFalse(manifest['annotations_reviewed']); self.assertEqual(manifest['purpose'],'evaluation')
            with av.open(str(recorder.path)) as video: frames=list(video.decode(video=0))
            self.assertEqual(len(frames),8); self.assertTrue(all(a.pts<b.pts for a,b in zip(frames,frames[1:])))
            second=TrainingRecorder(root,dict(label='NG04',product_id='part'),settings)
            self.assertNotEqual(second.folder,folder); second.close('INTERRUPTED')

    def test_profile_switch_waits_for_ack_and_retains_request_identity(self):
        camera=Mock(); camera.settings={'capture_profiles':{'overview':{},'detail':{}}}; camera.command.return_value='token'; camera.replies={}
        camera.get_latest.return_value=(None,0)
        port=ProcessCameraPort(camera,Mock(),{'max_frame_age_seconds':.1,'timing_validation_reference':'test'})
        request={'id':'a','kind':'capture_detail','issued_at':100,'deadline':110}
        port.submit(request,{}); port.tick(101); self.assertIsNone(port.gate)
        camera.replies['token']={'result':{'applied_at':102}}
        port.tick(102.1); self.assertEqual(port.gate.request,request); self.assertEqual(port.gate.not_before,102)
        port.cancel(); self.assertFalse(port.busy)

    def test_profile_change_updates_revision(self):
        old=default_equipment()['camera']; new=deepcopy(old)
        new['capture_profiles']={'detail':dict(auto_exposure=False,exposure=-7,gain=0)}
        assign_acquisition_revision(old,new,'changed'); self.assertEqual(new['acquisition_revision'],'changed')

    def test_dialog_limits_apply_and_saves_two_profiles(self):
        with tempfile.TemporaryDirectory() as root,patch.object(CaptureDialog,'connect_when_ready'):
            owner=QWidget(); owner.root=Path(root); owner.store=OperationStore(Path(root)/'runtime')
            owner.equipment=default_equipment(); owner.product=owner.store.save_product(new_product('ASH'))
            owner.camera=None; owner.refresh_equipment=Mock()
            d=CaptureDialog(owner); d.ranges=RANGES; d.load_profile()
            self.assertEqual((d.exposure.minimum(),d.exposure.maximum()),(-13,-1))
            self.assertEqual((d.gain.minimum(),d.gain.maximum()),(0,100))
            d.auto.setChecked(False); d.exposure.setValue(-8)
            d.profiles['overview']=d.values(); d.applied=('overview',d.values()); d.save_profiles()
            self.assertEqual(owner.store.equipment()['camera']['capture_profiles']['overview']['exposure'],-8)
            self.assertIn('detail',owner.store.equipment()['camera']['capture_profiles'])
            d.recording=True; d.changed(); self.assertFalse(d.profile.isEnabled()); self.assertFalse(d.apply_button.isEnabled())
            d.recording=False; d.close(); d.deleteLater(); owner.deleteLater()

    def test_modal_close_returns_without_destroying_live_worker(self):
        with tempfile.TemporaryDirectory() as root,patch.object(CaptureDialog,'connect_when_ready'):
            owner=QWidget(); owner.root=Path(root); owner.store=OperationStore(root); owner.equipment=default_equipment()
            owner.product=None; owner.camera=None
            d=CaptureDialog(owner); QTimer.singleShot(20,d.close); d.exec()
            self.assertTrue(d.closing); self.assertFalse(d.timer.isActive())
            d.deleteLater(); owner.deleteLater()

if __name__=='__main__': unittest.main()

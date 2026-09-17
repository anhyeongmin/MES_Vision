"""No hardware: exercise actual rectification, persistence and coordinate guards."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from copy import deepcopy
from dataclasses import replace
from datetime import datetime,timezone
import tempfile,time,unittest
import numpy as np
from mes_vision.inputs import Frame,SourceKind
from mes_vision.operation.catalog import default_equipment
from mes_vision.operation.acquisition import Acquisition,configure_equipment,identity,inspection_camera,verify_frame
from mes_vision.operation.quality import validate_equipment,in_workspace
from mes_vision.station.manual_capture import Rectifier,save_capture as save_manual
from mes_vision.station.evidence import save_capture
from mes_vision.station.worker import read_capture
from mes_vision.station.camera_port import CaptureGate
from mes_vision.inspection import Box
from test_station_devices import setup,request
from PySide6.QtWidgets import QApplication

APP=QApplication.instance() or QApplication([])

ROOT=Path(__file__).resolve().parents[1]

def equipment():
    e=default_equipment(); e['camera']['serial']='TEST'
    return configure_equipment(ROOT,e)

def raw_frame():
    y,x=np.indices((720,1280))
    rgb=np.stack((x%256,y%256,(x+y)%256),axis=-1).astype(np.uint8)
    rgb.setflags(write=False)
    return Frame('raw:1','raw',1,SourceKind.UVC,'uvc://TEST',datetime.now(timezone.utc),rgb,
                 encoded_size=(1280,720),is_live=True,host_read_completed_monotonic=101.)


class InjectedCamera:
    def __init__(self, settings): self.settings=settings; self.info={'serial':'TEST'}; self.n=0
    def open(self): pass
    def close(self): pass
    def read(self):
        time.sleep(.035); self.n+=1
        return replace(raw_frame(),frame_id='raw:'+str(self.n),sequence=self.n,
                       host_read_completed_monotonic=time.monotonic())

class AcquisitionTests(unittest.TestCase):
    def test_spawned_camera_keeps_atomic_raw_and_rectified_pair(self):
        from mes_vision.operation.camera import CameraProcess
        c=equipment()['camera']; camera=CameraProcess(c,factory=InjectedCamera)
        errors=[]; camera.failed.connect(errors.append); camera.start()
        try:
            deadline=time.monotonic()+15
            while camera.get_latest()[0] is None and not errors and time.monotonic()<deadline:
                APP.processEvents(); time.sleep(.005)
            self.assertFalse(errors,errors)
            view,stamp=camera.get_latest(); raw,raw_stamp=camera.get_latest_raw()
            self.assertIsNotNone(view)
            self.assertEqual(stamp,raw_stamp); self.assertEqual(view.sequence,raw.sequence)
            self.assertEqual((raw.width,view.width),(1280,720))
            np.testing.assert_array_equal(view.rgb,Acquisition(c).apply(raw).rgb)
            with tempfile.TemporaryDirectory() as folder:
                token=camera.command('record_start',{'root':folder,'metadata':{'label':'OK','product_id':'test'}})
                deadline=time.monotonic()+4
                while token not in camera.replies and time.monotonic()<deadline:
                    APP.processEvents(); time.sleep(.005)
                self.assertNotIn('error',camera.replies[token])
                first=camera.get_latest()[0].sequence
                while camera.get_latest()[0].sequence<first+3 and time.monotonic()<deadline:
                    APP.processEvents(); time.sleep(.005)
                token=camera.command('record_stop')
                while token not in camera.replies and time.monotonic()<deadline:
                    APP.processEvents(); time.sleep(.005)
                self.assertNotIn('error',camera.replies[token])
                import av,json
                directory=Path(camera.replies[token]['result']['folder'])
                metadata=json.loads((directory/'recording.json').read_text(encoding='utf-8'))
                self.assertEqual(metadata['preprocessing'],'raw_camera_rgb')
                self.assertGreater(metadata['frames'],0)
                with av.open(str(directory/'OK.mp4')) as video:
                    recorded=next(video.decode(video=0))
                    self.assertEqual((recorded.width,recorded.height),(1280,720))
        finally:
            camera.stopping.set(); deadline=time.monotonic()+6
            while not camera.disposed and time.monotonic()<deadline:
                APP.processEvents(); time.sleep(.01)
            if not camera.disposed:
                camera.process.terminate(); camera.process.join(timeout=2); camera.poll()
        self.assertIsNone(camera.get_latest_raw()[0])

    def test_manual_and_acquisition_pixels_identical_raw_preserved(self):
        raw=raw_frame(); acq=Acquisition(equipment()['camera']); view=acq.apply(raw)
        self.assertEqual((view.width,view.height),(720,720))
        self.assertFalse(view.transformations); self.assertNotEqual(view.frame_id,raw.frame_id)
        self.assertEqual(view.acquisition_source['crop_xyxy'],[280,0,1000,720])
        self.assertEqual((raw.width,raw.height),(1280,720)); self.assertIsNone(raw.acquisition_identity)
        with tempfile.TemporaryDirectory() as d:
            from PIL import Image
            path=save_manual(Path(d)/'manual',raw,acq.rectifier,True,{})
            np.testing.assert_array_equal(np.asarray(Image.open(path)),view.rgb)
            with self.assertRaises(ValueError): save_manual(Path(d)/'bad',view,acq.rectifier,True,{})
        with self.assertRaises(ValueError): acq.apply(view)

    def test_provenance_survives_saved_capture_and_wrong_pipeline_rejected(self):
        c=equipment()['camera']; view=Acquisition(c).apply(raw_frame())
        with tempfile.TemporaryDirectory() as d:
            receipt=save_capture(d,view,camera_serial='TEST',acquired_at=101.)
            restored=read_capture(receipt)
            self.assertEqual(restored.metadata(),view.metadata())
            np.testing.assert_array_equal(restored.rgb,view.rgb)
            verify_frame(restored,c)
            with self.assertRaises(ValueError): verify_frame(replace(restored,acquisition_identity='other'),c)

    def test_device_free_worker_accepts_raw_but_not_unregistered_rectified_input(self):
        from mes_vision.station.worker import verify_acquisition
        raw=raw_frame(); e=equipment(); view=Acquisition(e['camera']).apply(raw)
        verify_acquisition(raw,{})
        verify_acquisition(view,e)
        with self.assertRaises(ValueError): verify_acquisition(view,{})

    def test_rectified_snapshot_to_vlm_region_has_no_second_coordinate_shift(self):
        from mes_vision.inspection.contracts import Crop,Detection,Finding,CheckResult,CheckStatus,ModelRef,ObjectResult,RunResult,Mode
        from mes_vision.vlm.snapshots import save_snapshot
        from mes_vision.vlm.region_backend import region_plan
        from mes_vision.anomaly.features import fingerprint
        f=Acquisition(equipment()['camera']).apply(raw_frame())
        bounds=Box(100,100,300,300); local=Box(10,20,50,60)
        crop=Crop(f.frame_id,'object',bounds,bounds,f.rgb[100:300,100:300])
        model=ModelRef('test-only','1','model',training_scope='product_defects')
        finding=Finding(label='test-only',defect_code='NG05',score=.9,crop_box=local,original_box=local.translated(100,100))
        check=CheckResult('known_defects',f.frame_id,'object',CheckStatus.UNCERTAIN,model,(finding,))
        obj=ObjectResult('object',Detection(bounds,1.,0,'test-only'),bounds,crop.metadata(),[check])
        result=RunResult('test-only',f.metadata(),Mode.MODEL_FILE,model,{'product_id':'test-only'},[obj])
        with tempfile.TemporaryDirectory() as d:
            snapshot=Path(d)/'snapshot'
            manifest=save_snapshot(snapshot,f,result,kind='real')
            plan=region_plan({'snapshot_path':str(snapshot),'snapshot_digest':fingerprint(manifest),'object_id':'object'})
            self.assertEqual(plan['regions'][0]['box'],[0,10,60,70])

    def test_raw_camera_mode_and_new_roi_are_separate(self):
        original=default_equipment(); original['workspace'].update(roi=[[0,0],[1280,0],[1280,720],[0,720]],validation_reference='old')
        e=configure_equipment(ROOT,original)
        self.assertEqual(e['camera']['width'],1280); self.assertEqual(inspection_camera(e['camera'])['width'],720)
        self.assertEqual(original['workspace']['validation_reference'],'old')
        self.assertEqual(e['workspace']['validation_reference'],'')
        validate_equipment(e)
        self.assertTrue(in_workspace(Box(10,10,700,700),e['workspace']))
        self.assertFalse(in_workspace(Box(700,10,730,30),e['workspace']))
        e['workspace']['roi']=[[0,0],[721,0],[721,720],[0,720]]
        with self.assertRaises(ValueError): validate_equipment(e)

    def test_reconfigured_roi_is_retained_only_for_same_pipeline(self):
        e=equipment(); e['workspace']['roi']=[[10,10],[100,10],[100,100],[10,100]]
        e['workspace']['validation_reference']='new-measurements'
        self.assertEqual(configure_equipment(ROOT,e),e)
        e['workspace']['acquisition_identity']='old'
        self.assertEqual(configure_equipment(ROOT,e)['workspace']['validation_reference'],'')

    def test_wrong_resolution_file_or_driver_fails_closed(self):
        c=equipment()['camera']
        for field,value in [('calibration_sha256','0'*64),('sensor_size',[640,480])]:
            bad=deepcopy(c); bad['inspection_acquisition'][field]=value
            with self.assertRaises(ValueError): Acquisition(bad)
        bad=deepcopy(c); bad['driver']='d405'
        with self.assertRaises(ValueError): Acquisition(bad)

    def test_capture_gate_rejects_same_size_different_acquisition(self):
        e=equipment(); f=Acquisition(e['camera']).apply(raw_frame())
        p={'camera_serial':'TEST','camera_session':'raw','camera_driver':'uvc','image_size':[720,720],
           'acquisition_identity':identity(e['camera'])}
        gate=CaptureGate(request(now=100),p,.2,'test-only')
        self.assertIsNotNone(gate.observe(f,101.01,101.02))
        gate=CaptureGate(request(now=100),p,.2,'test-only')
        with self.assertRaises(ValueError): gate.observe(replace(f,acquisition_identity='other'),101.01,101.02)

    def test_existing_robot_map_is_rejected_without_removing_guards(self):
        with tempfile.TemporaryDirectory() as d:
            p,cal,_,e=setup(Path(d)); e['camera'].update(equipment()['camera'])
            with self.assertRaises(ValueError): cal.verify_context(p,e)

    def test_new_measured_domain_maps_without_offset_and_rejects_second_lens_model(self):
        from mes_vision.calibration.core import fit,Lens
        from mes_vision.calibration.fixtures import make_spec
        from mes_vision.station.calibration import build_bundle,MountedCalibration
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); p,old,_,e=setup(root)
            e['camera'].update(driver='uvc',width=1280,height=720,
                inspection_acquisition=equipment()['camera']['inspection_acquisition'])
            c=inspection_camera(e['camera']); spec=replace(make_spec(),kind='real',product_id=p['product_id'])
            spec=replace(spec,context=replace(spec.context,camera_id=c['serial'],mount_revision=c['mount_revision'],
                acquisition_revision=c['acquisition_revision'],image_size=(720,720)))
            args={k:v for k,v in old.value.items() if k not in {'kind','schema_version','capture_map','pick_map'}}
            for brown in (False,True):
                s=replace(spec,lens=Lens('opencv_brown5','UNIT-TEST',((1000.,0.,360.),(0.,1000.,360.),(0.,0.,1.)),(0.,)*5)) if brown else spec
                paths=[root/('new-'+str(brown)+'-'+role+'.json') for role in ('capture','pick')]
                for path in paths: fit(s).accept('UNIT-TEST-ONLY').save(path)
                cal=MountedCalibration(build_bundle(*paths,**args)); p.update(calibration_digest=cal.digest,image_size=[720,720])
                if brown:
                    with self.assertRaisesRegex(ValueError,'second lens correction'): cal.verify_context(p,e)
                else:
                    cal.verify_context(p,e)
                    req={'kind':'map_targets','payload':{'calibration_digest':cal.digest,'targets':[
                        {'id':'one','overview':{'box':{'x1':180,'y1':60,'x2':260,'y2':110}}}]}}
                    result=cal.map_targets(req,p,e)['targets'][0]
                    expected=cal.maps['capture_map'].map_xy((220,85),cal.context,plane_z_mm=cal.plane_z,kind='real')
                    self.assertAlmostEqual(result['capture_pose']['x'],expected[0])
                    self.assertAlmostEqual(result['capture_pose']['y'],expected[1])

    def test_no_processing_for_d405(self):
        e=default_equipment(); e['camera']['driver']='d405'
        self.assertEqual(configure_equipment(ROOT,e),e)
        raw=replace(raw_frame(),source_kind=SourceKind.D405)
        self.assertIs(Acquisition(e['camera']).apply(raw),raw)

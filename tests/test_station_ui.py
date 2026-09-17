"""Injected images/devices only; these tests are not physical acceptance."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
from copy import deepcopy
from dataclasses import asdict,replace
from types import SimpleNamespace
import tempfile,time,unittest
from unittest.mock import Mock,patch
from PySide6.QtWidgets import QApplication
from mes_vision.station.window import StationWindow
from mes_vision.station.worker import read_capture,StationProcess
from mes_vision.station.runtime import ScanRuntime
from mes_vision.station.process_camera import ProcessCameraPort
from mes_vision.station.settings import StationSettings
from mes_vision.station.sequence import ScanJournal
from mes_vision.station.evidence import save_capture,save_detail
from mes_vision.station.recipe import save_recipe,load_recipe
from mes_vision.operation.catalog import new_product,default_equipment
from mes_vision.qt_i18n import apply_language
from mes_vision.i18n import tr,language
from mes_vision.anomaly.features import fingerprint
from test_station_devices import setup,live
from test_station import profile,vision,workspace
from test_operation import InjectedModels,detection

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]


class InjectedStationModels:
    def __init__(self,*args,**kwargs):
        self.models=InjectedModels(); self.models.detections=(detection(x=180,y=30),)
    def load(self): pass
    def close(self): pass
    def service(self,**kwargs): return vision(self.models)


def wait(predicate,seconds=5):
    deadline=time.monotonic()+seconds
    while not predicate() and time.monotonic()<deadline: APP.processEvents(); time.sleep(.005)
    if not predicate(): raise AssertionError('Software work timeout')


class LocalEngine:
    """Same event contract, deterministic model output; no CUDA or process."""
    def __init__(self,root):
        self.root=root; self.service=InjectedStationModels().service(); self.pending=None; self.stopping=False; self.events=[]
        self.process=SimpleNamespace(is_alive=lambda:True); self.stop_at=None
    def submit(self,request,**extra):
        assert not self.stopping and self.pending is None
        self.pending=(request,extra)
    def poll(self):
        if not self.pending: return []
        r,extra=self.pending; self.pending=None
        if r['kind'].startswith('capture_'):
            payload=save_capture(self.root/'station-captures',extra['frame'],camera_serial=extra['camera_serial'],acquired_at=extra['timing']['acquired_at'])
            payload['timing']=extra['timing']
        else:
            frame=read_capture(r['payload']['capture'])
            if r['kind']=='detect_overview':
                payload={'frame_id':frame.frame_id,'model_digest':self.service.overview_detector.model.weights_sha256,'objects':self.service.locate(frame)}
            else:
                result=self.service.inspect_detail(frame,cycle_id=r['cycle_id'],target_id=r['target_id'],
                    overview_frame_id=r['payload']['overview_frame_id'],expected_label=r['payload']['expected_label'])
                payload=save_detail(self.root/'station-evidence',frame,result)
        return [{'type':'completed','request':r,'payload':payload}]
    def watchdog_error(self,now): return None
    def stop(self): self.stopping=True; self.stop_at=time.monotonic()
    def dispose(self): pass


class WindowTests(unittest.TestCase):
    def setUp(self):
        self.old=language(); self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.w=StationWindow(ROOT,self.root,start_worker=False); self.w.timer.stop(); self.w.analysis.timer.stop()
        wait(lambda:not self.w.tasks and not self.w.analysis.refresh_read.busy)
    def tearDown(self):
        self.w.scan=None; self.w.capture_port=None; self.w.camera=None; self.w.engine=None; self.w.robot=None; self.w.worker=None; self.w.running=False
        wait(lambda:not self.w.tasks and not self.w.analysis.refresh_read.busy and not self.w.vlm_read.busy)
        self.w.close(); self.w.close(); self.w.deleteLater(); APP.processEvents(); apply_language(APP,self.old); self.temp.cleanup()
    def install(self):
        p,cal,rp,e=setup(self.root)
        settings=StationSettings(self.root); value=deepcopy(settings.value)
        value.update(calibration_bundle=str(self.root/'bundle.json'),settle_seconds=.1,action_timeout_seconds=30,
            validation_reference='UNIT-TEST-ONLY',max_frame_age_seconds=.01,timing_validation_reference='UNIT-TEST-ONLY'); settings.save(value)
        w=self.w; w.equipment=default_equipment(); w.equipment.update(version=1); w.equipment['camera'].update(e['camera']); w.equipment['robot'].update(e['robot'])
        w.equipment['workspace'].update(workspace()); w.product=new_product(); w.product.update(id='test-part',version=1)
        w.engine=LocalEngine(self.root); w.engine_ready=True; w.model_info={'overview_model_digest':w.engine.service.overview_detector.model.weights_sha256,
            'detail_policy_digest':fingerprint(asdict(w.engine.service.policy))}
        w.robot=Mock(); w.robot_state={'state':'RECAPTURE','busy':False,'status':{'connection_epoch':'UNIT-TEST'}}; w.robot_seen_at=time.monotonic()
        w.camera=Mock(); w.camera_info={'serial':'TEST'}; w.frame=live(0,time.monotonic()); w.frame_received=time.monotonic()
        w.camera.get_latest.return_value=(w.frame,w.frame_received)
    def drive(self):
        w=self.w; w.begin(); counter=getattr(self,'sequence',1); last_move=None
        end=time.monotonic()+5; last_frame_at=0.
        while w.running and time.monotonic()<end:
            now=time.monotonic(); w.robot_seen_at=now
            if now-last_frame_at>=.034:
                w.camera.get_latest.return_value=(live(counter,now),now); counter+=1; last_frame_at=now
            pending=w.scan.sequence.data['pending']
            if pending and pending['kind'].startswith('move_') and pending['id']!=last_move:
                last_move=pending['id']; w.robot_completed({'request':pending,'payload':{'motion_complete':True,'actual_pose':pending['payload']['pose']}})
            w.tick(); APP.processEvents(); time.sleep(.005)
        self.sequence=counter
        self.assertEqual(w.scan.sequence.data['state'],'COMPLETED',str(w.scan.sequence.data.get('error')))
        return deepcopy(w.scan.sequence.data)
    def test_begin_drives_saved_overview_detail_verdict_and_click_selection(self):
        self.install(); data=self.drive(); w=self.w
        self.assertEqual(len(data['targets']),1); target=data['targets'][0]
        self.assertEqual(target['decision'],'OK'); self.assertNotEqual(data['overview']['frame_id'],target['detail']['frame_id'])
        w.select_track(target['id']); self.assertFalse(w.detail_image.canvas.pixmap.isNull()); self.assertTrue(w.live_details.toPlainText())
        self.assertEqual(w.canvas.tracks[0]['track_id'],target['id']); self.assertFalse(w.queue.list())
        # A new preview frame cannot replace the saved overview, or erase the result.
        key=w.canvas.pixmap.cacheKey(); w.camera.get_latest.return_value=(live(999,time.monotonic()),time.monotonic()); w.tick()
        self.assertEqual(w.canvas.pixmap.cacheKey(),key); self.assertEqual(w.display_data['targets'][0]['decision'],'OK')
    def test_multiple_objects_link_their_own_closeups_and_table_selection(self):
        self.install(); overview=InjectedModels(); overview.detections=(detection(x=40),detection(x=320))
        self.w.engine.service.overview_detector=overview; data=self.drive(); w=self.w
        self.assertEqual(len(data['targets']),2); first,second=data['targets']
        self.assertNotEqual(first['detail']['frame_id'],second['detail']['frame_id'])
        w.select_track(second['id']); self.assertEqual(w.object_table.currentRow(),1)
        self.assertEqual(w.detail_key[1],second['detail']['frame_id'])
        self.assertEqual(w.detail_image.canvas.tracks[0]['box'],[180,30,290,150])
        self.assertNotEqual(w.canvas.tracks[1]['box'],w.detail_image.canvas.tracks[0]['box'])
    def test_known_ng_reason_is_visible_with_vlm_off(self):
        self.install(); self.w.engine.service.detail_detector.defect=True; data=self.drive()
        self.assertEqual(data['targets'][0]['decision'],'NG'); self.assertIn('NG03',self.w.live_details.toPlainText())
        self.assertFalse(self.w.queue.list())
    def test_stop_revokes_motion_before_failed_persistence_and_keeps_camera(self):
        self.install(); w=self.w; w.begin(); before=w.camera
        with patch.object(w.scan.sequence,'stop',side_effect=OSError('disk full')): w.pause_inspection()
        self.assertFalse(w.running); self.assertFalse(w.scan.authorized); w.robot.stop_motion.assert_called(); self.assertIs(w.camera,before)
        self.assertTrue(w.engine.stopping)
    def test_late_completion_after_stop_cannot_publish_or_move(self):
        self.install(); w=self.w; w.begin(); old=deepcopy(w.scan.sequence.data['pending']); w.pause_inspection()
        calls=w.robot.request.call_count
        w.robot_completed({'request':old,'payload':{'motion_complete':True,'actual_pose':old['payload']['pose']}})
        self.assertEqual(w.robot.request.call_count,calls); self.assertEqual(w.scan.sequence.data['state'],'CANCELLED')
    def test_four_languages_preserve_current_selection_and_saved_images(self):
        self.install(); data=self.drive(); w=self.w; identity=w.selected_id; key=w.canvas.pixmap.cacheKey()
        for code in ('en','zh-CN','th','ko'):
            apply_language(APP,code); self.assertEqual(w.start.text(),str(tr('검사 시작')))
            self.assertEqual(w.selected_id,identity); self.assertEqual(w.canvas.pixmap.cacheKey(),key)
    def test_no_model_or_calibration_is_inferred_from_empty_installation(self):
        with self.assertRaises(ValueError): self.w.begin()
        self.assertIsNone(self.w.engine); self.assertIsNone(self.w.robot); self.assertIsNone(self.w.display_data)
    def test_vlm_off_has_basic_result_and_manual_on_links_same_detail(self):
        self.install(); data=self.drive(); w=self.w; original=deepcopy(data['targets'][0]['result'])
        w.queue.set_enabled(True); w.request_selected_vlm(); jobs=w.queue.list()
        self.assertEqual(len(jobs),1); self.assertEqual(jobs[0]['object_id'],original['object_id'])
        self.assertEqual(jobs[0]['snapshot_digest'],original['snapshot_digest'])
        self.assertEqual(w.scan_journal.read(data['id'])['targets'][0]['result'],original)
        claimed=w.queue.claim('UNIT-TEST'); w.queue.finish(claimed['id'],claimed['token'],'COMPLETED',result={'analysis':{'observation':'UNIT-TEST advisory only'}})
        w.refresh_selected_vlm(); self.assertIn('UNIT-TEST advisory only',w.vlm_details.toPlainText())
        self.assertEqual(w.display_data['targets'][0]['decision'],'OK')
    def test_backup_restores_station_images_without_restoring_motion_authority(self):
        from mes_vision.operation.maintenance import Maintenance
        self.install(); data=self.drive(); w=self.w
        with tempfile.TemporaryDirectory() as destination:
            destination=Path(destination); service=Maintenance(w.store)
            service.backup(destination/'backup'); result=service.restore(destination/'backup',destination/'restored')
            root=Path(result['runtime']); restored=ScanJournal(root).read(data['id'])
            self.assertFalse(restored['coordinate_valid']); self.assertIsNone(restored['pending']); self.assertTrue(restored['restored_history'])
            self.assertTrue(Path(restored['overview']['path']).is_relative_to(root)); read_capture(restored['overview'])
            self.assertEqual(restored['targets'][0]['decision'],'OK')
            self.assertIsNone(StationSettings(root).value['calibration_bundle'])
    def test_prior_cycle_selection_is_not_replaced_by_idle_tick(self):
        self.install(); first=self.drive(); w=self.w
        second=self.drive(); self.assertNotEqual(first['id'],second['id'])
        w.render_cycle(first,force=True); w.tick(); self.assertEqual(w.display_data['id'],first['id'])


class ProtocolTests(unittest.TestCase):
    def test_recipe_versions_conflicts_and_weights_are_verified(self):
        from mes_vision.training.data import sha256
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); p=new_product(); p.update(id='part',version=2,objects={'registered':True})
            e=default_equipment(); e['version']=3; weights=root/'weights.bin'; weights.write_bytes(b'UNIT-TEST')
            w,h=e['camera']['width'],e['camera']['height']
            recipe={'schema_version':1,'product_id':p['id'],'product_version':2,'equipment_version':3,'image_size':[w,h],
                'validation_reference':'UNIT-TEST','overview_asset':{'weights':str(weights),'sha256':sha256(weights),'class_names':['part']},
                'capture_domains':{'overview_objects':'overview','objects':'detail'},
                'detail_workspace':{'roi':[[0,0],[w,0],[w,h],[0,h]],'excluded':[],'validation_reference':'UNIT-TEST'}}
            save_recipe(root,recipe,p,e); self.assertEqual(load_recipe(root,p,e),recipe)
            with self.assertRaises(ValueError): save_recipe(root,recipe,p,e)
            e['version']=4
            with self.assertRaises(ValueError): load_recipe(root,p,e)
            e['version']=3; weights.write_bytes(b'changed')
            with self.assertRaises(ValueError): load_recipe(root,p,e)
    def test_tracking_settings_do_not_reappear_in_moving_camera_setup(self):
        from mes_vision.station.equipment_dialog import StationEquipmentDialog
        dialog=StationEquipmentDialog(default_equipment()); dialog.pages.advanced.setChecked(True)
        try:
            for i in range(dialog.pages.tabs.count()):
                if dialog.pages.tabs.tabText(i)==str(tr('추적 · 고급 설정')): self.assertFalse(dialog.pages.tabs.isTabVisible(i))
        finally: dialog.close(); dialog.deleteLater()
    def test_capture_writer_uses_process_and_stop_does_not_close_preview(self):
        camera=Mock(); engine=Mock(); engine.process.is_alive.return_value=True
        port=ProcessCameraPort(camera,engine,{'max_frame_age_seconds':.1,'timing_validation_reference':'UNIT-TEST'})
        req={'id':'x','kind':'capture_overview','issued_at':100,'deadline':130}; p=profile(InjectedModels())
        port.submit(req,p); camera.get_latest.return_value=(live(1,10),100); port.tick(100)
        camera.get_latest.return_value=(live(2,10.2),100.2); port.tick(100.2); self.assertTrue(port.busy)
        port.cancel(); engine.stop.assert_called_once(); camera.close.assert_not_called()
        engine.process.is_alive.return_value=False; self.assertFalse(port.busy)
    def test_capture_pixel_or_metadata_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            receipt=save_capture(root,live(1,10),camera_serial='TEST',acquired_at=100)
            self.assertEqual(read_capture(receipt).frame_id,receipt['frame_id'])
            changed=deepcopy(receipt); changed['frame_metadata']['source_uri']='realsense://OTHER'
            with self.assertRaises(ValueError): read_capture(changed)
            Path(receipt['path']).write_bytes(b'corrupt')
            with self.assertRaises(ValueError): read_capture(receipt)
    def test_resident_process_handles_saved_overview_and_detail_without_robot(self):
        with tempfile.TemporaryDirectory() as root:
            root=Path(root); product=new_product(); product.update(id='test-part',version=1)
            recipe={'overview_asset':{},'capture_domains':{},'detail_workspace':workspace()}
            process=StationProcess(ROOT,root,product,{'workspace':workspace()},recipe,model_factory=InjectedStationModels)
            events=[]
            def collect(kind):
                def got():
                    events.extend(process.poll()); return any(x['type']==kind for x in events)
                wait(got,30); return next(x for x in events if x['type']==kind)
            try:
                ready=collect('ready'); self.assertEqual(ready['overview_model_digest'],'d'*64)
                f=live(1,time.monotonic()); capture=save_capture(root/'captures',f,camera_serial='TEST',acquired_at=time.monotonic())
                req={'id':'locate','cycle_id':'cycle','generation':1,'target_id':None,'kind':'detect_overview','issued_at':time.monotonic(),
                    'deadline':time.monotonic()+30,'payload':{'capture':capture}}
                events=[]; process.submit(req); output=collect('completed'); self.assertEqual(output['payload']['objects'][0]['label'],'part')
                f=live(2,time.monotonic()); capture=save_capture(root/'captures',f,camera_serial='TEST',acquired_at=time.monotonic())
                req.update(id='inspect',kind='inspect_detail',target_id='target',payload={'capture':capture,'overview_frame_id':'injected-camera:1','expected_label':'part'})
                events=[]; process.submit(req); output=collect('completed'); self.assertIn('snapshot_path',output['payload'])
                self.assertTrue(Path(output['payload']['snapshot_path']).exists())
            finally: process.stop(); process.dispose()


if __name__=='__main__': unittest.main()

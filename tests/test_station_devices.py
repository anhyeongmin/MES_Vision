"""Injected software evidence only. No measured camera or robot parameters."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import json,tempfile,unittest,time
from concurrent.futures import Future
from unittest.mock import patch,Mock
from filelock import Timeout
from PySide6.QtWidgets import QApplication
from mes_vision.calibration.core import fit
from mes_vision.calibration.fixtures import make_spec
from mes_vision.robot import RobotController,Pose
from mes_vision.robot.contracts import RobotStatus,Completion
from mes_vision.robot.fixtures import make_fixture
from mes_vision.station.calibration import build_bundle,save_bundle,MountedCalibration
from mes_vision.station.settings import StationSettings
from mes_vision.station.camera_port import CaptureGate,StationCameraPort
from mes_vision.station.motion import CaptureMotion,capture_path
from mes_vision.station.owner import OwnedAdapter
from mes_vision.station.profile import build_scan_profile
from mes_vision.station.robot_port import authorize_move,StationRobotPort
from mes_vision.station import ScanJournal,ScanSequence
from mes_vision.station.settings_dialog import StationSettingsDialog,CalibrationBundleDialog
from mes_vision.qt_i18n import apply_language
from mes_vision.i18n import tr,language
from test_station import profile,vision
from test_operation import frame,InjectedModels

APP=QApplication.instance() or QApplication([])


def setup(root):
    spec=replace(make_spec(),kind='real',product_id='test-part')
    c=replace(spec.context,camera_id='TEST',mount_revision='test-mount'); spec=replace(spec,context=c)
    for name,dx in (('capture',0),('pick',20)):
        s=replace(spec,version='UNIT-TEST-'+name,pairs=tuple(replace(p,robot_xy_mm=(p.robot_xy_mm[0]+dx,p.robot_xy_mm[1])) for p in spec.pairs))
        calibration=fit(s).accept('UNIT-TEST-ONLY'); calibration.save(root/(name+'.json'))
    p=profile(InjectedModels())
    bundle=build_bundle(root/'capture.json',root/'pick.json',version='UNIT-TEST-ONLY',overview_pose=p['overview_pose'],
        detail_z=60,detail_r=0,pick_z=10,pick_r=0,travel_z=160,grasp_uv=[.5,.5],grasp_version='UNIT-TEST-GRASP',
        mount_link='UNIT-TEST-MOUNT',validation_reference='UNIT-TEST-ONLY')
    save_bundle(root/'bundle.json',bundle); calibration=MountedCalibration.load(root/'bundle.json'); p['calibration_digest']=calibration.digest
    rp=replace(make_fixture()['profile'],product_id='test-part',kind='real',calibration_version=calibration.identity,
               grasp_policy_version=bundle['grasp_version'],travel_z_mm=160)
    (root/'robot.json').write_text(json.dumps(asdict(rp)),encoding='utf-8')
    equipment={'version':1,'camera':{'driver':'d405','serial':c.camera_id,'mount_revision':c.mount_revision,'acquisition_revision':c.acquisition_revision,
        'robot_base_id':c.robot_base_id,'tool_frame_id':c.tool_frame_id,'width':480,'height':180},
        'robot':{'pose_tolerance_mm':1,'rotation_tolerance_deg':1,'profile':str(root/'robot.json')}}
    return p,calibration,rp,equipment


def request(kind='capture_detail',now=100,pose=None):
    return {'id':'request-1','cycle_id':'cycle-1','generation':1,'target_id':'target-1' if kind.endswith('detail') else None,
            'kind':kind,'issued_at':now,'deadline':now+30,'payload':{'pose':pose} if pose else {}}


def live(sequence,stamp,**kwargs):
    return replace(frame(sequence),source_uri='realsense://TEST',media_time_seconds=stamp,**kwargs)


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.p,self.cal,self.rp,self.e=setup(self.root)
    def tearDown(self): self.temp.cleanup()
    def mapping(self):
        return {'kind':'map_targets','payload':{'calibration_digest':self.cal.digest,
            'targets':[{'id':'one','overview':{'box':{'x1':180,'y1':60,'x2':260,'y2':110}}}]}}
    def test_distinct_capture_and_gripper_coordinates(self):
        result=self.cal.map_targets(self.mapping(),self.p,self.e)['targets'][0]
        self.assertAlmostEqual(result['pick_pose']['x']-result['capture_pose']['x'],20)
        self.assertEqual((result['capture_pose']['z'],result['pick_pose']['z']),(60,10))
    def test_changed_acquisition_overview_and_measured_file_rejected(self):
        e=deepcopy(self.e); e['camera']['acquisition_revision']='changed'
        with self.assertRaises(ValueError): self.cal.verify_context(self.p,e)
        p=deepcopy(self.p); p['overview_pose']['r']=1
        with self.assertRaises(ValueError): self.cal.verify_context(p,self.e)
        with (self.root/'capture.json').open('a') as stream: stream.write(' ')
        with self.assertRaises(ValueError): self.cal.verify_context(self.p,self.e)
    def test_outside_measured_region_is_not_extrapolated(self):
        req=self.mapping(); req['payload']['targets'][0]['overview']['box']={'x1':0,'y1':0,'x2':10,'y2':10}
        with self.assertRaises(ValueError): self.cal.map_targets(req,self.p,self.e)
    def test_unaccepted_and_synthetic_maps_rejected(self):
        synthetic=fit(make_spec()).accept('UNIT-TEST-ONLY'); synthetic.save(self.root/'synthetic.json')
        value=deepcopy(self.cal.value)
        from mes_vision.training.data import sha256
        value['capture_map']={'path':str(self.root/'synthetic.json'),'sha256':sha256(self.root/'synthetic.json')}
        with self.assertRaises(ValueError): MountedCalibration(value)
    def test_incomplete_settings_stay_unconfigured_and_conflicts_do_not_overwrite(self):
        first=StationSettings(self.root); second=StationSettings(self.root)
        first.save(first.value)
        self.assertIsNone(first.value['settle_seconds'])
        with self.assertRaises(ValueError): first.calibrated()
        with self.assertRaises(ValueError): second.save(second.value)
        self.assertEqual(StationSettings(self.root).value['version'],1)
    def ready_settings(self):
        s=StationSettings(self.root); v=deepcopy(s.value)
        v.update(calibration_bundle=str(self.root/'bundle.json'),settle_seconds=.5,action_timeout_seconds=30,
                 validation_reference='UNIT-TEST-ONLY',max_frame_age_seconds=.1,timing_validation_reference='UNIT-TEST-ONLY')
        s.save(v); return s
    def test_profile_freezes_registered_versions_and_geometry(self):
        s=self.ready_settings()
        p=build_scan_profile(s,self.e,product_id='test-part',product_version=1,overview_model_digest='a'*64,
            detail_policy_digest='b'*64,frame=live(1,10),max_objects=10,expected_count=None,auto_sort=False,robot_connection_epoch='UNIT-TEST')
        self.assertEqual(p['station_settings_version'],1); self.assertEqual(p['overview_pose'],self.cal.value['overview_pose'])
        self.assertEqual(p['calibration_digest'],self.cal.digest)
        self.assertEqual(p['motion_bounds']['z'],[0,200])
    def test_registered_bundle_tampering_rejected(self):
        s=self.ready_settings()
        with (self.root/'bundle.json').open('a') as f: f.write(' ')
        with self.assertRaises(ValueError): s.calibrated()
    def test_settings_atomic_write_failure_preserves_original(self):
        s=self.ready_settings(); old=s.path.read_bytes()
        with patch('mes_vision.station.settings.os.fsync',side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): s.save(s.value)
        self.assertEqual(s.path.read_bytes(),old); self.assertEqual(list(self.root.glob('.station-settings-*.tmp')),[])
    def test_homography_changes_fail_integrity_not_just_version(self):
        value=deepcopy(self.cal.value); value['capture_map']['sha256']='0'*64
        with self.assertRaises(ValueError): MountedCalibration(value)


class CaptureTests(unittest.TestCase):
    def setUp(self): self.p=profile(InjectedModels()); self.req=request()
    def gate(self): return CaptureGate(self.req,self.p,.1,'UNIT-TEST-LATENCY-ONLY')
    def test_pre_request_frame_and_anchor_are_not_saved(self):
        gate=self.gate()
        self.assertIsNone(gate.observe(live(1,10),99.9,100))
        self.assertIsNone(gate.observe(live(2,10.1),100.01,100.01))
        self.assertIsNone(gate.observe(live(3,10.15),100.06,100.06))
        proof=gate.observe(live(4,10.3),100.21,100.21)
        self.assertGreaterEqual(proof['acquired_at'],100); self.assertLess(proof['acquired_at'],100.21)
        self.assertIn('validated',proof['timing_basis'])
    def test_device_changes_geometry_and_missing_timestamp_rejected(self):
        cases=[replace(live(1,10),session_id='reconnected'),replace(live(1,10),source_uri='realsense://OTHER'),
            replace(live(1,10),media_time_seconds=None),replace(live(1,10),transformations=('crop',))]
        for f in cases:
            with self.subTest(frame=f.frame_id),self.assertRaises(ValueError): self.gate().observe(f,100,100)
    def test_clock_jump_backward_sequence_and_deadline_rejected(self):
        for f,received,now in [(live(2,1000),100.1,100.1),(live(0,10.1),100.1,100.1),(live(2,10.2),100.2,131)]:
            gate=self.gate(); gate.observe(live(1,10),100,100)
            with self.assertRaises(ValueError): gate.observe(f,received,now)
    def test_stale_host_frame_cannot_become_post_move_image(self):
        gate=self.gate(); gate.observe(live(1,10),100,100)
        with self.assertRaises(ValueError): gate.observe(live(2,10.2),100.2,101)
    def test_unvalidated_or_unbounded_timing_rejected(self):
        for bound,reference in [(0,'x'),(3,'x'),(.1,''),(float('nan'),'x')]:
            with self.assertRaises(ValueError): CaptureGate(self.req,self.p,bound,reference)
    def test_cancel_does_not_queue_unbounded_writes_or_disconnect_preview(self):
        camera=Mock(); port=StationCameraPort(camera,'.',{'max_frame_age_seconds':.1,'timing_validation_reference':'UNIT-TEST'})
        future=Future(); future.set_running_or_notify_cancel(); port.saving=(self.req,future)
        port.cancel(); self.assertTrue(port.busy)
        with self.assertRaises(ValueError): port.submit(self.req,self.p)
        future.set_result({'late':True}); self.assertIsNone(port.tick(101)); self.assertFalse(port.busy)
        port.close(); camera.close.assert_not_called(); camera.stopping.set.assert_not_called()
    def test_one_shot_capture_persists_real_frame_and_receipt(self):
        with tempfile.TemporaryDirectory() as root:
            camera=Mock(); camera.get_latest.return_value=(live(1,10),100)
            port=StationCameraPort(camera,root,{'max_frame_age_seconds':.1,'timing_validation_reference':'UNIT-TEST'})
            port.submit(self.req,self.p); self.assertIsNone(port.tick(100))
            camera.get_latest.return_value=(live(2,10.2),100.2); port.tick(100.2)
            port.saving[1].result(timeout=5); event=port.tick(100.3)
            self.assertTrue(Path(event['payload']['path']).is_file()); self.assertEqual(event['request'],self.req)
            self.assertIn('timing',event['payload']); self.assertFalse(port.busy); port.close()


class InjectedRobot:
    kind='real'
    def __init__(self):
        self.pose=Pose(0,0,100,0); self.epoch='UNIT-TEST'; self.state='READY'; self.holding=False
        self.commands=[]; self.stop_count=0; self.wrong_pose=False; self.pending=None
    def status(self): return RobotStatus(self.epoch,True,self.state,self.holding,self.pose)
    def submit(self,command): self.commands.append(command); self.pending=command; return True
    def poll(self,identity):
        self.pose=replace(self.pending.target,x=999) if self.wrong_pose else self.pending.target
        return Completion(identity,'DONE',pose=self.pending.target)
    def request_stop(self): self.stop_count+=1; self.state='STOPPED'; return True
    def close(self): self.request_stop()
    def configure(self,profile): self.configured=profile
    def ready_after_recovery(self): self.state='READY'; self.epoch+='-RECOVERED'


class MotionTests(unittest.TestCase):
    setUp=CalibrationTests.setUp
    tearDown=CalibrationTests.tearDown
    def test_overview_capture_map_and_detail_move_use_same_saved_target(self):
        robot=InjectedRobot(); journal=ScanJournal(self.root); sequence=ScanSequence(journal)
        req=sequence.start(self.p,100)
        def complete(req,payload,now): return sequence.complete(req['id'],req['cycle_id'],req['generation'],payload,now)
        with RobotController(self.root/'robot-owner',robot) as c:
            motion=CaptureMotion(c); motion.start(req,self.p,self.cal,self.rp,self.e,now=100)
            for i in range(6): event=motion.tick(now=100+i*.1)
            self.assertIsNone(complete(req,event['payload'],100.6)); capture_request=sequence.tick(101.2)
            camera=Mock(); camera.get_latest.return_value=(live(1,10),101.2)
            port=StationCameraPort(camera,self.root/'images',{'max_frame_age_seconds':.1,'timing_validation_reference':'UNIT-TEST'})
            try:
                port.submit(capture_request,self.p); port.tick(101.2)
                image=live(2,10.2); camera.get_latest.return_value=(image,101.4); port.tick(101.4)
                port.saving[1].result(timeout=5); event=port.tick(101.5)
                detect=complete(capture_request,event['payload'],101.5)
                mapping=complete(detect,{'frame_id':image.frame_id,'model_digest':self.p['overview_model_digest'],
                    'objects':vision(InjectedModels()).locate(image)},101.6)
                detail=complete(mapping,self.cal.map_targets(mapping,self.p,self.e),101.7)
                self.assertEqual(detail['kind'],'move_detail')
                authorize_move(journal,detail,self.cal,self.e)
                motion.start(detail,self.p,self.cal,self.rp,self.e,now=101.7)
                for i in range(6): result=motion.tick(now=101.7+i*.1)
                self.assertEqual(result['payload']['actual_pose'],detail['payload']['pose'])
                complete(detail,result['payload'],102.3)
                self.assertEqual(sequence.tick(102.9)['kind'],'capture_detail')
                self.assertEqual(len(robot.commands),6)
            finally: port.close()
    def test_thread_connect_requires_explicit_readiness_then_completes_journal_move(self):
        settings=CalibrationTests.ready_settings(self)
        robot=InjectedRobot(); robot.state='STOPPED'; worker=StationRobotPort(self.root,self.e,adapter_factory=lambda _:robot)
        statuses=[]; events=[]; errors=[]
        worker.changed.connect(statuses.append); worker.completed.connect(events.append); worker.failed.connect(errors.append)
        def wait(predicate):
            deadline=time.monotonic()+5
            while not predicate() and time.monotonic()<deadline: APP.processEvents(); time.sleep(.005)
            self.assertTrue(predicate(),str(errors))
        try:
            worker.start(); wait(lambda:any(x['state']=='RECOVERY' for x in statuses))
            self.assertFalse(robot.commands)
            worker.request('recover',reference='UNIT-TEST-ONLY'); wait(lambda:any(x['state']=='RECAPTURE' for x in statuses))
            p=build_scan_profile(settings,self.e,product_id='test-part',product_version=1,overview_model_digest='a'*64,
                detail_policy_digest='b'*64,frame=live(1,10),max_objects=10,expected_count=None,auto_sort=False,robot_connection_epoch=robot.epoch)
            seq=ScanSequence(ScanJournal(self.root)); req=seq.start(p,time.monotonic())
            worker.request('move',request=req); wait(lambda:bool(events))
            self.assertFalse(errors); self.assertTrue(events[0]['payload']['motion_complete'])
            self.assertEqual(events[0]['request'],req); self.assertEqual(len(robot.commands),3)
            robot.epoch+='-CHANGED'; worker.request('move',request=req); wait(lambda:len(events)==2)
            self.assertIn('reconnected',events[1]['error']); self.assertEqual(len(robot.commands),3)
        finally:
            worker.stop_motion(); worker.stopping.set(); self.assertTrue(worker.wait(5000)); APP.processEvents()
    def test_lift_travel_descend_uses_verified_actual_position(self):
        robot=InjectedRobot(); req=request('move_overview',pose=self.p['overview_pose'])
        with RobotController(self.root/'robot-owner',robot) as c:
            motion=CaptureMotion(c); motion.start(req,self.p,self.cal,self.rp,self.e,now=100)
            for i in range(6): event=motion.tick(now=100+i*.1)
            self.assertEqual([x.stage for x in robot.commands],['CAPTURE_LIFT','CAPTURE_TRAVEL','CAPTURE_DESCEND'])
            self.assertEqual([x.target.z for x in robot.commands],[160,160,150])
            self.assertTrue(event['payload']['motion_complete']); self.assertEqual(event['payload']['actual_pose'],self.p['overview_pose'])
            with self.assertRaises(Exception): motion.start(req,self.p,self.cal,self.rp,self.e,now=101)
            self.assertEqual(len(robot.commands),3)
    def test_done_ack_without_actual_arrival_faults_and_stops(self):
        robot=InjectedRobot(); robot.wrong_pose=True
        with RobotController(self.root/'robot-owner',robot) as c:
            motion=CaptureMotion(c); motion.start(request('move_overview',pose=self.p['overview_pose']),self.p,self.cal,self.rp,self.e,now=100)
            motion.tick(now=100); event=motion.tick(now=100.1)
            self.assertIn('error',event); self.assertEqual(c.state,'FAULT'); self.assertEqual(robot.stop_count,1)
            self.assertEqual(len(robot.commands),1); self.assertIsNone(motion.tick(now=100.2))
    def test_timeout_and_stop_never_advance_pending_command(self):
        robot=InjectedRobot()
        with RobotController(self.root/'robot-owner',robot) as c:
            motion=CaptureMotion(c); motion.start(request('move_overview',pose=self.p['overview_pose']),self.p,self.cal,self.rp,self.e,now=100)
            motion.tick(now=100); event=motion.tick(now=102)
            self.assertIn('error',event); self.assertEqual(len(robot.commands),1)
    def test_unready_holding_and_unvalidated_workspace_block_before_movement(self):
        for state,holding in [('STOPPED',False),('READY',True),('READY',None)]:
            robot=InjectedRobot(); robot.state=state; robot.holding=holding
            with RobotController(self.root/('owner-'+state+str(holding)),robot) as c:
                with self.assertRaises(ValueError): CaptureMotion(c).start(request('move_overview',pose=self.p['overview_pose']),self.p,self.cal,self.rp,self.e,now=100)
                self.assertFalse(robot.commands)
        with self.assertRaises(ValueError): capture_path(Pose(0,0,190,0),self.p['overview_pose'],self.p,self.cal,self.rp,self.e)
    def test_journal_failure_stops_before_dispatch(self):
        robot=InjectedRobot()
        with RobotController(self.root/'robot-owner',robot) as c:
            motion=CaptureMotion(c); motion.start(request('move_overview',pose=self.p['overview_pose']),self.p,self.cal,self.rp,self.e,now=100)
            with patch.object(c.journal,'record',side_effect=OSError('disk error')): event=motion.tick(now=100)
            self.assertIn('error',event); self.assertFalse(robot.commands); self.assertGreaterEqual(robot.stop_count,1)
    def test_exclusive_owner_prevents_even_opening_second_serial_adapter(self):
        factory=Mock(return_value=InjectedRobot())
        with OwnedAdapter(self.root/'shared',factory):
            second=Mock(return_value=InjectedRobot())
            with self.assertRaises(Timeout):
                with OwnedAdapter(self.root/'shared',second): pass
            second.assert_not_called()
        with OwnedAdapter(self.root/'shared',lambda:InjectedRobot()): pass
    def test_worker_stop_discards_queued_request_before_execution(self):
        worker=StationRobotPort(self.root,self.e,adapter_factory=Mock())
        worker.request('move',request=request('move_overview',pose=self.p['overview_pose']))
        worker.stop_motion(); self.assertTrue(worker.commands.empty())
        with self.assertRaises(ValueError): worker.request('recover',reference='UNIT-TEST')
        worker.factory.assert_not_called()
    def test_only_journal_issued_request_is_authorized(self):
        journal=ScanJournal(self.root); sequence=ScanSequence(journal); req=sequence.start(self.p,100)
        self.assertEqual(authorize_move(journal,req,self.cal,self.e),self.p)
        changed=deepcopy(req); changed['payload']['pose']['x']+=1
        with self.assertRaises(ValueError): authorize_move(journal,changed,self.cal,self.e)
        sequence.stop()
        with self.assertRaises(ValueError): authorize_move(journal,req,self.cal,self.e)


class DialogTests(unittest.TestCase):
    def test_draft_fields_remain_unregistered_and_live_languages_preserve_input(self):
        old=language()
        with tempfile.TemporaryDirectory() as root:
            dialog=StationSettingsDialog(root); dialog.reference.setText('USER-REFERENCE-정상')
            try:
                for code in ('ko','en','zh-CN','th'):
                    apply_language(APP,code)
                    self.assertEqual(dialog.windowTitle(),str(tr('로봇 촬영 설정')))
                    self.assertEqual(dialog.reference.text(),'USER-REFERENCE-정상')
                    self.assertIsNone(dialog.collect()['settle_seconds'])
                dialog.save(); saved=StationSettings(root)
                self.assertEqual(saved.value['validation_reference'],'USER-REFERENCE-정상')
                self.assertIsNone(saved.value['calibration_bundle'])
            finally: dialog.close(); dialog.deleteLater(); apply_language(APP,old)
    def test_cancel_does_not_persist_and_incomplete_bundle_has_no_fake_pose(self):
        with tempfile.TemporaryDirectory() as root:
            dialog=StationSettingsDialog(root); dialog.settle.setValue(1); dialog.reject()
            self.assertFalse((Path(root)/'station-settings.json').exists()); dialog.deleteLater()
            bundle=CalibrationBundleDialog(root)
            with self.assertRaises(ValueError): bundle.collect()
            self.assertIsNone(bundle.path); bundle.reject(); bundle.deleteLater()


if __name__=='__main__': unittest.main()

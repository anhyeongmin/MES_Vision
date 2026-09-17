"""Software-only sorting: injected frames, AI outputs and independent tool inputs."""
from copy import deepcopy
from dataclasses import replace,asdict
from pathlib import Path
import json,time,unittest,threading,tempfile
from unittest.mock import patch
from PySide6.QtCore import QTimer
from mes_vision.robot import RobotController,load_profile
from mes_vision.robot.contracts import Completion,Pose
from mes_vision.station.sorting import SortMotion,authorize_sort,verify_sort_receipt
from mes_vision.station.robot_port import StationRobotPort
from mes_vision.station.sequence import ScanSequence
from mes_vision.station.settings import StationSettings
from mes_vision.training.data import sha256
from mes_vision.i18n import tr
from mes_vision.qt_i18n import apply_language
import test_station_ui as ui_helpers
from test_station_devices import InjectedRobot
from test_operation import InjectedModels,detection


def write_test_profile(root,equipment,profile):
    # All fixtures live under TemporaryDirectory; never write through an arbitrary
    # equipment path or any deployed runtime configuration.
    root=Path(root).resolve(); temporary=Path(tempfile.gettempdir()).resolve()
    if root==temporary or not root.is_relative_to(temporary): raise AssertionError('Temporary test directory required')
    path=(root/'robot.json').resolve()
    if not path.is_relative_to(root) or Path(equipment['robot']['profile']).resolve()!=path:
        raise AssertionError('Robot profile must be the temporary fixture file')
    path.write_text(json.dumps(asdict(profile)),encoding='utf-8')
    return path


class ToolRobot(InjectedRobot):
    def __init__(self): super().__init__(); self.fail_verify=False; self.lost=False
    def poll(self,identity):
        c=self.pending
        if c.action=='move': self.pose=c.target
        if c.action=='engage': self.holding=True
        if c.action=='release': self.holding=False
        verified=not self.fail_verify if c.action.startswith('verify_') else None
        return Completion(identity,'DONE',pose=c.target,verified=verified)


class SortTests(unittest.TestCase):
    def setUp(self):
        self.ui=ui_helpers.WindowTests(); self.ui.setUp(); self.ui.install(); self.w=self.ui.w
        self.data=self.ui.drive(); self.root=self.ui.root; self.journal=self.w.scan_journal
        self.settings=StationSettings(self.root); self.cal=self.settings.calibrated()
        self.rp=replace(load_profile(Path(self.w.equipment['robot']['profile'])),frame_max_age_seconds=60)
        self.robot=ToolRobot(); self.robot.pose=Pose(**self.data['targets'][0]['capture_pose'])
        self.controller=RobotController(self.root/'robot',self.robot); self.sorter=SortMotion(self.controller,self.journal)
        seq=ScanSequence(self.journal); seq.data=deepcopy(self.data); seq.data['profile']['auto_sort']=True
        self.req=seq._persist('UNIT_TEST_SORT',lambda:seq._next(time.monotonic())); self.seq=seq
    def tearDown(self):
        self.sorter.active=None; self.controller.close(); self.ui.tearDown()
    def start(self): self.sorter.start(self.req,self.cal,self.w.equipment,self.rp,now=time.monotonic())
    def finish(self):
        event=None
        for _ in range(30):
            event=self.sorter.tick(now=time.monotonic())
            if event: break
        self.assertIsNotNone(event); return event
    def test_full_sort_lifts_first_verifies_tool_and_persists_completion(self):
        self.start(); event=self.finish(); self.assertNotIn('error',event)
        self.assertEqual([c.action for c in self.robot.commands],['move','move','move','engage','verify_pick','move','move','move','release','verify_place','move'])
        lift=self.robot.commands[0].target; self.assertEqual((lift.x,lift.y,lift.z),(self.data['targets'][0]['capture_pose']['x'],self.data['targets'][0]['capture_pose']['y'],160))
        verify_sort_receipt(self.journal,self.req,event['payload'])
        self.seq.complete(self.req['id'],self.req['cycle_id'],self.req['generation'],event['payload'],time.monotonic())
        self.assertEqual(self.seq.data['targets'][0]['status'],'SORTED'); self.assertEqual(self.seq.data['state'],'COMPLETED')
        self.assertEqual(self.robot.pose,self.rp.destinations['OK'].__class__(150,50,160,0))
        with self.assertRaises(Exception): self.start()
    def test_changed_pick_or_unfinished_inspection_never_moves(self):
        for change in ('pose','inspection','review'):
            with self.subTest(change=change):
                data=self.journal.read(self.req['cycle_id']); before=deepcopy(data)
                if change=='pose': data['targets'][0]['pick_pose']['x']+=1
                elif change=='inspection': data['targets'].append({'id':'uninspected','status':'WAITING'})
                else: data['targets'][0]['decision']='REVIEW'
                data['revision']+=1; self.journal.save(data,before['revision'],'UNIT_TEST_MUTATION')
                with self.assertRaises(ValueError): self.start()
                before['revision']=data['revision']+1; self.journal.save(before,data['revision'],'UNIT_TEST_RESET')
        self.assertFalse(self.robot.commands)
    def test_different_detail_receipt_and_changed_original_are_rejected(self):
        target=self.data['targets'][0]; path=Path(target['detail']['path']); old=path.read_bytes(); path.write_bytes(b'corrupt')
        with self.assertRaises(Exception): self.start()
        path.write_bytes(old)
        path=Path(target['result']['snapshot_path'])/'frame.png'; path.write_bytes(b'corrupt')
        with self.assertRaises(Exception): self.start()
        self.assertFalse(self.robot.commands)
    def test_expired_overview_is_not_refreshed_by_detailed_image(self):
        expired=replace(self.rp,frame_max_age_seconds=.00001)
        with self.assertRaisesRegex(ValueError,'expired'):
            self.sorter.start(self.req,self.cal,self.w.equipment,expired,now=time.monotonic())
        self.assertFalse(self.robot.commands)
    def test_false_verification_does_not_place_or_report_success(self):
        self.robot.fail_verify=True; self.start(); event=self.finish()
        self.assertIn('error',event); self.assertGreater(self.robot.stop_count,0)
        self.assertNotIn('release',[c.action for c in self.robot.commands])
        with self.controller.journal.db() as db: self.assertIsNone(db.execute('SELECT receipt FROM station_sorts').fetchone()[0])
    def test_object_loss_during_transport_stops_before_destination(self):
        self.start()
        while self.sorter.active['index']<5: self.sorter.tick(now=time.monotonic())
        self.robot.holding=False; event=self.sorter.tick(now=time.monotonic())
        self.assertIn('lost',event['error']); self.assertNotIn('MOVE_PLACE',[c.stage for c in self.robot.commands])
    def test_cancelled_journal_prevents_next_command(self):
        self.start(); self.sorter.tick(now=time.monotonic()); count=len(self.robot.commands); self.seq.stop()
        event=self.sorter.tick(now=time.monotonic()); self.assertIn('error',event); self.assertEqual(len(self.robot.commands),count)
        self.assertEqual(self.seq.data['targets'][0]['status'],'SORT_UNCONFIRMED')
    def test_restart_keeps_verdict_but_does_not_claim_placement(self):
        self.journal.recover_interrupted(); data=self.journal.read(self.req['cycle_id'])
        self.assertEqual(data['targets'][0]['decision'],'OK'); self.assertEqual(data['targets'][0]['status'],'SORT_UNCONFIRMED')
        self.assertFalse(data['coordinate_valid'])
    def test_storage_failure_stops_and_never_accepts_boolean_only_receipt(self):
        self.start()
        with patch.object(self.controller.journal,'record',side_effect=OSError('disk full')):
            event=self.sorter.tick(now=time.monotonic())
        self.assertIn('error',event); self.assertGreater(self.robot.stop_count,0); self.assertFalse(self.robot.commands)
        with self.assertRaises(ValueError): verify_sort_receipt(self.journal,self.req,{'object_id':self.req['target_id'],'placement_confirmed':True})
    def test_duplicate_reservation_is_rejected_after_ambiguous_stop(self):
        self.start(); self.sorter.stop(); self.controller.state='RECAPTURE'; self.robot.state='READY'
        with self.assertRaises(Exception): self.start()
        self.assertFalse(self.robot.commands)
    def test_stop_arriving_during_intent_write_prevents_submit(self):
        self.start(); allowed=[True]; self.sorter.is_authorized=lambda:allowed[0]
        original=self.controller.journal.record
        def record(*args,**kwargs):
            result=original(*args,**kwargs); allowed[0]=False; return result
        with patch.object(self.controller.journal,'record',side_effect=record): event=self.sorter.tick(now=time.monotonic())
        self.assertIn('error',event); self.assertFalse(self.robot.commands)
    def test_reconnect_fails_without_success(self):
        self.start(); self.sorter.tick(now=time.monotonic()); self.robot.epoch+='changed'
        event=self.sorter.tick(now=time.monotonic()); self.assertIn('error',event)
        self.assertEqual(len(self.robot.commands),1)
    def test_measured_pose_mismatch_is_not_accepted(self):
        self.start(); self.sorter.tick(now=time.monotonic())
        command=self.robot.pending
        def wrong(_):
            self.robot.pose=replace(command.target,x=999)
            return Completion(command.command_id,'DONE',pose=command.target)
        with patch.object(self.robot,'poll',side_effect=wrong): event=self.sorter.tick(now=time.monotonic())
        self.assertIn('error',event); self.assertEqual(len(self.robot.commands),1)
    def test_command_timeout_never_retries(self):
        self.start(); self.sorter.tick(now=time.monotonic()); deadline=self.sorter.active['deadline']
        event=self.sorter.tick(now=deadline); self.assertIn('error',event); self.assertEqual(len(self.robot.commands),1)
    def run_device_sort(self,*,block_validation=False):
        self.controller.close(); self.robot.state='STOPPED'
        profile_path=write_test_profile(self.root,self.w.equipment,self.rp)
        worker=StationRobotPort(self.root,self.w.equipment,adapter_factory=lambda _:self.robot)
        states=[]; events=[]; errors=[]; entered=threading.Event(); release=threading.Event()
        worker.changed.connect(states.append); worker.completed.connect(events.append); worker.failed.connect(errors.append)
        def delayed(*args,**kwargs):
            entered.set(); release.wait(4); return authorize_sort(*args,**kwargs)
        context=patch('mes_vision.station.sorting.authorize_sort',side_effect=delayed) if block_validation else patch('time.sleep',wraps=time.sleep)
        try:
            worker.start(); ui_helpers.wait(lambda:any(s['state']=='RECOVERY' for s in states))
            worker.request('recover',reference='UNIT-TEST-ONLY'); ui_helpers.wait(lambda:any(s['state']=='RECAPTURE' for s in states))
            data=self.journal.read(self.req['cycle_id']); previous=data['revision']; data['revision']+=1
            data['profile'].update(robot_connection_epoch=self.robot.epoch,robot_profile_sha256=sha256(profile_path))
            self.journal.save(data,previous,'UNIT_TEST_RECOVERY')
            with context:
                worker.request('sort',request=self.req)
                if block_validation:
                    ui_helpers.wait(entered.is_set); before=self.robot.stop_count; worker.stop_motion()
                    ui_helpers.wait(lambda:self.robot.stop_count>before,seconds=1)
                    self.assertFalse(release.is_set()); self.assertFalse(self.robot.commands); release.set()
                else:
                    ui_helpers.wait(lambda:bool(events)); self.assertFalse(errors,str(errors)); self.assertNotIn('error',events[0])
                    verify_sort_receipt(self.journal,self.req,events[0]['payload'])
                    self.assertEqual(len(self.robot.commands),11)
        finally:
            release.set(); worker.stop_motion(); worker.stopping.set(); self.assertTrue(worker.wait(5000)); ui_helpers.APP.processEvents()
    def test_device_thread_completes_validated_sort(self): self.run_device_sort()
    def test_device_stop_is_serviced_while_evidence_read_is_blocked(self): self.run_device_sort(block_validation=True)


class UiSortTests(unittest.TestCase):
    def run_ui_sort(self,*,mixed=False,capture=None):
        ui=ui_helpers.WindowTests(); ui.setUp()
        try:
            ui.install(); w=ui.w; w.sort_enabled.setChecked(True)
            if mixed:
                overview=InjectedModels(); overview.detections=tuple(detection(x=x,y=30) for x in (40,180,320))
                w.engine.service.overview_detector=overview
                original=w.engine.service.inspect_detail
                def inspect(frame,**kwargs):
                    identity=kwargs['target_id']
                    w.engine.service.detail_detector.defect=identity.endswith('0002')
                    w.engine.service.detail_detector.anomaly_score=.5 if identity.endswith('0003') else .1
                    return original(frame,**kwargs)
                w.engine.service.inspect_detail=inspect
            rp=replace(load_profile(Path(w.equipment['robot']['profile'])),frame_max_age_seconds=60)
            write_test_profile(ui.root,w.equipment,rp)
            robot=ToolRobot(); c=RobotController(ui.root/'robot',robot); sorter=SortMotion(c,w.scan_journal)
            def dispatch(kind,**kwargs):
                if kind!='sort': return
                request=kwargs['request']; cycle=w.scan_journal.read(request['cycle_id'])
                self.assertTrue(all(t['detail'] and t['result'] for t in cycle['targets']))
                if not robot.commands: robot.pose=Pose(**cycle['targets'][-1]['capture_pose'])
                sorter.start(request,StationSettings(ui.root).calibrated(),w.equipment,rp,now=time.monotonic())
                event=None
                for _ in range(30):
                    event=sorter.tick(now=time.monotonic())
                    if event: break
                self.assertIsNotNone(event); self.assertNotIn('error',event)
                QTimer.singleShot(0,lambda:w.robot_completed(event))
            w.robot.request.side_effect=dispatch
            try:
                data=ui.drive(); self.assertEqual(data['targets'][0]['status'],'SORTED'); self.assertFalse(w.queue.list())
                self.assertEqual(w.object_table.item(0,2).text(),'이송 완료')
                if mixed:
                    self.assertEqual([t['status'] for t in data['targets']],['SORTED','SORTED','REVIEW'])
                    self.assertEqual([t['decision'] for t in data['targets']],['OK','NG','REVIEW'])
                    self.assertEqual(len(robot.commands),22)
                    self.assertEqual(robot.commands[7].target,rp.destinations['OK'])
                    self.assertEqual(robot.commands[18].target,rp.destinations['NG'])
                for code in ('en','zh-CN','th','ko'):
                    apply_language(ui_helpers.APP,code)
                    self.assertEqual(w.sort_enabled.text(),str(tr('검사 후 자동 분류')))
                    self.assertEqual(w.object_table.item(0,2).text(),str(tr('이송 완료')))
                    self.assertTrue(w.sort_enabled.isChecked())
                if capture: capture(w)
            finally: c.close()
        finally: ui.tearDown()
    def test_checkbox_dispatches_sort_after_detail_and_keeps_vlm_off(self): self.run_ui_sort()
    def test_three_objects_are_all_inspected_before_ok_ng_sort_review_stays(self): self.run_ui_sort(mixed=True)

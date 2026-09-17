"""Injected device tests only. Never opens a physical serial port."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import math
import struct
import time
import unittest

from PySide6.QtWidgets import QApplication
from filelock import FileLock
from mes_vision.operation.catalog import default_equipment
from mes_vision.robot.contracts import Pose,JointPose,RobotStatus,Completion
from mes_vision.robot.magician import Magician
from mes_vision.station.robot_port import StationRobotPort
from mes_vision.station.teaching import TeachingStore,stable_pose
from mes_vision.station.teaching_dialog import TeachingDialog

APP=QApplication.instance() or QApplication([])
BOUNDS={'x':[-300,300],'y':[-300,300],'z':[-20,200],'r':[-90,90]}
CONTEXT={'installation_id':'test','robot_base_id':'base','tool_frame_id':'tool','mount_revision':'one'}

def wait(predicate):
    deadline=time.monotonic()+4
    while not predicate() and time.monotonic()<deadline:
        APP.processEvents(); time.sleep(.005)
    if not predicate(): raise AssertionError('worker timeout')

class FakeAdapter:
    kind='real'
    def __init__(self):
        self.epoch='epoch'; self.pose=Pose(150,0,20,0); self.motion_state='STOPPED'
        self.actual_joints=JointPose(0,20,40,0)
        self.commands=[]; self.stops=0; self.closed=False; self.pending=None
        self.auto_complete=True; self.moving=False; self.broken=False
    def status(self):
        if self.broken: raise TimeoutError('injected disconnect')
        if self.moving: self.pose=Pose(self.pose.x+1,0,20,0)
        return RobotStatus(self.epoch,True,self.motion_state,None,self.pose)
    def holding(self): return None
    def request_stop(self): self.stops+=1; self.pending=None; self.motion_state='STOPPED'; return True
    def prepare_teaching_move(self,speed,*,joint=False): self.motion_state='READY'
    def submit(self,command): self.commands.append(command); self.pending=command; return True
    def poll(self,command_id):
        if not self.auto_complete: return Completion(command_id,'PENDING')
        if isinstance(self.pending.target,JointPose): self.actual_joints=self.pending.target
        else: self.pose=self.pending.target
        self.pending=None; return Completion(command_id,'DONE',self.pose)
    def close(self): self.request_stop(); self.closed=True

class StoreTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name)
        self.store=TeachingStore(self.root,CONTEXT)
    def tearDown(self): self.temp.cleanup()
    def save(self,role,pose): self.store.save(role,pose,'epoch')
    def test_precision_restore_draft_and_overwrite(self):
        p=Pose(123.12345678,0,2,0); self.save('home',p)
        restored=TeachingStore(self.root,CONTEXT)
        self.assertEqual(restored.point('home','epoch'),p)
        self.assertEqual(restored.value['state'],'draft')
        self.assertNotIn('validated',restored.value)
        with self.assertRaises(ValueError): self.save('home',p)
        self.store.save('home',Pose(100,0,0,0),'epoch',True)
        self.store.delete('home'); self.assertNotIn('home',self.store.value['points'])
    def test_reconnection_and_context_change_reject_old_points(self):
        self.save('home',Pose(150,0,0,0))
        with self.assertRaises(ValueError): self.store.point('home','new-epoch')
        other=TeachingStore(self.root,{**CONTEXT,'tool_frame_id':'changed'})
        with self.assertRaises(ValueError): other.point('home','epoch')
    def test_atomic_failure_preserves_old_file_and_memory(self):
        self.save('home',Pose(150,0,0,0)); original=self.store.path.read_bytes()
        with patch('mes_vision.station.teaching.os.replace',side_effect=OSError('disk')):
            with self.assertRaises(OSError): self.save('pick',Pose(155,0,0,0))
        self.assertEqual(self.store.path.read_bytes(),original)
        self.assertNotIn('pick',self.store.value['points'])
    def test_offset_is_pose_delta_not_automatic_calibration(self):
        self.save('camera_reference',Pose(150,0,80,0)); self.save('tool_reference',Pose(175,-12,10,0))
        d=self.store.offset('epoch'); self.assertEqual((d['dx'],d['dy'],d['dz']),(25,-12,-70))
        self.assertFalse(d['camera_rotates_with_r']); self.assertEqual(d['state'],'draft')
        self.store.save('tool_reference',Pose(175,-12,10,3),'epoch',True)
        self.assertIsNone(self.store.value['offset'])
        with self.assertRaises(ValueError): self.store.offset('epoch')
    def test_route_has_clearance_and_rejects_limits_rotation_and_low_height(self):
        self.save('travel',Pose(150,0,100,0)); self.save('home',Pose(180,20,10,0))
        route=self.store.route('home',Pose(150,0,20,0),'epoch',BOUNDS)
        self.assertEqual([(p.x,p.y,p.z) for p in route],[(150,0,100),(180,20,100),(180,20,10)])
        for current,bounds in [(Pose(150,0,20,10),BOUNDS),(Pose(150,0,150,0),BOUNDS),
                               (Pose(150,0,20,0),{**BOUNDS,'z':[0,50]}),
                               (Pose(150,0,20,0),{**BOUNDS,'x':[0,math.nan]})]:
            with self.assertRaises(ValueError): self.store.route('home',current,'epoch',bounds)
    def test_crossed_and_degenerate_corners_rejected(self):
        names=['bottom_left','bottom_right','top_right','top_left']
        for role,(x,y) in zip(names,[(0,0),(100,0),(100,100),(0,100)]): self.save(role,Pose(x,y,0,0))
        self.assertEqual(len(self.store.board('epoch')),4)
        self.store.save('top_right',Pose(0,0,0,0),'epoch',True)
        with self.assertRaises(ValueError): self.store.board('epoch')
    def test_nonfinite_pose_rejected(self):
        with self.assertRaises(ValueError): Pose(float('nan'),0,0,0)

class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp=TemporaryDirectory(); self.root=Path(self.temp.name); self.adapter=FakeAdapter()
        self.events=[]; self.states=[]; self.errors=[]; self.created=[]
        def factory(settings): self.created.append(True); return self.adapter
        self.port=StationRobotPort(self.root,default_equipment(),adapter_factory=factory,teaching_context=CONTEXT)
        self.port.completed.connect(self.events.append); self.port.changed.connect(self.states.append); self.port.failed.connect(self.errors.append)
    def tearDown(self):
        self.port.stop_motion(); self.port.stopping.set(); self.port.wait(4000); APP.processEvents(); self.temp.cleanup()
    def start(self): self.port.start(); wait(lambda:len(self.states)>0)
    def request(self,kind,**kw):
        count=len(self.events); self.port.request(kind,**kw); wait(lambda:len(self.events)>count or bool(self.errors))
    def points(self):
        store=TeachingStore(self.root,CONTEXT)
        store.save('home',Pose(180,20,10,0),'epoch'); store.save('travel',Pose(150,0,100,0),'epoch')
    def move_payload(self):
        store=TeachingStore(self.root,CONTEXT)
        return dict(role='home',bounds=BOUNDS,speed=5,empty_tool_confirmed=True,path_confirmed=True,
                    expected_start=asdict(self.adapter.pose),draft_revision=store.value['updated_at'])
    def test_save_reads_current_pose_without_move_or_profile(self):
        self.start(); self.adapter.pose=Pose(170,5,12,0)
        self.request('teach_save',role='home')
        self.assertEqual(TeachingStore(self.root,CONTEXT).point('home','epoch'),self.adapter.pose)
        self.assertEqual(self.adapter.commands,[]); self.assertFalse(self.errors)
    def test_moving_and_disconnected_pose_cannot_be_saved(self):
        self.start(); self.adapter.moving=True; self.request('teach_save',role='home')
        self.assertIn('error',self.events[-1]); self.assertFalse((self.root/'robot-teaching/draft.json').exists())
        self.adapter.moving=False; self.adapter.broken=True; wait(lambda:bool(self.errors))
        wait(lambda:not self.port.isRunning()); self.assertEqual(self.adapter.commands,[])
    def test_motion_is_three_legs_and_never_grips(self):
        self.points(); self.start(); self.request('teach_move',**self.move_payload())
        self.assertFalse(self.errors); self.assertNotIn('error',self.events[-1])
        self.assertEqual([c.action for c in self.adapter.commands],['move']*3)
        self.assertEqual(self.adapter.pose,Pose(180,20,10,0)); self.assertGreater(self.adapter.stops,0)
    def test_changed_start_and_missing_confirmation_send_no_move(self):
        self.points(); self.start(); payload=self.move_payload(); payload['expected_start']['x']=0
        self.request('teach_move',**payload); self.assertIn('error',self.events[-1]); self.assertEqual(self.adapter.commands,[])
        payload=self.move_payload(); payload['empty_tool_confirmed']=False
        self.request('teach_move',**payload); self.assertEqual(self.adapter.commands,[])
    def test_stop_discards_remaining_path(self):
        self.points(); self.adapter.auto_complete=False; self.start()
        self.port.request('teach_move',**self.move_payload()); wait(lambda:len(self.adapter.commands)==1)
        self.port.stop_motion(); wait(lambda:self.adapter.stops>0)
        time.sleep(.1); APP.processEvents(); self.assertEqual(len(self.adapter.commands),1)
    def test_lock_prevents_second_connection(self):
        folder=self.root/'robot'; folder.mkdir()
        with FileLock(str(folder/'connection.lock'),timeout=0):
            self.port.start(); wait(lambda:bool(self.errors)); self.port.wait(2000)
        self.assertEqual(self.created,[])
    def test_auto_commands_are_rejected_in_teaching_and_reverse(self):
        with self.assertRaises(ValueError): self.port.request('recover',reference='test')
        normal=StationRobotPort(self.root,default_equipment(),adapter_factory=lambda _:self.adapter)
        with self.assertRaises(ValueError): normal.request('teach_save',role='home')

class ProtocolTests(unittest.TestCase):
    def test_absolute_speed_readback_and_no_ptp_on_prepare(self):
        class Protocol:
            def __init__(self): self.calls=[]; self.values={}
            def request(self,code,write=False,queued=False,parameters=b'',**kw):
                self.calls.append(code)
                if write: self.values[code]=parameters; return b''
                if code==10: return struct.pack('<8f',*([0]*8))
                return self.values[code]
            def close(self): pass
        protocol=Protocol(); robot=Magician(default_equipment()['robot'],protocol=protocol)
        robot.prepare_teaching_move(5)
        self.assertEqual(struct.unpack('<4f',protocol.values[81]),(5,5,20,20))
        self.assertNotIn(84,protocol.calls); self.assertNotIn(63,protocol.calls); self.assertNotIn(31,protocol.calls)
        with self.assertRaises(ValueError): robot.prepare_teaching_move(11)
        robot.close()

class DialogTests(unittest.TestCase):
    def test_blank_invalid_and_reversed_limits_have_actionable_messages(self):
        with TemporaryDirectory() as root, patch('serial.tools.list_ports.comports',return_value=[]):
            dialog=TeachingDialog(Path(root),default_equipment())
            try:
                for value in ('','  ','abc','nan','inf'):
                    dialog.bounds['x'][0].setText(value)
                    with self.assertRaisesRegex(ValueError,'X 최소값.*숫자'): dialog.read_bounds()
                for axis,values in BOUNDS.items():
                    for edit,value in zip(dialog.bounds[axis],values): edit.setText(str(value))
                self.assertEqual(dialog.read_bounds(),BOUNDS)
                dialog.bounds['y'][0].setText('400')
                with self.assertRaisesRegex(ValueError,'Y 최소값.*최대값'): dialog.read_bounds()
            finally: dialog.close(); dialog.deleteLater(); APP.processEvents()

    def test_connect_button_pose_save_disconnect_and_close(self):
        from types import SimpleNamespace
        device=SimpleNamespace(device='TEST',description='Injected',hwid='TEST')
        adapter=FakeAdapter()
        def factory(*args,**kwargs):
            return StationRobotPort(*args,**kwargs,adapter_factory=lambda _:adapter)
        with TemporaryDirectory() as root, patch('serial.tools.list_ports.comports',return_value=[device]), \
                patch('mes_vision.station.teaching_dialog.StationRobotPort',side_effect=factory):
            dialog=TeachingDialog(Path(root),default_equipment())
            try:
                dialog.connect_button.click()
                wait(lambda:dialog.ready())
                self.assertTrue(dialog.save_buttons[0].isEnabled())
                self.assertIn('150.0',dialog.pose_label.text())
                dialog.save_buttons[0].click()
                wait(lambda:(Path(root)/'robot-teaching/draft.json').exists())
                self.assertEqual(adapter.commands,[])
                dialog.disconnect_button.click(); wait(lambda:dialog.worker is None)
                self.assertTrue(dialog.connect_button.isEnabled())
                self.assertFalse(dialog.save_buttons[0].isEnabled())
            finally:
                if dialog.worker:
                    dialog.disconnect_robot(); wait(lambda:dialog.worker is None)
                dialog.close(); dialog.deleteLater(); APP.processEvents()

    def test_connection_setup_failure_does_not_leave_unstarted_worker(self):
        from types import SimpleNamespace
        device=SimpleNamespace(device='TEST',description='Injected',hwid='TEST')
        with TemporaryDirectory() as root, patch('serial.tools.list_ports.comports',return_value=[device]), \
                patch.object(StationRobotPort,'start',side_effect=RuntimeError('start failed')):
            dialog=TeachingDialog(Path(root),default_equipment())
            dialog.connect_button.click()
            self.assertIsNone(dialog.worker); self.assertTrue(dialog.connect_button.isEnabled())
            self.assertIn('start failed',dialog.feedback.text())
            dialog.close(); dialog.deleteLater(); APP.processEvents()

    def test_production_start_and_connection_blocked_during_teaching(self):
        from types import SimpleNamespace
        from mes_vision.station.window import StationWindow
        window=SimpleNamespace(manual_robot_active=lambda:True)
        with self.assertRaises(ValueError): StationWindow.begin(window)
        with self.assertRaises(ValueError): StationWindow.connect_robot(window)

    def test_disconnected_ui_and_stale_pose(self):
        with TemporaryDirectory() as root, patch('serial.tools.list_ports.comports',return_value=[]):
            dialog=TeachingDialog(Path(root),default_equipment()); dialog.show(); APP.processEvents()
            self.assertFalse(dialog.move_button.isEnabled()); self.assertFalse(dialog.save_buttons[0].isEnabled())
            dialog.last_state={'state':'TEACHING','status':{'pose':asdict(Pose(1,2,3,0))},'busy':False}
            dialog.received=time.monotonic()-2; dialog.tick(); self.assertFalse(dialog.ready())
            self.assertIn('응답 지연',dialog.pose_label.text()); dialog.close(); dialog.deleteLater(); APP.processEvents()

if __name__=='__main__': unittest.main()

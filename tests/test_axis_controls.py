"""Axis commands and the actual sidebar entry, with injected hardware only."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import struct
import time
import unittest

import test_teaching as support
from mes_vision.robot.contracts import Pose,JointPose,Command
from mes_vision.robot.magician import Magician
from mes_vision.operation.catalog import default_equipment
from mes_vision.station.teaching import axes_target
from mes_vision.station.teaching_dialog import TeachingDialog
from mes_vision.station.robot_port import StationRobotPort

JOINT_BOUNDS={k:[-90,90] for k in ('j1','j2','j3','j4')}

def message(space='cartesian',**changes):
    joints=space=='joint'
    return {'space':space,'mode':'step','axis':'j1' if joints else 'x','delta':1,
            'bounds':deepcopy(JOINT_BOUNDS if joints else support.BOUNDS),
            'expected':asdict(JointPose(0,20,40,0) if joints else Pose(150,0,20,0)),
            'epoch':'epoch','issued_at':100,'speed':5,
            'empty_tool_confirmed':True,'path_confirmed':True,**changes}

class AxisValidationTests(unittest.TestCase):
    def target(self,m): return axes_target(m,Pose(150,0,20,0),JointPose(0,20,40,0),'epoch',100)
    def test_steps_use_measured_coordinate_and_keep_other_axes(self):
        self.assertEqual(self.target(message()),Pose(151,0,20,0))
        self.assertEqual(self.target(message('joint',axis='j3',delta=-2)),JointPose(0,20,38,0))
    def test_one_axis_and_all_axis_absolute_targets(self):
        self.assertEqual(self.target(message(mode='absolute',target={'r':10})),Pose(150,0,20,10))
        self.assertEqual(self.target(message('joint',mode='absolute',target={'j1':10,'j2':22,'j3':45,'j4':-5})),JointPose(10,22,45,-5))
    def test_rejects_stale_epoch_large_step_nonfinite_outside_bounds_unarmed(self):
        for changes in ({'issued_at':97},{'epoch':'old'},{'delta':6},{'delta':float('nan')},
                        {'empty_tool_confirmed':False},{'expected':asdict(Pose(140,0,20,0))},
                        {'mode':'absolute','target':{'x':301}}, {'mode':'absolute','target':{'x':float('nan')}}):
            with self.subTest(changes=changes),self.assertRaises(ValueError): self.target(message(**changes))
    def test_joint_and_cartesian_types_are_not_interchanged(self):
        with self.assertRaises(ValueError): self.target(message('joint',mode='absolute',target={'x':10}))
        with self.assertRaises(ValueError): self.target(message('joint',axis='r'))

class AxisWorkerTests(unittest.TestCase):
    setUp=support.WorkerTests.setUp
    tearDown=support.WorkerTests.tearDown
    start=support.WorkerTests.start
    request=support.WorkerTests.request
    def test_cartesian_and_joint_requests_dispatch_distinct_commands(self):
        self.start()
        self.request('teach_axes',**message(issued_at=time.monotonic()))
        self.assertEqual(self.adapter.commands[-1].action,'move')
        self.assertEqual(self.adapter.commands[-1].target,Pose(151,0,20,0))
        self.request('teach_axes',**message('joint',issued_at=time.monotonic(),axis='j2',delta=1))
        self.assertEqual(self.adapter.commands[-1].action,'move_joints')
        self.assertEqual(self.adapter.actual_joints,JointPose(0,21,40,0))
        self.assertFalse(self.errors)
    def test_stopped_joint_move_has_no_replay(self):
        self.adapter.auto_complete=False; self.start()
        self.port.request('teach_axes',**message('joint',issued_at=time.monotonic()))
        support.wait(lambda:len(self.adapter.commands)==1)
        self.port.stop_motion(); support.wait(lambda:self.adapter.stops>0)
        time.sleep(.1); support.APP.processEvents()
        self.assertEqual(len(self.adapter.commands),1)
    def test_expired_message_cannot_send_a_move(self):
        self.start(); self.request('teach_axes',**message(issued_at=time.monotonic()-10))
        self.assertIn('error',self.events[-1]); self.assertEqual(self.adapter.commands,[])

class JointProtocolTests(unittest.TestCase):
    def test_joint_packet_and_measured_completion(self):
        class Protocol:
            def __init__(self): self.values={}; self.calls=[]; self.joints=(1,2,3,4)
            def request(self,code,write=False,queued=False,parameters=b'',**kw):
                self.calls.append((code,write,queued,parameters))
                if code==84: self.payload=parameters; return struct.pack('<Q',8)
                if code==246: return struct.pack('<Q',8)
                if code==10: return struct.pack('<8f',100,20,30,5,*self.joints)
                if write: self.values[code]=parameters; return b''
                return self.values[code]
            def close(self): pass
        protocol=Protocol(); robot=Magician(default_equipment()['robot'],protocol=protocol)
        self.assertEqual(robot.actual_joints,JointPose(1,2,3,4))
        robot.prepare_teaching_move(3,joint=True)
        self.assertEqual(struct.unpack('<8f',protocol.values[80]),(3,3,3,3,20,20,20,20))
        cmd=Command('j','manual','axis','move_joints',JointPose(1,10,3,4))
        robot.submit(cmd)
        self.assertEqual(struct.unpack('<B4f',protocol.payload),(4,1,10,3,4))
        self.assertEqual(robot.poll('j').state,'PENDING')
        protocol.joints=(1,10,3,4); self.assertEqual(robot.poll('j').state,'DONE')
        self.assertFalse(any(code in (31,62,63) for code,*_ in protocol.calls)); robot.close()

class AxisUiTests(unittest.TestCase):
    def test_click_and_numeric_entry_run_only_after_explicit_action(self):
        device=SimpleNamespace(device='TEST',description='Injected',hwid='TEST'); adapter=support.FakeAdapter()
        def factory(*a,**kw): return StationRobotPort(*a,**kw,adapter_factory=lambda _:adapter)
        with TemporaryDirectory() as root,patch('serial.tools.list_ports.comports',return_value=[device]), \
                patch('mes_vision.station.teaching_dialog.StationRobotPort',side_effect=factory):
            dialog=TeachingDialog(Path(root),default_equipment(),embedded=True)
            try:
                dialog.connect_button.click(); support.wait(dialog.ready)
                group=dialog.axes.groups['cartesian']
                for key,row in group['rows'].items():
                    row['lower'].setText(str(support.BOUNDS[key][0])); row['upper'].setText(str(support.BOUNDS[key][1]))
                dialog.axes.armed.setChecked(True)
                group['rows']['x']['target'].setText('160')
                support.APP.processEvents(); self.assertEqual(adapter.commands,[])
                group['buttons'][1].click(); support.wait(lambda:len(adapter.commands)==1 and dialog.ready())
                self.assertEqual(adapter.commands[-1].target.x,151)
                self.assertFalse(group['buttons'][1].autoRepeat())
                with patch('mes_vision.station.axis_controls.QMessageBox.question',return_value=__import__('PySide6.QtWidgets',fromlist=['QMessageBox']).QMessageBox.Yes):
                    group['buttons'][2].click()
                support.wait(lambda:len(adapter.commands)==2 and dialog.ready())
                self.assertEqual(adapter.commands[-1].target.x,160)
                dialog.stop(); self.assertFalse(dialog.axes.armed.isChecked())
            finally:
                dialog.disconnect_robot(); support.wait(lambda:dialog.worker is None)
                dialog.timer.stop(); dialog.deleteLater(); support.APP.processEvents()

    def test_sidebar_page_is_embedded_and_global_stop_reaches_manual_worker(self):
        from mes_vision.station.window import StationWindow
        from unittest.mock import Mock
        with TemporaryDirectory() as root,patch('serial.tools.list_ports.comports',return_value=[]):
            window=StationWindow(Path(__file__).resolve().parents[1],Path(root),start_worker=False)
            try:
                window.timer.stop(); window.analysis.timer.stop()
                support.wait(lambda:not window.tasks and not window.analysis.refresh_read.busy)
                self.assertEqual(window.nav.count(),6); self.assertEqual(window.pages.count(),6)
                window.sidebar.items[5].click()
                self.assertEqual(window.pages.currentWidget(),window.robot_panel)
                self.assertFalse(window.robot_panel.isWindow())
                self.assertEqual(window.robot_panel.modes.tabText(0),'축 제어')
                worker=Mock(); window.robot_panel.worker=worker
                from mes_vision.operation.status_hooks import update_context
                window.robot_panel.last_state={'state':'TEACHING','busy':False}
                window.robot_panel.received=time.monotonic(); update_context(window)
                self.assertEqual(window.context.values['로봇'].text(),'수동 제어 연결')
                window.global_stop.click(); worker.stop_motion.assert_called()
                with self.assertRaises(ValueError): window.begin()
                with self.assertRaises(ValueError): window.require_idle(robot=True)
                worker.reset_mock(); window.nav.setCurrentRow(0); worker.stop_motion.assert_called()
            finally:
                window.robot_panel.worker=None
                support.wait(lambda:not window.tasks and not window.analysis.refresh_read.busy and not window.vlm_read.busy)
                window.close(); window.close(); window.deleteLater(); support.APP.processEvents()

if __name__=='__main__': unittest.main()

import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile, threading, time, unittest
from types import SimpleNamespace
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from PySide6.QtTest import QTest
from PySide6.QtCore import Qt
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.catalog import OperationStore,new_product,default_equipment
from mes_vision.operation.readiness_controls import check_storage,check_coordinates
from mes_vision.calibration import fit
from mes_vision.calibration.fixtures import make_spec
from test_operation import frame

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]

def wait(predicate):
    end=time.monotonic()+8
    while not predicate() and time.monotonic()<end: APP.processEvents(); time.sleep(.005)
    APP.processEvents()
    if not predicate(): raise AssertionError('readiness read did not finish')


class ReadinessUiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.w=DesktopWindow(ROOT,self.temp.name,start_worker=False)
        self.w.timer.stop(); self.w.analysis.timer.stop(); self.commands=[]
    def tearDown(self):
        self.w.timer.stop(); self.w.analysis.timer.stop()
        self.w.closing=True; self.w.readiness_read.invalidate()
        wait(lambda:not self.w.readiness_read.busy and not self.w.vlm_read.busy and not self.w.analysis.refresh_read.busy and not self.w.tasks)
        self.w.engine=self.w.camera=self.w.robot=None; self.w.close(); APP.processEvents(); self.temp.cleanup()
    def ready(self):
        w=self.w; w.product=new_product(); w.frame=frame(); w.camera_info={'serial':'test'}; w.frame_received=time.monotonic()
        w.engine=SimpleNamespace(command=lambda *args,**kw:self.commands.append((args,kw))); w.engine_ready=True
        w.readiness_key=w.readiness_inputs()[0]; w.readiness_at=time.monotonic()
        w.readiness_report={'settings':[],'storage':None,'coordinates':'no calibration','robot':'not connected'}
    def test_missing_requirements_are_visible_and_shortcuts_do_not_start_devices(self):
        w=self.w; w.readiness_tick(time.monotonic()); wait(lambda:not w.readiness_read.busy)
        w.render_readiness(time.monotonic()); self.assertFalse(w.start.isEnabled())
        self.assertIn('시작 전 확인',w.readiness_summary.text()); self.assertEqual(len(w.readiness_buttons),6)
        for key,page in [('product',1),('models',1),('camera',2),('coordinates',2),('storage',3),('robot',2)]:
            w.readiness_buttons[key][0].click(); self.assertEqual(w.pages.currentIndex(),page)
        self.assertIsNone(w.camera); self.assertIsNone(w.robot); self.assertIsNone(w.engine)
    def test_inspection_only_does_not_require_robot_or_coordinates(self):
        self.ready(); self.w.render_readiness(time.monotonic())
        self.assertTrue(self.w.start.isEnabled()); self.assertIn('자동 분류 시 필수',self.w.readiness_buttons['coordinates'][0].detail.text())
        self.w.begin(); self.assertTrue(self.w.running); self.assertTrue(self.commands)
    def test_auto_mode_requires_validated_coordinates_and_robot(self):
        self.ready(); w=self.w
        w.auto.blockSignals(True); w.auto.setChecked(True); w.auto.blockSignals(False)
        w.render_readiness(time.monotonic()); self.assertFalse(w.start.isEnabled())
        with self.assertRaises(ValueError): w.begin()
        self.assertFalse(self.commands)
    def test_auto_cannot_be_enabled_during_inspection_with_missing_calibration(self):
        self.ready(); w=self.w; w.running=True
        w.robot=SimpleNamespace(); w.robot_seen_at=time.monotonic(); w.robot_state={'state':'IDLE','busy':False}
        with patch.object(w,'error') as error: w.auto.setChecked(True)
        self.assertFalse(w.auto.isChecked()); self.assertTrue(error.called)
    def test_storage_failure_or_expired_check_blocks_button_and_begin(self):
        self.ready(); w=self.w; w.readiness_report['storage']='disk full'
        w.render_readiness(time.monotonic()); self.assertFalse(w.start.isEnabled())
        with self.assertRaisesRegex(ValueError,'disk full'): w.begin()
        w.readiness_report['storage']=None; w.readiness_at-=16
        w.render_readiness(time.monotonic()); self.assertFalse(w.start.isEnabled())
        with self.assertRaises(ValueError): w.begin()
    def test_fresh_video_requirement_matches_start_button_and_command(self):
        self.ready(); self.w.frame_received-=.7
        self.w.render_readiness(time.monotonic()); self.assertFalse(self.w.start.isEnabled())
        with self.assertRaises(ValueError): self.w.begin()
        self.assertFalse(self.commands)
    def test_changed_settings_cannot_accept_old_successful_probe(self):
        self.ready(); w=self.w; entered,release=threading.Event(),threading.Event()
        report=deepcopy(w.readiness_report)
        def slow(*args): entered.set(); release.wait(5); return report
        with patch('mes_vision.operation.readiness_controls.probe_readiness',slow):
            try:
                w.readiness_tick(time.monotonic()); wait(entered.is_set)
                w.equipment['version']+=1; w.readiness_tick(time.monotonic())
                self.assertIsNone(w.readiness_report)
            finally: release.set(); wait(lambda:not w.readiness_read.busy)
        self.assertIsNone(w.readiness_report); self.assertFalse(w.start.isEnabled())
    def test_unchanged_inputs_do_not_repeat_filesystem_checks_on_each_tick(self):
        w=self.w
        with patch('mes_vision.operation.readiness_controls.probe_readiness',return_value={'settings':[],'storage':None,'coordinates':None,'robot':None}) as probe:
            w.readiness_tick(time.monotonic()); wait(lambda:not w.readiness_read.busy)
            for _ in range(25): w.readiness_tick(time.monotonic())
            self.assertEqual(probe.call_count,1)
    def test_active_fault_is_first_start_blocker(self):
        self.ready(); w=self.w; w.faults.raise_fault('TEST','test fault')
        w.render_readiness(time.monotonic()); self.assertIn('미해결',w.readiness_summary.text()); self.assertFalse(w.start.isEnabled())
    def test_storage_dialog_waits_for_probe_and_invalidates_previous_success(self):
        w=self.w; entered,release=threading.Event(),threading.Event(); visits=[]
        def slow(*args):
            entered.set(); release.wait(5)
            return {'settings':[],'storage':None,'coordinates':None,'robot':None}
        with patch('mes_vision.operation.readiness_controls.probe_readiness',slow),patch('mes_vision.operation.storage_dialog.StorageDialog.exec',lambda d:visits.append(True) or 0):
            try:
                w.readiness_tick(time.monotonic()); wait(entered.is_set)
                w.show_storage_dialog(); w.show_storage_dialog(); self.assertFalse(visits)
            finally: release.set()
            wait(lambda:bool(visits)); w.timer.stop(); w.analysis.timer.stop()
            self.assertEqual(len(visits),1)
            self.assertIsNone(w.readiness_report)


class ReadinessStorageTests(unittest.TestCase):
    def test_write_probe_leaves_no_inspection_or_database_record(self):
        with tempfile.TemporaryDirectory() as root:
            store=OperationStore(root)
            with patch('mes_vision.operation.readiness_controls.check_free_space'):
                check_storage(store)
            self.assertEqual(list((store.root/'inspections').iterdir()),[])
            with store.connect() as db:
                self.assertEqual(db.execute('SELECT COUNT(*) FROM preferences').fetchone()[0],0)
                self.assertEqual(db.execute('SELECT COUNT(*) FROM inspections').fetchone()[0],0)
    def test_low_space_is_not_reported_as_writable(self):
        with tempfile.TemporaryDirectory() as root:
            with patch('mes_vision.operation.readiness_controls.check_free_space',side_effect=ValueError('disk full')):
                with self.assertRaisesRegex(ValueError,'disk full'): check_storage(OperationStore(root))


class ReadinessCalibrationTests(unittest.TestCase):
    def test_registration_requires_matching_product_mount_and_height(self):
        with tempfile.TemporaryDirectory() as root:
            spec=replace(make_spec(),kind='real'); calibration=fit(spec).accept('UNIT TEST ONLY')
            path=Path(root)/'calibration.json'; calibration.save(path)
            e=default_equipment(); c=spec.context
            e['calibration']=str(path)
            e['camera'].update(serial=c.camera_id,mount_revision=c.mount_revision,acquisition_revision=c.acquisition_revision,
                               robot_base_id=c.robot_base_id,tool_frame_id=c.tool_frame_id)
            p=new_product(); p['id']=spec.product_id; p['grasp']['plane_z_mm']=spec.plane_z_mm
            geometry={'width':c.image_size[0],'height':c.image_size[1],'transformations':list(c.transformations)}
            self.assertEqual(check_coordinates(p,e,geometry),calibration.identity)
            for change in ('product','mount','height'):
                pp,ee=deepcopy(p),deepcopy(e)
                if change=='product': pp['id']='different'
                elif change=='mount': ee['camera']['mount_revision']='changed'
                else: pp['grasp']['plane_z_mm']+=100
                with self.assertRaises(ValueError): check_coordinates(pp,ee,geometry)


if __name__=='__main__': unittest.main()

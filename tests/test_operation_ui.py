import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from PySide6.QtCore import Qt,QPoint
from PySide6.QtWidgets import QApplication,QLabel,QPushButton,QTableWidgetItem
from PySide6.QtTest import QTest
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.catalog import new_product,default_equipment
from mes_vision.operation.product_dialog import ProductDialog
from mes_vision.operation.setup_dialogs import EquipmentDialog,CalibrationDialog,RobotProfileDialog,GeometryDialog
from mes_vision.operation.training_dialog import TrainingDialog
from mes_vision.calibration.fixtures import make_spec
from mes_vision.calibration import fit
from mes_vision.training.data import write_json
from test_operation import frame
import test_operation

ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])

class OperationUiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.window=DesktopWindow(ROOT,self.root,start_worker=False); self.window.timer.stop(); self.window.analysis.timer.stop()
        self.window.show(); APP.processEvents()
    def tearDown(self):
        end=time.monotonic()+5
        while (self.window.vlm_read.busy or self.window.analysis.refresh_read.busy or self.window.readiness_read.busy) and time.monotonic()<end:
            APP.processEvents(); time.sleep(.005)
        self.window.engine=None; self.window.camera=None; self.window.robot=None; self.window.closing=True
        self.window.close(); APP.processEvents(); self.temp.cleanup()
    def test_five_pages_no_demo_controls_and_disabled_start(self):
        self.window.tick(); self.assertEqual(self.window.pages.count(),5); self.assertFalse(self.window.start.isEnabled())
        labels=[w.text() for cls in (QLabel,QPushButton) for w in self.window.findChildren(cls)]
        self.assertFalse(any(any(s in label for s in ("데모","모의 집기","합성 예제","COCO","사진 모델 시험")) for label in labels))
    def test_click_bbox_selects_id_and_changed_revision_clears_verdict(self):
        w=self.window; w.canvas.set_rgb(np.full((180,480,3),120,np.uint8))
        t={"track_id":"session:T1","box":[20,30,130,150],"status":"WAITING","revision":1,"missing":False,"result":None}
        w.tracks=[t]; w.canvas.tracks=[t]; APP.processEvents(); scale,x,y=w.canvas.transform()
        QTest.mouseClick(w.canvas,Qt.LeftButton,pos=QPoint(int(x+75*scale),int(y+80*scale)))
        self.assertEqual(w.selected_id,"session:T1"); self.assertIn("현재 판정이 없습니다",w.live_details.toPlainText())
        w.handle_engine({"type":"tracks","tracks":[],"generation":w.generation,"updated":time.monotonic(),"loop_ms":10})
        self.assertIn("추적이 종료",w.selection.text())
    def test_pause_keeps_camera_and_rejects_old_generation(self):
        w=self.window; commands=[]; w.engine=SimpleNamespace(command=lambda *a,**k:commands.append((a,k)))
        sentinel=object(); w.camera=sentinel; w.running=True; before=w.generation; w.pause_inspection()
        self.assertIs(w.camera,sentinel); self.assertFalse(w.running); self.assertGreater(w.generation,before)
        w.handle_engine({"type":"tracks","generation":before,"tracks":[{"bad":"old"}]})
        self.assertEqual(w.tracks,[]); self.assertEqual(commands[-1][1]["value"],False)
    def test_product_add_edit_copy_and_apply_version(self):
        w=self.window; p=w.store.save_product(new_product("품목 A")); w.refresh_products(); w.apply_product()
        self.assertEqual(w.product["version"],1)
        dialog=ProductDialog(p,w.equipment,ROOT,self.root,w); dialog.name.setText("수정 품목"); dialog.save()
        saved=w.store.save_product(dialog.value); self.assertEqual(saved["version"],2)
        self.assertEqual(w.product["name"],"품목 A"); w.refresh_products(); w.apply_product(); self.assertEqual(w.product["name"],"수정 품목")
        dialog.deleteLater()
    def test_equipment_roi_draft_and_acquisition_revision_cannot_bypass(self):
        e=default_equipment(); e['camera'].update(width=640,height=480); d=EquipmentDialog(e,frame(),self.window)
        d.set_region({"mode":"roi","polygon":[[5,5],[450,5],[450,170],[5,170]]}); d.save(); self.assertTrue(d.value["workspace"]["roi"])
        d=EquipmentDialog(e,frame(),self.window); d.resolution.setCurrentIndex(d.resolution.findData('1280x720')); d.save(); self.assertEqual(d.result(),1)
        self.assertNotEqual(d.value['camera']['acquisition_revision'],e['camera']['acquisition_revision'])
        self.assertTrue(d.camera_text['acquisition_revision'].isReadOnly())
    def test_calibration_fit_register_and_changed_measurements_invalidate(self):
        spec=make_spec(); p=new_product(); p.update(id=spec.product_id); p["grasp"]["plane_z_mm"]=20
        e=default_equipment(); c=spec.context; e["camera"].update(serial=c.camera_id,mount_revision=c.mount_revision,acquisition_revision=c.acquisition_revision,
            robot_base_id=c.robot_base_id,tool_frame_id=c.tool_frame_id,width=480,height=180); e["workspace"]["roi"]=[list(x) for x in spec.application_polygon]
        d=CalibrationDialog(p,e,frame(),{"distortion":"distortion.none","serial":"unit-test","firmware":"unit-test"},self.root,self.window)
        for pair in spec.pairs:
            i=d.points.rowCount(); d.points.insertRow(i)
            for j,val in enumerate([pair.point_id,pair.role,pair.measurement_session,*pair.pixel,*pair.robot_xy_mm]): d.points.setItem(i,j,QTableWidgetItem(str(val)))
        d.version.setText("unit-test"); d.max_error.setValue(.1); d.rms_error.setValue(.05); d.calculate(); self.assertIsNotNone(d.calibration,d.message.text())
        d.reference.setText("UNIT-TEST-ONLY"); self.assertIsNotNone(d.calibration); d.register(); self.assertEqual(d.result(),1,d.message.text())
        self.assertTrue(Path(d.path).exists()); d.points.item(0,5).setText("99"); self.assertIsNone(d.calibration)
    def test_live_selection_and_history_use_same_record_and_never_offer_history_motion(self):
        t=test_operation.OperationTests(); t.setUp()
        try:
            committed=t.engine.commit(t.inspect()); self.window.store=t.store; link=committed["links"][0]
            self.window.tracks=[x.data() for x in t.engine.tracker.tracks.values()]; self.window.select_track(link["track_id"])
            self.assertIn("정상",self.window.live_details.toPlainText()); self.window.open_selected_record()
            self.assertFalse(self.window.saved_canvas.pixmap.isNull()); self.assertEqual(self.window.pages.currentIndex(),3)
            self.assertIn("최종 판정: 정상",self.window.history_details.toPlainText())
        finally: t.tearDown()
    def test_vlm_global_toggle_sync(self):
        w=self.window; w.toggle_vlm(True); w.tick(); self.assertTrue(w.vlm.isChecked()); self.assertTrue(w.analysis.toggle.isChecked())
        w.analysis.toggle_vlm(False); w.tick(); self.assertFalse(w.vlm.isChecked())
    def test_training_open_and_runtime_guards(self):
        d=TrainingDialog(ROOT,self.root,self.window); d.show(); APP.processEvents(); self.assertFalse(d.busy)
        d.busy=True; self.window.children.append(d); self.assertFalse(self.window.close()); d.busy=False; d.close()
    def test_models_ready_syncs_new_process_generation(self):
        w=self.window; commands=[]; w.generation=8; w.engine=SimpleNamespace(command=lambda *a,**k:commands.append((a,k)))
        w.handle_engine({"type":"ready","concurrent_vlm":False}); self.assertEqual(commands[-1][1]["generation"],9); self.assertTrue(w.engine_ready)

if __name__=="__main__": unittest.main()

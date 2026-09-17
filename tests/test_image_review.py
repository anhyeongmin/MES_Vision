import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
from types import SimpleNamespace as NS
import numpy as np
from PySide6.QtCore import Qt,QPoint,QPointF
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QWidget
from mes_vision.operation.image_view import ImageCanvas,ImagePanel
from mes_vision.operation.evidence_dialog import EvidenceDialog,bounds
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.snapshots import load_snapshot,save_snapshot
from mes_vision.vlm.viewer import AnalysisViewer
from mes_vision.training.data import sha256

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]


class NavigationTests(unittest.TestCase):
    def setUp(self):
        self.panel=ImagePanel(); self.panel.resize(640,510); self.panel.show(); APP.processEvents()
        self.canvas=self.panel.canvas; self.rgb=np.full((900,1200,3),120,np.uint8); self.canvas.set_rgb(self.rgb)

    def tearDown(self): self.panel.close(); self.panel.deleteLater(); APP.processEvents()

    def test_zoom_keeps_pixel_under_pointer_and_hit_test_uses_original_coordinates(self):
        c=self.canvas; point=QPointF(c.width()/2+20,c.height()/2+10); before=c.pixel(point)
        c.zoom(3,point); after=c.pixel(point)
        self.assertAlmostEqual(before[0],after[0]); self.assertAlmostEqual(before[1],after[1])
        c.tracks=[{'track_id':'object-A','box':[before[0]-20,before[1]-20,before[0]+20,before[1]+20],'status':'NG'}]
        selected=[]; c.selected.connect(selected.append); QTest.mouseClick(c,Qt.LeftButton,pos=point.toPoint())
        self.assertEqual(selected,['object-A'])

    def test_wheel_changes_view_without_modifying_source_pixels(self):
        c=self.canvas; before=c.pixmap.toImage().copy(); scale=c.transform()[0]
        event=QWheelEvent(QPointF(c.width()/2,c.height()/2),QPointF(0,0),QPoint(),QPoint(0,120),Qt.NoButton,Qt.NoModifier,Qt.NoScrollPhase,False)
        QApplication.sendEvent(c,event)
        self.assertGreater(c.transform()[0],scale); self.assertEqual(c.pixmap.toImage(),before)
        self.assertTrue(np.array_equal(self.rgb,np.full((900,1200,3),120,np.uint8)))

    def test_drag_pans_without_selecting_and_cannot_lose_image(self):
        c=self.canvas; c.native_size(); selected=[]; c.selected.connect(selected.append)
        c.tracks=[{'track_id':'A','box':[0,0,1200,900],'status':'OK'}]
        before=c.transform(); center=QPoint(c.width()//2,c.height()//2)
        QTest.mousePress(c,Qt.LeftButton,pos=center); QTest.mouseMove(c,center+QPoint(80,30)); QTest.mouseRelease(c,Qt.LeftButton,pos=center+QPoint(80,30))
        self.assertFalse(selected); self.assertNotEqual(c.transform(),before)
        c._center=(-1e9,1e9); scale,x,y=c.transform()
        self.assertLessEqual(x,0); self.assertGreaterEqual(x+c.pixmap.width()*scale,c.width())
        self.assertLessEqual(y,0); self.assertGreaterEqual(y+c.pixmap.height()*scale,c.height())

    def test_native_pixels_account_for_high_dpi_and_resize(self):
        c=self.canvas
        with patch.object(c,'devicePixelRatioF',return_value=2.):
            c.native_size(); self.assertEqual(c.transform()[0],.5); self.assertEqual(self.panel.scale_label.text(),'100%')
            self.panel.resize(720,550); APP.processEvents(); self.assertEqual(c.transform()[0],.5)

    def test_live_frames_keep_zoom_until_geometry_changes(self):
        c=self.canvas; c.zoom(2); before=c.transform(); c.set_rgb(self.rgb.copy())
        self.assertEqual(before,c.transform())
        c.set_rgb(np.zeros((180,320,3),np.uint8)); self.assertIsNone(c._scale); self.assertFalse(c._native)

    def test_fit_and_keyboard_restore_view_and_overlays_leave_pixels_unchanged(self):
        c=self.canvas; before=c.pixmap.toImage(); c.zoom(5); QTest.keyClick(c,Qt.Key_F)
        self.assertIsNone(c._scale); QTest.keyClick(c,Qt.Key_1)
        self.assertAlmostEqual(c.transform()[0]*c.devicePixelRatioF(),1)
        self.panel.overlays.setChecked(False); c.grab(); self.assertFalse(c.show_overlays)
        self.assertEqual(before,c.pixmap.toImage())

    def test_blank_canvas_is_inert_and_toolbar_disabled(self):
        other=ImagePanel(); c=other.canvas; before=c.transform(); c.zoom(3); c.native_size()
        self.assertEqual(before,c.transform()); self.assertFalse(other.native_button.isEnabled()); other.deleteLater()

    def test_zoom_limits_reject_nonfinite_and_extreme_values(self):
        c=self.canvas; before=c.transform(); c.zoom(float('nan')); c.zoom(-1)
        self.assertEqual(before,c.transform()); c.zoom(1e20)
        self.assertLessEqual(c.transform()[0]*c.devicePixelRatioF(),32)
        c.zoom(1e-30); self.assertGreaterEqual(c.transform()[0]*c.devicePixelRatioF(),.01)


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.fixture=make_vlm_fixture(self.root/'fixture'); self.snapshot=self.fixture['snapshot']
        self.manifest,self.inspection,self.digest=load_snapshot(self.snapshot)
        self.identity=self.fixture['result'].objects[1].object_id; self.windows=[]

    def tearDown(self):
        for w in self.windows:
            if isinstance(w,AnalysisViewer):
                w.timer.stop(); deadline=time.monotonic()+3
                while w.refresh_read.busy and time.monotonic()<deadline: APP.processEvents(); time.sleep(.005)
            w.close(); w.deleteLater()
        APP.processEvents(); self.temp.cleanup()

    def dialog(self,root=None,digest=None):
        d=EvidenceDialog(root or self.snapshot,self.identity,expected_digest=digest or self.digest)
        self.windows.append(d); d.show(); APP.processEvents(); return d

    def test_comparison_uses_saved_reference_selected_crop_and_original_finding_boxes(self):
        d=self.dialog(); source=self.manifest['objects'][1]; record=self.inspection['objects'][1]
        self.assertEqual(d.inspected.canvas.pixmap.width(),source['image_size'][0])
        self.assertEqual(d.inspected.canvas.pixmap.height(),source['image_size'][1])
        self.assertEqual(d.normal.canvas._source,str(self.snapshot/'reference.png'))
        self.assertEqual(d.full.canvas.tracks[0]['box'],bounds(record['effective_box']))
        finding=next(check for check in record['checks'] if check['findings'])['findings'][0]
        self.assertEqual(d.inspected.canvas.tracks[0]['box'],bounds(finding['crop_box']))
        self.assertIn(self.manifest['reference']['collection_id'],d.reference_status.text())

    def test_navigation_is_independent_and_does_not_rewrite_evidence_or_enqueue_vlm(self):
        before={p.name:sha256(p) for p in self.snapshot.iterdir() if p.is_file()}
        jobs=self.fixture['queue'].list(); d=self.dialog(); normal=d.normal.canvas.transform()
        d.inspected.canvas.zoom(3); d.inspected.overlays.setChecked(False); d.full.canvas.native_size()
        self.assertEqual(d.normal.canvas.transform(),normal)
        self.assertEqual(before,{p.name:sha256(p) for p in self.snapshot.iterdir() if p.is_file()})
        self.assertEqual(jobs,self.fixture['queue'].list()); self.assertFalse(self.fixture['queue'].enabled())

    def test_original_view_handles_evidence_without_attached_final_verdict(self):
        d=self.dialog(); d.tabs.setCurrentIndex(0); d.full.canvas.native_size(); APP.processEvents()
        self.assertEqual(d.full.canvas.tracks[0]['status'],'판정 연결 전')
        self.assertFalse(d.full.canvas.grab().isNull())

    def test_missing_reference_is_explicit_without_loading_current_product_reference(self):
        root=self.root/'without-reference'; save_snapshot(root,self.fixture['frame'],self.fixture['result'],kind='synthetic')
        _,_,digest=load_snapshot(root); d=self.dialog(root,digest)
        self.assertTrue(d.normal.canvas.pixmap.isNull()); self.assertFalse(d.normal.native_button.isEnabled())
        self.assertIn('저장되어 있지 않습니다',d.reference_status.text())

    def test_tampered_source_and_wrong_snapshot_digest_are_rejected(self):
        with self.assertRaises(ValueError): EvidenceDialog(self.snapshot,self.identity,expected_digest='invalid')
        (self.snapshot/'reference.png').write_bytes(b'tampered')
        with self.assertRaises(ValueError): EvidenceDialog(self.snapshot,self.identity,expected_digest=self.digest)

    def test_wrong_object_identity_is_rejected(self):
        with self.assertRaises(ValueError): EvidenceDialog(self.snapshot,'wrong-object',expected_digest=self.digest)

    def test_vlm_detail_available_while_off_and_opens_selected_snapshot(self):
        viewer=AnalysisViewer(self.fixture['queue'],ROOT,run_worker=False); self.windows.append(viewer); viewer.timer.stop()
        self.assertTrue(viewer.evidence_button.isEnabled())
        row=self.fixture['queue'].get(self.fixture['job_id'])
        with patch('mes_vision.operation.evidence_dialog.show_evidence') as dialog:
            viewer.open_evidence()
            dialog.assert_called_once_with(viewer,row['snapshot_path'],self.identity,row['snapshot_digest'])

    def test_history_detail_opens_registered_snapshot(self):
        from mes_vision.operation.window import DesktopWindow
        owner=NS(record=({'path':str(self.snapshot),'object_id':self.identity,'digest':self.digest},))
        with patch('mes_vision.operation.evidence_dialog.show_evidence') as dialog:
            DesktopWindow.open_evidence(owner)
            dialog.assert_called_once_with(owner,str(self.snapshot),self.identity,self.digest)
        owner.record=None
        with self.assertRaises(ValueError): DesktopWindow.open_evidence(owner)

    def test_review_window_is_nonmodal_bounded_and_does_not_disable_owner_controls(self):
        from mes_vision.operation.evidence_dialog import show_evidence,close_evidence
        owner=QWidget(); self.windows.append(owner); owner.show()
        first=show_evidence(owner,self.snapshot,self.identity,self.digest); APP.processEvents()
        self.assertFalse(first.isModal()); self.assertTrue(owner.isEnabled()); self.assertIsNone(APP.activeModalWidget())
        second=show_evidence(owner,self.snapshot,self.identity,self.digest)
        self.assertFalse(first.isVisible()); self.assertIs(owner.evidence_window,second)
        close_evidence(owner); self.assertIsNone(owner.evidence_window)


if __name__=='__main__': unittest.main()

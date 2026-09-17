import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile,time,unittest,threading
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication,QWidget,QPushButton,QScrollArea
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.responsive import FlowLayout
from mes_vision.operation.storage import StorageService

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]

def wait(predicate):
    end=time.monotonic()+5
    while not predicate() and time.monotonic()<end: APP.processEvents(); time.sleep(.005)
    APP.processEvents()
    if not predicate(): raise AssertionError('UI work timeout')


class LaptopTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.w=DesktopWindow(ROOT,self.root,start_worker=False); self.w.timer.stop(); self.w.analysis.timer.stop()
        self.w.show(); APP.processEvents()

    def tearDown(self):
        wait(lambda:not self.w.tasks and not any(r.busy for r in (self.w.readiness_read,self.w.vlm_read,self.w.analysis.refresh_read)))
        self.w.closing=True; self.w.close(); APP.processEvents(); self.temp.cleanup()

    def test_compact_menu_tracks_both_navigation_directions(self):
        # w.nav is no longer shown: it is a hidden QListWidget kept as the selection
        # model that the grouped sidebar mirrors, so visibility is asserted on
        # w.sidebar. w.nav still drives the page stack and the compact selector.
        w=self.w; w.resize(960,640); APP.processEvents()
        self.assertTrue(w.compact_nav.isVisible()); self.assertFalse(w.sidebar.isVisible())
        w.compact_nav.setCurrentIndex(3); self.assertEqual(w.pages.currentIndex(),3)
        w.nav.setCurrentRow(1); self.assertEqual(w.compact_nav.currentIndex(),1)
        w.resize(1440,900); APP.processEvents()
        self.assertTrue(w.sidebar.isVisible()); self.assertFalse(w.compact_nav.isVisible())

    def test_all_pages_and_global_stop_fit_small_window_at_large_text(self):
        w=self.w; w.preferences.value['text_scale']=130; w.apply_preferences(); w.resize(960,640); APP.processEvents()
        for index in range(5):
            w.nav.setCurrentRow(index); APP.processEvents()
            self.assertLessEqual(w.width(),960); self.assertLessEqual(w.height(),640)
            self.assertTrue(w.global_stop.isVisible()); self.assertTrue(w.global_stop.isEnabled())
            self.assertIsInstance(w.pages.currentWidget(),QScrollArea)
            top=w.global_stop.mapTo(w,w.global_stop.rect().topLeft()); bottom=w.global_stop.mapTo(w,w.global_stop.rect().bottomRight())
            self.assertTrue(w.rect().contains(top)); self.assertTrue(w.rect().contains(bottom))

    def test_global_stop_uses_existing_stop_handler(self):
        with patch.object(DesktopWindow,'stop_robot') as stop:
            # Existing button is connected to the method captured during construction.
            other=DesktopWindow(ROOT,self.root/'other',start_worker=False)
            other.timer.stop(); other.analysis.timer.stop(); other.global_stop.click(); stop.assert_called_once()
            wait(lambda:not any(r.busy for r in (other.readiness_read,other.vlm_read,other.analysis.refresh_read)))
            other.closing=True; other.close(); other.deleteLater()

    def test_compact_readiness_starts_collapsed_and_respects_user_choice(self):
        w=self.w; w.resize(960,640); APP.processEvents(); self.assertFalse(w.readiness_toggle.isChecked())
        w.readiness_toggle.click(); self.assertTrue(w.readiness_toggle.isChecked())
        w.resize(1000,660); APP.processEvents(); self.assertTrue(w.readiness_toggle.isChecked())

    def test_filter_edits_do_not_query_until_enter_and_summary_uses_applied_values(self):
        w=self.w; w.nav.setCurrentRow(3)
        with patch.object(StorageService,'search',wraps=StorageService(w.store).search) as search:
            w.history_product.setText('part-A'); self.assertEqual(search.call_count,0)
            self.assertIn('변경',w.history_dirty.text()); self.assertNotIn('part-A',w.history_applied.text())
            QTest.keyClick(w.history_product,Qt.Key_Return); wait(lambda:not w.history_busy)
            self.assertEqual(search.call_count,1); self.assertIn('part-A',w.history_applied.text())
            w.history_product.setText('part-B'); self.assertIn('part-A',w.history_applied.text()); self.assertNotIn('part-B',w.history_applied.text())

    def test_reset_clears_hidden_filters_and_queries_once(self):
        w=self.w; w.history_product.setText('part'); w.history_query.setText('old')
        for field in (w.history_filter,w.history_defect,w.history_review,w.history_vlm,w.history_robot,w.history_storage): field.setCurrentIndex(1)
        w.history_period.setChecked(True)
        with patch.object(StorageService,'search',wraps=StorageService(w.store).search) as search:
            w.reset_history_filters(); wait(lambda:not w.history_busy); self.assertEqual(search.call_count,1)
        self.assertFalse(w.history_period.isChecked()); self.assertEqual(w.history_inputs()['text'],'')
        self.assertFalse(any(w.history_inputs()[key] for key in ('product','decision','defect','review','vlm','robot','archived')))
        self.assertIn('전체 조건',w.history_applied.text())

    def test_invalid_period_does_not_replace_applied_query(self):
        w=self.w; previous=w.history_search; w.history_period.setChecked(True); w.history_since.setDateTime(w.history_until.dateTime())
        with self.assertRaises(ValueError): w.search_history()
        self.assertEqual(w.history_search,previous)

    def test_new_query_clears_previous_evidence_and_empty_results_have_explanation(self):
        import numpy as np
        w=self.w; w.record=({'object_id':'old'},); w.saved_canvas.set_rgb(np.zeros((200,300,3),dtype=np.uint8))
        w.history_details.setPlainText('old evidence'); w.search_history(); wait(lambda:not w.history_busy)
        self.assertIsNone(w.record); self.assertTrue(w.saved_canvas.pixmap.isNull()); self.assertNotIn('old evidence',w.history_details.toPlainText())
        self.assertIn('조건에 맞는',w.history_summary.text())

    def test_failed_new_search_does_not_export_old_result(self):
        w=self.w; w.history_product.setText('new')
        with patch.object(StorageService,'search',side_effect=RuntimeError('query failed')):
            w.search_history(); wait(lambda:not w.history_busy)
        with self.assertRaises(ValueError): w.export_history()

    def test_late_failure_does_not_overwrite_current_query_summary(self):
        w=self.w; started=threading.Event(); release=threading.Event(); original=StorageService.search; calls=[]
        def slow(service,*args,**kwargs):
            calls.append(1)
            if len(calls)==1: started.set(); release.wait(3); raise RuntimeError('old failure')
            return original(service,*args,**kwargs)
        with patch.object(StorageService,'search',slow):
            w.search_history(); wait(started.is_set); w.history_product.setText('new'); w.search_history(); release.set()
            wait(lambda:len(calls)>=2 and not w.history_busy)
        self.assertNotIn('old failure',w.history_summary.text()); self.assertIn('new',w.history_applied.text())

    def test_flow_actions_wrap_without_overlap(self):
        widget=QWidget(); layout=FlowLayout(); widget.setLayout(layout)
        for i in range(7): layout.addWidget(QPushButton('Action '+str(i)))
        widget.resize(280,250); widget.show(); APP.processEvents()
        rects=[layout.itemAt(i).geometry() for i in range(layout.count())]
        self.assertGreater(len({r.y() for r in rects}),1)
        for i,a in enumerate(rects):
            self.assertGreaterEqual(a.left(),0); self.assertLessEqual(a.right(),widget.width())
            for b in rects[i+1:]: self.assertFalse(a.intersects(b))
        widget.close(); widget.deleteLater()


if __name__=='__main__': unittest.main()

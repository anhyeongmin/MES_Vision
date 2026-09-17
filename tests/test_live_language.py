import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json,pickle,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
from PySide6.QtCore import QTimer
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication,QLabel,QComboBox,QLineEdit,QTableWidget,QTableWidgetItem,QDialogButtonBox
from mes_vision.i18n import tr,trf,text_join,language,set_language,LANGUAGES,qt_button_text
from mes_vision.qt_i18n import ui_text,apply_language,refresh_translations
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.preferences import Preferences
from mes_vision.operation.preferences_dialog import PreferencesDialog

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]


def wait(predicate):
    end=time.monotonic()+5
    while not predicate() and time.monotonic()<end: APP.processEvents(); time.sleep(.005)
    if not predicate(): raise AssertionError('UI work timeout')


class TextBindingTests(unittest.TestCase):
    def setUp(self): self.previous=language(); set_language('ko'); self.widgets=[]
    def tearDown(self):
        for widget in self.widgets: widget.close(); widget.deleteLater()
        apply_language(APP,self.previous); APP.processEvents()
    def keep(self,w): self.widgets.append(w); return w

    def test_nested_text_roundtrip_preserves_user_values_and_numeric_formats(self):
        source=trf('{v0}  ·  버전 {v1}',v0='정상 / English / ไทย',v1=42)
        label=self.keep(ui_text(QLabel,text_join('\n',[tr('검사 시작'),source,trf('  점수: {v0:.4g}',v0=.125)])))
        for code in ('en','zh-CN','th','ko'):
            apply_language(APP,code)
            self.assertEqual(label.text(),str(tr('검사 시작')+'\n'+source+'\n'+trf('  점수: {v0:.4g}',v0=.125)))
            self.assertIn('정상 / English / ไทย',label.text()); self.assertIn('0.125',label.text())

    def test_raw_user_text_matching_catalog_is_never_translated(self):
        label=self.keep(ui_text(QLabel,'검사 시작'))
        edit=self.keep(QLineEdit('정상'))
        ui_text(edit.setPlaceholderText,tr('품목'))
        apply_language(APP,'th')
        self.assertEqual(label.text(),'검사 시작'); self.assertEqual(edit.text(),'정상')
        self.assertEqual(edit.placeholderText(),str(tr('품목')))

    def test_later_unbound_write_and_clear_do_not_restore_old_text(self):
        label=self.keep(ui_text(QLabel,tr('정상'))); label.setText('불량')
        other=self.keep(ui_text(QLabel,tr('검사 시작'))); ui_text(other.setToolTip,tr('품목')); ui_text(other.clear)
        apply_language(APP,'en')
        self.assertEqual(label.text(),'불량'); self.assertEqual(other.text(),''); self.assertEqual(other.toolTip(),str(tr('품목')))

    def test_combo_selection_identifiers_and_manual_entry_survive_without_signals(self):
        combo=self.keep(QComboBox()); combo.setEditable(True)
        ui_text(combo.addItem,tr('미등록'),'saved-id'); ui_text(combo.addItem,tr('등록'),'other-id')
        combo.setCurrentIndex(0); combo.setEditText('manual-정상')
        signals=QSignalSpy(combo.currentTextChanged); edits=QSignalSpy(combo.lineEdit().textChanged)
        apply_language(APP,'en')
        self.assertEqual(combo.itemText(0),str(tr('미등록'))); self.assertEqual(combo.currentData(),'saved-id')
        self.assertEqual(combo.currentText(),'manual-정상'); self.assertEqual(signals.count(),0); self.assertEqual(edits.count(),0)
        ui_text(combo.clear); ui_text(combo.addItem,'미등록','raw')
        apply_language(APP,'th'); self.assertEqual(combo.itemText(0),'미등록')

    def test_table_header_cells_and_selection_change_in_place(self):
        table=self.keep(QTableWidget(2,2)); ui_text(table.setHorizontalHeaderLabels,[tr('품목'),tr('기본 판정')])
        table.setItem(0,0,ui_text(QTableWidgetItem,'정상')); table.setItem(0,1,ui_text(QTableWidgetItem,tr('불량')))
        table.selectRow(0); item=table.item(0,1); changed=QSignalSpy(table.itemSelectionChanged)
        for code in ('en','zh-CN','th','ko'):
            apply_language(APP,code)
            self.assertEqual(table.horizontalHeaderItem(0).text(),str(tr('품목')))
            self.assertIs(table.item(0,1),item); self.assertEqual(item.text(),str(tr('불량')))
            self.assertEqual(table.item(0,0).text(),'정상'); self.assertEqual(table.currentRow(),0)
        self.assertEqual(changed.count(),0)

    def test_queued_source_and_module_constant_render_in_current_language(self):
        from mes_vision.operation.widgets import STATES
        source=pickle.loads(pickle.dumps(tr('검사 시작')))
        label=self.keep(QLabel()); apply_language(APP,'en'); ui_text(label.setText,source)
        self.assertEqual(label.text(),'Start inspection')
        ui_text(label.setText,STATES['NG']); self.assertEqual(label.text(),str(tr('불량')))
        self.assertIsInstance(json.loads(json.dumps(source,ensure_ascii=False)),str)

    def test_deleted_widgets_and_repeated_switches_are_safe(self):
        import shiboken6
        label=ui_text(QLabel,tr('품목')); shiboken6.delete(label)
        for code in LANGUAGES: apply_language(APP,code)


class LiveLanguageTests(unittest.TestCase):
    def setUp(self):
        self.previous=language(); set_language('ko'); self.temp=tempfile.TemporaryDirectory()
        self.w=DesktopWindow(ROOT,Path(self.temp.name),start_worker=False)
        self.w.timer.stop(); self.w.analysis.timer.stop()
        wait(lambda:not self.w.tasks and not any(r.busy for r in (self.w.readiness_read,self.w.vlm_read,self.w.analysis.refresh_read)))
    def tearDown(self):
        self.w.camera=self.w.engine=self.w.robot=self.w.worker=None; self.w.running=False
        self.w.close(); self.w.deleteLater(); APP.processEvents(); apply_language(APP,self.previous); self.temp.cleanup()
    def switch(self,code): self.w.preferences.value['language']=code; self.w.apply_preferences()

    def test_all_pages_update_without_reconstruction_or_operational_side_effects(self):
        w=self.w; w.nav.setCurrentRow(3); w.history_query.setText('정상'); w.history_filter.setCurrentIndex(2)
        w.selected_id='track-1'; w.session='session-1'; w.running=True
        objects=[object() for _ in range(4)]; w.camera,w.engine,w.robot,w.worker=objects
        w.tracks=[{'track_id':'track-1','status':'NG'}]; tracks=w.tracks; record=object(); w.record=record
        pages=[w.pages.widget(i) for i in range(5)]; index=w.history_filter.currentIndex()
        cached=w.history_search; changes=QSignalSpy(w.history_filter.currentIndexChanged)
        with patch.object(w,'refresh_history',side_effect=AssertionError('unexpected query')), patch.object(w,'pause_inspection',side_effect=AssertionError('unexpected stop')):
            for code in ('en','zh-CN','th','ko'):
                self.switch(code)
                self.assertEqual(w.start.text(),str(tr('검사 시작'))); self.assertEqual(w.settings_button.text(),str(tr('환경설정')))
                self.assertEqual(w.nav.item(3).text(),str(tr('4  검사 이력'))); self.assertEqual(w.compact_nav.itemText(3),w.nav.item(3).text())
                self.assertEqual(w.products_table.horizontalHeaderItem(0).text(),str(tr('품목')))
                self.assertEqual(w.analysis.table.horizontalHeaderItem(1).text(),str(tr('기본 판정')))
                self.assertEqual(w.pages.currentIndex(),3); self.assertEqual(w.history_query.text(),'정상')
                self.assertEqual(w.history_filter.currentIndex(),index); self.assertEqual(w.history_search,cached)
                self.assertEqual([w.pages.widget(i) for i in range(5)],pages)
                self.assertEqual([w.camera,w.engine,w.robot,w.worker],objects); self.assertTrue(w.running)
                self.assertIs(w.tracks,tracks); self.assertIs(w.record,record); self.assertEqual(w.selected_id,'track-1')
        self.assertEqual(changes.count(),0); w.session=None

    def test_actual_preferences_save_applies_and_cancel_does_not(self):
        def choose():
            dialog=QApplication.activeModalWidget(); self.assertIsInstance(dialog,PreferencesDialog)
            dialog.language.setCurrentIndex(dialog.language.findData('th')); dialog.save()
        self.w.show(); QTimer.singleShot(0,choose); self.w.open_preferences()
        self.assertEqual(language(),'th'); self.assertEqual(self.w.start.text(),str(tr('검사 시작')))
        self.assertEqual(Preferences(self.w.runtime).value['language'],'th')
        def cancel():
            dialog=QApplication.activeModalWidget(); dialog.language.setCurrentIndex(dialog.language.findData('en')); dialog.reject()
        QTimer.singleShot(0,cancel); self.w.open_preferences()
        self.assertEqual(language(),'th'); self.assertEqual(Preferences(self.w.runtime).value['language'],'th')

    def test_open_preferences_form_tabs_standard_buttons_and_drafts_update(self):
        dialog=PreferencesDialog(Preferences(self.w.runtime),self.w.runtime,self.w)
        dialog.scale.setCurrentIndex(2); dialog.start_page.setCurrentIndex(4)
        dialog.show(); APP.processEvents()
        for code in ('en','zh-CN','th','ko'):
            self.switch(code); APP.processEvents()
            self.assertEqual(dialog.windowTitle(),str(tr('환경설정')))
            self.assertEqual(dialog.buttons.button(QDialogButtonBox.Save).text().replace('&',''),qt_button_text('Save').replace('&',''))
            self.assertEqual(dialog.scale.currentData(),130); self.assertEqual(dialog.start_page.currentData(),4)
            self.assertEqual(dialog.start_page.itemText(4),str(tr('VLM 추가 분석')))
        dialog.close(); dialog.deleteLater()

    def test_modeless_evidence_keeps_images_and_zoom(self):
        from mes_vision.vlm.fixtures import make_vlm_fixture
        from mes_vision.operation.evidence_dialog import EvidenceDialog
        fixture=make_vlm_fixture(Path(self.temp.name)/'fixture'); row=fixture['queue'].list()[0]
        dialog=EvidenceDialog(row['snapshot_path'],row['object_id'],expected_digest=row['snapshot_digest'],parent=self.w)
        dialog.show(); APP.processEvents()
        # Every image canvas remains the same object with the same view and pixels.
        from mes_vision.operation.image_view import ImageCanvas
        canvases=dialog.findChildren(ImageCanvas)
        for canvas in canvases:
            if not canvas.pixmap.isNull(): canvas.zoom(2)
        states=[(c,c.pixmap.cacheKey(),c._scale,c._center) for c in canvases]
        for code in ('en','zh-CN','th','ko'):
            self.switch(code)
            self.assertEqual([(c,c.pixmap.cacheKey(),c._scale,c._center) for c in canvases],states)
        dialog.close(); dialog.deleteLater()

    def test_vlm_evidence_labels_retranslate_without_reload_or_rewriting_observation(self):
        from mes_vision.vlm.fixtures import make_vlm_fixture
        from mes_vision.vlm.viewer import AnalysisViewer
        fixture=make_vlm_fixture(Path(self.temp.name)/'fixture'); queue=fixture['queue']
        queue.set_enabled(True); queue.retry(fixture['job_id']); job=queue.claim('language-test')
        observation='검사 시작 / 정상 / VLM original observation'
        queue.finish(job['id'],job['token'],'COMPLETED',result={'analysis':{'observation':observation,'needs_review':True},'limitations':[]})
        viewer=AnalysisViewer(queue,ROOT,run_worker=False)
        viewer.timer.stop(); source=viewer.reasons.toPlainText(); key=viewer.render_version; image=viewer.picture.pixmap().cacheKey()
        with patch('mes_vision.vlm.viewer.load_snapshot',side_effect=AssertionError('unexpected evidence read')):
            self.switch('en')
            self.assertNotEqual(viewer.reasons.toPlainText(),source)
            self.assertIn(observation,viewer.analysis.toPlainText()); self.assertEqual(viewer.table.item(0,3).text(),observation)
            self.assertEqual(viewer.render_version,key); self.assertEqual(viewer.picture.pixmap().cacheKey(),image)
            self.switch('ko'); self.assertEqual(viewer.reasons.toPlainText(),source)
        viewer.close(); viewer.deleteLater()


if __name__=='__main__': unittest.main()

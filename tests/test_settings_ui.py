import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from copy import deepcopy
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication,QPushButton
from mes_vision.operation.catalog import default_equipment,new_product,OperationStore
from mes_vision.operation.setup_dialogs import EquipmentDialog
from mes_vision.operation.product_dialog import ProductDialog
from mes_vision.operation.settings_ui import changes,SettingsHistoryDialog
from mes_vision.operation.robot_service import geometry_context

APP=QApplication.instance() or QApplication([])
ROOT=Path(__file__).resolve().parents[1]


class SettingsUiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.windows=[]
        self.equipment=default_equipment(); self.equipment['camera'].update(driver='d405',width=640,height=480,acquisition_revision='existing')

    def tearDown(self):
        for w in self.windows: w.close(); w.deleteLater()
        APP.processEvents(); self.temp.cleanup()

    def equipment_dialog(self):
        d=EquipmentDialog(self.equipment); self.windows.append(d); d.show(); APP.processEvents(); return d

    def product_dialog(self,product=None):
        product=product or new_product(); product['workspace_id']='main'
        d=ProductDialog(product,self.equipment,ROOT,self.root); self.windows.append(d); d.show(); APP.processEvents(); return d

    def test_advanced_tabs_hidden_by_default_and_toggle_preserves_values(self):
        d=self.equipment_dialog(); tabs=d.pages.tabs
        self.assertEqual(sum(tabs.isTabVisible(i) for i in range(tabs.count())),3)
        d.pages.advanced.setChecked(True); self.assertTrue(all(tabs.isTabVisible(i) for i in range(tabs.count())))
        d.tracking['motion_px'].setValue(9); tabs.setCurrentIndex(d.pages.advanced_pages[-1]); d.pages.advanced.setChecked(False)
        self.assertEqual(tabs.currentIndex(),0); self.assertEqual(d.collect()['tracking']['motion_px'],9)
        self.assertTrue(d.camera_text['acquisition_revision'].isReadOnly())

    def test_camera_revision_is_automatic_stable_and_reverts_with_original_settings(self):
        d=self.equipment_dialog(); old=deepcopy(self.equipment)
        self.assertEqual(d.collect()['camera']['acquisition_revision'],'existing')
        d.resolution.setCurrentIndex(d.resolution.findData('1280x720'))
        first=d.collect(); second=d.collect(); self.assertEqual(first,second)
        self.assertNotEqual(first['camera']['acquisition_revision'],'existing')
        d.resolution.setCurrentIndex(d.resolution.findData('640x480'))
        self.assertEqual(d.collect()['camera']['acquisition_revision'],'existing'); self.assertEqual(self.equipment,old)

    def test_unrelated_settings_do_not_rotate_capture_revision(self):
        d=self.equipment_dialog(); d.mm_width.setValue(700); d.tracking['motion_px'].setValue(8)
        self.assertEqual(d.collect()['camera']['acquisition_revision'],'existing')
        d.serial.setEditText('new-camera'); self.assertNotEqual(d.collect()['camera']['acquisition_revision'],'existing')

    def test_changed_capture_cannot_reuse_old_calibration_or_mutate_failed_draft(self):
        self.equipment['camera'].update(serial='camera-A',mount_revision='mount-A',robot_base_id='base-A',tool_frame_id='tool-A')
        self.equipment['calibration']='existing.json'; d=self.equipment_dialog()
        context=geometry_context(self.equipment,{'width':640,'height':480,'transformations':[]})
        d.resolution.setCurrentIndex(d.resolution.findData('1280x720'))
        with patch('mes_vision.calibration.load',return_value=NS(ready=True,data={'specification':{'kind':'real'}})),patch('mes_vision.calibration.spec_from_dict',return_value=NS(context=context)):
            d.save()
        self.assertEqual(d.result(),0); self.assertIn('보정이 다릅니다',d.message.text())
        self.assertEqual(d.value,self.equipment)

    def test_exposure_and_fixed_count_controls_follow_modes(self):
        d=self.equipment_dialog(); self.assertFalse(d.exposure.isEnabled()); d.auto_exposure.setChecked(False); self.assertTrue(d.exposure.isEnabled())
        p=self.product_dialog(); self.assertFalse(p.expected.isEnabled()); p.count_mode.setCurrentIndex(1); self.assertTrue(p.expected.isEnabled())

    def test_product_hidden_unset_values_stay_unregistered_on_basic_save(self):
        product=new_product(); product['workspace_id']='main'; original=deepcopy(product); d=self.product_dialog(product)
        self.assertEqual(d.collect(),original)
        d.name.setText('changed'); result=d.collect()
        self.assertEqual(result['quality'],original['quality']); self.assertEqual(result['grasp'],original['grasp'])
        self.assertEqual(sum(d.pages.tabs.isTabVisible(i) for i in range(d.pages.tabs.count())),2)

    def test_explicit_zero_is_distinct_from_unregistered(self):
        d=self.product_dialog(); self.assertIsNone(d.collect()['quality']['blur_min'])
        d.blur.setValue(0); d.grasp_z.setValue(0)
        self.assertEqual(d.collect()['quality']['blur_min'],0); self.assertEqual(d.collect()['grasp']['plane_z_mm'],0)

    def test_validated_threshold_or_grasp_change_requires_a_new_revision(self):
        product=new_product(); product['quality'].update(version='q1',blur_min=1); product['grasp'].update(version='g1',plane_z_mm=20)
        d=self.product_dialog(product); d.blur.setValue(2)
        with self.assertRaises(ValueError): d.collect()
        d.quality_version.setText('q2'); d.grasp_z.setValue(21)
        with self.assertRaises(ValueError): d.collect()
        d.grasp_version.setText('g2'); self.assertEqual(d.collect()['grasp']['plane_z_mm'],21)

    def test_preview_changes_are_read_only_and_hidden_when_draft_changes(self):
        d=self.equipment_dialog(); d.mm_width.setValue(500)
        next(b for b in d.findChildren(QPushButton) if b.text()=='변경 내용 확인').click()
        self.assertTrue(d.change_preview.isVisible()); self.assertIn('500',d.change_preview.toPlainText())
        self.assertEqual(d.value['workspace']['width_mm'],None)
        d.mm_width.setValue(600); self.assertFalse(d.change_preview.isVisible())

    def test_diff_omits_store_version_but_keeps_criteria_versions(self):
        rows=changes({'version':1,'quality':{'version':'q1'}},{'version':2,'quality':{'version':'q2'}})
        self.assertEqual(len(rows),1); self.assertEqual(rows[0][1:],('q1','q2'))

    def test_history_uses_saved_versions_paginates_and_never_writes(self):
        store=OperationStore(self.root/'store'); e=default_equipment()
        e['camera']['driver']='d405'
        for i in range(52): e['camera']['fps']=15 if i%2 else 30; e=store.save_equipment(e)
        before=store.equipment(); d=SettingsHistoryDialog(store,'equipment'); self.windows.append(d)
        self.assertEqual(len(d.rows),50); self.assertTrue(d.next.isEnabled()); self.assertIn('초당 프레임',d.details.toPlainText())
        d.move(1); self.assertEqual(len(d.rows),2); self.assertFalse(d.next.isEnabled()); self.assertEqual(store.equipment(),before)

    def test_product_history_is_scoped_to_selected_product(self):
        store=OperationStore(self.root/'store'); a=store.save_product(new_product('A')); b=store.save_product(new_product('B'))
        a['name']='A changed'; store.save_product(a)
        d=SettingsHistoryDialog(store,'product',a['id']); self.windows.append(d)
        self.assertEqual(len(d.rows),2); self.assertIn('A changed',d.details.toPlainText()); self.assertNotIn(b['id'],d.details.toPlainText())


if __name__=='__main__': unittest.main()

import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import json
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication, QWidget, QMessageBox
from mes_vision.training.data import write_json,read_json,sha256
from mes_vision.operation.catalog import default_equipment,OperationStore
from mes_vision.operation.acquisition import configure_equipment,acquisition_config_path
from mes_vision.operation.lens_binding import plan_binding,bound_equipment,persist_binding
from mes_vision.station.equipment_dialog import StationEquipmentDialog

APP=QApplication.instance() or QApplication([])

class BindingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(ignore_cleanup_errors=True);self.root=Path(self.tmp.name)
        self.lens=self.root/'lens.json';write_json(self.lens,dict(image_size=[1280,720],K=[[900,0,640],[0,900,360],[0,0,1]],D=[0,0,0,0,0]))
        self.config=self.root/'configs/camera/acquisition.json';self.config.parent.mkdir(parents=True)
        write_json(self.config,dict(schema_version=1,processing=dict(kind='undistorted_center_square_v1',sensor_size=[1280,720],calibration_file='lens.json',calibration_sha256=sha256(self.lens),camera_serial='uvc:old')))
        self.config_bytes=self.config.read_bytes();self.lens_bytes=self.lens.read_bytes()
        self.store=OperationStore(self.root/'runtime-test')
        self.candidate=default_equipment();self.candidate['camera']['serial']='uvc:new'
    def tearDown(self):self.tmp.cleanup()
    def plan(self):return plan_binding(self.root,self.candidate['camera'],confirmed=True)
    def test_rebind_preserves_optics_and_survives_reload(self):
        with self.assertRaisesRegex(ValueError,'another camera'):configure_equipment(self.root,self.candidate)
        p=self.plan();c=bound_equipment(self.root,self.candidate,p)
        self.assertEqual(acquisition_config_path(self.root),self.config)
        saved=persist_binding(self.root,c,p,self.store)
        loaded=configure_equipment(self.root,self.store.equipment())
        self.assertEqual(loaded['camera']['inspection_acquisition']['camera_serial'],'uvc:new')
        self.assertEqual(loaded['camera']['inspection_acquisition']['calibration_sha256'],sha256(self.lens))
        self.assertEqual(loaded['camera']['acquisition_revision'],p['acquisition_revision'])
        self.assertEqual(self.config.read_bytes(),self.config_bytes);self.assertEqual(self.lens.read_bytes(),self.lens_bytes)
        self.assertIsNone(saved['calibration']);self.assertEqual(saved['workspace']['validation_reference'],'')
        self.assertEqual(len(list((self.root/'artifacts/operation/camera-binding-history').iterdir())),1)
        wrong=deepcopy(saved);wrong['camera']['serial']='uvc:third'
        with self.assertRaisesRegex(ValueError,'another camera'):configure_equipment(self.root,wrong)
    def test_confirmation_size_and_hash_guards(self):
        with self.assertRaises(ValueError):plan_binding(self.root,self.candidate['camera'])
        self.candidate['camera']['width']=640
        with self.assertRaises(ValueError):self.plan()
        self.candidate['camera']['width']=1280;self.lens.write_text('{}')
        with self.assertRaises(ValueError):self.plan()
        self.assertFalse((self.root/'artifacts/operation/camera-acquisition.json').exists())
    def test_changed_selection_and_stale_plan_rejected(self):
        p=self.plan();other=deepcopy(self.candidate);other['camera']['serial']='uvc:third'
        with self.assertRaises(ValueError):bound_equipment(self.root,other,p)
        self.config.write_bytes(self.config_bytes+b' ')
        with self.assertRaises(ValueError):persist_binding(self.root,self.candidate,p,self.store)
        self.assertFalse((self.root/'artifacts/operation/camera-acquisition.json').exists())
    def test_database_failure_rolls_back_override(self):
        p=self.plan()
        with patch.object(self.store,'save_equipment',side_effect=RuntimeError('DB conflict')):
            with self.assertRaisesRegex(RuntimeError,'DB conflict'):persist_binding(self.root,self.candidate,p,self.store)
        self.assertEqual(acquisition_config_path(self.root),self.config)
        self.assertEqual(self.config.read_bytes(),self.config_bytes)
    def test_second_binding_failure_preserves_previous_override(self):
        saved=persist_binding(self.root,self.candidate,self.plan(),self.store)
        override=acquisition_config_path(self.root);previous=override.read_bytes()
        saved['camera']['serial']='uvc:third';p=plan_binding(self.root,saved['camera'],confirmed=True)
        with patch.object(self.store,'save_equipment',side_effect=RuntimeError('DB conflict')):
            with self.assertRaises(RuntimeError):persist_binding(self.root,saved,p,self.store)
        self.assertEqual(override.read_bytes(),previous)
    def test_dialog_cancel_and_confirm_are_not_early_writes(self):
        parent=QWidget();parent.root=self.root
        dialog=StationEquipmentDialog(self.candidate,parent=parent,discover=False)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.No):dialog.request_lens_binding()
        self.assertIsNone(dialog.lens_binding)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):dialog.request_lens_binding()
        self.assertIsNotNone(dialog.lens_binding,dialog.message.text())
        value=dialog.collect()
        self.assertEqual(value['camera']['inspection_acquisition']['camera_serial'],'uvc:new')
        self.assertEqual(acquisition_config_path(self.root),self.config)
        dialog.reject();self.assertEqual(acquisition_config_path(self.root),self.config)
        dialog.deleteLater();parent.deleteLater();APP.processEvents()

    def test_dialog_save_then_persist_uses_confirmed_binding(self):
        parent=QWidget();parent.root=self.root
        dialog=StationEquipmentDialog(self.candidate,parent=parent,discover=False)
        with patch.object(QMessageBox,'question',return_value=QMessageBox.Yes):dialog.request_lens_binding()
        with patch.object(dialog,'selected_mode_supported',return_value=True):dialog.save()
        self.assertEqual(dialog.result(),1,dialog.message.text())
        saved=dialog.persist(self.store)
        self.assertEqual(saved['camera']['inspection_acquisition']['camera_serial'],'uvc:new')
        self.assertEqual(acquisition_config_path(self.root),self.root/'artifacts/operation/camera-acquisition.json')
        dialog.deleteLater();parent.deleteLater();APP.processEvents()

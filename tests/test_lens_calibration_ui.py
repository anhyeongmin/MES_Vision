import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
import numpy as np
from PySide6.QtWidgets import QApplication,QWidget,QLabel
from mes_vision.operation.catalog import default_equipment,OperationStore
from mes_vision.operation.lens_calibration_dialog import LensCalibrationDialog
from mes_vision.training.data import write_json,read_json,sha256

APP=QApplication.instance() or QApplication([])

class CalibrationUITests(unittest.TestCase):
    def test_activate_backs_up_and_invalidates_old_pixel_mapping(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs/camera').mkdir(parents=True)
            cfg=root/'configs/camera/acquisition.json';write_json(cfg,{'old':True})
            owner=QWidget();owner.root=root;owner.store=OperationStore(root/'runtime')
            owner.equipment=default_equipment();owner.equipment['camera']['serial']='uvc:test'
            owner.equipment['workspace']['validation_reference']='old-approved';owner.refresh_equipment=Mock()
            studio=QWidget();studio.owner=owner;studio.equipment=owner.equipment;studio.camera=Mock()
            studio.reopen=Mock();studio.message=QLabel()
            d=LensCalibrationDialog(studio);d.timer.stop();d.folder.mkdir(parents=True)
            frame=NS(width=1280,height=720,session_id='session',transformations=(),acquisition_identity=None)
            studio.camera.get_latest_raw.return_value=(frame,time.monotonic())
            d.result=dict(quality_passed=True,image_size=[1280,720],K=[[900,0,640],[0,900,360],[0,0,1]],D=[[0,0,0,0,0]])
            d.path=d.folder/'calibration.json';write_json(d.path,d.result);d.confirm.setChecked(True)
            d.apply_profile()
            self.assertEqual(read_json(d.folder/'previous-acquisition.json'),{'old':True})
            self.assertEqual(owner.equipment['workspace']['validation_reference'],'')
            self.assertIsNone(owner.equipment['calibration'])
            self.assertEqual(owner.equipment['camera']['inspection_acquisition']['calibration_sha256'],sha256(d.path))
            studio.reopen.assert_called_once();d.deleteLater();studio.deleteLater();owner.deleteLater()

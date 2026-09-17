"""UVC negotiation, identity and freshness tests; never open physical devices."""
import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from dataclasses import replace
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch
import tempfile
import unittest
import cv2
import numpy as np
from PySide6.QtWidgets import QApplication
from mes_vision.inputs import Frame,SourceKind
from mes_vision.inputs.camera_identity import camera_uri,camera_driver
from mes_vision.operation.uvc_camera import UVCCamera,device_identity
from mes_vision.operation.catalog import default_equipment
from mes_vision.operation.setup_dialogs import EquipmentDialog
from mes_vision.operation.settings_ui import assign_acquisition_revision
from mes_vision.operation.quality import validate_equipment
from mes_vision.station.camera_port import CaptureGate,StationCameraPort
from mes_vision.station.evidence import save_capture
from mes_vision.station.worker import read_capture
from mes_vision.i18n import language
from mes_vision.qt_i18n import apply_language

APP=QApplication.instance() or QApplication([])


def usb_frame(sequence=1,stamp=100.,**kwargs):
    rgb=np.zeros((720,1280,3),np.uint8); rgb.setflags(write=False)
    frame=Frame(f's:{sequence}','s',sequence,SourceKind.UVC,camera_uri('uvc','uvc:TEST'),datetime.now(timezone.utc),rgb,
                encoded_size=(1280,720),is_live=True,host_read_completed_monotonic=stamp)
    return replace(frame,**kwargs)


class DriverTests(unittest.TestCase):
    def setUp(self):
        self.platform=patch('mes_vision.operation.uvc_camera.platform.system',return_value='Windows'); self.platform.start(); self.addCleanup(self.platform.stop)
        self.path='\\\\?\\usb#vid_1234&pid_5678#device-A'; self.identifier=device_identity(cv2.CAP_DSHOW,self.path)
        self.settings=default_equipment()['camera']; self.settings['serial']=self.identifier
        self.props={cv2.CAP_PROP_FRAME_WIDTH:640.,cv2.CAP_PROP_FRAME_HEIGHT:480.,cv2.CAP_PROP_FPS:30.,cv2.CAP_PROP_FOURCC:0.}
        self.cap=Mock(); self.cap.isOpened.return_value=True; self.cap.getBackendName.return_value='DSHOW'
        self.cap.set.side_effect=lambda key,value:self.props.update({key:value}) is None
        self.cap.get.side_effect=lambda key:-1. if key==cv2.CAP_PROP_AUTO_EXPOSURE else self.props.get(key,0.)
        self.cap.read.return_value=(True,np.full((720,1280,3),[1,2,3],np.uint8))
        self.cv=NS(**{name:getattr(cv2,name) for name in dir(cv2) if name.startswith('CAP_')})
        self.cv.VideoCapture=Mock(return_value=self.cap); self.cv.VideoWriter_fourcc=cv2.VideoWriter_fourcc
        self.enum=Mock(return_value=[NS(index=2,path=self.path,name='U20CAM',backend=cv2.CAP_DSHOW)])
        self.camera=UVCCamera(self.settings,cv=self.cv,enumerate_fn=self.enum,clock=lambda:100.)
        self.addCleanup(self.camera.close)

    def test_selected_device_and_actual_format_rgb_are_verified(self):
        self.camera.open(); self.cv.VideoCapture.assert_called_once_with(2,cv2.CAP_DSHOW)
        frame=self.camera.read()
        self.assertEqual(frame.rgb[0,0].tolist(),[3,2,1]); self.assertFalse(frame.rgb.flags.writeable)
        self.assertEqual(frame.source_uri,camera_uri('uvc',self.identifier)); self.assertIsNone(frame.media_time_seconds)
        self.assertIsNone(frame.captured_at_utc); self.assertEqual(frame.host_read_completed_monotonic,100.)
        self.assertEqual(self.camera.info['distortion'],'unverified')
        self.assertFalse(self.camera.info['auto_exposure_readback_available'])
        self.assertIn(unittest.mock.call(cv2.CAP_PROP_AUTO_EXPOSURE,1.),self.cap.set.call_args_list)

    def test_missing_selected_device_does_not_fall_back_to_webcam_zero(self):
        self.enum.return_value=[]
        with self.assertRaises(ValueError): self.camera.open()
        self.cv.VideoCapture.assert_not_called()

    def test_mjpeg_is_applied_after_size_changes_that_reset_directshow_format(self):
        original=self.cap.set.side_effect
        def setter(key,value):
            result=original(key,value)
            if key in (cv2.CAP_PROP_FRAME_WIDTH,cv2.CAP_PROP_FRAME_HEIGHT):
                self.props[cv2.CAP_PROP_FOURCC]=cv2.VideoWriter_fourcc(*'YUY2')
            return result
        self.cap.set.side_effect=setter
        self.camera.open()
        self.assertEqual(self.props[cv2.CAP_PROP_FOURCC],cv2.VideoWriter_fourcc(*'MJPG'))

    def test_device_index_reorder_during_open_is_rejected_and_released(self):
        self.enum.side_effect=[self.enum.return_value,[NS(index=0,path=self.path,name='U20CAM')]]
        with self.assertRaises(ValueError): self.camera.open()
        self.cap.release.assert_called_once()

    def test_silent_resolution_or_fps_fallback_is_rejected(self):
        original=self.cap.get.side_effect
        self.cap.get.side_effect=lambda key:15. if key==cv2.CAP_PROP_FPS else original(key)
        with self.assertRaises(ValueError): self.camera.open()
        self.cap.release.assert_called_once()

    def test_wrong_format_is_rejected(self):
        original=self.cap.get.side_effect
        self.cap.get.side_effect=lambda key:cv2.VideoWriter_fourcc(*'YUY2') if key==cv2.CAP_PROP_FOURCC else original(key)
        with self.assertRaises(ValueError): self.camera.open()

    def test_manual_exposure_accepts_negative_directshow_units(self):
        self.camera.settings.update(auto_exposure=False,exposure=-6.)
        self.camera.open(); self.assertEqual(self.props[cv2.CAP_PROP_EXPOSURE],-6.)
        self.assertIn(unittest.mock.call(cv2.CAP_PROP_AUTO_EXPOSURE,0.),self.cap.set.call_args_list)

    def test_failed_read_and_size_change_do_not_publish_frames(self):
        self.camera.open()
        for value in ((False,None),(True,np.zeros((480,640,3),np.uint8))):
            self.cap.read.return_value=value
            with self.assertRaises(ValueError): self.camera.read()


class FreshnessTests(unittest.TestCase):
    def setUp(self):
        self.request={'id':'r','kind':'capture_detail','issued_at':100.,'deadline':110.}
        self.profile={'camera_driver':'uvc','camera_serial':'uvc:TEST','camera_session':'s','image_size':[1280,720]}
    def gate(self): return CaptureGate(self.request,self.profile,.1,'UNIT-TEST-BOUND')

    def test_post_request_receipt_alone_is_insufficient(self):
        gate=self.gate()
        self.assertIsNone(gate.observe(usb_frame(1,100.02),100.03,100.03))
        proof=gate.observe(usb_frame(2,100.15),100.16,100.17)
        self.assertAlmostEqual(proof['acquired_at'],100.05)
        self.assertIn('host_read_completion',proof['timing_basis'])

    def test_stale_duplicate_future_or_renamed_exposure_metadata_rejected(self):
        frames=[usb_frame(stamp=99.5),usb_frame(stamp=101.),usb_frame(stamp=None),
                usb_frame(media_time_seconds=100.),usb_frame(source_uri=camera_uri('uvc','other')),
                usb_frame(session_id='reconnected'),usb_frame(transformations=('crop',))]
        for frame in frames:
            with self.subTest(frame=frame.metadata()),self.assertRaises(ValueError): self.gate().observe(frame,100.2,100.2)
        gate=self.gate(); gate.observe(usb_frame(2,100.2),100.2,100.2)
        self.assertIsNone(gate.observe(usb_frame(2,100.2),100.2,100.2))
        with self.assertRaises(ValueError): gate.observe(usb_frame(1,100.3),100.3,100.3)

    def test_usb_cannot_satisfy_a_legacy_realsense_profile(self):
        self.profile.pop('camera_driver')
        with self.assertRaises(ValueError): self.gate().observe(usb_frame(1,100.2),100.2,100.2)

    def test_capture_save_and_reload_preserves_identity_and_read_time(self):
        with tempfile.TemporaryDirectory() as root:
            frame=usb_frame(1,100.2); receipt=save_capture(root,frame,camera_serial='uvc:TEST',acquired_at=100.1)
            loaded=read_capture(receipt)
            self.assertEqual(loaded.metadata(),frame.metadata()); np.testing.assert_array_equal(loaded.rgb,frame.rgb)
            receipt['camera_serial']='other'
            with self.assertRaises(ValueError): read_capture(receipt)

    def test_cancel_capture_keeps_preview_and_discards_pending_selection(self):
        camera=Mock(); camera.get_latest.return_value=(usb_frame(1,100.02),100.02)
        with tempfile.TemporaryDirectory() as root:
            port=StationCameraPort(camera,root,{'max_frame_age_seconds':.1,'timing_validation_reference':'UNIT-TEST'})
            port.submit(self.request,self.profile); self.assertIsNone(port.tick(100.02)); port.cancel()
            self.assertIsNone(port.gate); camera.close.assert_not_called(); port.close()


class ControlsTests(unittest.TestCase):
    def dialog(self,e=None):
        d=EquipmentDialog(e or default_equipment()); self.addCleanup(d.deleteLater); return d
    def test_default_usb_native_resolution_and_format_candidates(self):
        d=self.dialog(); self.assertEqual(d.driver.currentData(),'uvc')
        self.assertEqual(d.selected_resolution(),(1280,720)); self.assertEqual(d.fps.currentData(),30)
        d.pixel_format.setCurrentIndex(d.pixel_format.findData('YUY2'))
        self.assertEqual(d.fps.currentData(),10)
        d.resolution.setCurrentIndex(d.resolution.findData('800x600'))
        self.assertEqual(d.fps.currentData(),20); validate_equipment(d.collect())
    def test_legacy_settings_remain_realsense_and_switch_changes_revision(self):
        e=default_equipment(); e['camera'].pop('driver'); e['camera'].pop('pixel_format'); e['camera']['serial']='OLD'
        d=self.dialog(e); self.assertEqual(d.driver.currentData(),'d405')
        self.assertEqual(camera_driver(e['camera']),'d405')
        d.driver.setCurrentIndex(d.driver.findData('uvc')); self.assertEqual(d.serial.identifier(),'')
        self.assertNotEqual(d.collect()['camera']['acquisition_revision'],e['camera']['acquisition_revision'])
    def test_transfer_format_invalidates_acquisition_revision(self):
        old=default_equipment()['camera']; new=dict(old,pixel_format='YUY2',fps=10)
        assign_acquisition_revision(old,new,'changed'); self.assertEqual(new['acquisition_revision'],'changed')
    def test_new_controls_translate_live_without_resetting_selection(self):
        original=language(); d=self.dialog(); self.addCleanup(lambda:apply_language(APP,original))
        d.show(); APP.processEvents()
        for locale in ('en','zh-CN','th','ko'):
            apply_language(APP,locale); APP.processEvents()
            self.assertEqual(d.driver.currentData(),'uvc'); self.assertEqual(d.pixel_format.currentData(),'MJPG')
        d.close()

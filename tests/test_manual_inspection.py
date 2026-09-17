import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import numpy as np
import cv2
from PIL import Image
from PySide6.QtWidgets import QApplication
from mes_vision.inputs.sources import Frame, SourceKind
from mes_vision.training.data import write_json, read_json
from mes_vision.station.manual_capture import Rectifier, fresh_after, save_capture
from mes_vision.station.manual_dialog import ManualInspectionDialog
from mes_vision.station.photo_inspection import run_photos
from test_photo_inspection import Registry

ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])


class ManualTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)
        self.cal=self.path/'cal.json'
        self.k=np.array([[90.,0,80],[0,90,50],[0,0,1]])
        self.d=np.array([-.12,.01,0,0,0])
        write_json(self.cal,dict(K=self.k.tolist(),D=self.d.tolist(),image_size=[160,100]))
        self.rect=Rectifier(self.cal)
        yy,xx=np.mgrid[:100,:160]
        rgb=np.stack((xx,yy,(xx+yy)//2),axis=-1).astype(np.uint8)
        self.frame=Frame('f','s',1,SourceKind.UVC,'test',datetime.now(timezone.utc),rgb,is_live=True)

    def test_full_rectify_before_crop_and_archive_provenance(self):
        full,crop,box=self.rect.apply(self.frame.rgb)
        expected=cv2.undistort(self.frame.rgb,self.k,self.d,None,self.k)
        np.testing.assert_allclose(full,expected,atol=1)
        self.assertEqual(box,(30,0,130,100))
        np.testing.assert_array_equal(crop,full[:,30:130])
        image=save_capture(self.path/'capture',self.frame,self.rect,True,dict(target='2'))
        np.testing.assert_array_equal(np.array(Image.open(image)),crop)
        meta=read_json(image.parent/'capture.json')
        self.assertEqual(meta['target'],'2'); self.assertFalse(meta['robot_commands_enabled'])
        self.assertEqual(meta['crop_xyxy'],list(box)); self.assertEqual(len(meta['files']),3)

    def test_wrong_resolution_and_double_rectification_rejected(self):
        with self.assertRaises(ValueError): self.rect.apply(np.zeros((50,80,3),np.uint8))
        from dataclasses import replace
        with self.assertRaises(ValueError): save_capture(self.path/'bad',replace(self.frame,transformations=('corrected',)),self.rect,True,{})
        self.assertFalse((self.path/'bad').exists())

    def test_freshness(self):
        self.assertFalse(fresh_after(self.frame,9,10,10.1))
        self.assertFalse(fresh_after(self.frame,10,10,11))
        self.assertFalse(fresh_after(self.frame,12,10,11))
        self.assertTrue(fresh_after(self.frame,10.1,10,10.2))

    def dialog(self):
        camera=Mock(); camera.disposed=False; camera.stopping.is_set.return_value=False
        camera.command.return_value='token'; camera.replies={}; camera.get_latest.return_value=(self.frame,100.)
        dialog=ManualInspectionDialog(ROOT,self.path/'runtime',{'driver':'uvc'},camera=camera)
        dialog.timer.stop(); dialog.rectifier=self.rect
        self.addCleanup(dialog.deleteLater)
        return dialog,camera

    def test_arrival_waits_profile_ack_settle_and_new_frame_cancel_keeps_preview(self):
        dialog,camera=self.dialog(); dialog.run_capture=Mock()
        with patch('mes_vision.station.manual_dialog.time.monotonic',return_value=100.):
            dialog.arrive('overview'); dialog.tick()
        self.assertIsNone(dialog.pending['not_before']); dialog.run_capture.assert_not_called()
        camera.replies['token']={'result':{'applied_at':100.}}
        with patch('mes_vision.station.manual_dialog.time.monotonic',return_value=100.1): dialog.tick()
        self.assertAlmostEqual(dialog.pending['not_before'],101.6)
        with patch('mes_vision.station.manual_dialog.time.monotonic',return_value=101.7): dialog.tick()
        dialog.run_capture.assert_not_called()
        camera.get_latest.return_value=(self.frame,101.7)
        with patch('mes_vision.station.manual_dialog.time.monotonic',return_value=101.8): dialog.tick()
        dialog.run_capture.assert_called_once()
        self.assertTrue((dialog.cycle/'bundle.json').exists())
        dialog.pending={'role':'detail'}; dialog.stop(); self.assertIsNone(dialog.pending)
        camera.stopping.set.assert_not_called()

    def test_full_to_selected_detail_and_click_results_without_robot(self):
        dialog,camera=self.dialog(); dialog.square.setChecked(False)
        dialog.cycle=self.path/'cycle'; dialog.cycle.mkdir()
        image=self.path/'input.png'; Image.fromarray(self.frame.rgb).save(image)
        bundle=dict(schema_version=1,purpose='saved_photo_inspection',product_id='ASH',production_ready=False,
                    models=dict(overview={},objects={},defects={}))
        from mes_vision.training.data import sha256
        def complete(role,count,target=None):
            output=self.path/(role+'-result')
            request=dict(output=str(output),runtime=str(self.path/'runtime'),bundle='unused',image_kind='real',
                images=[dict(path=str(image),sha256=sha256(image),role=role)])
            with patch('mes_vision.station.photo_inspection.load_bundle',return_value=bundle):
                run_photos(request,ROOT,registry=Registry(count=count))
            dialog.capture=dict(role=role,target=target); dialog.output=output; dialog.accept_result()
        complete('overview',2)
        self.assertEqual([o['manual_id'] for o in dialog.objects],['1','2'])
        dialog.select_object('2'); complete('detail',1,'2')
        self.assertEqual(set(dialog.details),{'2'}); self.assertIn('NG03',dialog.reasons.toPlainText())
        self.assertEqual(dialog.overview.canvas.selected_id,'2')
        dialog.select_object('1'); self.assertIn('옮긴 뒤',dialog.reasons.toPlainText())
        dialog.select_object('2'); self.assertIn('NG03',dialog.reasons.toPlainText())
        saved=read_json(dialog.cycle/'session.json'); self.assertFalse(saved['robot_commands_enabled'])
        previous=dialog.cycle; dialog.reset_cycle(); self.assertFalse(dialog.details); self.assertTrue((previous/'session.json').exists())
        camera.command.assert_not_called()

    def test_close_waits_own_camera_disposal(self):
        dialog,camera=self.dialog(); dialog.owns_camera=True
        dialog.reject(); camera.stopping.set.assert_called_once()
        self.assertTrue(dialog.closing)
        camera.disposed=True; dialog.tick(); self.assertFalse(dialog.timer.isActive())

    def test_cancelled_completion_cannot_attach_result(self):
        from PySide6.QtCore import QProcess
        dialog,camera=self.dialog(); child=Mock(); dialog.process=child
        dialog.accept_result=Mock()
        dialog.stop(); child.kill.assert_called_once()
        dialog.finished(0,QProcess.NormalExit)
        dialog.accept_result.assert_not_called(); self.assertIsNone(dialog.process)
        self.assertTrue(dialog.full_button.isEnabled())

    def test_resident_ready_and_matching_completion_keep_server_alive(self):
        dialog,camera=self.dialog(); child=Mock(); dialog.server=dialog.process=child
        dialog.server_directory=self.path/'worker';dialog.server_directory.mkdir()
        dialog.request_id=None
        write_json(dialog.server_directory/'state.json',dict(state='READY'))
        dialog.poll_server(); self.assertIsNone(dialog.process); self.assertIs(dialog.server,child)
        dialog.process=child;dialog.request_id='current';dialog.accept_result=Mock()
        write_json(dialog.server_directory/'state.json',dict(state='COMPLETED',id='old',elapsed_ms=10))
        dialog.poll_server();dialog.accept_result.assert_not_called()
        write_json(dialog.server_directory/'state.json',dict(state='COMPLETED',id='current',elapsed_ms=10))
        dialog.poll_server();dialog.accept_result.assert_called_once()
        self.assertIsNone(dialog.process);self.assertIs(dialog.server,child);child.kill.assert_not_called()
        dialog.reject();child.kill.assert_called_once();dialog.finished(0,0)


if __name__=='__main__': unittest.main()

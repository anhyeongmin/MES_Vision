"""Photo-domain, model identity, coordinate and cancellation regression checks."""
import json, os, sys, tempfile, time, unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
import numpy as np
from PIL import Image
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication
from mes_vision.inspection import Box,CheckResult,CheckStatus
from mes_vision.inspection.contracts import ModelRef,Detection,DetectionBatch,Finding
from mes_vision.inspection.model_registry import ModelHandle
from mes_vision.station.photo_inspection import run_photos,open_results,load_bundle
from mes_vision.operation.finding_display import object_overlays
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.training.data import write_json,read_json,sha256
from mes_vision.station.photo_dialog import PhotoInspectionDialog
from mes_vision.qt_i18n import apply_language

ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])


class Adapter:
    def __init__(self,role,owner):
        self.owner=owner; self.role=role
        self.model=ModelRef(role,'test','model','a'*64,'product_defects' if role=='defects' else 'product_objects')
        self.check_id='known_defects'
    def load(self): self.owner.loaded.append(self.role)
    def close(self): self.owner.closed.append(self.role)
    def detect(self,frame):
        detections=tuple(Detection(Box(20+i*60,20,70+i*60,80),.95,0,'ASH') for i in range(self.owner.count))
        return DetectionBatch(frame.frame_id,(frame.width,frame.height),detections,self.model,candidate_threshold=.2)
    def inspect(self,crop):
        self.owner.calls+=1
        findings=(Finding('NG03','NG03',.8,Box(5,7,15,17)),) if self.owner.defect else ()
        return CheckResult('known_defects',crop.frame_id,crop.object_id,CheckStatus.UNCERTAIN,self.model,findings,
            messages=('CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED',),candidate_threshold=.1,
            details={'class_codes':{'0':'NG03'}})


class Registry:
    def __init__(self,count=1,defect=True): self.count=count; self.defect=defect; self.calls=0; self.loaded=[]; self.closed=[]
    def create(self,spec,role):
        resource=Adapter(role,self); return ModelHandle(resource,resource,lambda:None)


class PhotoTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root=Path(self.tmp.name)
        self.path=self.root/'photo.png'; Image.fromarray(np.full((100,160,3),160,np.uint8)).save(self.path)
        self.bundle={'schema_version':1,'purpose':'saved_photo_inspection','product_id':'ASH','production_ready':False,
            'models':{'overview':{},'objects':{},'defects':{}}}
        self.request={'output':str(self.root/'out'),'runtime':str(self.root/'runtime'),'bundle':'unused','image_kind':'synthetic',
            'images':[{'role':'detail','path':str(self.path),'sha256':sha256(self.path)}]}

    def run_job(self,registry=None):
        with patch('mes_vision.station.photo_inspection.load_bundle',return_value=deepcopy(self.bundle)):
            return run_photos(self.request,ROOT,registry=registry or Registry())

    def test_saved_candidates_mapped_and_no_robot_or_ok(self):
        registry=Registry(); report=self.run_job(registry); out=self.root/'out'
        _,verified=open_results(str(out/'results.json')); self.assertEqual(report,verified)
        _,result,_=load_snapshot(out/'photo-0000')
        self.assertEqual(result['mode'],'model_file'); self.assertFalse(result['frame']['is_live'])
        self.assertFalse(result['robot_commands_enabled']); obj=result['objects'][0]
        self.assertEqual(obj['final_decision'],'REVIEW'); self.assertEqual(registry.calls,1)
        overlays=object_overlays(obj); crop=object_overlays(obj,crop=True)
        self.assertEqual(overlays[1]['box'],[25,27,35,37]); self.assertEqual(crop[1]['box'],[5,7,15,17])
        self.assertEqual(overlays[1]['track_id'],obj['object_id']); self.assertEqual(overlays[1]['status'],'REVIEW')
        self.assertCountEqual(registry.loaded,registry.closed)

    def test_no_candidates_still_review(self):
        self.run_job(Registry(defect=False)); _,result,_=load_snapshot(self.root/'out/photo-0000')
        self.assertEqual(result['objects'][0]['final_decision'],'REVIEW')

    def test_ambiguous_detail_does_not_inspect_wrong_object(self):
        registry=Registry(count=2); report=self.run_job(registry)
        self.assertFalse(report['records'][0]['association_confirmed']); self.assertEqual(registry.calls,0)

    def test_overview_localizes_only(self):
        self.request['images'][0]['role']='overview'; registry=Registry(count=2); report=self.run_job(registry)
        self.assertEqual(report['records'][0]['objects'],2); self.assertEqual(registry.calls,0)
        self.assertEqual(registry.loaded,['objects'])

    def test_changed_photo_fails_and_releases_models(self):
        self.request['images'][0]['sha256']='b'*64; registry=Registry()
        with self.assertRaisesRegex(ValueError,'Photograph changed'): self.run_job(registry)
        self.assertCountEqual(registry.loaded,registry.closed)
        self.assertFalse((self.root/'out/results.json').exists())

    def test_changed_snapshot_or_role_rejected(self):
        self.run_job(); path=self.root/'out/results.json'; report=read_json(path)
        report['records'][0]['role']='overview'; write_json(path,report)
        with self.assertRaisesRegex(ValueError,'domain changed'): open_results(path)

    def test_model_hash_and_class_order_rejected(self):
        spec=read_json(ROOT/'configs/inspection/ash-trained-v3.json')
        for value,expected in [('b'*64,'model changed')]:
            spec['models']['overview']['sha256']=value; path=self.root/'bundle.json'; write_json(path,spec)
            with self.assertRaisesRegex(ValueError,expected): load_bundle(path,ROOT)
        spec=read_json(ROOT/'configs/inspection/ash-trained-v3.json'); spec['models']['overview']['class_names']=['wrong']
        write_json(path,spec)
        with self.assertRaisesRegex(ValueError,'class order'): load_bundle(path,ROOT)

    def test_photo_window_selection_languages_and_reopen(self):
        self.run_job(); w=PhotoInspectionDialog(ROOT,self.root/'runtime'); self.addCleanup(w.deleteLater)
        w.load_results(self.root/'out/results.json'); APP.processEvents()
        self.assertIn('NG03',w.reasons.toPlainText()); identity=w.selected_id
        for language in ('en','zh-CN','th','ko'):
            apply_language(APP,language); self.assertEqual(w.selected_id,identity)
            self.assertIn('NG03',w.reasons.toPlainText()); self.assertEqual(w.detail.canvas.tracks[1]['box'],[5,7,15,17])
        self.assertFalse(w.original.canvas.pixmap.isNull()); self.assertEqual(w.objects.currentRow(),0)
        w.load_results(self.root/'out/results.json'); self.assertEqual(w.selected_id,identity)

    def test_cancel_child_and_close_waits_for_exit(self):
        w=PhotoInspectionDialog(ROOT,self.root/'runtime'); self.addCleanup(w.deleteLater)
        process=QProcess(w); w.process=process; process.finished.connect(w.process_finished)
        process.start(sys.executable,['-c','import time; time.sleep(30)']); self.assertTrue(process.waitForStarted(5000))
        w.reject(); self.assertTrue(w.cancelled); self.assertTrue(w.pending_close)
        deadline=time.monotonic()+5
        while w.process is not None and time.monotonic()<deadline: APP.processEvents(); time.sleep(.01)
        self.assertIsNone(w.process)


if __name__=='__main__': unittest.main()

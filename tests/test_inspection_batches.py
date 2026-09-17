from dataclasses import replace
from types import SimpleNamespace
import unittest
import numpy as np
from mes_vision.inspection import Box,Detection,CheckResult,CheckStatus,Finding,InspectionPipeline,Mode,ModelRef
from mes_vision.inspection.adapters import MockDetector
from mes_vision.inspection.geometry import extract_crop
from mes_vision.inspection.batching import validate_crops
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend,RFDETRDefectInspector,model_reference
from mes_vision.operation.engine import SelectiveInspector
from test_inspection import frame


class Recorder:
    check_id='known_defects'; model=ModelRef('batch fixture','1','mock')
    def __init__(self,fault=None): self.groups=[]; self.fault=fault
    def inspect(self,crop):
        self.groups.append([crop.object_id]); return self.result(crop)
    def result(self,crop):
        return CheckResult(self.check_id,crop.frame_id,crop.object_id,CheckStatus.FAIL,self.model,
            findings=(Finding('crack','NG03',.9,Box(1,2,3,4)),),criteria_version='synthetic-only')
    def inspect_many(self,crops):
        self.groups.append([c.object_id for c in crops]); rows=tuple(self.result(c) for c in crops)
        if self.fault=='count': return rows[:-1]
        if self.fault=='order': return rows[::-1]
        if self.fault=='duplicate': return (rows[0],)*len(rows)
        if self.fault=='stale': return tuple(replace(r,frame_id='old') for r in rows)
        if self.fault=='raise': raise RuntimeError('batch failed')
        return rows


class BatchTests(unittest.TestCase):
    def pipeline(self,inspector,count=9,cap=4):
        boxes=tuple(Detection(Box(2+i*15,10,12+i*15,30),.9,0,'part') for i in range(count))
        return InspectionPipeline(MockDetector(boxes),(inspector,),mode=Mode.SIMULATION,inspection_batch_size=cap)
    def test_bounded_groups_tail_and_original_coordinates(self):
        inspector=Recorder(); result=self.pipeline(inspector).run(frame())
        self.assertEqual([len(g) for g in inspector.groups],[4,4,1])
        for i,item in enumerate(result.objects):
            self.assertEqual(item.checks[0].object_id,item.object_id)
            self.assertEqual(item.checks[0].findings[0].original_box,Box(3+i*15,12,5+i*15,14))
    def test_batch_identity_count_and_exception_fail_closed(self):
        for fault in ('count','order','duplicate','stale','raise'):
            with self.subTest(fault=fault):
                rows=self.pipeline(Recorder(fault),4).run(frame()).objects
                self.assertEqual([r.checks[0].status for r in rows],[CheckStatus.ERROR]*4)
    def test_single_and_empty_scenes_do_not_wait(self):
        inspector=Recorder(); self.pipeline(inspector,0).run(frame()); self.assertEqual(inspector.groups,[])
        self.pipeline(inspector,1).run(frame()); self.assertEqual([len(g) for g in inspector.groups],[1])
    def test_only_selected_tracks_reach_inference(self):
        recorder=Recorder(); wrapper=SelectiveInspector(recorder,{1,4,8})
        rows=self.pipeline(wrapper).run(frame()).objects
        self.assertEqual([int(x.rsplit('OBJ',1)[1]) for g in recorder.groups for x in g],[2,5,9])
        self.assertEqual([i for i,r in enumerate(rows) if r.checks[0].status==CheckStatus.FAIL],[1,4,8])
        self.assertTrue(all(rows[i].checks[0].status==CheckStatus.NOT_RUN for i in (0,2,3,5,6,7)))
    def test_mixed_capture_duplicate_and_oversize_rejected(self):
        a=extract_crop(frame(),'a',Box(1,1,8,8)); b=replace(a,object_id='b',frame_id='other')
        for values in ((a,a),(a,b),tuple(replace(a,object_id=str(i)) for i in range(5))):
            with self.assertRaises(ValueError): validate_crops(values)
    def test_profile_capacity_is_part_of_policy_identity(self):
        self.assertNotEqual(model_reference('a'*64,'product_defects',max_batch_size=1),model_reference('a'*64,'product_defects',max_batch_size=4))
        self.assertEqual(model_reference('a'*64,'product_objects').version,'1.9.4+fp32_jit')
        self.assertEqual(model_reference('a'*64,'product_defects').version,'1.9.4+fp32_jit+batch4')
    def test_invalid_batch_limits(self):
        for cap in (True,0,3,8):
            with self.assertRaises(ValueError): self.pipeline(Recorder(),cap=cap)
            with self.assertRaises(ValueError): RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general',max_batch_size=cap)
    def test_backend_batches_preserve_order_and_reserved_slots(self):
        def output(image):
            width,height=image.size
            return SimpleNamespace(xyxy=np.array([[1,1,width-1,height-1],[1,1,2,2]],float),confidence=np.array([.8,.9]),class_id=np.array([0,1]),data={})
        calls=[]
        class Network:
            def predict(self,images,**kwargs):
                if isinstance(images,list): calls.append(len(images)); return [output(image) for image in images]
                calls.append(1); return output(images)
        backend=RFDETRBackend('unused','a'*64,threshold=.4,training_scope='product_defects',class_names=('crack',))
        network=Network(); backend._network=network; backend._batch_networks={1:network,2:network,4:network}
        images=[np.zeros((10+i,20+i,3),np.uint8) for i in range(3)]
        rows=backend.predict_many_rgb(images)
        self.assertEqual(calls,[2,1]); self.assertEqual([r[0].box.x2 for r in rows],[19,20,21])
        self.assertEqual(backend.excluded_reserved_slots_batch,(1,1,1)); self.assertEqual(backend.actual_batch_sizes,(2,2,1))
        crops=tuple(extract_crop(frame(),str(i),Box(1+i*20,1,12+i*20,12)) for i in range(3))
        inspector=RFDETRDefectInspector(backend,{0:'NG03'}); results=inspector.inspect_many(crops)
        self.assertEqual([r.object_id for r in results],['0','1','2'])
        self.assertTrue(all(r.status==CheckStatus.UNCERTAIN and len(r.findings)==1 for r in results))
    def test_backend_wrong_output_count_rejected(self):
        backend=RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general',max_batch_size=2)
        network=SimpleNamespace(predict=lambda *a,**kw:[]); backend._network=network; backend._batch_networks={2:network}
        with self.assertRaises(ValueError): backend.predict_many_rgb([np.zeros((10,10,3),np.uint8)]*2)


if __name__=='__main__': unittest.main()

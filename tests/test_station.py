"""Software protocol tests only; injected outputs are not physical validation."""
from copy import deepcopy
from dataclasses import asdict,replace
from pathlib import Path
from types import SimpleNamespace
import tempfile,unittest
from unittest.mock import patch
import numpy as np
from mes_vision.station import ScanJournal,ScanSequence,StationVision
from mes_vision.station.evidence import save_capture,save_detail
from mes_vision.station.sequence import valid_profile
from mes_vision.inspection import Detection,Box
from mes_vision.inspection.model_registry import ModelRegistry,Provider,ModelHandle,default_registry
from mes_vision.anomaly.features import fingerprint
from test_operation import InjectedModels,frame,detection


def profile(models):
    return {'schema_version':1,'mode':'overview_detail','profile_id':'profile-test',
        'validation_reference':'UNIT-TEST-ONLY','camera_serial':'TEST','mount_revision':'test-mount',
        'camera_session':'injected-camera','calibration_kind':'robot_mounted_overview_detail',
        'calibration_digest':'c'*64,'overview_model_digest':models.model.weights_sha256,
        'detail_policy_digest':fingerprint(asdict(models.policy)),'product_id':'test-part',
        'product_version':1,'equipment_version':1,'image_size':[480,180],
        'settle_seconds':.5,'action_timeout_seconds':30,'pose_tolerance_mm':1,'rotation_tolerance_deg':1,
        'max_objects':100,'expected_count':None,'auto_sort':True,
        'motion_bounds':{'x':[-100,400],'y':[-200,200],'z':[0,200],'r':[-90,90]},
        'overview_pose':{'x':200,'y':0,'z':150,'r':0}}


def workspace():
    return {'roi':[[0,0],[480,0],[480,180],[0,180]],'excluded':[],'validation_reference':'UNIT-TEST-ONLY'}


def vision(models,overview=None):
    return StationVision(overview or models,models,models.inspectors,models.policy,product_id='test-part',
        overview_workspace=workspace(),detail_workspace=workspace(),
        detail_quality={'version':'test-quality','validation_reference':'UNIT-TEST-ONLY',
                        'blur_min':0,'brightness_min':0,'brightness_max':255})


class RegistryTests(unittest.TestCase):
    def test_default_lists_only_implemented_small_provider(self):
        providers=default_registry().describe()
        self.assertEqual([p['backend'] for p in providers],['rfdetr-small'])
        self.assertIn('Small',providers[0]['weights_license'])

    def test_unknown_backend_never_loads_rfdetr_or_imports_configured_module(self):
        with patch('mes_vision.inspection.rfdetr_adapter.RFDETRBackend') as backend:
            for name in ('rtdetrv2','os.system',None,[]):
                with self.assertRaises(ValueError): default_registry().create({'backend':name},'objects')
            backend.assert_not_called()

    def test_separate_architectures_and_roles_do_not_change_common_contract(self):
        registry=ModelRegistry(); built=[]
        def factory(spec,role):
            spec['nested'].append('changed'); built.append(role)
            adapter=InjectedModels() if role=='objects' else InjectedModels().inspectors[0]
            return ModelHandle(SimpleNamespace(load=lambda:None,close=lambda:None),adapter,lambda:None)
        for name,role in (('overview-test','objects'),('detail-test','defects')):
            registry.register(Provider(name,frozenset({role}),'test-code','test-weights',factory))
        original={'backend':'overview-test','nested':[]}
        self.assertTrue(callable(registry.create(original,'objects').adapter.detect)); self.assertEqual(original['nested'],[])
        self.assertTrue(callable(registry.create({'backend':'detail-test','nested':[]},'defects').adapter.inspect))
        with self.assertRaises(ValueError): registry.create(original,'defects')
        self.assertEqual(built,['objects','defects'])

    def test_provider_requires_unique_name_and_license_metadata(self):
        registry=ModelRegistry(); factory=lambda *_:None
        with self.assertRaises(ValueError): registry.register(Provider('x',frozenset({'objects'}),'','',factory))
        registry.register(Provider('x',frozenset({'objects'}),'test','test',factory))
        with self.assertRaises(ValueError): registry.register(Provider('x',frozenset({'objects'}),'test','test',factory))

    def test_loading_failure_releases_model_once(self):
        calls=[]
        def fail(): raise RuntimeError('warmup failed')
        handle=ModelHandle(SimpleNamespace(load=lambda:calls.append('load'),close=lambda:calls.append('close')),None,fail)
        with self.assertRaises(RuntimeError): handle.load()
        handle.close(); self.assertEqual(calls,['load','close'])

    def test_scan_models_require_explicit_training_domains(self):
        from mes_vision.station.models import StationModels
        product={'objects':{'weights':'not-loaded'},'defects':None,'anomaly':None,'geometry':None}
        with self.assertRaises(ValueError): StationModels('.',{},product,capture_domains={})
        with self.assertRaises(ValueError): StationModels('.',{},product,capture_domains={'overview_objects':'overview','objects':'overview'})
        owned=StationModels('.',{},product,capture_domains={'overview_objects':'overview','objects':'detail'})
        self.assertFalse(owned.ready); self.assertIsNone(owned.overview)

    def test_anomaly_provider_is_replaceable_without_changing_check_identity(self):
        registry=ModelRegistry(); model=InjectedModels().inspectors[1]
        registry.register(Provider('anomaly-test',frozenset({'anomaly'}),'test','test',lambda spec,role:
            ModelHandle(SimpleNamespace(load=lambda:None,close=lambda:None),model,lambda:None)))
        self.assertIs(registry.create({'backend':'anomaly-test'},'anomaly').adapter,model)
        model.check_id='known_defects'
        with self.assertRaises(ValueError): registry.create({'backend':'anomaly-test'},'anomaly')

    def test_existing_model_manager_uses_registered_provider_and_policy_identity(self):
        from mes_vision.operation.engine import Models
        models=InjectedModels(); registry=ModelRegistry(); calls=[]
        registry.register(Provider('test-object',frozenset({'objects'}),'test','test',lambda spec,role:
            ModelHandle(SimpleNamespace(load=lambda:calls.append('load'),close=lambda:calls.append('close')),models,lambda:None)))
        product={'objects':{'backend':'test-object'},'defects':None,'anomaly':None,'geometry':None,'policy':'test'}
        # Use the existing required policy to prove model substitution cannot bypass it.
        owned=Models('.',product,registry=registry)
        with patch('mes_vision.operation.engine.load_policy',return_value=models.policy),self.assertRaises(ValueError): owned.load()
        owned.close(); self.assertEqual(calls,['load','close'])


class StationVisionTests(unittest.TestCase):
    def setUp(self): self.models=InjectedModels(); self.service=vision(self.models)
    def inspect(self,f=None,**values):
        return self.service.inspect_detail(f or frame(2),cycle_id='cycle',target_id='target',overview_frame_id='overview',expected_label='part',**values)

    def test_overview_only_localizes_multiple_objects_and_never_inspects_defects(self):
        self.models.detections=(detection(x=230),detection(x=10))
        result=self.service.locate(frame())
        self.assertEqual(len(result),2); self.assertLess(result[0]['box']['x1'],result[1]['box']['x1'])
        self.assertFalse(any('decision' in r for r in result)); self.assertEqual(self.models.calls,0)

    def test_overlap_is_review_instead_of_silent_drop(self):
        self.models.detections=(detection(x=10),detection(x=20))
        result=self.service.locate(frame()); self.assertEqual(len(result),2)
        self.assertTrue(all(t['localization_status']=='REVIEW' for t in result))

    def test_detail_requires_single_centered_matching_product(self):
        for detections in ((),(detection(),detection(x=200)),(detection(x=0),),
                           (Detection(Box(170,30,300,150),.9,0,'different'),)):
            self.models.detections=detections
            with self.subTest(detections=detections):
                result=self.inspect(); self.assertEqual(result['decision'],'REVIEW'); self.assertIsNone(result['inspection'])
        self.assertEqual(self.models.calls,0)

    def test_independent_class_order_does_not_break_product_association(self):
        self.models.detections=(Detection(Box(170,30,300,150),.99,4,'part'),)
        result=self.inspect(); self.assertIsNotNone(result['inspection'])

    def test_detail_ng_comes_from_closeup_and_has_separate_coordinate_link(self):
        self.models.detections=(detection(x=170),); self.models.defect=True
        result=self.inspect(); self.assertEqual(result['decision'],'NG')
        self.assertEqual(result['inspection'].config['station_link']['overview_frame_id'],'overview')
        self.assertEqual(result['inspection'].frame['frame_id'],frame(2).frame_id)
        self.assertEqual(result['inspection'].objects[0].decision_details['defect_codes'],['NG03'])

    def test_unexpected_anomaly_stays_review_without_named_defect(self):
        self.models.detections=(detection(x=170),); self.models.anomaly_score=.5
        result=self.inspect(); self.assertEqual(result['decision'],'REVIEW')
        self.assertFalse(result['inspection'].objects[0].decision_details['defect_codes'])

    def test_bad_quality_does_not_invoke_inspectors(self):
        self.models.detections=(detection(x=170),); self.service.detail_quality['blur_min']=1
        result=self.inspect(); self.assertEqual(result['reason'],'DETAIL_IMAGE_QUALITY'); self.assertEqual(self.models.calls,0)

    def test_policy_detector_mismatch_rejected(self):
        other=InjectedModels(); other.model=replace(other.model,weights_sha256='f'*64)
        with self.assertRaises(ValueError): vision(other)

    def test_overview_frame_cannot_be_reused_for_detail(self):
        with self.assertRaises(ValueError): self.service.inspect_detail(frame(),cycle_id='c',target_id='t',overview_frame_id=frame().frame_id,expected_label='part')


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.models=InjectedModels(); self.profile=profile(self.models)
        self.journal=ScanJournal(self.root); self.scan=ScanSequence(self.journal)
        self.now=10.; self.sequence=0; self.request=self.scan.start(self.profile,self.now)
    def tearDown(self): self.temp.cleanup()
    def done(self,payload,**options):
        request=self.request; self.now+=.1
        self.request=self.scan.complete(request['id'],request['cycle_id'],request['generation'],payload,self.now,**options)
        return self.request
    def moved(self):
        self.done({'actual_pose':self.request['payload']['pose'],'motion_complete':True})
        self.assertEqual(self.scan.data['state'],'SETTLING'); self.assertIsNone(self.scan.tick(self.now+.49))
        self.now+=.5; self.request=self.scan.tick(self.now)
    def captured(self):
        self.sequence+=1; f=frame(self.sequence); self.now+=.01
        capture=save_capture(self.root/'camera',f,camera_serial='TEST',acquired_at=self.now)
        self.done(capture); return f,capture
    def localized(self,count=2):
        self.moved(); self.captured(); self.models.detections=tuple(detection(x=10+i*180) for i in range(count))
        self.done({'frame_id':self.scan.data['overview']['frame_id'],'model_digest':self.profile['overview_model_digest'],
                   'objects':vision(self.models).locate(frame(self.sequence))})
        if self.request and self.request['kind']=='map_targets':
            targets=[]
            for i,t in enumerate(self.request['payload']['targets']):
                targets.append({'id':t['id'],'capture_pose':{'x':200+i*10,'y':0,'z':80,'r':0},
                                'pick_pose':{'x':210+i*10,'y':0,'z':20,'r':0}})
            self.done({'calibration_digest':self.profile['calibration_digest'],'targets':targets})
    def inspected(self,*,ng=False,review=False):
        self.moved(); f,capture=self.captured(); self.models.detections=(detection(x=170),)
        self.models.defect=ng; self.models.anomaly_score=.5 if review else .1
        req=self.request
        output=vision(self.models).inspect_detail(f,cycle_id=req['cycle_id'],target_id=req['target_id'],
            overview_frame_id=req['payload']['overview_frame_id'],expected_label=req['payload']['expected_label'])
        receipt=save_detail(self.root/'details',f,output)
        self.done(receipt); return receipt

    def test_complete_two_objects_then_sort_only_known_verdicts_and_link_vlm(self):
        self.localized(); self.inspected(ng=True)
        self.assertEqual(self.request['kind'],'move_detail')
        self.assertFalse(any(t['status']=='SORTING' for t in self.scan.data['targets']))
        self.inspected(review=True); self.assertEqual(self.request['kind'],'sort_object')
        first,second=self.scan.data['targets']; before=deepcopy(self.scan.data)
        self.journal.attach_vlm(before['id'],first['id'],first['detail']['frame_id'],'Additional observation')
        self.assertEqual(self.journal.read(before['id']),before)
        # Sequence-only test; the real durable receipt is exercised by test_station_sort.
        with patch('mes_vision.station.sorting.verify_sort_receipt'):
            self.done({'object_id':first['id'],'placement_confirmed':True})
        self.assertEqual(self.scan.data['state'],'COMPLETED')
        self.assertEqual([t['status'] for t in self.scan.data['targets']],['SORTED','REVIEW'])

    def test_stop_invalidates_coordinates_and_discards_late_completion(self):
        old=self.request; self.scan.stop()
        self.scan.complete(old['id'],old['cycle_id'],old['generation'],{},11)
        self.assertEqual(self.scan.data['state'],'CANCELLED'); self.assertFalse(self.scan.data['coordinate_valid'])
        self.assertIsNone(self.scan.tick(100))

    def test_scene_change_requires_new_overview(self):
        self.localized(1); self.scan.scene_changed()
        self.assertEqual(self.scan.data['error'],'SCENE_CHANGED_RECAPTURE_REQUIRED')
        new=self.scan.start(self.profile,self.now+1)
        self.assertEqual(new['kind'],'move_overview'); self.assertFalse(self.scan.data['targets'])

    def test_motion_without_arrival_never_captures(self):
        pose=dict(self.profile['overview_pose'],x=220)
        self.done({'actual_pose':pose,'motion_complete':True})
        self.assertEqual(self.scan.data['state'],'FAILED'); self.assertIsNone(self.scan.tick(100))

    def test_capture_rejects_stale_sequence_camera_and_altered_file(self):
        self.moved(); f=frame(1); capture=save_capture(self.root/'camera',f,camera_serial='TEST',acquired_at=self.now-.01)
        self.done(capture); self.assertEqual(self.scan.data['state'],'FAILED')
        self.assertIn('stale',self.scan.data['error'])

    def test_reconnected_camera_never_reuses_overview_coordinates(self):
        self.moved(); capture=save_capture(self.root/'camera',frame(1,session='replacement'),camera_serial='TEST',acquired_at=self.now)
        self.done(capture); self.assertEqual(self.scan.data['state'],'FAILED')
        self.assertIn('Camera changed',self.scan.data['error'])

    def test_unsaved_or_modified_capture_blocks_detection(self):
        self.moved(); capture=save_capture(self.root/'camera',frame(1),camera_serial='TEST',acquired_at=self.now)
        Path(capture['path']).write_bytes(b'changed')
        self.done(capture); self.assertEqual(self.scan.data['state'],'FAILED')
        self.assertIsNone(self.request)

    def test_no_sort_option_finishes_after_basic_results_without_vlm(self):
        self.scan.stop(); self.profile['auto_sort']=False; self.request=self.scan.start(self.profile,self.now)
        self.localized(1); self.inspected(ng=True)
        self.assertEqual(self.scan.data['state'],'COMPLETED'); self.assertEqual(self.scan.data['targets'][0]['decision'],'NG')

    def test_changed_detail_policy_cannot_authorize_sorting(self):
        self.scan.stop(); self.profile['detail_policy_digest']='e'*64; self.request=self.scan.start(self.profile,self.now)
        self.localized(1); self.inspected(ng=True)
        self.assertEqual(self.scan.data['state'],'FAILED'); self.assertIn('policy changed',self.scan.data['error'])

    def test_timeout_is_terminal_without_retry(self):
        self.scan.tick(self.request['deadline'])
        self.assertEqual(self.scan.data['state'],'FAILED'); self.assertIsNone(self.scan.tick(1000))

    def test_wrong_and_duplicate_completion_cannot_advance(self):
        old=deepcopy(self.request); revision=self.scan.data['revision']
        self.scan.complete('wrong',old['cycle_id'],old['generation'],{},11)
        self.assertEqual(self.scan.data['revision'],revision)
        self.moved(); current=self.scan.snapshot()
        self.scan.complete(old['id'],old['cycle_id'],old['generation'],{},self.now)
        self.assertEqual(self.scan.snapshot(),current)

    def test_restart_marks_inflight_cycle_interrupted_without_motion(self):
        self.localized(1); identity=self.scan.data['id']
        self.assertEqual(self.journal.recover_interrupted(),[identity])
        recovered=self.journal.read(identity)
        self.assertEqual(recovered['state'],'INTERRUPTED'); self.assertIsNone(recovered['pending']); self.assertFalse(recovered['coordinate_valid'])
        with self.assertRaises(ValueError): self.scan.stop()

    def test_two_owners_cannot_start_overlapping_cycles(self):
        other=ScanSequence(ScanJournal(self.root))
        with self.assertRaises(ValueError): other.start(self.profile,12)
        self.assertIsNone(other.data)

    def test_failed_save_does_not_publish_request_or_advance_memory(self):
        before=self.scan.snapshot()
        with patch.object(self.journal,'save',side_effect=OSError('disk full')),self.assertRaises(OSError):
            self.done({'actual_pose':self.profile['overview_pose'],'motion_complete':True})
        self.assertEqual(self.scan.snapshot(),before)

    def test_wrong_detail_link_and_vlm_link_are_rejected(self):
        self.localized(1); self.moved(); self.captured()
        self.done({'target_id':'other','cycle_id':self.scan.data['id'],'detail_frame_id':'wrong',
                   'review_reason':'DETAIL_TARGET_UNCONFIRMED','decision':'REVIEW'})
        self.assertEqual(self.scan.data['state'],'FAILED')
        with self.assertRaises(ValueError): self.journal.attach_vlm(self.scan.data['id'],self.scan.data['targets'][0]['id'],'wrong','text')

    def test_model_mismatch_blocks_mapping(self):
        self.moved(); self.captured()
        self.done({'frame_id':self.scan.data['overview']['frame_id'],'model_digest':'f'*64,'objects':[]})
        self.assertEqual(self.scan.data['state'],'FAILED')

    def test_fixed_camera_calibration_and_missing_physical_values_rejected(self):
        for key,value in (('calibration_kind','fixed_camera'),('overview_pose',None),('settle_seconds',float('nan')),
                          ('validation_reference',''),('camera_session','')):
            p=deepcopy(self.profile); p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError): valid_profile(p)

    def test_snapshot_frame_tampering_is_not_accepted_as_a_verdict(self):
        self.localized(1); self.moved(); f,capture=self.captured(); req=self.request
        self.models.detections=(detection(x=170),)
        different=replace(f,rgb=np.zeros_like(f.rgb))
        output=vision(self.models).inspect_detail(different,cycle_id=req['cycle_id'],target_id=req['target_id'],
            overview_frame_id=req['payload']['overview_frame_id'],expected_label='part')
        receipt=save_detail(self.root/'details',different,output); self.done(receipt)
        self.assertEqual(self.scan.data['state'],'FAILED'); self.assertIn('pixels',self.scan.data['error'])


if __name__=='__main__': unittest.main()

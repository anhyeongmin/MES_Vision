"""Full-view reuse must preserve saved pixels, identities and crop coordinates."""
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch
import test_photo_inspection as fixtures
from test_photo_inspection import Registry, ROOT
from mes_vision.station.photo_inspection import run_photos, open_results
from mes_vision.station.manual_dialog import ManualInspectionDialog
from mes_vision.training.data import read_json, write_json, sha256
from mes_vision.vlm.snapshots import load_snapshot


class OverviewTests(unittest.TestCase):
    setUp = fixtures.PhotoTests.setUp
    run_job = fixtures.PhotoTests.run_job

    def prepare(self, count=2):
        self.request['images'][0]['role']='overview'
        report=self.run_job(Registry(count=count))
        self.origin=dict(snapshot=str(self.root/'out/photo-0000'),snapshot_digest=report['records'][0]['snapshot_digest'])
        image=self.root/'out/photo-0000/frame.png'
        self.batch=deepcopy(self.request)
        self.batch.update(output=str(self.root/'batch'))
        self.batch['images']=[dict(role='overview_inspection',path=str(image),sha256=sha256(image),overview_origin=self.origin)]

    def run_batch(self):
        self.registry=Registry(count=99)
        with patch('mes_vision.station.photo_inspection.load_bundle',return_value=deepcopy(self.bundle)):
            return run_photos(self.batch,ROOT,registry=self.registry)

    def test_reuses_objects_without_detection_and_preserves_pixels(self):
        self.prepare(); report=self.run_batch()
        self.assertEqual(self.registry.loaded,['defects']); self.assertEqual(self.registry.calls,2)
        open_results(self.root/'batch/results.json')
        _,source,_=load_snapshot(self.origin['snapshot'])
        _,result,_=load_snapshot(self.root/'batch/photo-0000')
        self.assertEqual(sha256(self.root/'batch/photo-0000/frame.png'),self.batch['images'][0]['sha256'])
        pairs=report['records'][0]['object_mapping']
        self.assertEqual([p['source_object_id'] for p in pairs],[o['object_id'] for o in source['objects']])
        self.assertEqual([o['effective_box'] for o in source['objects']],[o['effective_box'] for o in result['objects']])
        self.assertTrue(all(o['final_decision']=='REVIEW' for o in result['objects']))
        self.assertFalse(result['robot_commands_enabled'])

    def test_changed_source_pixels_rejected(self):
        self.prepare(); self.batch['images'][0].update(path=str(self.path),sha256=sha256(self.path))
        # A different photograph of the same dimensions cannot reuse old detections.
        from PIL import Image
        Image.new('RGB',(160,100),'white').save(self.path)
        self.batch['images'][0]['sha256']=sha256(self.path)
        with self.assertRaisesRegex(ValueError,'original saved overview pixels'):self.run_batch()

    def test_changed_bundle_rejected(self):
        self.prepare();self.bundle['models']['defects']['revision']='changed'
        with self.assertRaisesRegex(ValueError,'model bundle changed'):self.run_batch()

    def test_empty_overview_rejected(self):
        self.prepare(count=0)
        with self.assertRaisesRegex(ValueError,'No overview objects'):self.run_batch()

    def test_mapping_tamper_rejected(self):
        self.prepare();self.run_batch();path=self.root/'batch/results.json'
        report=read_json(path);report['records'][0]['object_mapping'][0]['source_object_id']='wrong';write_json(path,report)
        with self.assertRaisesRegex(ValueError,'binding changed'):open_results(path)

    def test_manual_button_all_results_and_individual_vlm_ids(self):
        self.prepare()
        camera=Mock();camera.disposed=False;camera.stopping.is_set.return_value=False
        dialog=ManualInspectionDialog(ROOT,self.root/'runtime',{'driver':'uvc'},camera=camera)
        dialog.timer.stop();self.addCleanup(dialog.deleteLater)
        self.assertFalse(dialog.overview_inspect_button.isEnabled())
        dialog.cycle=self.root/'cycle';dialog.cycle.mkdir();write_json(dialog.cycle/'bundle.json',self.bundle)
        dialog.capture=dict(role='overview',target=None);dialog.output=self.root/'out';dialog.accept_result()
        self.assertTrue(dialog.overview_inspect_button.isEnabled())
        ids=[o['manual_id'] for o in dialog.objects]
        dialog.select_object('2');dialog.submit_request=Mock();dialog.inspect_overview()
        request=read_json(dialog.submit_request.call_args.args[0])
        with patch('mes_vision.station.photo_inspection.load_bundle',return_value=deepcopy(self.bundle)):
            run_photos(request,ROOT,registry=Registry(count=99))
        dialog.accept_result()
        self.assertEqual([o['manual_id'] for o in dialog.objects],ids)
        self.assertEqual(set(dialog.details),{'1','2'});self.assertEqual(dialog.selected,'2')
        self.assertEqual(dialog.detail.canvas.pixmap.width(),50)
        self.assertEqual(dialog.detail.canvas.tracks[1]['box'],[5,7,15,17])
        self.assertIn('풀뷰',dialog.detail_title.text());self.assertIn('NG03',dialog.reasons.toPlainText())
        dialog.vlm_queue.set_enabled(True);dialog.ensure_vlm_worker=Mock()
        with patch('mes_vision.vlm.region_backend.region_plan',return_value={'regions':[{}]}):
            for identity in ids:dialog.request_vlm(identity)
        targets=[]
        for identity in ids:
            detail=dialog.details[identity];job=dialog.vlm_queue.get(detail['vlm_job'])
            self.assertEqual(job['object_id'],detail['snapshot_object_id']);targets.append(job['object_id'])
        self.assertEqual(len(set(targets)),2)
        dialog.vlm_queue.set_enabled(False)
        dialog.overview_capture=dict(dialog.overview_capture,snapshot_digest='0'*64)
        with self.assertRaisesRegex(ValueError,'회차'):dialog.accept_result()
        # A later real close-up replaces only the selected object's UI result.
        request=deepcopy(self.request);request['output']=str(self.root/'closeup')
        request['images'][0]['role']='detail'
        with patch('mes_vision.station.photo_inspection.load_bundle',return_value=deepcopy(self.bundle)):
            run_photos(request,ROOT,registry=Registry(count=1))
        dialog.capture=dict(role='detail',target='2');dialog.output=self.root/'closeup';dialog.accept_result()
        self.assertEqual(dialog.details['2']['capture_method'],'detail')
        self.assertEqual(dialog.details['1']['capture_method'],'overview_inspection')
        self.assertEqual(dialog.detail.canvas.pixmap.width(),160)
        camera.command.assert_not_called()

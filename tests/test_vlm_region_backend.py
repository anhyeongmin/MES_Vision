import unittest
import tempfile
from pathlib import Path
from mes_vision.training.data import read_json,write_json,sha256
from mes_vision.anomaly.features import fingerprint
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.region_backend import region_plan,parse_region,RegionBackend

class RegionBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.fixture=make_vlm_fixture(Path(self.temp.name)/'fixture')
        self.job=self.fixture['queue'].get(self.fixture['job_id'])
        self.root=Path(self.job['snapshot_path'])
    def change(self,mutate):
        inspection=read_json(self.root/'inspection.json');mutate(inspection['objects'][1])
        write_json(self.root/'inspection.json',inspection)
        m=read_json(self.root/'snapshot.json');m['files']['inspection.json']=sha256(self.root/'inspection.json')
        write_json(self.root/'snapshot.json',m);self.job['snapshot_digest']=fingerprint(m)
    def supported(self,obj):obj['checks'][0]['findings'][0]['defect_code']='NG05'
    def test_unsupported_is_not_turned_into_normal(self):
        plan=region_plan(self.job);self.assertEqual(plan['regions'],[]);self.assertIn('NG03',plan['unsupported'])
    def test_real_coordinate_pair_and_padding(self):
        self.change(self.supported);p=region_plan(self.job)
        self.assertEqual(p['regions'][0]['code'],'NG05')
        self.assertEqual(p['regions'][0]['box'],[7,1,53,84])
        self.assertEqual(p['object_id'],self.job['object_id'])
    def test_mismatched_original_coordinate_rejected(self):
        def mutate(o):
            self.supported(o);o['checks'][0]['findings'][0]['original_box']['x1']+=1
        self.change(mutate)
        with self.assertRaisesRegex(ValueError,'coordinate mismatch'):region_plan(self.job)
    def test_resized_crop_rejected(self):
        self.change(lambda o:o['crop'].update(resized=True))
        with self.assertRaisesRegex(ValueError,'coordinate transform'):region_plan(self.job)
    def test_tampered_snapshot_rejected(self):
        with (self.root/'inspection.json').open('a') as f:f.write(' ')
        with self.assertRaises(ValueError):region_plan(self.job)
    def test_missing_object_rejected(self):
        self.job['object_id']='wrong'
        with self.assertRaisesRegex(ValueError,'object not'):region_plan(self.job)
    def test_strict_observation_schema(self):
        self.assertEqual(parse_region('{"observation":"확인할 수 없습니다."}'),'확인할 수 없습니다.')
        for text in ['{"observation":"a","observation":"b"}','{"observation":"a","code":"OK"}','{"observation":false}','{"observation":""}']:
            with self.assertRaises(ValueError):parse_region(text)

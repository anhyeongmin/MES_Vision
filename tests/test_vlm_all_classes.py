"""Expanded capability is opt-in; old model semantics and pixel guards remain."""
import tempfile,unittest
from pathlib import Path
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.region_backend import region_plan,supported_codes,QUESTIONS
from mes_vision.training.data import read_json,write_json,sha256
from mes_vision.anomaly.features import fingerprint

class AllClassTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  f=make_vlm_fixture(Path(self.tmp.name)/'fixture');self.job=f['queue'].get(f['job_id'])
  self.root=Path(self.job['snapshot_path'])
 def change(self,fn):
  result=read_json(self.root/'inspection.json');fn(result['objects'][1]);write_json(self.root/'inspection.json',result)
  meta=read_json(self.root/'snapshot.json');meta['files']['inspection.json']=sha256(self.root/'inspection.json')
  write_json(self.root/'snapshot.json',meta);self.job['snapshot_digest']=fingerprint(meta)
 def test_all_defect_regions_and_old_capabilities(self):
  for code in ['NG01','NG02','NG03','NG04','NG05','NG06']:
   self.change(lambda o:o['checks'][0]['findings'][0].update(defect_code=code))
   plan=region_plan(self.job,set(QUESTIONS));self.assertEqual(plan['regions'][0]['code'],code)
  self.assertEqual(region_plan(self.job)['regions'],[])
 def test_missing_feature_uses_whole_object(self):
  self.change(lambda o:o['checks'][0]['findings'][0].update(defect_code='NG01'))
  plan=region_plan(self.job,set(QUESTIONS));b=plan['regions'][0]['box']
  self.assertEqual(b[:2],[0,0]);self.assertGreater(b[2],53)
 def test_no_findings_observes_without_approving(self):
  self.change(lambda o:o['checks'][0].update(findings=[]))
  plan=region_plan(self.job,set(QUESTIONS))
  self.assertEqual(plan['regions'][0]['code'],'OK')
  self.assertNotIn('decision',plan);self.assertEqual(region_plan(self.job)['regions'],[])
 def test_unknown_defect_is_not_normal(self):
  self.change(lambda o:o['checks'][0]['findings'][0].update(defect_code='NG99'))
  plan=region_plan(self.job,set(QUESTIONS));self.assertFalse(plan['regions']);self.assertIn('NG99',plan['unsupported'])
 def test_whole_missing_feature_still_checks_coordinate_pair(self):
  def bad(o):
   f=o['checks'][0]['findings'][0];f['defect_code']='NG01';f['original_box']['x1']+=1
  self.change(bad)
  with self.assertRaisesRegex(ValueError,'coordinate mismatch'):region_plan(self.job,set(QUESTIONS))
 def test_declared_capabilities_only(self):
  self.assertEqual(supported_codes({}),{'NG04','NG05'})
  self.assertEqual(supported_codes({'supported_codes':list(QUESTIONS)}),set(QUESTIONS))
  for codes in [[],['NG99'],['OK','OK'],'OK']:
   with self.assertRaises(ValueError):supported_codes({'supported_codes':codes})

if __name__=='__main__':unittest.main()

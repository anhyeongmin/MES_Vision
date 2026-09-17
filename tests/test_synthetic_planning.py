import unittest
from collections import defaultdict
import numpy as np
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from PIL import Image
from mes_vision.synthetic.planning import make_plan,fingerprint,CODES
from mes_vision.synthetic.dataset_export import appearance,export,sha


class PlanningTests(unittest.TestCase):
    def test_reproducible_plan_and_real_variation(self):
        a=make_plan(); self.assertEqual(fingerprint(a),fingerprint(make_plan()))
        self.assertNotEqual(fingerprint(a),fingerprint(make_plan(42)))
        self.assertEqual(len(a['tasks']),540)
        over=[t for t in a['tasks'] if t['domain']=='overview']
        self.assertTrue(any(not t['objects'] for t in over))
        self.assertGreater(len({len(t['objects']) for t in over}),5)
    def test_counterfactual_group_never_crosses_splits(self):
        groups=defaultdict(list)
        for t in make_plan()['tasks']: groups[t['group']].append(t)
        for values in groups.values():
            self.assertEqual(len({v['split'] for v in values}),1)
            if values[0]['domain']=='detail':
                self.assertEqual({v['objects'][0]['code'] for v in values},set(CODES))
                self.assertEqual(len({fingerprint(v['condition']) for v in values}),1)
                self.assertEqual(len({fingerprint(v['camera']) for v in values}),1)
    def test_known_shape_reuse_is_declared_not_hidden_as_physical_holdout(self):
        plan=make_plan(); self.assertIn('shared',plan['cad_split']); self.assertIn('never independent physical',plan['evaluation_scope'])
    def test_appearance_rejects_invisible_defect_despite_nonempty_geometry(self):
        normal=np.full((8,8,3),80,dtype=np.uint8); mask=np.ones((8,8),bool)
        self.assertFalse(appearance(normal,normal,mask)['passed'])
        self.assertTrue(appearance(normal+20,normal,mask)['passed'])
        with self.assertRaises(ValueError): appearance(normal,normal,np.zeros((8,8),bool))
    def test_shadow_outside_defect_does_not_make_it_visible(self):
        normal=np.zeros((8,8,3),dtype=np.uint8); actual=normal.copy(); actual[:4]=255
        mask=np.zeros((8,8),bool); mask[4:]=True
        self.assertFalse(appearance(actual,normal,mask)['passed'])


class ExportTests(unittest.TestCase):
    def fixture(self,root,groups=(1,1,1)):
        plan=make_plan(groups=groups,overviews=(1,1,1)); plan['cad_lineage']={c:{'source_sha256':c} for c in CODES}
        for t in plan['tasks']:
            if t['domain']=='overview': t['objects']=[dict(code='OK',x=0,y=0,angle=0)]
        (root/'raw').mkdir()
        for i,t in enumerate(plan['tasks']):
            Image.new('RGB',(1280,720),(i*7%255,i*13%255,i*23%255)).save(root/'raw'/(t['id']+'.png'))
        (root/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
        (root/'geometry.json').write_text('{}',encoding='utf-8')
        (root/'contract.json').write_text(json.dumps(dict(plan_sha256=sha(root/'plan.json'),geometry_sha256=sha(root/'geometry.json'))),encoding='utf-8')
        return plan
    def quality(self,root,t,*args):
        labels=[dict(code=o['code'],bbox=[10,10,200,200]) for o in t['objects']]
        defects=[dict(code=t['objects'][0]['code'],bbox=[50,50,20,20])] if t['domain']=='detail' and t['objects'][0]['code']!='OK' else []
        return dict(accepted=True,image_sha256=sha(root/'raw'/(t['id']+'.png')),objects=labels,defects=defects)
    def test_export_is_accepted_by_existing_training_validator_and_preserves_lineage(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root)
            with patch('mes_vision.synthetic.dataset_export.quality',self.quality): report=export(root)
            self.assertEqual(report['accepted_rgb'],24)
            self.assertTrue(report['shared_cad_across_splits'])
            meta=json.loads((root/'training/detail-defects/dataset.json').read_text(encoding='utf-8'))
            self.assertEqual(set(meta['cad_lineage']),set(CODES)); self.assertEqual(meta['kind'],'synthetic')
            stats=report['groups']['detail-defects']['splits']
            self.assertEqual(stats['valid']['negative_images'],1)
            self.assertEqual(stats['test']['annotations'],6)
    def test_split_leak_is_rejected_before_copy(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); plan=self.fixture(root); plan['tasks'][1]['split']='test'
            (root/'plan.json').write_text(json.dumps(plan),encoding='utf-8')
            contract=json.loads((root/'contract.json').read_text()); contract['plan_sha256']=sha(root/'plan.json')
            (root/'contract.json').write_text(json.dumps(contract),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'leaks'): export(root)
            self.assertFalse((root/'training').exists())
    def test_one_bad_defect_excludes_its_whole_condition_group(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); self.fixture(root,groups=(2,1,1))
            def fail_one(root,t,*args):
                q=self.quality(root,t,*args)
                if t['id']=='train-d0000-NG03': q.update(accepted=False,reason='invisible test defect')
                return q
            with patch('mes_vision.synthetic.dataset_export.quality',fail_one): report=export(root)
            self.assertEqual(report['rejected_rgb'],7)
            self.assertEqual(report['groups']['detail-defects']['splits']['train']['images'],7)
            self.assertEqual(set(report['rejected_groups']),{'train-d0000'})


if __name__=='__main__': unittest.main()

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"src"))
from mes_vision.synthetic.reinforcement import make_plan, crop_variant, acceptable
from mes_vision.synthetic.planning import fingerprint, CODES
from mes_vision.synthetic.reinforcement_export import crop_labels, group_tasks
import numpy as np


class ReinforcementTests(unittest.TestCase):
    def test_pairs_and_fresh_split_independence(self):
        plan = make_plan()
        self.assertEqual(len(plan["tasks"]), 588)
        self.assertEqual(fingerprint(plan), fingerprint(make_plan()))
        groups = {}
        for task in plan["tasks"]:
            groups.setdefault(task["group"], []).append(task)
        for tasks in groups.values():
            self.assertEqual(len({t["split"] for t in tasks}), 1)
            self.assertEqual({t["objects"][0]["code"] for t in tasks}, set(CODES))
            self.assertEqual(len({fingerprint(t["condition"]) for t in tasks}), 1)
        # Changing training volume must not change held-out render conditions.
        a = [t for t in plan["tasks"] if t["split"] != "train"]
        b = [t for t in make_plan((50,18,18))["tasks"] if t["split"] != "train"]
        self.assertEqual(a, b)

    def test_crop_preserves_counterpart_sites_and_repeats(self):
        objects = [[400,200,300,260]]*7
        defects = [[430,230,25,25], [630,390,30,30]]
        for i in range(4):
            crop = crop_variant("group", i, objects, defects)
            self.assertTrue(acceptable(crop["bounds"],objects,defects))
            self.assertEqual(crop,crop_variant("group",i,objects,defects))
        for i in range(3):
            self.assertTrue(acceptable(crop_variant("group",i,objects,defects,stress=True)["bounds"], objects, defects))

    def test_cut_defect_or_context_or_background_only_rejected(self):
        objects = [[100,100,100,100]]
        self.assertFalse(acceptable([100,100,190,200],objects,[[180,160,15,20]]))
        self.assertFalse(acceptable([100,100,200,200],objects,[[102,130,10,10]]))
        self.assertFalse(acceptable([110,110,190,190],objects,[]))
        self.assertFalse(acceptable([-1,100,200,200],objects,[]))

    def test_crop_pixels_and_label_round_trip(self):
        rgb = np.arange(40*50*3,dtype=np.uint8).reshape(40,50,3)
        mask = np.zeros((40,50),dtype=np.uint8)
        mask[15:20,18:25] = 255
        crop, cut, labels = crop_labels(rgb,mask,[10,8,35,30],[dict(code="NG01",bbox=[16,13,11,9])])
        np.testing.assert_array_equal(crop,rgb[8:30,10:35])
        self.assertEqual(labels[0]["bbox"],[6,5,11,9])
        self.assertEqual(int(np.count_nonzero(cut)),35)
        with self.assertRaisesRegex(ValueError,"truncates"):
            crop_labels(rgb,mask,[20,8,35,30],[dict(code="NG01",bbox=[16,13,11,9])])

    def test_normal_cannot_have_unlabelled_defect_mask(self):
        rgb = np.zeros((20,20,3),dtype=np.uint8)
        mask = np.zeros((20,20),dtype=np.uint8)
        _,_,labels = crop_labels(rgb,mask,[0,0,20,20],[])
        self.assertEqual(labels,[])
        mask[5,5] = 255
        with self.assertRaisesRegex(ValueError,"disagrees"):
            crop_labels(rgb,mask,[0,0,20,20],[])

    def test_split_leak_or_mismatched_pair_rejected(self):
        plan = make_plan((1,1,1))
        plan["tasks"][1]["split"] = "test"
        with self.assertRaisesRegex(ValueError,"crosses"):
            group_tasks(plan)
        plan = make_plan((1,1,1))
        plan["tasks"][1] = {**plan["tasks"][1],"camera":{"different":True}}
        with self.assertRaisesRegex(ValueError,"differ"):
            group_tasks(plan)

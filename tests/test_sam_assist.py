import unittest,tempfile,json
from pathlib import Path
import numpy as np
from mes_vision.data_management.sam_assist import mask_box,validate_prompt


class SamGeometryTests(unittest.TestCase):
    def test_retained_mask_is_copied_with_provenance(self):
        from PIL import Image
        from mes_vision.data_management.collection import Collection
        from mes_vision.data_management.sam_dialog import retain_proposal
        from mes_vision.training.data import sha256
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);collection=Collection.create(root/'collection','test',kind='synthetic')
            mask=root/'mask.png';Image.fromarray(np.ones((5,6),dtype=np.uint8)*255).save(mask)
            candidate={'mask':'mask.png','sha256':sha256(mask),'box':[0,0,6,5]}
            entry=retain_proposal(collection,{'directory':str(root),'candidate':candidate,'result':{'kind':'synthetic-test'}})
            self.assertEqual(sha256(collection.root/entry['mask']),sha256(mask))
            self.assertFalse(entry['reviewed_automatically'])
            mask.write_bytes(b'changed')
            with self.assertRaises(ValueError):retain_proposal(collection,{'directory':str(root),'candidate':candidate,'result':{}})

    def test_mask_box_uses_exclusive_right_bottom(self):
        mask=np.zeros((30,50),dtype=bool);mask[7:15,9:22]=True
        self.assertEqual(mask_box(mask),[9,7,22,15])
        mask[:]=False;mask[29,49]=True
        self.assertEqual(mask_box(mask),[49,29,50,30])

    def test_empty_and_nonbinary_rejected(self):
        for mask in [np.zeros((3,3),dtype=bool),np.ones((3,3),dtype=float),np.ones((3,3,3),dtype=bool)]:
            with self.assertRaises(ValueError):mask_box(mask)

    def test_positive_negative_and_box_prompts(self):
        validate_prompt({'points':[[10,12],[4,3]],'labels':[1,0]},50,30)
        validate_prompt({'box':[0,0,50,30]},50,30)

    def test_invalid_or_negative_only_prompt_rejected(self):
        for request in [{},{'points':[[2,2]],'labels':[0]},
                        {'points':[[50,2]],'labels':[1]},
                        {'points':[[float('nan'),2]],'labels':[1]},
                        {'points':[[2,2]],'labels':[]},{'box':[8,4,2,12]},
                        {'box':[0,0,51,30]}]:
            with self.assertRaises(ValueError):validate_prompt(request,50,30)

if __name__=='__main__':unittest.main()

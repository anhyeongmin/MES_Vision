from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from mes_vision.anomaly.features import DinoFeatures


class DinoBatchTests(unittest.TestCase):
    def test_device_features_keep_same_values_and_batch_limits(self):
        import torch
        extractor,groups=self.extractor(2)
        images=[np.full((15+i,21+i,3),i*60,np.uint8) for i in range(3)]
        device=extractor.extract_many_device(images); self.assertEqual(groups,[2,1])
        host=extractor.extract_many(images)
        for a,b in zip(device,host):
            self.assertIsInstance(a,torch.Tensor); self.assertEqual(a.dtype,torch.float32)
            np.testing.assert_array_equal(a.numpy(),b)

    def test_reference_signature_distinguishes_batch_capacity(self):
        with patch.object(DinoFeatures,'verify_assets',return_value={}):
            single=DinoFeatures('unused',device='cpu',max_batch_size=1)
            batch=DinoFeatures('unused',device='cpu',max_batch_size=4)
        self.assertNotIn('batch_limit',single.signature)
        self.assertEqual(batch.signature['batch_limit'],4)
        self.assertEqual({k:v for k,v in batch.signature.items() if k!='batch_limit'},single.signature)
    def extractor(self,limit):
        import torch
        groups=[]
        class Model:
            def __call__(self,pixel_values):
                groups.append(pixel_values.shape[0])
                self_dtype=pixel_values.dtype
                if self_dtype!=torch.float32: raise AssertionError('precision changed')
                means=pixel_values.mean(dim=(2,3))
                tokens=torch.ones((len(means),65,384),dtype=torch.float32)
                tokens[:,:,:3]=means[:,None,:]+4
                return SimpleNamespace(last_hidden_state=tokens)
        extractor=object.__new__(DinoFeatures)
        extractor.device='cpu'; extractor.image_size=112; extractor.max_batch_size=limit; extractor.model=Model()
        return extractor,groups
    def test_batch_keeps_input_order_and_single_results(self):
        extractor,groups=self.extractor(4)
        images=[np.full((15+i,21+i,3),i*60,np.uint8) for i in range(4)]
        batch=extractor.extract_many(images); self.assertEqual(groups,[4])
        for rgb,result in zip(images,batch,strict=True):
            np.testing.assert_allclose(result,extractor.extract(rgb),atol=1e-7)
            self.assertEqual(result.shape,(8,8,384)); self.assertEqual(result.dtype,np.float32)
        self.assertFalse(np.array_equal(batch[0],batch[3]))
    def test_limit_and_empty_validation(self):
        extractor,groups=self.extractor(2); images=[np.zeros((5,5,3),np.uint8)]*3
        self.assertEqual(extractor.extract_many(()),()); self.assertEqual(groups,[])
        extractor.extract_many(images); self.assertEqual(groups,[2,1])
        with self.assertRaises(ValueError): extractor.extract_many(images*2)
        with self.assertRaises(ValueError): extractor.extract_many([np.zeros((5,5,3),np.float32)])


if __name__=='__main__': unittest.main()

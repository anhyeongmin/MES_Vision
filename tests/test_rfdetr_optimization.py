import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend,model_reference


class OptimizationTests(unittest.TestCase):
    def test_cuda_default_and_cpu_default(self):
        a=RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general')
        b=RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general',device='cpu')
        self.assertEqual(a.inference_profile,'fp32_jit'); self.assertEqual(b.inference_profile,'standard')
    def test_execution_profiles_have_different_policy_identities(self):
        refs={model_reference('a'*64,'product_objects',inference_profile=p) for p in ('standard','fp32_jit','fp16','fp16_jit')}
        self.assertEqual(len(refs),4)
        self.assertEqual(model_reference('a'*64,'product_objects').version,'1.9.4+fp32_jit')
        self.assertEqual(model_reference('a'*64,'product_objects',inference_profile='standard').version,'1.9.4')
    def test_invalid_or_cpu_optimized_profile_rejected(self):
        for profile in ('INT8',[],False):
            with self.assertRaises(ValueError): RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general',inference_profile=profile)
        with self.assertRaises(ValueError): RFDETRBackend('unused','a'*64,threshold=.4,training_scope='coco_general',device='cpu',inference_profile='fp16')
    def test_load_optimizes_once_and_keeps_file(self):
        import torch
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'weights.pth'; path.write_bytes(b'fixture'); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            calls=[]
            network=SimpleNamespace(inference=lambda **kwargs:calls.append(kwargs))
            with patch('mes_vision.inspection.rfdetr_adapter.version',return_value='1.9.4'),patch.dict('sys.modules',{'rfdetr':SimpleNamespace(RFDETRSmall=lambda **kwargs:network)}):
                backend=RFDETRBackend(path,digest,threshold=.4,training_scope='coco_general')
                backend.load(); backend.load()
                self.assertEqual(calls,[{'compile':True,'batch_size':1,'dtype':torch.float32}])
                self.assertEqual(path.read_bytes(),b'fixture')
                backend.close(); self.assertIsNone(backend._network)
    def test_optimization_failure_does_not_publish_half_loaded_network(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'weights.pth'; path.write_bytes(b'fixture'); digest=hashlib.sha256(path.read_bytes()).hexdigest()
            def fail(**kwargs): raise RuntimeError('compile failed')
            network=SimpleNamespace(inference=fail)
            with patch('mes_vision.inspection.rfdetr_adapter.version',return_value='1.9.4'),patch.dict('sys.modules',{'rfdetr':SimpleNamespace(RFDETRSmall=lambda **kwargs:network)}):
                backend=RFDETRBackend(path,digest,threshold=.4,training_scope='coco_general')
                with self.assertRaisesRegex(RuntimeError,'compile failed'): backend.load()
                self.assertIsNone(backend._network)


if __name__=='__main__': unittest.main()

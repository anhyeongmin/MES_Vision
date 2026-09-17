from types import SimpleNamespace
import unittest
from unittest.mock import patch
import torch
from mes_vision.vlm.optimization import configure_patch_projection,linear_patch_forward,validate_patch


class VlmProjectionTests(unittest.TestCase):
    def make_patch(self):
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionPatchEmbed
        return Qwen3_5VisionPatchEmbed(SimpleNamespace(patch_size=2,temporal_patch_size=2,in_channels=3,hidden_size=5)).double().eval()

    def model(self,projection):
        return SimpleNamespace(model=SimpleNamespace(visual=SimpleNamespace(patch_embed=projection)))

    def test_linear_uses_same_layout_parameters_and_bias(self):
        projection=self.make_patch();validate_patch(projection)
        torch.manual_seed(11)
        state={k:v.clone() for k,v in projection.state_dict().items()}
        with torch.inference_mode():
            for count in (1,3,17):
                values=torch.randn(count,24,dtype=torch.float64)
                torch.testing.assert_close(linear_patch_forward(projection,values),projection(values),rtol=1e-12,atol=1e-12)
                noncontiguous=values.T.contiguous().T
                torch.testing.assert_close(linear_patch_forward(projection,noncontiguous),projection(values),rtol=1e-12,atol=1e-12)
        for key,value in state.items(): self.assertTrue(torch.equal(value,projection.state_dict()[key]))

    def test_configure_is_idempotent_and_restores_class_forward(self):
        projection=self.make_patch();model=self.model(projection)
        original_weight=projection.proj.weight
        first=configure_patch_projection(model,'linear_patch_v1')
        forward=projection.forward
        self.assertEqual(configure_patch_projection(model,'linear_patch_v1'),first)
        self.assertIs(projection.forward,forward)
        self.assertIs(projection.proj.weight,original_weight)
        configure_patch_projection(model,'standard');configure_patch_projection(model,'standard')
        self.assertNotIn('forward',projection.__dict__)
        self.assertNotIn('_mes_original_forward',projection.__dict__)

    def test_original_instance_override_is_restored(self):
        projection=self.make_patch();model=self.model(projection)
        original=lambda values: values
        projection.forward=original
        configure_patch_projection(model,'linear_patch_v1');configure_patch_projection(model,'standard')
        self.assertIs(projection.forward,original)

    def test_instance_scope_does_not_modify_other_models(self):
        first=self.make_patch();second=self.make_patch()
        configure_patch_projection(self.model(first),'linear_patch_v1')
        self.assertNotIn('forward',second.__dict__)
        configure_patch_projection(self.model(first),'standard')

    def test_geometry_changes_are_rejected(self):
        for field,value in (('stride',(1,1,1)),('padding',(1,1,1)),('dilation',(2,2,2)),('groups',3)):
            projection=self.make_patch();setattr(projection.proj,field,value)
            with self.assertRaisesRegex(ValueError,'geometry'): validate_patch(projection)

    def test_runtime_source_and_unknown_profile_rejected(self):
        projection=self.make_patch();model=self.model(projection)
        with self.assertRaises(ValueError): configure_patch_projection(model,'unknown')
        with patch('mes_vision.vlm.optimization.importlib.metadata.version',return_value='different'):
            with self.assertRaisesRegex(ValueError,'runtime'): configure_patch_projection(model,'linear_patch_v1')
        with patch('mes_vision.vlm.optimization.SOURCE_SHA256','0'*64):
            with self.assertRaisesRegex(ValueError,'implementation changed'): configure_patch_projection(model,'linear_patch_v1')
        self.assertNotIn('forward',projection.__dict__)

    def test_backend_rejects_unknown_profile_before_loading_assets(self):
        from mes_vision.vlm.backend import QwenBackend
        with self.assertRaisesRegex(ValueError,'profile'): QwenBackend('unused',inference_profile='unknown')


if __name__=='__main__':unittest.main()

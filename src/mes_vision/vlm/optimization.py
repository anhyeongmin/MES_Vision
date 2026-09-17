"""Pinned Qwen patch projection: one whole patch per convolution output."""
import importlib.metadata
import hashlib
from pathlib import Path
from types import MethodType
from mes_vision.training.data import require

PROFILES=('standard','linear_patch_v1')
SOURCE_SHA256='788d4bad50a8d39be2fe79125f0f40134773cd23b1791606fb6b3ab0bc6d2263'


def linear_patch_forward(self, hidden_states):
    from torch.nn.functional import linear
    patch_width=self.in_channels*self.temporal_patch_size*self.patch_size*self.patch_size
    # Conv3d has exactly one output position for each input patch. Reuse its
    # original parameters; only the execution kernel changes.
    patches=hidden_states.reshape(-1,patch_width).to(dtype=self.proj.weight.dtype)
    return linear(patches,self.proj.weight.flatten(1),self.proj.bias)


def validate_patch(patch):
    import torch
    conv=patch.proj
    kernel=(patch.temporal_patch_size,patch.patch_size,patch.patch_size)
    require(isinstance(conv,torch.nn.Conv3d) and conv.groups==1
            and tuple(conv.kernel_size)==kernel and tuple(conv.stride)==kernel
            and tuple(conv.padding)==(0,0,0) and tuple(conv.dilation)==(1,1,1)
            and conv.padding_mode=='zeros' and conv.in_channels==patch.in_channels
            and conv.out_channels==patch.embed_dim,
            'Unsupported Qwen patch projection geometry')
    require(tuple(conv.weight.shape)==(patch.embed_dim,patch.in_channels,*kernel),
            'Qwen patch weight layout changed')


def configure_patch_projection(model,profile):
    require(profile in PROFILES,'Unsupported VLM inference profile')
    patch=model.model.visual.patch_embed
    if profile=='standard':
        if hasattr(patch,'_mes_original_forward'):
            original=patch._mes_original_forward
            if original is None: del patch.forward
            else: patch.forward=original
            del patch._mes_original_forward
        return {'profile':profile,'vision_patch_projection':'conv3d'}
    from transformers.models.qwen3_5 import modeling_qwen3_5 as module
    require(importlib.metadata.version('transformers')=='5.16.1'
            and importlib.metadata.version('torch')=='2.9.1+cu128',
            'VLM projection optimization requires the validated runtime versions')
    require(hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()==SOURCE_SHA256,
            'Qwen implementation changed; revalidate the projection optimization')
    require(type(patch) is module.Qwen3_5VisionPatchEmbed,'Unsupported Qwen patch module')
    validate_patch(patch)
    if not hasattr(patch,'_mes_original_forward'):
        patch._mes_original_forward=patch.__dict__.get('forward')
        patch.forward=MethodType(linear_patch_forward,patch)
    return {'profile':profile,'vision_patch_projection':'linear_same_conv3d_parameters',
            'implementation_sha256':SOURCE_SHA256,'numerically_bitwise_equivalent':False}

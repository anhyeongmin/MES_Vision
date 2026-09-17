"""Explicit, code-registered model providers. Configuration never imports code.

Providers return the existing Detector/Inspector contracts, so replacing weights
or architecture does not grant a model authority over decisions or robot motion.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import Callable
import numpy as np
from mes_vision.training.data import require


@dataclass
class ModelHandle:
    resource: object
    adapter: object
    warmup: Callable[[], None]
    closed: bool = False

    def load(self):
        require(not self.closed,'Model handle has been closed')
        try:
            self.resource.load(); self.warmup()
        except BaseException:
            self.close()
            raise

    def close(self):
        if not self.closed:
            self.closed=True; self.resource.close()


@dataclass(frozen=True)
class Provider:
    name: str
    roles: frozenset[str]
    code_license: str
    weights_license: str
    build: Callable[[dict, str], ModelHandle]


class ModelRegistry:
    def __init__(self): self._providers = {}

    def register(self, provider):
        require(isinstance(provider, Provider) and provider.name not in self._providers, 'Duplicate or invalid model provider')
        require(provider.roles and provider.roles <= {'objects','defects','anomaly'} and callable(provider.build), 'Invalid provider roles')
        require(provider.code_license and provider.weights_license, 'Code and weight license references are required')
        self._providers[provider.name] = provider

    def create(self, spec, role):
        require(isinstance(spec, dict), 'Model registration is required')
        name = spec.get('backend', 'rfdetr-small' if role in {'objects','defects'} else None)
        require(isinstance(name,str) and name in self._providers, 'Unsupported model backend: '+str(name))
        provider = self._providers[name]
        require(role in provider.roles, 'Model backend does not support this role')
        handle = provider.build(deepcopy(spec), role)
        require(isinstance(handle,ModelHandle), 'Invalid model resource contract')
        required = 'detect' if role == 'objects' else 'inspect'
        if not callable(getattr(handle.adapter,required,None)) or not hasattr(handle.adapter,'model'):
            handle.close(); raise ValueError('Model adapter does not implement '+required)
        if role!='objects' and getattr(handle.adapter,'check_id',None)!={'defects':'known_defects','anomaly':'anomaly'}[role]:
            handle.close(); raise ValueError('Inspector check identity does not match its registered role')
        return handle

    def describe(self):
        return tuple({'backend':p.name,'roles':sorted(p.roles),'code_license':p.code_license,
                      'weights_license':p.weights_license} for p in self._providers.values())


def _rfdetr(spec, role):
    from .rfdetr_adapter import RFDETRBackend,RFDETRDetector,RFDETRDefectInspector
    options = {k:spec[k] for k in ('device','inference_profile','max_batch_size') if k in spec}
    backend = RFDETRBackend(spec['weights'],spec['sha256'],threshold=spec['threshold'],
                           training_scope='product_'+role,class_names=tuple(spec['class_names']),**options)
    adapter = RFDETRDetector(backend) if role == 'objects' else RFDETRDefectInspector(
        backend,{int(k):v for k,v in spec['class_codes'].items()})
    def warmup():
        for size in (1,2,4):
            if size <= backend.max_batch_size:
                backend.predict_many_rgb([np.zeros((512,512,3),np.uint8)]*size)
    return ModelHandle(backend,adapter,warmup)


def default_registry():
    registry = ModelRegistry()
    registry.register(Provider('rfdetr-small',frozenset({'objects','defects'}),
                              'Apache-2.0','Apache-2.0 (RF-DETR Small only)',_rfdetr))
    return registry

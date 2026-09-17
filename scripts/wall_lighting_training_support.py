"""Training-only lighting variation, identical for normal and defective classes."""
import random
import numpy as np
from PIL import Image
from rotation_training_support import RotationAugment, SETTINGS as BASE, self_test as rotation_test

SETTINGS=dict(BASE,lighting_probability=.6,gamma_range=[.8,1.25],
              directional_shading_min=.65,lighting_class_independent=True)

def relight(image,gamma,axis,strength,reverse):
    a=np.asarray(image,dtype=np.float32)/255.
    h,w=a.shape[:2]
    ramp=np.linspace(strength,1.,w if axis==0 else h,dtype=np.float32)
    if reverse:ramp=ramp[::-1]
    field=ramp[None,:,None] if axis==0 else ramp[:,None,None]
    return Image.fromarray(np.uint8(np.clip(a**gamma*field*255.,0,255)))

class LightingAugment(RotationAugment):
    def __call__(self,image,target):
        # No added geometry or class-dependent effects: normal remains unlabelled.
        if random.random()<.6:
            image=relight(image,random.uniform(.8,1.25),random.randrange(2),
                          random.uniform(.65,1.),random.choice([False,True]))
        return super().__call__(image,target)

def install(dm):
    original=dm.setup
    def setup(stage):
        original(stage)
        dataset=dm._dataset_train
        if dataset is not None and not isinstance(dataset._transforms,LightingAugment):
            dataset._transforms=LightingAugment(dataset._transforms)
    dm.setup=setup

def self_test():
    rotation_test()
    for axis in (0,1):
        im=Image.fromarray(np.full((40,60,3),128,np.uint8))
        out=np.asarray(relight(im,1,axis,.65,False))
        assert out.shape==(40,60,3) and out.dtype==np.uint8
        assert out.min()<out.max() and out.max()<=128
    print('Lighting augmentation preserves geometry; applies equally to every class')

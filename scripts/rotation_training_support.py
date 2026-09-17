"""CPU augmentation before RF-DETR resize/normalization; absolute XYXY boxes."""
import random
import numpy as np
import torch
import cv2
from PIL import Image, ImageEnhance, ImageFilter

SETTINGS = dict(quarter_turns=[0,90,180,270],jitter_probability=.3,jitter_degrees=10,
                brightness_probability=.4,brightness_range=[.85,1.15],blur_probability=.15,blur_radius=[.2,.6])

def geometry(image, boxes, turns, angle=0):
    a=np.array(image);h,w=a.shape[:2];b=boxes.clone().reshape(-1,4)
    for _ in range(turns):
        if len(b):b=torch.stack([b[:,1],w-b[:,2],b[:,3],w-b[:,0]],dim=1)
        w,h=h,w
    a=np.rot90(a,turns).copy()
    if angle:
        # Transform edge coordinates, then shift the matrix for pixel-center sampling.
        m=cv2.getRotationMatrix2D((w/2,h/2),angle,1.)
        corners=np.array([[0,0,1],[w,0,1],[w,h,1],[0,h,1]],dtype=float)@m.T
        lo=np.floor(corners.min(0));hi=np.ceil(corners.max(0));m[:,2]-=lo
        ow,oh=(hi-lo).astype(int)
        sampling=m.copy();sampling[:,2]+=(m[:,:2]@np.array([.5,.5]))-.5
        a=cv2.warpAffine(a,sampling,(int(ow),int(oh)),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE)
        if len(b):
            pts=torch.stack([b[:,[0,1]],b[:,[2,1]],b[:,[2,3]],b[:,[0,3]]],dim=1).numpy()
            mapped=pts@m[:,:2].T+m[:,2]
            b=torch.tensor(np.concatenate([mapped.min(1),mapped.max(1)],axis=1),dtype=boxes.dtype)
        w,h=int(ow),int(oh)
    assert len(b)==0 or bool(((b[:,0]>=0)&(b[:,1]>=0)&(b[:,2]<=w+.01)&(b[:,3]<=h+.01)).all())
    return Image.fromarray(a),b

class RotationAugment:
    def __init__(self,following):self.following=following
    def __call__(self,image,target):
        assert 'masks' not in target and 'keypoints' not in target, 'Box-only augmentation'
        turns=random.randrange(4);angle=random.uniform(-10,10) if random.random()<.3 else 0
        image,boxes=geometry(image,target['boxes'],turns,angle)
        target=dict(target);target['boxes']=boxes
        target['area']=(boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
        target['size']=torch.tensor([image.height,image.width])
        if random.random()<.4:image=ImageEnhance.Brightness(image).enhance(random.uniform(.85,1.15))
        if random.random()<.15:image=image.filter(ImageFilter.GaussianBlur(random.uniform(.2,.6)))
        return self.following(image,target)

def install(dm):
    original=dm.setup
    def setup(stage):
        original(stage)
        dataset=dm._dataset_train
        if dataset is not None and not isinstance(dataset._transforms,RotationAugment):
            dataset._transforms=RotationAugment(dataset._transforms)
    dm.setup=setup

def self_test():
    im=Image.fromarray(np.zeros((10,20,3),dtype=np.uint8));box=torch.tensor([[1.,2.,5.,6.]])
    for r,expected in [(1,[2,15,6,19]),(2,[15,4,19,8]),(3,[4,1,8,5])]:
        _,b=geometry(im,box,r);assert b.tolist()==[expected]
    for r in range(4):
        for angle in [-10,0,10]:
            _,b=geometry(im,box,r,angle);assert torch.isfinite(b).all()
            _,empty=geometry(im,torch.empty((0,4)),r,angle);assert empty.shape==(0,4)
    # Compare transformed foreground pixels against transformed box, allowing interpolation edges.
    a=np.zeros((80,100,3),dtype=np.uint8);a[20:40,10:35]=255
    for angle in [-10,10]:
        pic,b=geometry(Image.fromarray(a),torch.tensor([[10.,20.,35.,40.]]),1,angle)
        yy,xx=np.where(np.array(pic)[:,:,0]>127);x1,y1,x2,y2=b[0].tolist()
        assert xx.min()>=x1-1 and yy.min()>=y1-1 and xx.max()<=x2+1 and yy.max()<=y2+1
    print('Rotation box geometry and empty-normal tests passed')

if __name__=='__main__':self_test()

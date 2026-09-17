"""Paired inference-only undistortion ablation with fixed frames/models/threshold."""
from pathlib import Path
import sys,os,json,gc
from contextlib import ExitStack
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key,folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch'}.items():os.environ[key]=str(ROOT/'.cache'/folder)
os.environ['HF_HUB_OFFLINE']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from mes_vision.training.data import sha256
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
from mes_vision.vlm.gpu import GpuCoordinator
from filelock import FileLock

def corrected_crop(rgb,box,K,D):
 x,y,x2,y2=box;t=np.linspace(0,1,257)
 edges=np.concatenate([np.c_[x+(x2-x)*t,np.full_like(t,y)],np.c_[x+(x2-x)*t,np.full_like(t,y2)],np.c_[np.full_like(t,x),y+(y2-y)*t],np.c_[np.full_like(t,x2),y+(y2-y)*t]])
 points=cv2.undistortPoints(edges.reshape(-1,1,2),K,D,P=K,criteria=(3,200,1e-10)).reshape(-1,2)
 assert np.isfinite(points).all()
 lo=np.floor(points.min(0)).astype(int);hi=np.ceil(points.max(0)).astype(int);size=hi-lo
 assert np.all(size>0) and np.all(size<5000)
 # Shift the rectified output origin, retaining original full-frame intrinsics.
 target=K.copy();target[:2,2]-=lo
 mx,my=cv2.initUndistortRectifyMap(K,D,None,target,tuple(size),cv2.CV_32FC1)
 output=cv2.remap(rgb,mx,my,cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
 valid=(mx>=0)&(mx<=rgb.shape[1]-1)&(my>=0)&(my<=rgb.shape[0]-1)
 normalized=np.c_[(points[:,0]-K[0,2])/K[0,0],(points[:,1]-K[1,2])/K[1,1],np.ones(len(points))]
 back,_=cv2.projectPoints(normalized,np.zeros(3),np.zeros(3),K,D)
 err=float(np.max(np.linalg.norm(back.reshape(-1,2)-edges,axis=1)));assert err<.02
 return output,{'rectified_box':[*lo.tolist(),*hi.tolist()],'valid_source_fraction':float(valid.mean()),'boundary_roundtrip_max_px':err}

def main():
 import torch
 base=ROOT/'artifacts/real-eval-20260914';out=base/'undistortion';out.mkdir(exist_ok=False)
 calpath=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json';cal=json.loads(calpath.read_text());K=np.array(cal['K']);D=np.array(cal['D'])
 jobs=json.loads((base/'frozen-selection.json').read_text());assert len(jobs)==25
 for j in jobs:
  assert sha256(Path(j['crop']))==j['crop_sha256']
  rgb=np.array(Image.open(base/'frames'/(j['name']+'.png')).convert('RGB'));assert rgb.shape==(720,1280,3)
  crop,info=corrected_crop(rgb,j['box'],K,D);dest=out/(j['name']+'.png');Image.fromarray(crop).save(dest)
  j.update(corrected_crop=str(dest),corrected_sha256=sha256(dest),**info)
 (out/'manifest.json').write_text(json.dumps({'calibration_sha256':sha256(calpath),'selection_sha256':sha256(base/'frozen-selection.json'),'K':K.tolist(),'D':D.tolist(),'interpolation':'bilinear','model_threshold':.1,'training_performed':False,'method':'Rectify original full-frame coordinates then extract transformed ROI; no crop-local intrinsics; no auto zoom or same-box reuse','jobs':jobs},indent=2))
 models={'baseline':('artifacts/training/ash-reinforced-v3/detail-defects/inference.pth','d748f4a1736568553a17c8196f7d0f5a8b92f36ceb389fa7b779f3326c8f6b37'),'bootstrap':('artifacts/training/ash-real-bootstrap-v1/detail-defects/inference-unvalidated.pth','b865c613dc43140c97deefef64c5325d6709150200ddaab34034561a887b1975')}
 codes=tuple(f'NG{i:02}' for i in range(1,7));coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
 with ExitStack() as stack:
  stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
  for model,(path,digest) in models.items():
   backend=RFDETRBackend(ROOT/path,digest,threshold=.1,training_scope='product_defects',class_names=codes,inference_profile='standard',max_batch_size=1);backend.load();results=[]
   for j in jobs:
    pair={}
    for variant,key in [('raw','crop'),('corrected','corrected_crop')]:
     ds=backend.predict_rgb(np.array(Image.open(j[key]).convert('RGB')));pair[variant]=[asdict(d) for d in ds]
    results.append(dict(j,predictions=pair))
   (out/(model+'-predictions.json')).write_text(json.dumps(results,indent=2));backend.close();del backend;gc.collect();torch.cuda.empty_cache();print(model,'complete',flush=True)

if __name__=='__main__':main()

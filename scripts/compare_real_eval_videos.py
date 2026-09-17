"""Frozen crop-level defect comparison on new videos, never training data."""
from pathlib import Path
import sys,os,json,shutil,gc
from dataclasses import asdict
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key,folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch'}.items():os.environ[key]=str(ROOT/'.cache'/folder)
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from mes_vision.training.data import sha256
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
from mes_vision.vlm.gpu import GpuCoordinator
from filelock import FileLock
# Boxes selected from 320x180 overview frames, prior to any predictions.
SELECT={
 'OK':{1:[117,40,210,122],9:[68,42,123,108],11:[205,40,272,110]},
 'NG01':{0:[109,32,180,120],7:[52,20,134,112],11:[112,12,207,92]},
 'NG02':{0:[104,36,187,133],4:[109,50,207,130],11:[54,39,134,119]},
 'NG03':{0:[117,31,194,124],3:[56,34,131,125],9:[65,42,155,117],11:[154,51,266,162]},
 'NG04':{0:[96,22,181,129],3:[29,14,110,118],11:[29,41,121,119]},
 'NG05':{1:[196,40,265,121],3:[78,88,151,171],6:[45,1,131,75],11:[64,14,162,103]},
 'NG06':{0:[109,42,183,130],3:[60,80,136,166],5:[70,0,159,80],7:[205,0,285,68],11:[71,110,160,179]}}

def main():
 import torch
 out=ROOT/'artifacts/real-eval-20260914';out.mkdir(exist_ok=True)
 for folder in ['originals','frames','crops']: (out/folder).mkdir(exist_ok=True)
 jobs=[]
 for code,choices in SELECT.items():
  source=Path('C:/Users/alexi/OneDrive/사진/카메라 앨범')/(code+'_1.mp4');dest=out/'originals'/source.name
  if not dest.exists():shutil.copyfile(source,dest)
  digest=sha256(source);assert sha256(dest)==digest
  c=cv2.VideoCapture(str(dest));n=int(c.get(7));fps=c.get(5)
  for index,b in choices.items():
   frame=round((n-1)*(index+.5)/12);c.set(cv2.CAP_PROP_POS_FRAMES,frame);ok,a=c.read();assert ok and a.shape[:2]==(720,1280)
   box=[v*4 for v in b];im=Image.fromarray(cv2.cvtColor(a,cv2.COLOR_BGR2RGB));name=f'{code}-{frame}';im.save(out/'frames'/(name+'.png'));crop=im.crop(box);crop.save(out/'crops'/(name+'.png'))
   jobs.append({'name':name,'expected_code':code,'frame_index':frame,'estimated_seconds':frame/fps,'source_sha256':digest,'box':box,'crop':str(out/'crops'/(name+'.png')),'crop_sha256':sha256(out/'crops'/(name+'.png'))})
  c.release()
 (out/'frozen-selection.json').write_text(json.dumps(jobs,indent=2))
 sheet=Image.new('RGB',(1400,230*((len(jobs)+6)//7)),'white');draw=ImageDraw.Draw(sheet)
 for i,j in enumerate(jobs):
  im=Image.open(j['crop']);im.thumbnail((195,205));x=i%7*200;y=i//7*230;sheet.paste(im,(x,y));draw.text((x+2,y+208),j['name'],fill='black')
 sheet.save(out/'selected-crops.jpg')
 models={'baseline':('artifacts/training/ash-reinforced-v3/detail-defects/inference.pth','d748f4a1736568553a17c8196f7d0f5a8b92f36ceb389fa7b779f3326c8f6b37'),
         'bootstrap':('artifacts/training/ash-real-bootstrap-v1/detail-defects/inference-unvalidated.pth','b865c613dc43140c97deefef64c5325d6709150200ddaab34034561a887b1975')}
 codes=tuple(f'NG{i:02}' for i in range(1,7));coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
 with ExitStack() as stack:
  stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
  for model,(path,digest) in models.items():
   backend=RFDETRBackend(ROOT/path,digest,threshold=.1,training_scope='product_defects',class_names=codes,inference_profile='standard',max_batch_size=1);backend.load();results=[]
   for j in jobs:
    detections=backend.predict_rgb(np.array(Image.open(j['crop']).convert('RGB')))
    results.append(dict(j,predictions=[asdict(d) for d in detections]))
   (out/(model+'-predictions.json')).write_text(json.dumps(results,indent=2))
   backend.close();del backend;gc.collect();torch.cuda.empty_cache();print(model,'complete',len(results),flush=True)
 print('Comparison complete',flush=True)

if __name__=='__main__':main()

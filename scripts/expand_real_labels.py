"""Generate resumable SAM object proposals for the 140 real review frames."""
import os,sys,json
from pathlib import Path
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from filelock import FileLock
from mes_vision.training.data import sha256
from mes_vision.data_management.sam_assist import mask_box
from mes_vision.vlm.gpu import GpuCoordinator

def prompt_box(picture):
 a=np.asarray(picture).astype(float);h,w=a.shape[:2]
 mask=((a[:,:,2]>a[:,:,1]*1.025)&(a.mean(2)>105)).astype('uint8')
 mask[:,:int(w*.12)]=0;mask[:,int(w*.9):]=0
 mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
 n,labels,stats,centers=cv2.connectedComponentsWithStats(mask)
 candidates=[]
 for s,c in zip(stats[1:],centers[1:]):
  x,y,bw,bh,area=s
  if area<500 or bw<35 or bh<35:continue
  score=area/(1+6*((c[0]/w-.5)**2+(c[1]/h-.5)**2))
  candidates.append((score,[max(0,int(x)-8),max(0,int(y)-8),min(w,int(x+bw)+8),min(h,int(y+bh)+8)]))
 return max(candidates,key=lambda v:v[0])[1] if candidates else [int(w*.25),int(h*.1),int(w*.8),int(h*.95)]

def main():
 import torch
 from transformers import Sam2Model,Sam2Processor
 base=ROOT/'artifacts/real-videos-20260914';out=base/'expanded';out.mkdir(exist_ok=True)
 source=json.loads((base/'review-priority.json').read_text(encoding='utf-8'))
 directory=ROOT/'models/sam2.1-hiera-small';manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
 for name,v in manifest['files'].items():assert sha256(directory/name)==v['sha256']
 coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
 with ExitStack() as stack:
  stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
  model=Sam2Model.from_pretrained(directory,local_files_only=True).to('cuda').eval();proc=Sam2Processor.from_pretrained(directory,local_files_only=True)
  for group in source:
   results=[]
   for index,r in enumerate(group['review_priority']):
    dest=out/(r['record_id']+'.json')
    if dest.exists():results.append(json.loads(dest.read_text()));continue
    im=Image.open(r['image']).convert('RGB');box=prompt_box(im)
    inputs=proc(images=im,input_boxes=[[box]],return_tensors='pt').to('cuda')
    with torch.inference_mode():pred=model(**inputs,multimask_output=True)
    masks=proc.post_process_masks(pred.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][0].numpy().astype(bool)
    scores=pred.iou_scores[0,0].detach().cpu().numpy();candidates=[]
    for k in np.argsort(-scores):
     if not masks[k].any():continue
     path=out/f"{r['record_id']}-mask-{k}.png";Image.fromarray(masks[k].astype('uint8')*255).save(path)
     candidates.append(dict(mask=str(path),sha256=sha256(path),box=mask_box(masks[k]),score=float(scores[k])))
    row=dict(r,code=group['source_code_hint'],index=index,prompt_box=box,candidates=candidates,image_sha256=sha256(Path(r['image'])),model_sha256=manifest['files']['model.safetensors']['sha256'],status='UNREVIEWED')
    dest.write_text(json.dumps(row,indent=2));results.append(row)
   sheet=Image.new('RGB',(1500,5*260),'white');d=ImageDraw.Draw(sheet)
   for i,r in enumerate(results):
    im=Image.open(r['image']).convert('RGB');b=r['candidates'][0]['box'];x,y,x2,y2=b
    crop=im.crop((max(0,x-10),max(0,y-10),min(im.width,x2+10),min(im.height,y2+10)));crop.thumbnail((295,230))
    px=i%5*300;py=i//5*260;sheet.paste(crop,(px,py));d.text((px+3,py+235),f"{i}: {r['time_s']:.1f}s / {x2-x}x{y2-y}",fill='black')
   sheet.save(out/(group['source_code_hint']+'-crops.jpg'))
   print(group['source_code_hint'],len(results),flush=True)

if __name__=='__main__':main()

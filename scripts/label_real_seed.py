"""Visually prompted initial real-image annotation drafts; no auto review/training."""
from pathlib import Path
import sys,json,os
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
import numpy as np
from PIL import Image,ImageDraw
from filelock import FileLock
from mes_vision.training.data import sha256
from mes_vision.data_management.sam_assist import mask_box
from mes_vision.vlm.gpu import GpuCoordinator

# Prompts measured on displayed 1/3-size full frames; multiply back to source.
BOXES={1:[163,33,301,198],2:[160,14,324,210],3:[140,0,313,180],
4:[141,18,307,220],5:[137,8,312,217],7:[198,50,299,168],
8:[153,27,316,224],9:[154,5,312,187],10:[167,33,335,240],
12:[144,8,296,192],13:[174,41,303,189]}

def main():
 import torch
 from transformers import Sam2Model,Sam2Processor
 out=ROOT/'artifacts/real-videos-20260914/labeling';selected=json.loads((out/'selected.json').read_text())
 directory=ROOT/'models/sam2.1-hiera-small';manifest=json.loads((directory/'manifest.json').read_text())
 for name,v in manifest['files'].items(): assert sha256(directory/name)==v['sha256']
 coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
 results=[]
 with ExitStack() as stack:
  stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
  model=Sam2Model.from_pretrained(directory,local_files_only=True).to('cuda').eval();proc=Sam2Processor.from_pretrained(directory,local_files_only=True)
  for i,box in BOXES.items():
   r=selected[i];im=Image.open(r['image']).convert('RGB');box=[min(v*3,[1280,720,1280,720][j]) for j,v in enumerate(box)]
   inputs=proc(images=im,input_boxes=[[box]],return_tensors='pt').to('cuda')
   with torch.inference_mode():pred=model(**inputs,multimask_output=True)
   masks=proc.post_process_masks(pred.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][0].numpy().astype(bool)
   scores=pred.iou_scores[0,0].detach().cpu().numpy();candidates=[]
   for k in np.argsort(-scores):
    if not masks[k].any():continue
    p=out/f"{r['record_id']}-mask-{k}.png";Image.fromarray(masks[k].astype('uint8')*255).save(p)
    candidates.append({'mask':str(p),'sha256':sha256(p),'box':mask_box(masks[k]),'score':float(scores[k])})
   results.append(dict(r,prompt_box=box,candidates=candidates,image_sha256=sha256(Path(r['image'])),model_sha256=manifest['files']['model.safetensors']['sha256']))
   print(r['code'],r['time_s'],flush=True)
 (out/'sam-seed.json').write_text(json.dumps(results,indent=2))
 sheet=Image.new('RGB',(1280,205*((len(results)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(results):
  a=np.array(Image.open(r['image']).convert('RGB'));mask=np.array(Image.open(r['candidates'][0]['mask']))>0;a[mask]=(a[mask]*.7+np.array([0,210,240])*.3).astype('uint8')
  im=Image.fromarray(a);im.thumbnail((320,180));x=i%4*320;y=i//4*205;sheet.paste(im,(x,y));draw.text((x+3,y+182),f"{i}: {r['code']} {r['time_s']:.1f}s",fill='black')
 sheet.save(out/'sam-seed-review.jpg')

if __name__=='__main__':main()

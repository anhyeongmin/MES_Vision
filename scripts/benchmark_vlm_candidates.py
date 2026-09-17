"""Offline candidate VLM benchmark. No production configuration changes."""
from pathlib import Path
import sys,os,json,time,gc,hashlib,argparse,statistics,html
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration,Qwen3VLForConditionalGeneration
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection
SYSTEM='''Inspect the pictured white 3D-printed tray. Normal: two cylindrical bosses in opposite corners, both with open holes; straight walls; no extra internal block, crack or dent. Normal print layer lines are not defects. Candidate categories: OK=normal; NG01=missing boss; NG02=extra material/block; NG03=crack; NG04=deformed wall/reduced boss-to-wall connection; NG05=blocked or malformed hole; NG06=surface dent; UNCERTAIN=not visible enough. Do not infer exact measurements or causes. Return compact JSON only: {"code":"candidate category","observation":"one brief Korean phrase describing visible evidence"}. No thinking or markdown. This is advisory observation, not a production decision.'''
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(p,data):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--model',required=True,choices=['Qwen3.5-4B','Qwen3.5-2B','Qwen3.5-0.8B','Qwen3-VL-2B-Instruct']);parser.add_argument('--repeats',type=int,default=2);args=parser.parse_args()
 torch.set_num_threads(4);assert torch.cuda.is_available()
 out=ROOT/'artifacts/vlm-candidate-comparison-v1';out.mkdir(exist_ok=True)
 source=ROOT/'datasets/ash-recorded-low-exposure-reviewed-v2/provenance.json';meta=json.loads(source.read_text(encoding='utf-8'));rows=[]
 for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
  group=[r for r in meta['items'] if r['label']==code];rows.extend([group[0],group[-1]])
 for r in rows:assert sha(Path(r['crop']))==r['crop_sha256']
 if args.model=='Qwen3.5-4B':directory=ROOT/'models/qwen3.5-4b';revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'
 else:
  record=next(r for r in json.loads((ROOT/'models/vlm-benchmark-downloads.json').read_text()) if r['repo']=='Qwen/'+args.model);directory=Path(record['path']);revision=record['revision']
 provenance={'model':args.model,'revision':revision,'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'transformers':__import__('transformers').__version__,'dtype':'bfloat16','prompt':SYSTEM,'edge':448,'token_limit':64,'repeats':args.repeats,'source_sha256':sha(source),'script_sha256':sha(Path(__file__)),'scope':'14 images, 2 per class; development diagnostic, not independent industrial accuracy; no DETR results given; isolated GPU; preparation+generation+decode included; load and warmup excluded','assets':{p.name:sha(p) for p in directory.glob('*.safetensors')}}
 results=[];coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');dest=out/(args.model+'.json')
 assert not dest.exists(),'Use a new output version rather than overwrite results'
 with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
  cls=Qwen3VLForConditionalGeneration if 'Qwen3-VL' in args.model else Qwen3_5ForConditionalGeneration
  t=time.perf_counter();processor=AutoProcessor.from_pretrained(directory,local_files_only=True,trust_remote_code=False);model=cls.from_pretrained(directory,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,device_map='cuda',attn_implementation='sdpa').eval()
  if 'Qwen3.5' in args.model:provenance['optimization']=configure_patch_projection(model,'linear_patch_v1')
  else:provenance['optimization']='standard sdpa'
  torch.cuda.synchronize();provenance['load_seconds']=time.perf_counter()-t
  try:
   for repeat in range(-1,args.repeats):
    for row in (rows[:1] if repeat==-1 else rows):
     with Image.open(row['crop']) as im:im=im.convert('RGB');im.thumbnail((448,448));im=im.copy()
     messages=[{'role':'system','content':SYSTEM},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'Describe the visible condition.'}]}]
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
     inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda');torch.cuda.synchronize();prep=time.perf_counter()-t
     with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False)
     torch.cuda.synchronize();gen=tokens[:,inputs['input_ids'].shape[1]:];text=processor.batch_decode(gen,skip_special_tokens=True)[0].strip();elapsed=time.perf_counter()-t
     eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];truncated=gen.shape[1]>=64 and int(gen[0,-1]) not in eos
     try:
      parsed=json.loads(text.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'code','observation'} and parsed['code'] in ['OK','UNCERTAIN']+[f'NG{i:02}' for i in range(1,7)] and isinstance(parsed['observation'],str)
     except Exception:parsed=None;valid=False
     results.append(dict(id=row['id'],image=row['crop'],expected=row['label'],repeat=repeat,warmup=repeat==-1,seconds=elapsed,prepare_seconds=prep,tokens=int(gen.shape[1]),truncated=truncated,valid=valid,parsed=parsed,raw=text,peak_mib=torch.cuda.max_memory_allocated()/2**20))
     save(dest,dict(provenance=provenance,results=results));print(args.model,row['label'],repeat,round(elapsed,2),text,flush=True)
  finally:del model;gc.collect();torch.cuda.empty_cache()
 print('FINISHED',dest,flush=True)
if __name__=='__main__':main()

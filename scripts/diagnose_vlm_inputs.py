"""Frozen LoRA input diagnostics; no training and no production changes."""
from train_vlm_lora import ROOT,save,sha
from pathlib import Path
import json,time,gc,statistics,html
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection
run=ROOT/'artifacts/training/vlm-2b-lora-20260915-000817-709363'
data=json.loads((ROOT/'datasets/ash-vlm-lora-v1/dataset.json').read_text(encoding='utf-8'))
rows=[r for r in data['items'] if r['split']=='eval' and r['target']['code'] in ['OK','NG04','NG05']]
reference=next(r for r in data['items'] if r['split']=='train' and r['target']['code']=='OK')
out=ROOT/'artifacts/vlm-input-diagnostics-v1';out.mkdir(exist_ok=False);(out/'inputs').mkdir()
meta=json.loads((run/'run.json').read_text(encoding='utf-8'));base=Path(meta['base']['path']);processor=AutoProcessor.from_pretrained(base,local_files_only=True);torch.set_num_threads(4)
previous=json.loads((run/'comparison/after.json').read_text(encoding='utf-8'))['results'];results=[dict(mode='original',**r) for r in previous if not r['warmup'] and r['target']['code'] in ['OK','NG04','NG05']]
save(out/'manifest.json',dict(rows=[r['id'] for r in rows],reference=reference['image'],reference_sha256=reference['image_sha256'],scope='24 development images, fixed adapter; ROI uses ground-truth regions as oracle diagnostic, not deployable detector performance. OK uses predefined context region. Reference not pose registered. Reused evaluation images now development diagnostics. Original timing from prior run.'))
coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
 model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');model=PeftModel.from_pretrained(model,run/'adapter').merge_and_unload(safe_merge=True).eval();configure_patch_projection(model,'linear_patch_v1')
 try:
  for mode in ['rotate180','context_crop','normal_reference']:
   for n,row in enumerate([rows[0]]+rows):
    assert sha(Path(row['image']))==row['image_sha256']
    with Image.open(row['image']) as im:im=im.convert('RGB')
    crop=None
    if mode=='rotate180':im=im.transpose(Image.Transpose.ROTATE_180)
    if mode=='context_crop':
     w,h=im.size;b=row['source_region'] or [w*.1,h*.1,w*.5,h*.8];x,y,bw,bh=b;pad=max(bw,bh)*.45;crop=[max(0,int(x-pad)),max(0,int(y-pad)),min(w,int(x+bw+pad)),min(h,int(y+bh+pad))];im=im.crop(crop)
    im.thumbnail((448,448));content=[]
    if mode=='normal_reference':
     with Image.open(reference['image']) as ref:ref=ref.convert('RGB');ref.thumbnail((448,448));ref=ref.copy()
     content.extend([{'type':'text','text':'정상 기준사진입니다. 촬영 방향은 다를 수 있습니다.'},{'type':'image','image':ref},{'type':'text','text':'다음 사진이 검사 대상입니다. 기준사진의 상태를 답하지 마세요.'}])
    content.extend([{'type':'image','image':im},{'type':'text','text':'사진의 상태를 관찰하세요.'}]);messages=[{'role':'system','content':data['system']},{'role':'user','content':content}]
    torch.cuda.synchronize();t=time.perf_counter();inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
    with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
    torch.cuda.synchronize();gen=tokens[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(gen,skip_special_tokens=True)[0].strip();seconds=time.perf_counter()-t;eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];truncated=gen.shape[1]>=64 and int(gen[0,-1]) not in eos
    try:parsed=json.loads(raw.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'code','observation'}
    except Exception:parsed=None;valid=False
    r=dict(mode=mode,id=row['id'],image=row['image'],target=row['target'],raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=seconds,warmup=n==0,code_match=valid and not truncated and parsed['code']==row['target']['code'],crop=crop)
    if n:im.save(out/'inputs'/f'{mode}-{n:02}.png');r['preview']=f'inputs/{mode}-{n:02}.png'
    results.append(r);save(out/'results.json',results);print(mode,n,'/24',row['target']['code'],round(seconds,2),raw,flush=True)
 finally:del model;gc.collect();torch.cuda.empty_cache()
summary=[]
for mode in ['original','rotate180','context_crop','normal_reference']:
 for code in ['ALL','OK','NG04','NG05']:
  selected=[r for r in results if r['mode']==mode and not r['warmup'] and (code=='ALL' or r['target']['code']==code)]
  summary.append(dict(mode=mode,code=code,images=len(selected),matches=sum(r['code_match'] for r in selected),mean_seconds=statistics.mean(r['seconds'] for r in selected)))
save(out/'summary.json',summary)
page='<meta charset="utf-8"><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}img{max-width:240px}article{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap}</style><h1>LoRA 입력 방식 점검</h1><p>NG04 8장, NG05 7장, 정상9장. 정답코드는 입력하지 않았습니다. 확대는 정답 영역을 사용한 원인 분석용이며 실제 DETR 성능이 아닙니다. 정상 기준은 자세 정합하지 않은 한 장입니다. 원본 시간은 이전 실행, 나머지는 각 방식 예열 제외. 코드 일치와 설명 정확도는 별개입니다.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
for r in results:
 if not r['warmup']:page+='<article><h3>'+html.escape(r['mode']+' / '+r['id'])+'</h3>'+('<img src="'+r['preview']+'">' if 'preview' in r else '')+'<p>'+html.escape(r['raw'])+'</p></article>'
(out/'report.html').write_text(page,encoding='utf-8');print('FINISHED',out,flush=True)

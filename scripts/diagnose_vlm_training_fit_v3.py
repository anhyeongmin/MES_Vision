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
run=ROOT/'artifacts/training/vlm-2b-caption-v3-20260915-021709-909031'
data=json.loads((ROOT/'datasets/ash-vlm-caption-v3/dataset.json').read_text(encoding='utf-8'))
rows=[r for r in data['items'] if r['split']=='train' and (r['target']['code']=='NG04' or Path(r['image']).name in ['2_OK_204448_f0870.png','2_OK_204448_f0900.png'])]
# Two matched-orientation normal controls; explicit selection is persisted.
if len(rows)<4:
 candidates=[r for r in data['items'] if r['split']=='train' and r['target']['code']=='OK' and r not in rows]
 rows.extend(candidates[:4-len(rows)])
assert len(rows)==4 and sum(r['target']['code']=='NG04' for r in rows)==2
out=ROOT/'artifacts/vlm-training-fit-diagnostics-v3';out.mkdir(exist_ok=False);(out/'inputs').mkdir()
meta=json.loads((run/'run.json').read_text(encoding='utf-8'));base=Path(meta['base']['path']);processor=AutoProcessor.from_pretrained(base,local_files_only=True);torch.set_num_threads(4)
results=[]
save(out/'manifest.json',dict(rows=[dict(id=r['id'],image=r['image'],sha256=r['image_sha256'],target=r['target']) for r in rows],adapter=str(run/'adapter'),scope='Training-fit diagnostic ONLY: 2 trained NG04 and 2 trained normal controls. Same fixed upper-right ROI for all; no class supplied in prompts. Crop-only loses global context. Bilinear enlargement creates no new detail. Not independent accuracy or actual DETR localization. Timing excludes image preparation and disk I/O.',crop_normalized=[.52,.04,.98,.48],max_new_tokens=64))
coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
 model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');model=PeftModel.from_pretrained(model,run/'adapter').merge_and_unload(safe_merge=True).eval();configure_patch_projection(model,'linear_patch_v1')
 try:
  for mode in ['original','crop_only','full_and_crop']:
   for n,row in enumerate([rows[0]]+rows):
    assert sha(Path(row['image']))==row['image_sha256']
    with Image.open(row['image']) as im:im=im.convert('RGB')
    w,h=im.size;crop=[int(w*.52),int(h*.04),int(w*.98),int(h*.48)]
    detail=im.crop(crop);detail=detail.resize((448,max(1,round(448*detail.height/detail.width))),Image.Resampling.BILINEAR);detail.thumbnail((448,448))
    im.thumbnail((448,448));content=[]
    if mode=='original':content=[{'type':'image','image':im}]
    elif mode=='crop_only':content=[{'type':'image','image':detail}]
    else:content=[{'type':'text','text':'첫 사진은 전체 모습, 두 번째 사진은 같은 물체의 오른쪽 위 부분 확대입니다.'},{'type':'image','image':im},{'type':'image','image':detail}]
    content.append({'type':'text','text':'사진의 상태를 관찰하세요.'});messages=[{'role':'system','content':data['system']},{'role':'user','content':content}]
    torch.cuda.synchronize();t=time.perf_counter();inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
    with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
    torch.cuda.synchronize();gen=tokens[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(gen,skip_special_tokens=True)[0].strip();seconds=time.perf_counter()-t;eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];truncated=gen.shape[1]>=64 and int(gen[0,-1]) not in eos
    try:parsed=json.loads(raw.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'code','observation'}
    except Exception:parsed=None;valid=False
    r=dict(mode=mode,id=row['id'],image=row['image'],target=row['target'],raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=seconds,warmup=n==0,code_match=valid and not truncated and parsed['code']==row['target']['code'],crop=crop)
    if n:
     im.save(out/'inputs'/f'{mode}-{n:02}.png');detail.save(out/'inputs'/f'{mode}-{n:02}-crop.png');r['preview']=f'inputs/{mode}-{n:02}.png';r['crop_preview']=f'inputs/{mode}-{n:02}-crop.png'
    results.append(r);save(out/'results.json',results);print(mode,n,'/4',row['target']['code'],round(seconds,2),raw,flush=True)
 finally:del model;gc.collect();torch.cuda.empty_cache()
summary=[]
for mode in ['original','crop_only','full_and_crop']:
 for code in ['ALL','OK','NG04']:
  selected=[r for r in results if r['mode']==mode and not r['warmup'] and (code=='ALL' or r['target']['code']==code)]
  summary.append(dict(mode=mode,code=code,images=len(selected),matches=sum(r['code_match'] for r in selected),mean_seconds=statistics.mean(r['seconds'] for r in selected)))
save(out/'summary.json',summary)
page='<meta charset="utf-8"><style>body{font-family:Malgun Gothic,sans-serif;max-width:1100px;margin:30px auto}img{max-width:300px}article{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap}</style><h1>학습 사진 NG04: 전체 / 확대 / 함께 입력</h1><p>학습 NG04 2장 + 정상 2장. 고정된 오른쪽 위 영역을 동일하게 확대. 정답 코드 미입력. 학습 사진 점검이며 독립 성능 검증 아님. 확대 입력은 별도 학습하지 않은 조건. 이미지 준비·디스크 입출력은 시간에서 제외.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
for r in results:
 if not r['warmup']:page+='<article><h3>'+html.escape(r['mode']+' / '+r['id'])+'</h3>'+('<img src="'+r['preview']+'">' if 'preview' in r else '')+'<img src="'+r['crop_preview']+'">'+'<p>'+html.escape(r['raw'])+'</p></article>'
(out/'report.html').write_text(page,encoding='utf-8');print('FINISHED',out,flush=True)

"""Frozen DETR->VLM diagnostics with explicit false-candidate controls."""
from train_vlm_lora import ROOT,save,sha
import json,time,gc,statistics,html,math
from dataclasses import asdict
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
run=ROOT/'artifacts/training/vlm-2b-lora-aug-20260915-013632-237393';meta=json.loads((run/'run.json').read_text(encoding='utf-8'));assert meta['status']=='COMPLETED_UNVALIDATED'
data=json.loads((ROOT/'datasets/ash-vlm-lora-v2/dataset.json').read_text(encoding='utf-8'));rows=[r for r in data['items'] if r['split']=='eval'];assert len(rows)==55
out=ROOT/'artifacts/detr-vlm-context-v1';out.mkdir(exist_ok=False)
weights=ROOT/'artifacts/training/ash-recorded-20260914-212558-590266/detail-defects/inference-unvalidated.pth';digest='4e58419200d37109981f697db4502c0db4a1ac4c494bcda08735cee748a84f55';assert sha(weights)==digest
coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');torch.set_num_threads(4);detections={};results=[]
system=data['system']+' 검출 후보는 정답이 아니며 틀릴 수 있습니다. 사진에서 확인한 특징만 설명하세요. 후보 코드만 따라 말하지 마세요. 후보를 뒷받침할 근거가 부족하면 UNCERTAIN, 실제 정상으로 보이면 OK로 답하세요.'
save(out/'manifest.json',dict(detr_sha256=digest,vlm_run=str(run),dataset_sha256=sha(ROOT/'datasets/ash-vlm-lora-v2/dataset.json'),detr_threshold=.3,system=system,scope='Development only; these55 images were used by DETR training and previously inspected for VLM development. Not independent accuracy. False candidates deliberately injected only in labeled stress-control arm. DETR and VLM timed separately; sum excludes model load and orchestration.'))
with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
 detector=RFDETRBackend(weights,digest,threshold=.3,training_scope='product_defects',class_names=tuple(f'NG{i:02}' for i in range(1,7)),inference_profile='standard',max_batch_size=1)
 try:
  detector.load()
  for n,row in enumerate([rows[0]]+rows):
   assert sha(__import__('pathlib').Path(row['image']))==row['image_sha256'];rgb=np.array(Image.open(row['image']).convert('RGB'));h,w=rgb.shape[:2];torch.cuda.synchronize();t=time.perf_counter();pred=[asdict(v) for v in detector.predict_rgb(rgb)];torch.cuda.synchronize();dt=time.perf_counter()-t
   if n:
    candidates=[]
    for p in sorted(pred,key=lambda d:-d['score'])[:3]:
     b=p['box'];candidates.append(dict(code=p['label'],score=round(p['score'],3),box_normalized=[round(b['x1']/w,3),round(b['y1']/h,3),round(b['x2']/w,3),round(b['y2']/h,3)]))
    detections[row['id']]=dict(candidates=candidates,raw_predictions=pred,seconds=dt)
  save(out/'detections.json',detections)
 finally:detector.close();del detector;gc.collect();torch.cuda.empty_cache()
 jobs=[dict(row=r,arm='actual_detr',candidates=detections[r['id']]['candidates']) for r in rows]
 for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
  selected=[r for r in rows if r['target']['code']==code];selected=selected if code=='OK' else selected[:1]
  for r in selected:
   wrong='NG04' if code=='OK' else f'NG{int(code[2:])%6+1:02}';b=detections[r['id']]['candidates'];box=b[0]['box_normalized'] if b else [.05,.1,.35,.9]
   jobs.append(dict(row=r,arm='false_candidate_control',candidates=[dict(code=wrong,score=.9,box_normalized=box)],injected_code=wrong))
 base=__import__('pathlib').Path(meta['base']['path']);processor=AutoProcessor.from_pretrained(base,local_files_only=True);model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda');model=PeftModel.from_pretrained(model,run/'adapter').merge_and_unload(safe_merge=True).eval();configure_patch_projection(model,'linear_patch_v1')
 try:
  for n,job in enumerate([jobs[0]]+jobs):
   row=job['row'];im=Image.open(row['image']).convert('RGB');im.thumbnail((448,448));context=json.dumps({'unverified_detector_candidates':job['candidates'],'box_coordinates':'normalized xyxy in inspected image; candidate score is detector score, not truth'},ensure_ascii=False)
   messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'사진의 상태를 관찰하세요. 다음은 확인이 필요한 검출 후보입니다: '+context}]}]
   torch.cuda.synchronize();t=time.perf_counter();inputs=processor.apply_chat_template(messages,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False).to('cuda')
   with torch.inference_mode():tokens=model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)
   torch.cuda.synchronize();gen=tokens[:,inputs['input_ids'].shape[1]:];raw=processor.batch_decode(gen,skip_special_tokens=True)[0].strip();elapsed=time.perf_counter()-t;eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];truncated=gen.shape[1]>=64 and int(gen[0,-1]) not in eos
   try:parsed=json.loads(raw.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'code','observation'} and isinstance(parsed['observation'],str)
   except Exception:parsed=None;valid=False
   results.append(dict(id=row['id'],image=row['image'],target=row['target'],arm=job['arm'],candidates=job['candidates'],injected_code=job.get('injected_code'),warmup=n==0,raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=elapsed,serial_seconds=elapsed+detections[row['id']]['seconds'],code_match=valid and not truncated and parsed['code']==row['target']['code']))
   save(out/'results.json',results);print(n,'/',len(jobs),job['arm'],row['target']['code'],round(elapsed,2),raw,flush=True)
 finally:del model;gc.collect();torch.cuda.empty_cache()
summary=[]
for arm in ['actual_detr','false_candidate_control']:
 for code in ['ALL','OK']+[f'NG{i:02}' for i in range(1,7)]:
  selected=[r for r in results if not r['warmup'] and r['arm']==arm and (code=='ALL' or r['target']['code']==code)]
  times=sorted(r['seconds'] for r in selected);summary.append(dict(arm=arm,code=code,n=len(selected),matches=sum(r['code_match'] for r in selected),mean_seconds=statistics.mean(times),p95_seconds=times[math.ceil(.95*len(times))-1],within2=sum(t<=2 for t in times),mean_serial_seconds=statistics.mean(r['serial_seconds'] for r in selected),followed_false_code=sum(r['valid'] and r['parsed']['code']==r['injected_code'] for r in selected),uncertain=sum(r['valid'] and r['parsed']['code']=='UNCERTAIN' for r in selected)))
save(out/'summary.json',summary)
page='<meta charset="utf-8"><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}pre{white-space:pre-wrap}article{border:1px solid #ccc;padding:12px}</style><h1>실제 DETR 후보를 받은 VLM 점검</h1><p>실제 후보55장 + 고의 오답 후보15장. 후자는 오류내성 점검이며 실제 DETR 오탐이 아닙니다. DETR 학습에 사용된 사진이므로 독립 성능 아님. 코드 일치와 설명 정확도는 별개. VLM 단독 시간과 별도로 측정한 DETR 시간을 더한 참고값 제공.</p><pre>'+html.escape(json.dumps(summary,ensure_ascii=False,indent=2))+'</pre>'
for r in results:
 if not r['warmup']:page+='<article><h3>'+html.escape(r['arm']+' '+r['id'])+'</h3><p>정답초안 '+html.escape(json.dumps(r['target'],ensure_ascii=False))+'</p><p>후보 '+html.escape(json.dumps(r['candidates'],ensure_ascii=False))+'</p><p>응답 '+html.escape(r['raw'])+'</p></article>'
(out/'report.html').write_text(page,encoding='utf-8');print('FINISHED:',out,flush=True)

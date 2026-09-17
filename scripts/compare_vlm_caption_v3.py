"""Compare base and merged LoRA on reserved prior-session photos; no deployment."""
from pathlib import Path
import sys,json,time,gc,argparse,statistics,html,math
from train_vlm_lora import ROOT,encode,save,sha
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();run=args.run.resolve();meta=json.loads((run/'run.json').read_text(encoding='utf-8'));assert meta['status']=='COMPLETED_UNVALIDATED'
 dataset=Path(meta['dataset']);assert sha(dataset)==meta['dataset_sha256'];data=json.loads(dataset.read_text(encoding='utf-8'));rows=[r for r in data['items'] if r['split']=='eval'];assert len(rows)==55
 out=run/'comparison';out.mkdir(exist_ok=True);base=Path(meta['base']['path']);processor=AutoProcessor.from_pretrained(base,local_files_only=True);torch.set_num_threads(4);coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');summaries=[];cards=[]
 with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
  for mode in ['before','after']:
   dest=out/(mode+'.json');assert not dest.exists(),'Refuse to overwrite evaluation'
   model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda').eval()
   if mode=='after':model=PeftModel.from_pretrained(model,run/'adapter').merge_and_unload(safe_merge=True);model.eval()
   configure_patch_projection(model,'linear_patch_v1');results=[]
   try:
    for n,row in enumerate([rows[0]]+rows):
     assert sha(Path(row['image']))==row['image_sha256'];torch.cuda.synchronize();t=time.perf_counter();batch=encode(processor,row,data['system'],answer=False).to('cuda');torch.cuda.synchronize()
     with torch.inference_mode():tokens=model.generate(**batch,max_new_tokens=64,do_sample=False,use_cache=True)
     torch.cuda.synchronize();gen=tokens[:,batch['input_ids'].shape[1]:];raw=processor.batch_decode(gen,skip_special_tokens=True)[0].strip();seconds=time.perf_counter()-t
     eos=model.generation_config.eos_token_id;eos=eos if isinstance(eos,list) else [eos];truncated=gen.shape[1]>=64 and int(gen[0,-1]) not in eos
     try:
      parsed=json.loads(raw.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'code','observation'} and isinstance(parsed['observation'],str)
     except Exception:parsed=None;valid=False
     results.append(dict(id=row['id'],image=row['image'],target=row['target'],raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=seconds,warmup=n==0,code_match=valid and not truncated and parsed['code']==row['target']['code']))
     save(dest,dict(scope=data['scope'],model=mode,run=str(run),results=results));print(mode,n,'/55',round(seconds,2),raw,flush=True)
   finally:del model;gc.collect();torch.cuda.empty_cache()
   measured=[r for r in results if not r['warmup']];times=sorted(r['seconds'] for r in measured)
   summary=dict(model=mode,images=len(measured),code_matches=sum(r['code_match'] for r in measured),mean_seconds=statistics.mean(times),p95_seconds=times[math.ceil(.95*len(times))-1],within_2s=sum(t<=2 for t in times),truncated=sum(r['truncated'] for r in measured),normal_false=sum(r['target']['code']=='OK' and r['valid'] and r['parsed']['code'] not in ['OK','UNCERTAIN'] for r in measured),ng_as_ok=sum(r['target']['code']!='OK' and r['valid'] and r['parsed']['code']=='OK' for r in measured))
   summaries.append(summary);save(out/'summary.json',summaries)
   for r in measured:cards.append('<article><h3>'+html.escape(mode+' '+r['id'])+'</h3><p>정답 초안: '+html.escape(json.dumps(r['target'],ensure_ascii=False))+'</p><p>출력: '+html.escape(r['raw'])+'</p></article>')
 page='<meta charset="utf-8"><style>body{font-family:sans-serif;max-width:1100px;margin:30px auto}article{border:1px solid #ccc;padding:10px;margin:10px}pre{white-space:pre-wrap}</style><h1>Qwen3.5-2B LoRA 전후 비교</h1><p>학습 장수와 설정은 run.json 참조. 이전 촬영55장은 반복 사용한 개발용 평가 자료입니다. 같은 실제 부품이므로 독립 현장 성능 검증은 아님. 정답 설명은 기존 영역과 시각 검토를 반영한 초안. 코드 일치와 설명 정확도는 별개. 예열 제외, 모델 상주, 생성 완료까지 측정. LoRA를 메모리에서 병합한 추론 결과이며 원본 모델은 수정하지 않음.</p><pre>'+html.escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
 (out/'report.html').write_text(page,encoding='utf-8');print('COMPARISON FINISHED:',out/'report.html',flush=True)
if __name__=='__main__':main()

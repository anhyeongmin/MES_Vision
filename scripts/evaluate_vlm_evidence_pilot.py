"""Compare base and merged LoRA on reserved prior-session photos; no deployment."""
from pathlib import Path
import sys,json,time,gc,argparse,statistics,html,math
from train_vlm_lora import ROOT,save,sha
from vlm_evidence_support import encode
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();run=args.run.resolve();meta=json.loads((run/'run.json').read_text(encoding='utf-8'));assert meta['status']=='COMPLETED_UNVALIDATED'
 dataset=ROOT/'datasets/ash-vlm-evidence-pilot-v1/evaluation.json';data=json.loads(dataset.read_text(encoding='utf-8'));rows=data['items'];assert len(rows)==70
 out=run/'evidence-evaluation';out.mkdir(exist_ok=False);base=Path(meta['base']['path']);processor=AutoProcessor.from_pretrained(base,local_files_only=True);torch.set_num_threads(4);coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');summaries=[];cards=[]
 save(out/'manifest.json',dict(scope=data['scope'],dataset_sha256=sha(dataset),run=str(run),adapter_sha256=sha(run/'adapter/adapter_model.safetensors'),max_new_tokens=64,full_edge=448,region_edge=224))
 with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
  for mode in ['after']:
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
      parsed=json.loads(raw.removeprefix('```json').removesuffix('```').strip());valid=set(parsed)=={'observation','needs_review'} and isinstance(parsed['observation'],str) and isinstance(parsed['needs_review'],bool)
     except Exception:parsed=None;valid=False
     results.append(dict(id=row['id'],image=row['image'],specimen_code=row['specimen_code'],arm=row['arm'],candidates=row['candidates'],inspection_region=row['inspection_region'],raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=seconds,warmup=n==0))
     save(dest,dict(scope=data['scope'],model=mode,run=str(run),results=results));print(mode,n,'/70',round(seconds,2),raw,flush=True)
   finally:del model;gc.collect();torch.cuda.empty_cache()
   measured=[r for r in results if not r['warmup']];times=sorted(r['seconds'] for r in measured)
   for arm in ['actual_detr','controlled_wrong_code','missing_image_control']:
    subset=[x for x in measured if x['arm']==arm];ts=sorted(x['seconds'] for x in subset)
    summaries.append(dict(arm=arm,n=len(subset),valid=sum(x['valid'] for x in subset),needs_review=sum(x['valid'] and x['parsed']['needs_review'] for x in subset),mean_seconds=statistics.mean(ts),p95_seconds=ts[math.ceil(.95*len(ts))-1],within_2s=sum(t<=2 for t in ts),truncated=sum(x['truncated'] for x in subset)))
   save(out/'summary.json',summaries)
   for x in measured:cards.append('<article><h3>'+html.escape(x['arm']+' '+x['id'])+'</h3><img style="max-width:320px" src="'+Path(x['image']).as_uri()+'"><p>物品:'+html.escape(str(x['specimen_code']))+'</p><p>候補:'+html.escape(json.dumps(x['candidates'],ensure_ascii=False))+'</p><p>'+html.escape(x['raw'])+'</p></article>')
 page='<meta charset="utf-8"><style>body{font-family:Malgun Gothic,sans-serif;max-width:1100px;margin:30px auto}article{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap}</style><h1>검사 근거 설명 파일럿</h1><p>기존 개발용31장에 실제 후보/고의 오답 후보를 입력하고,8개 영상 없음 통제를 추가. 독립 현장 평가 아님. needs_review 빈도는 설명 정확도가 아니며 사진별 검토 필요. 정답 불량 코드는 입력하지 않음. 영상 없음 통제는 인공 사례로 실제 반사/애매함 검증을 대체하지 않음.</p><pre>'+html.escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
 page=page.replace('物品:','물품 라벨:').replace('候補:','후보:')
 (out/'report.html').write_text(page,encoding='utf-8');print('COMPARISON FINISHED:',out/'report.html',flush=True)
if __name__=='__main__':main()

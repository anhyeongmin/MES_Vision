"""Compare base and merged LoRA on reserved prior-session photos; no deployment."""
from pathlib import Path
import sys,json,time,gc,argparse,statistics,html,math
from train_vlm_lora import ROOT,save,sha
from vlm_evidence_balanced_support import encode
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--run',required=True,type=Path);args=parser.parse_args();run=args.run.resolve();meta=json.loads((run/'run.json').read_text(encoding='utf-8'));assert meta['status']=='COMPLETED_UNVALIDATED'
 dataset=ROOT/'datasets/ash-vlm-evidence-balanced-v2/evaluation.json';data=json.loads(dataset.read_text(encoding='utf-8'));rows=data['items'];assert len(rows)==189
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
     results.append(dict(id=row['id'],image=row['image'],specimen_code=row['specimen_code'],arm=row['arm'],candidate_code=row['candidate_code'],expected_review_by_label=row['target']['needs_review'],expected_draft=row['target']['observation'],raw=raw,parsed=parsed,valid=valid,truncated=truncated,seconds=seconds,warmup=n==0))
     save(dest,dict(scope=data['scope'],model=mode,run=str(run),results=results));print(mode,n,'/189',round(seconds,2),raw,flush=True)
   finally:del model;gc.collect();torch.cuda.empty_cache()
   measured=[r for r in results if not r['warmup']];times=sorted(r['seconds'] for r in measured)
   for arm in ['fit','transfer','missing']:
    subset=[x for x in measured if x['arm']==arm];ts=sorted(x['seconds'] for x in subset)
    summaries.append(dict(arm=arm,n=len(subset),label_based_flag_agreement_not_caption_accuracy=sum(x['valid'] and not x['truncated'] and x['parsed']['needs_review']==x['expected_review_by_label'] for x in subset),valid=sum(x['valid'] for x in subset),needs_review=sum(x['valid'] and x['parsed']['needs_review'] for x in subset),mean_seconds=statistics.mean(ts),p95_seconds=ts[math.ceil(.95*len(ts))-1],within_2s=sum(t<=2 for t in ts),truncated=sum(x['truncated'] for x in subset)))
   save(out/'summary.json',summaries)
   for x in measured:cards.append('<article><h3>'+html.escape(x['arm']+' '+x['id'])+'</h3><img style="max-width:320px" src="'+Path(x['image']).as_uri()+'"><p>物品:'+html.escape(str(x['specimen_code']))+'</p><p>候補:'+html.escape(x['candidate_code'])+'</p><p>'+html.escape(x['raw'])+'</p></article>')
 page='<meta charset="utf-8"><style>body{font-family:Malgun Gothic,sans-serif;max-width:1100px;margin:30px auto}article{border:1px solid #ccc;padding:12px}pre{white-space:pre-wrap}</style><h1>후보 메타데이터 통제 실험</h1><p>학습12장×후보3종, 학습제외50장×후보3종, 빈 영상3종. 후보코드만 제공하며 점수·좌표는 제공하지 않음. 모든 입력은 같은 중앙영역 추출 사용. 독립 현장 평가 아님. needs_review 빈도는 설명 정확도가 아니며 사진별 검토 필요. 물품 정답 코드는 입력하지 않음. 후보코드는 모든 사진에 동일하게 NONE/NG04/NG05를 각각 입력. 라벨 기반 확인플래그 일치는 사진 설명의 정확도가 아님. 영상 없음 통제는 인공 사례로 실제 반사/애매함 검증을 대체하지 않음.</p><pre>'+html.escape(json.dumps(summaries,ensure_ascii=False,indent=2))+'</pre>'+''.join(cards)
 page=page.replace('物品:','물품 라벨:').replace('候補:','후보:')
 (out/'report.html').write_text(page,encoding='utf-8');print('COMPARISON FINISHED:',out/'report.html',flush=True)
if __name__=='__main__':main()

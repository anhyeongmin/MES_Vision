"""Qwen3.5-2B LoRA pilot. Local files only, assistant responses alone are supervised."""
from pathlib import Path
import os,sys,json,time,random,math,argparse
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
from prepare_vlm_lora_aug import sha
def prepare():return ROOT/'datasets/ash-vlm-caption-v3/dataset.json'
from PIL import Image
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import LoraConfig,get_peft_model
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator

def save(p,d):
 temp=p.with_suffix('.tmp');temp.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8');temp.replace(p)
from vlm_caption_v3_support import encode,self_test,SETTINGS

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=5);parser.add_argument('--check-only',action='store_true');parser.add_argument('--smoke-test',action='store_true');args=parser.parse_args();assert 1<=args.epochs<=30
 self_test();source=prepare();data=json.loads(source.read_text(encoding='utf-8'));train=[r for r in data['items'] if r['split']=='train'];ev=[r for r in data['items'] if r['split']=='eval'];record=next(r for r in json.loads((ROOT/'models/vlm-benchmark-downloads.json').read_text()) if r['repo']=='Qwen/Qwen3.5-2B');base=Path(record['path']);assert record['revision']=='15852e8c16360a2fea060d615a32b45270f8a8fc'
 from datetime import datetime
 out=ROOT/'artifacts/training'/('vlm-2b-caption-v3-'+datetime.now().strftime('%Y%m%d-%H%M%S-%f'));out.mkdir(parents=True)
 manifest=dict(status='PREPARING',base=record,dataset=str(source),dataset_sha256=sha(source),train_images=len(train),eval_images=len(ev),epochs=args.epochs,production_ready=False,independent_evaluation=False,caption_status='assistant drafts',augmentation=SETTINGS,base_init='original2B not previous adapter',output=str(out),started_at=datetime.now().isoformat())
 save(out/'run.json',manifest);print('OUTPUT:',out,flush=True);torch.set_num_threads(4);random.seed(42);torch.manual_seed(42)
 processor=AutoProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=False)
 for r in data['items']:
  assert sha(Path(r['image']))==r['image_sha256'];batch=encode(processor,r,data['system']);assert (batch['labels']!=-100).sum()>0
 print('PASS: all dataset images and assistant-only target masks; no split overlap',flush=True)
 if args.check_only:manifest['status']='CHECK_ONLY_PASSED';save(out/'run.json',manifest);return
 coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
 with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
  model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  from mes_vision.vlm.optimization import configure_patch_projection
  manifest['vision_execution']=configure_patch_projection(model,'linear_patch_v1')
  # Freeze vision and base weights; adapt language attention, recurrent projections and MLP.
  targets=[n for n,m in model.named_modules() if isinstance(m,torch.nn.Linear) and 'language_model' in n and not n.endswith('lm_head')]
  assert targets
  model=get_peft_model(model,LoraConfig(r=8,lora_alpha=16,lora_dropout=.05,target_modules=targets,bias='none',task_type='CAUSAL_LM'))
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
  params=[p for p in model.parameters() if p.requires_grad];assert params and all('lora_' in n for n,p in model.named_parameters() if p.requires_grad)
  manifest.update(trainable_parameters=sum(p.numel() for p in params),target_modules=targets,gpu=torch.cuda.get_device_name(),status='RUNNING');save(out/'run.json',manifest)
  optimizer=torch.optim.AdamW(params,lr=1e-4,weight_decay=.01);start=time.perf_counter();step=0
  try:
   for epoch in range(1,args.epochs+1):
    model.train();order=list(train);random.Random(42+epoch).shuffle(order);losses=[];optimizer.zero_grad(set_to_none=True)
    if args.smoke_test:order=order[:1]
    for i,row in enumerate(order):
     batch=encode(processor,row,data['system']).to('cuda');group_start=(i//4)*4;divisor=min(4,len(order)-group_start)
     with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**batch,use_cache=False).loss
     assert torch.isfinite(loss);(loss/divisor).backward();losses.append(float(loss.detach()));del batch,loss
     if (i+1)%4==0 or i+1==len(order):
      norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm) and float(norm)>0
      optimizer.step();optimizer.zero_grad(set_to_none=True);step+=1
     if (i+1)%10==0:print('EPOCH',epoch,'IMAGE',i+1,'/',len(order),flush=True)
    if args.smoke_test:
     manifest.update(status='SMOKE_TEST_PASSED',optimizer_steps=step,loss=losses[0],seconds=time.perf_counter()-start);save(out/'run.json',manifest);print('PASS: real forward/backward and one LoRA update; disposable smoke model not saved',flush=True);return
    model.save_pretrained(out/f'adapter-epoch-{epoch:02}',safe_serialization=True)
    manifest.update(epoch=epoch,mean_train_loss=sum(losses)/len(losses),optimizer_steps=step,elapsed_seconds=time.perf_counter()-start);save(out/'run.json',manifest);print('EPOCH FINISHED',epoch,manifest['mean_train_loss'],flush=True)
   model.save_pretrained(out/'adapter',safe_serialization=True);processor.save_pretrained(out/'processor');manifest.update(status='COMPLETED_UNVALIDATED',adapter=str(out/'adapter'),elapsed_seconds=time.perf_counter()-start);save(out/'run.json',manifest)
  except BaseException as e:manifest.update(status='FAILED_OR_INTERRUPTED',error=repr(e));save(out/'run.json',manifest);raise
 print('TRAINING FINISHED:',out/'adapter',flush=True)
if __name__=='__main__':main()

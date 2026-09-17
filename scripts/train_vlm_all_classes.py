"""Continue the selected 2B LoRA with balanced all-class captions; no auto deployment."""
import os,sys,json,time,random,argparse,hashlib
from pathlib import Path
from datetime import datetime
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
from PIL import Image,ImageEnhance
import torch
from transformers import AutoProcessor,Qwen3_5ForConditionalGeneration
from peft import PeftModel
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.optimization import configure_patch_projection
from mes_vision.training.data import sha256,read_json,write_json
from contextlib import ExitStack
OUT=ROOT/'artifacts/vlm-all-classes-v1'

def encode(processor,row,system,answer=True):
 with Image.open(row['image']) as im:im=im.convert('RGB')
 if answer:
  im=im.rotate(row.get('rotation',0),expand=True)
  im=ImageEnhance.Brightness(im).enhance(random.uniform(.9,1.1))
 im.thumbnail((448,448))
 messages=[{'role':'system','content':system},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':row['question_text']}]}]
 kw=dict(tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=False)
 prefix=processor.apply_chat_template(messages,**kw)
 if not answer:return prefix
 kw['add_generation_prompt']=False
 full=processor.apply_chat_template(messages+[{'role':'assistant','content':json.dumps(row['target'],ensure_ascii=False,separators=(',',':'))}],**kw)
 n=prefix['input_ids'].shape[1];assert torch.equal(full['input_ids'][:,:n],prefix['input_ids'])
 labels=full['input_ids'].clone();labels[:,:n]=-100;full['labels']=labels
 assert (labels!=-100).sum()>5
 return full

def evaluate(model,processor,rows,system,path):
 model.eval();model.config.use_cache=True;results=[]
 for row in rows:
  t=time.perf_counter();batch=encode(processor,row,system,False).to('cuda')
  with torch.inference_mode():tokens=model.generate(**batch,max_new_tokens=64,do_sample=False,use_cache=True)
  torch.cuda.synchronize();raw=processor.batch_decode(tokens[:,batch['input_ids'].shape[1]:],skip_special_tokens=True)[0].strip()
  try:parsed=json.loads(raw);valid=set(parsed)=={'observation'} and isinstance(parsed['observation'],str)
  except Exception:parsed=None;valid=False
  results.append(dict(id=row['id'],label=row['label'],image=row['image'],target=row['target'],question=row['question_text'],raw=raw,parsed=parsed,valid=valid,seconds=time.perf_counter()-t))
  write_json(path,results)
 print('EVALUATED',path,len(results),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=6);args=p.parse_args();assert 1<=args.epochs<=30
 data=read_json(OUT/'dataset.json');train=[r for r in data['items'] if r['split']=='train'];ev=[]
 for label in sorted({r['label'] for r in data['items'] if r['split']=='eval'}):
  ev += [r for r in data['items'] if r['split']=='eval' and r['label']==label][:2]
 old=Path('C:/Users/alexi/Documents/ChatGPT/MES 비전/artifacts/vlm-region-abstain-v1/evaluation.json')
 prior=read_json(old)['items']
 for label in sorted({r['label'] for r in prior if r.get('label') is not None}):ev += [r for r in prior if r['label']==label][:2]
 # These are diagnostic old-session images, not an independent production test.
 out=OUT/('training-'+datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir()
 selected=read_json(ROOT/'configs/vlm/backend.json');base=ROOT/selected['base'];adapter=ROOT/selected['adapter']
 meta=dict(status='CHECKING',output=str(out),epochs=args.epochs,initial_adapter=str(adapter),
  initial_sha256=sha256(adapter/'adapter_model.safetensors'),dataset_sha256=sha256(OUT/'dataset.json'),
  train_examples=len(train),diagnostic_examples=len(ev),production_ready=False,independent_evaluation=False)
 write_json(out/'run.json',meta);write_json(OUT/'active.json',meta);print('OUTPUT',out,flush=True)
 torch.set_num_threads(4);torch.manual_seed(42);random.seed(42)
 for row in train+ev:assert sha256(Path(row['image']))==row['image_sha256']
 processor=AutoProcessor.from_pretrained(base,local_files_only=True,trust_remote_code=False)
 for row in train:encode(processor,row,data['system'])
 with ExitStack() as stack:
  for runtime in (ROOT/'artifacts/operation',ROOT/'.cache/real-labeling'):
   coord=GpuCoordinator(runtime/'vlm/gpu-coordination')
   stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0))
   stack.enter_context(coord.foreground(timeout=120))
  model=Qwen3_5ForConditionalGeneration.from_pretrained(base,local_files_only=True,trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to('cuda')
  configure_patch_projection(model,'linear_patch_v1')
  model=PeftModel.from_pretrained(model,adapter,is_trainable=True)
  evaluate(model,processor,ev,data['system'],out/'before.json')
  model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
  params=[p for p in model.parameters() if p.requires_grad];assert params
  optimizer=torch.optim.AdamW(params,lr=2e-5,weight_decay=.01);started=time.perf_counter()
  try:
   for epoch in range(1,args.epochs+1):
    model.train();model.config.use_cache=False;order=[]
    for label in sorted({r['label'] for r in train}):
     pool=[r for r in train if r['label']==label]
     for j in range(4):order.append(dict(pool[((epoch-1)*4+j)%len(pool)],rotation=((epoch+j)%4)*90))
    random.shuffle(order);optimizer.zero_grad(set_to_none=True);losses=[]
    for i,row in enumerate(order):
     batch=encode(processor,row,data['system']).to('cuda')
     loss=model(**batch,use_cache=False).loss;assert torch.isfinite(loss)
     group=min(4,len(order)-(i//4)*4);(loss/group).backward();losses.append(float(loss.detach()));del loss,batch
     if (i+1)%4==0 or i+1==len(order):
      norm=torch.nn.utils.clip_grad_norm_(params,1.);assert torch.isfinite(norm)
      optimizer.step();optimizer.zero_grad(set_to_none=True)
     if (i+1)%12==0:print('EPOCH',epoch,i+1,'/',len(order),flush=True)
    model.save_pretrained(out/f'adapter-epoch-{epoch:02}',safe_serialization=True)
    meta.update(status='TRAINING',epoch=epoch,mean_loss=sum(losses)/len(losses),elapsed_seconds=time.perf_counter()-started)
    write_json(out/'run.json',meta);write_json(OUT/'active.json',meta);print('EPOCH FINISHED',epoch,meta['mean_loss'],flush=True)
   model.save_pretrained(out/'adapter',safe_serialization=True)
   model.gradient_checkpointing_disable();model.eval()
   evaluate(model,processor,ev,data['system'],out/'after.json')
   meta.update(status='COMPLETED_UNVALIDATED',adapter=str(out/'adapter'),elapsed_seconds=time.perf_counter()-started)
   write_json(out/'run.json',meta);write_json(OUT/'active.json',meta)
  except BaseException as exc:
   meta.update(status='FAILED_OR_INTERRUPTED',error=repr(exc));write_json(out/'run.json',meta);write_json(OUT/'active.json',meta);raise
 print('FINISHED',out,flush=True)
if __name__=='__main__':main()

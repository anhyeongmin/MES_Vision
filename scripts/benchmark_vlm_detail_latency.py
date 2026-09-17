from pathlib import Path
import sys,os,json,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HOME']=str(ROOT/'.cache/huggingface')
from PIL import Image
from filelock import FileLock
from mes_vision.vlm.gpu import GpuCoordinator
import mes_vision.vlm.backend as api
from datetime import datetime
out=ROOT/'artifacts'/('vlm-latency-'+datetime.now().strftime('%Y%m%d-%H%M%S'));out.mkdir()
meta=json.loads((ROOT/'datasets/ash-recorded-low-exposure-reviewed-v2/provenance.json').read_text(encoding='utf-8'))
rows=[]
for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:rows.append(next(r for r in meta['items'] if r['label']==code))
def messages(job,config):
 with Image.open(job['image']) as im:
  im=im.convert('RGB');im.thumbnail((config.image_max_edge,config.image_max_edge));im=im.copy()
 system='Describe only visible surface or structural irregularities. Do not invent measurements or manufacturing causes. If unclear, say uncertain. Return only JSON with observation (one short Korean sentence) and needs_review (boolean).'
 return [{'role':'system','content':system},{'role':'user','content':[{'type':'image','image':im},{'type':'text','text':'Inspect this part. Brief observation only.'}]}],{'image_sizes':[list(im.size)],'label_withheld':True,'reference':False}
api.make_messages=messages # Isolated benchmark process only; production module on disk unchanged.
results=[];coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
print('OUTPUT:',out,flush=True)
with FileLock(str(coord.root/'resident.lock'),timeout=0),coord.foreground(timeout=10):
 backend=api.QwenBackend(ROOT)
 try:
  for edge,limit,warmup in [(448,64,True),(448,64,False),(256,64,False)]:
   for row in (rows[:1] if warmup else rows):
    job={'image':row['crop'],'payload':{'generation':{'image_max_edge':edge,'max_new_tokens':limit,'timeout_seconds':60,'language':'ko'}}}
    t=time.perf_counter();r=backend.generate(job);elapsed=time.perf_counter()-t
    try:parsed=api.parse_response(r['raw_text']);valid=True
    except Exception:parsed=None;valid=False
    results.append({'image':row['crop'],'expected_label_withheld':row['label'],'edge':edge,'max_tokens':limit,'warmup':warmup,'wall_seconds':elapsed,'valid_json':valid,'parsed':parsed,**r})
    (out/'results.json').write_text(json.dumps({'scope':'VLM-only isolated timing; one sample per class; not quality validation or shared-GPU timing','results':results},ensure_ascii=False,indent=2),encoding='utf-8')
    print(row['label'],edge,'warmup' if warmup else 'measured',round(elapsed,2),r['raw_text'],flush=True)
 finally:backend.close()
print('FINISHED',out,flush=True)

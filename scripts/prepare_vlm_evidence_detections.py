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
data=json.loads((ROOT/'datasets/ash-vlm-lora-v2/dataset.json').read_text(encoding='utf-8'));rows=[r for r in data['items'] if r['target']['code'] in ['OK','NG04']];assert len(rows)==39
out=ROOT/'artifacts/vlm-evidence-pilot-v1';out.mkdir(exist_ok=False)
weights=ROOT/'artifacts/training/ash-recorded-20260914-212558-590266/detail-defects/inference-unvalidated.pth';digest='4e58419200d37109981f697db4502c0db4a1ac4c494bcda08735cee748a84f55';assert sha(weights)==digest
coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');torch.set_num_threads(4);detections={};results=[]
system=data['system']+' 검출 후보는 정답이 아니며 틀릴 수 있습니다. 사진에서 확인한 특징만 설명하세요. 후보 코드만 따라 말하지 마세요. 후보를 뒷받침할 근거가 부족하면 UNCERTAIN, 실제 정상으로 보이면 OK로 답하세요.'
save(out/'manifest.json',dict(detr_sha256=digest,vlm_run=str(run),dataset_sha256=sha(ROOT/'datasets/ash-vlm-lora-v2/dataset.json'),detr_threshold=.3,system=system,scope='Development only; these39 images were used by DETR training and previously inspected for VLM development. Not independent accuracy. False candidates deliberately injected only in labeled stress-control arm. DETR and VLM timed separately; sum excludes model load and orchestration.'))
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

print("DETR candidate extraction finished",flush=True)

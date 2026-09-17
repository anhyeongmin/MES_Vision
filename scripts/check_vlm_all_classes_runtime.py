"""Actual saved-photo DETR findings -> immutable snapshots -> candidate VLM."""
import argparse,sys,time,json
from pathlib import Path
from dataclasses import replace
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.inspection.contracts import Box,Crop,Detection,ModelRef,ObjectResult,RunResult,Mode
from mes_vision.inspection.model_registry import default_registry
from mes_vision.station.photo_inspection import load_bundle
from mes_vision.inputs import ImageSource
from mes_vision.vlm.snapshots import save_snapshot
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.region_backend import RegionBackend
from mes_vision.vlm.backend import parse_response
from mes_vision.vlm.gpu import GpuCoordinator
from filelock import FileLock

def main():
 p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
 out=args.output.resolve();require(not out.exists(),'Use a new output directory');out.mkdir()
 dataset=read_json(ROOT/'datasets/ash-vlm-caption-v3/dataset.json')['items'];selected=read_json(args.candidate)
 queue=AnalysisQueue(out/'queue');queue.set_enabled(True);jobs=[]
 with ExitStack() as stack:
  for runtime in (ROOT/'artifacts/operation',ROOT/'.cache/real-labeling'):
   coord=GpuCoordinator(runtime/'vlm/gpu-coordination')
   stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=120))
  bundle=load_bundle(ROOT/'configs/inspection/ash-trained-v3.json',ROOT)
  handle=default_registry().create(bundle['models']['defects'],'defects');handle.load()
  try:
   for code in ['OK','NG01','NG02','NG03','NG04','NG05','NG06']:
    row=next(r for r in dataset if r['target']['code']==code and r['split']=='eval')
    require(sha256(Path(row['image']))==row['image_sha256'],'Image changed')
    with ImageSource(row['image']) as source:frame=source.read().frame
    box=Box(0,0,frame.width,frame.height);oid='object-'+code;crop=Crop(frame.frame_id,oid,box,box,frame.rgb)
    check=handle.adapter.inspect(crop)
    check=replace(check,findings=tuple(replace(f,original_box=f.crop_box) for f in check.findings))
    obj=ObjectResult(oid,Detection(box,1.,0,'preselected-detail'),box,crop.metadata(),[check])
    result=RunResult('all-'+code,frame.metadata(),Mode.MODEL_FILE,ModelRef('preselected-detail','1','unconfigured'),{'product_id':'ASH'},[obj])
    snapshot=out/code;save_snapshot(snapshot,frame,result,kind='real')
    jid=queue.enqueue(snapshot,oid,dict(max_new_tokens=64,language='ko'))
    jobs.append(dict(code=code,job=queue.get(jid),detected=[f.defect_code for f in check.findings]))
  finally:handle.close()
  model=RegionBackend(ROOT,selected);answers=[]
  try:
   # First invocation is warm-up and reported separately, never included as steady state.
   warm=model.generate(jobs[0]['job']);write_json(out/'warmup.json',warm)
   for item in jobs:
    response=model.generate(item['job']);analysis=parse_response(response['raw_text'])
    require(not response['truncated'] and not response['deadline_expired'],'Generation incomplete')
    require(analysis['needs_review'],'Advisory result must retain review')
    answers.append(dict(code=item['code'],detected=item['detected'],analysis=analysis,response=response))
    write_json(out/'results.json',answers)
    print(item['code'],item['detected'],analysis,response['metrics'],flush=True)
  finally:model.close()
 write_json(out/'manifest.json',dict(passed=True,candidate_sha256=sha256(args.candidate),
   scope='Seven old-session real object photos and current DETR findings. No physical capture, no independent accuracy claim.',
   hardware_used=False,base_decision_changed=False))
 print('PASS',out,flush=True)
if __name__=='__main__':main()

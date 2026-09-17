"""Same 14 scenes per trial; two repeats, reversed order; separate worker directories."""
from pathlib import Path
import sys,time,json,subprocess
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_today_plan import prepare
def main():
    from filelock import FileLock
    from mes_vision.vlm.gpu import GpuCoordinator
    out=ROOT/'artifacts/cad-parallel-benchmark';out.mkdir(exist_ok=False)
    coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');results=[]
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
        for trial,workers in enumerate([1,2,2,1],1):
            procs=[];logs=[];dirs=[]
            for w in range(workers):
                d=out/f'trial{trial}-w{w}';prepare(d);dirs.append(d)
            start=time.perf_counter()
            for w,d in enumerate(dirs):
                log=(d/'render.log').open('w',encoding='utf-8');logs.append(log)
                first=0 if workers==1 else w*7;last=14 if workers==1 else (w+1)*7
                cmd=[str(ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe'),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/'scripts/render_cad_today.py'),'--','--output',str(d),'--start',str(first),'--limit',str(last)]
                procs.append(subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT))
            codes=[p.wait() for p in procs];elapsed=time.perf_counter()-start
            for log in logs:log.close()
            assert codes==[0]*workers,(trial,codes)
            records=[json.loads(p.read_text()) for d in dirs for p in (d/'receipts').rglob('*.json')]
            assert len(records)==14 and len({r['id'] for r in records})==14
            results.append(dict(trial=trial,workers=workers,images=14,seconds=elapsed,seconds_per_image=elapsed/14,directories=[str(d) for d in dirs]))
            (out/'timings.json').write_text(json.dumps(results,indent=2));print(results[-1],flush=True)
    import numpy as np
    from PIL import Image
    reference={}
    for t in results:
        for d in map(Path,t['directories']):
            for p in (d/'receipts').rglob('*.json'):
                r=json.loads(p.read_text());rgb=np.array(Image.open(d/r['relative_rgb']).convert('RGB'))
                if t['trial']==1:reference[r['id']]=(r,rgb)
                else:
                    old,raw=reference[r['id']];assert r['difference_mask_sha256']==old['difference_mask_sha256'] and r['object_mask_sha256']==old['object_mask_sha256']
                    assert rgb.shape==raw.shape
    means={w:sum(t['seconds'] for t in results if t['workers']==w)/2 for w in [1,2]}
    report=dict(timings=results,mean_seconds_per_14=means,speedup_two_workers=means[1]/means[2],selected_workers=2 if means[2]<means[1]*.95 else 1,mask_consistency_passed=True,note='Startup and saving included; small benchmark, long-run performance may differ. 5% minimum gain required for two workers.')
    (out/'summary.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
if __name__=='__main__':main()

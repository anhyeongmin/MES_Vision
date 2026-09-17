"""Two disjoint Blender workers on one GPU; image generation only."""
from pathlib import Path
import argparse,json,os,sys,subprocess,threading,time,shutil
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cad_today_plan import prepare

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',default='datasets/ash-cad-today-parallel-v1')
    parser.add_argument('--images',type=int,default=1008,help='Use 14 for a smoke check; default generates all selected 1008 images')
    args=parser.parse_args()
    if not 14<=args.images<=1008 or args.images%14:parser.error('--images must be a multiple of 14 between 14 and 1008')
    from filelock import FileLock
    from mes_vision.vlm.gpu import GpuCoordinator
    root=ROOT/args.output;root.mkdir(parents=True,exist_ok=True)
    coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(root/'generation.lock'),timeout=0))
        stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
        if shutil.disk_usage(root).free<10*1024**3:raise RuntimeError('At least 10 GiB free disk space required')
        ranges=[(0,args.images//2),(args.images//2,args.images)]
        spec=dict(images=args.images,workers=2,ranges=ranges,scope='Selected grid coverage, not Cartesian grid; generation only, no training or label approval')
        config=root/'run-config.json'
        if config.exists():assert json.loads(config.read_text())==json.loads(json.dumps(spec)),'Use the same --images on resume or a different output folder'
        else:config.write_text(json.dumps(spec,indent=2))
        dirs=[root/f'worker-{i}' for i in range(2)]
        for d in dirs:prepare(d)
        procs=[];threads=[];started=time.time()
        print(f'OUTPUT: {root}\n2 workers / {args.images} images / no time cutoff',flush=True)
        def consume(process,d,w):
            with (d/'render.log').open('a',encoding='utf-8') as log:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith(('RENDER','SKIP')) or any(x in line for x in ['Error','Traceback','Assertion']):
                        print(f'[worker {w}] {line.rstrip()}',flush=True);log.flush()
        try:
            for w,(d,(first,last)) in enumerate(zip(dirs,ranges)):
                cmd=[str(ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe'),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/'scripts/render_cad_today.py'),'--','--output',str(d),'--start',str(first),'--limit',str(last)]
                proc=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=dict(os.environ,PYTHONIOENCODING='utf-8'));procs.append(proc)
                thread=threading.Thread(target=consume,args=(proc,d,w));thread.start();threads.append(thread)
            while any(p.poll() is None for p in procs):
                if any(p.poll() not in (None,0) for p in procs):raise RuntimeError('Worker failed; completed frames retained. See worker render.log')
                time.sleep(.5)
            if any(p.returncode for p in procs):raise RuntimeError('Worker failed; see worker render.log')
        finally:
            for p in procs:
                if p.poll() is None:p.terminate()
            for p in procs:p.wait()
            for t in threads:t.join()
        records=[json.loads(p.read_text()) for d in dirs for p in (d/'receipts').rglob('*.json')]
        assert len(records)==len({r['id'] for r in records})==args.images
        (root/'completed.json').write_text(json.dumps(dict(status='COMPLETED',images=len(records),seconds=time.time()-started,labels_approved=False,training_performed=False),indent=2))
        print('GENERATION FINISHED:',root,flush=True)
        print('Report generation and mixed training are separate; images are retained.',flush=True)
if __name__=='__main__':main()

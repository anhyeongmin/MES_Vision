"""Launch resumable Blender generation locally; no model training or deployment."""
from pathlib import Path
import sys,os,subprocess,time,shutil,argparse
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['pilot','full'],default='full')
    parser.add_argument('--limit',type=int,default=0,help='Optional initial task limit for a short generation check')
    args=parser.parse_args()
    from filelock import FileLock
    from mes_vision.vlm.gpu import GpuCoordinator
    blender=ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe';assert blender.exists()
    out=ROOT/('artifacts/cad-material-pilot-v2' if args.mode=='pilot' else 'datasets/ash-cad-5deg-v1')
    out.mkdir(parents=True,exist_ok=True)
    if args.mode=='full' and shutil.disk_usage(ROOT).free<10*1024**3:raise RuntimeError('At least 10 GiB free disk space required; actual size depends on renders.')
    coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(out/'generation.lock'),timeout=0))
        stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0))
        stack.enter_context(coord.foreground(timeout=10))
        print('OUTPUT:',out,flush=True)
        print('Full mode: 72 angles x 3 viewpoints x 4 lights x 7 parts = 6048 images.',flush=True)
        print('No time cutoff. Repeat this command to resume verified completed frames.',flush=True)
        cmd=[str(blender),'--background','--factory-startup','--python-exit-code','1','--python',str(ROOT/'scripts/render_ash_5deg.py'),'--','--mode',args.mode]
        if args.limit:cmd+=['--limit',str(args.limit)]
        env=dict(os.environ,PYTHONIOENCODING='utf-8')
        with (out/'render.log').open('a',encoding='utf-8') as log:
            process=subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',env=env)
            try:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith(('RENDER','SKIP')) or any(t in line for t in ['Error','Traceback','Assertion']):print(line.rstrip(),flush=True);log.flush()
                code=process.wait()
                if code:raise RuntimeError(f'Blender exited {code}; see {out / "render.log"}')
            except KeyboardInterrupt:
                if process.poll() is None:process.terminate();process.wait()
                print('Stopped by user. Completed receipts are retained.');return
        print('GENERATION FINISHED:',out,flush=True)
        print('Images and geometry proposals only; no training. Inspect samples before approving labels.',flush=True)
if __name__=='__main__':main()

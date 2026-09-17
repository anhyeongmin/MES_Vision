"""Train real-only then real75/CAD25, then compare on existing real images."""
from pathlib import Path
import sys,argparse,json,subprocess,hashlib
ROOT=Path(__file__).resolve().parents[1]
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--check-only',action='store_true');args=p.parse_args()
    root=ROOT/'artifacts/training'/('ratio-ablation-check' if args.check_only else 'ratio-ablation-v1');root.mkdir(parents=True,exist_ok=True)
    from filelock import FileLock
    with FileLock(str(root/'pipeline.lock'),timeout=0):
        for mode in ['real','mixed25']:
            out=root/mode;record=out/'run.json'
            if record.exists():
                r=json.loads(record.read_text())
                if args.check_only and r['status']=='CHECK_ONLY_PASSED':continue
                if r['status']=='COMPLETED_UNVALIDATED' and r['condition']==mode and r['global_step']==1600:
                    assert hashlib.sha256(Path(r['weights']).read_bytes()).hexdigest()==r['weights_sha256'];print('Already completed:',mode,flush=True);continue
                raise RuntimeError(f'Incomplete run at {out}. Retained for diagnosis; no overwrite.')
            cmd=[sys.executable,'-X','utf8',str(ROOT/'scripts/train_ratio_ablation.py'),'--mode',mode,'--output',str(out),'--epochs','50']
            if args.check_only:cmd+=['--check-only']
            print('START:',mode,flush=True)
            subprocess.run(cmd,cwd=ROOT,check=True)
        if args.check_only:print('PREFLIGHT FINISHED. No model training.');return
        subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/compare_ratio_ablation.py'),'--baseline',str(root/'real'),'--run',str(root/'mixed25'),'--output',str(root/'comparison')],cwd=ROOT,check=True)
        print('ABLATION FINISHED:',root/'comparison/report.html',flush=True)
if __name__=='__main__':main()

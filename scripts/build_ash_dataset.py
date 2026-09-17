"""Prepare, render or export a versioned CAD synthetic dataset. No model training."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mes_vision.synthetic.planning import make_plan,fingerprint


def write(path,value): path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root,seed,samples):
    from inspect_ash_cad import mesh
    registry_path=ROOT/'datasets/cad-sources/ash-v1/cad-sources.json'; registry=json.loads(registry_path.read_text(encoding='utf-8'))
    geometry={}; lineage={}
    for v in registry['variants']:
        ref=next(p for p in v['files'] if p['path'].endswith('.stl')); path=registry_path.parent/ref['path']
        if sha(path)!=ref['sha256']: raise ValueError('CAD source changed')
        tri,_=mesh(path); geometry[v['code']]=(tri[:,:,[0,2,1]]*np.array([.001,-.001,.001])).tolist()
        lineage[v['code']]=dict(source_sha256=ref['sha256'],source_registry=str(registry_path),source_file=ref['path'])
    plan=make_plan(seed); plan['cad_lineage']=lineage
    root.mkdir(parents=True,exist_ok=False)
    for folder in ('raw','receipts','quality','masks'): (root/folder).mkdir()
    write(root/'geometry.json',geometry); write(root/'plan.json',plan)
    sources=['scripts/render_ash_samples.py','scripts/render_ash_dataset.py','scripts/build_ash_dataset.py',
        'src/mes_vision/synthetic/planning.py','src/mes_vision/synthetic/projection.py']
    contract=dict(plan_sha256=sha(root/'plan.json'),geometry_sha256=sha(root/'geometry.json'),
        source_registry_sha256=sha(registry_path),samples=samples,source_hashes={p:sha(ROOT/p) for p in sources})
    write(root/'contract.json',contract)
    write(root/'status.json',dict(status='prepared',planned_images=len(plan['tasks']),trained=False))
    print('Prepared '+str(len(plan['tasks']))+' images: '+str(root))


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('action',choices=['prepare','render','export'])
    parser.add_argument('--root',type=Path,required=True); parser.add_argument('--seed',type=int,default=20260911)
    parser.add_argument('--samples',type=int,default=32); parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(); root=args.root.resolve()
    if args.action=='prepare':
        if not 8<=args.samples<=256: raise ValueError('Samples must be between 8 and 256')
        prepare(root,args.seed,args.samples)
    elif args.action=='render':
        import subprocess
        command=[str(ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe'),'--background','--factory-startup',
            '--python-exit-code','1','--python',str(ROOT/'scripts/render_ash_dataset.py'),'--','--root',str(root)]
        if args.limit: command+=['--limit',str(args.limit)]
        raise SystemExit(subprocess.call(command,cwd=ROOT))
    else:
        from mes_vision.synthetic.dataset_export import export
        print(json.dumps(export(root),ensure_ascii=False,indent=2))


if __name__=='__main__': main()

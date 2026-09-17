"""Blender worker for an immutable scene plan. Completed frames are hash-verified on resume."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import bpy
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_ash_samples import scene_setup,add_part,camera_record


def write(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(path)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--root',type=Path,required=True); parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:]); root=args.root.resolve()
    plan=json.loads((root/'plan.json').read_text()); geometry=json.loads((root/'geometry.json').read_text())
    contract=json.loads((root/'contract.json').read_text()); plan_hash=hashlib.sha256((root/'plan.json').read_bytes()).hexdigest()
    if plan_hash!=contract['plan_sha256'] or hashlib.sha256((root/'geometry.json').read_bytes()).hexdigest()!=contract['geometry_sha256']:
        raise ValueError('Immutable plan/geometry changed')
    for name,expected in contract['source_hashes'].items():
        if hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Generator source changed; create a new dataset')
    for i,task in enumerate(plan['tasks']):
        if args.limit and i>=args.limit: break
        image=root/'raw'/(task['id']+'.png'); receipt=root/'receipts'/(task['id']+'.json')
        if receipt.exists():
            r=json.loads(receipt.read_text())
            if r['plan_sha256']!=plan_hash or hashlib.sha256(image.read_bytes()).hexdigest()!=r['sha256']:
                raise ValueError('Completed frame changed')
            continue
        scene,mat,devices=scene_setup(task['condition'],False,contract['samples'])
        scene.camera.location=task['camera']['position']
        for o in task['objects']: add_part(o['code'],np.asarray(geometry[o['code']]),mat,o['x'],o['y'],o['angle'])
        bpy.context.view_layer.update(); actual=camera_record(scene)
        if not np.allclose(actual['position'],task['camera']['position'],atol=1e-8): raise ValueError('Camera position changed')
        scene.render.filepath=str(image); bpy.ops.render.render(write_still=True)
        write(receipt,dict(id=task['id'],plan_sha256=plan_hash,sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
            camera=actual,blender=bpy.app.version_string,devices=devices,samples=contract['samples']))
        print(f"DATASET_PROGRESS {i+1}/{len(plan['tasks'])} {task['id']}",flush=True)


if __name__=='__main__': main()

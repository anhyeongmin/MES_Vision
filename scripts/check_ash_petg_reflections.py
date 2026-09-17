"""User-specified JAMG HE PETG colors: hypothetical gloss sensitivity check."""
from pathlib import Path
import argparse,copy,subprocess,sys
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from mes_vision.training.data import read_json,write_json,sha256,require
BASE=ROOT/'artifacts/preprint-olive-white-v1'; OUT=ROOT/'artifacts/preprint-petg-reflections-v1'
FINISHES={'diffuse-control':(.75,.9),'semigloss-assumption':(.25,.25),'gloss-assumption':(.1,.1)}


def prepare():
    plan=read_json(BASE/'plan.json'); tasks=[]
    OUT.mkdir(exist_ok=False)
    for name in ('raw','receipts','quality','masks','review'): (OUT/name).mkdir()
    for finish,(part,table) in FINISHES.items():
        for lighting in ('side','near-axis'):
            source=[t for t in plan['tasks'] if t['group']=='olive-medium-d1']
            for original in source:
                task=copy.deepcopy(original); code=task['objects'][0]['code']; group=f'{finish}-{lighting}'
                task.update(id=group+'-'+code,group=group,finish=finish,lighting=lighting)
                c=task['condition']; c.update(id=group,roughness=part,table_roughness=table)
                if lighting=='near-axis': c.update(light=[.001,.001,.075],energy=.55)
                tasks.append(task)
        task=copy.deepcopy(next(t for t in plan['tasks'] if t['id']=='olive-medium-o7'))
        task.update(id=finish+'-overview',group=finish+'-overview',finish=finish,lighting='side')
        task['condition'].update(id=task['id'],roughness=part,table_roughness=table)
        tasks.append(task)
    plan={'schema_version':1,'tasks':tasks,'user_materials':{'table':'JAMG HE PETG Olive Green','parts':'JAMG HE PETG White'},
        'finishes':FINISHES,'source_plan_sha256':sha256(BASE/'plan.json'),
        'optical_properties_measured':False,'purpose':'roughness and light-direction sensitivity; no PETG material accuracy claim',
        'assumptions':'opaque generic dielectric shader; no measured color, transmission, actual extrusion or layer structure',
        'manufacturer_reference':'https://www.jamghe.com/products/petg-filament-1-75mm-1kg'}
    write_json(OUT/'plan.json',plan); write_json(OUT/'geometry.json',read_json(BASE/'geometry.json'))
    sources=['scripts/check_ash_petg_reflections.py','scripts/render_ash_petg_reflections.py','scripts/render_ash_samples.py',
             'scripts/render_ash_dataset.py','src/mes_vision/synthetic/planning.py','src/mes_vision/synthetic/projection.py']
    write_json(OUT/'contract.json',{'plan_sha256':sha256(OUT/'plan.json'),'geometry_sha256':sha256(OUT/'geometry.json'),
        'samples':32,'source_hashes':{s:sha256(ROOT/s) for s in sources}})
    print('PREPARED',len(tasks),'gloss/light sensitivity images',flush=True)


def render_all():
    subprocess.run([str(ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe'),'--background','--factory-startup',
        '--python-exit-code','1','--python',str(ROOT/'scripts/render_ash_petg_reflections.py'),'--','--root',str(OUT)],check=True,cwd=ROOT)


def inspect():
    import check_ash_before_print as base
    base.OUT=OUT; base.label_and_inspect()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['prepare','render','inspect']); args=p.parse_args()
    {'prepare':prepare,'render':render_all,'inspect':inspect}[args.action]()

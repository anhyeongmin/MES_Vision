"""Quality gates and split-preserving COCO export; synthetic lineage is never hidden."""
from pathlib import Path
from collections import Counter,defaultdict
import hashlib,json,shutil
import numpy as np
from PIL import Image
from .planning import geometry_labels,fingerprint,CODES
from ..training.data import validate_dataset

GATES=dict(min_defect_pixels=80,min_rgb_difference_mean=1.5,min_rgb_difference_p90=4.,
    max_object_occlusion_fraction=.005,border_pixels=4,min_object_box_side=30,
    note='Fixed synthetic generation gates; not validated product inspection thresholds')


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8'); tmp.replace(path)


def appearance(actual,normal,mask):
    if actual.shape!=normal.shape or actual.shape[:2]!=mask.shape or not mask.any(): raise ValueError('Aligned nonempty comparison required')
    delta=np.abs(actual.astype(float)-normal.astype(float)).mean(axis=2)[mask]
    result=dict(mean_rgb_difference=float(delta.mean()),p90_rgb_difference=float(np.quantile(delta,.9)))
    result['passed']=result['mean_rgb_difference']>=GATES['min_rgb_difference_mean'] and result['p90_rgb_difference']>=GATES['min_rgb_difference_p90']
    return result


def quality(root,task,geometry,plan_hash):
    path=root/'quality'/(task['id']+'.json'); image=root/'raw'/(task['id']+'.png')
    receipt=read(root/'receipts'/(task['id']+'.json'))
    if receipt['plan_sha256']!=plan_hash or sha(image)!=receipt['sha256']: raise ValueError('Rendered source changed')
    normal_name=task['group']+'-OK' if task['domain']=='detail' else None
    gate_hash=fingerprint(GATES)
    algorithm_hash=fingerprint({name:sha(Path(__file__).parent/name) for name in ('dataset_export.py','planning.py','projection.py')})
    if path.exists():
        old=read(path)
        if old['image_sha256']!=receipt['sha256'] or old['gate_hash']!=gate_hash or old.get('algorithm_hash')!=algorithm_hash: raise ValueError('Quality input, algorithm or gates changed')
        return old
    value=dict(id=task['id'],image_sha256=receipt['sha256'],gate_hash=gate_hash,algorithm_hash=algorithm_hash,accepted=False)
    try:
        objects,defects,mask=geometry_labels(task,geometry)
        value.update(objects=objects,defects=defects)
        if mask is not None:
            Image.fromarray(mask.astype(np.uint8)*255).save(root/'masks'/(task['id']+'.png'))
            value['mask_sha256']=sha(root/'masks'/(task['id']+'.png'))
        if defects:
            normal_path=root/'raw'/(normal_name+'.png'); r=read(root/'receipts'/(normal_name+'.json'))
            if r['plan_sha256']!=plan_hash or sha(normal_path)!=r['sha256']: raise ValueError('Counterfactual changed')
            with Image.open(image) as a,Image.open(normal_path) as b:
                value['appearance']=appearance(np.asarray(a),np.asarray(b),mask)
            if not value['appearance']['passed']: raise ValueError('Low visible RGB difference in geometric defect region')
        value['accepted']=True
    except ValueError as exc:
        value['reason']=str(exc)
    write(path,value)
    return value


def export(root):
    root=Path(root).resolve(); plan=read(root/'plan.json'); contract=read(root/'contract.json'); ph=sha(root/'plan.json')
    if ph!=contract['plan_sha256'] or sha(root/'geometry.json')!=contract['geometry_sha256']: raise ValueError('Plan or geometry changed')
    geometry={k:np.asarray(v) for k,v in read(root/'geometry.json').items()}
    tasks=plan['tasks']; groups=defaultdict(list)
    for task in tasks: groups[task['group']].append(task)
    for group in groups.values():
        if len({t['split'] for t in group})!=1: raise ValueError('Condition group leaks across splits')
        if group[0]['domain']=='detail':
            if {t['objects'][0]['code'] for t in group}!=set(CODES): raise ValueError('Incomplete detail group')
            if len({fingerprint([t['camera'],t['condition'],{k:v for k,v in t['objects'][0].items() if k!='code'}]) for t in group})!=1:
                raise ValueError('Counterfactual conditions differ')
    qualities={}
    for i,t in enumerate(tasks):
        qualities[t['id']]=quality(root,t,geometry,ph)
        if (i+1)%25==0: print(f"QUALITY_PROGRESS {i+1}/{len(tasks)}",flush=True)
    rejected={g:[dict(id=t['id'],reason=qualities[t['id']]['reason']) for t in ts if not qualities[t['id']]['accepted']]
              for g,ts in groups.items() if any(not qualities[t['id']]['accepted'] for t in ts)}
    accepted=[t for t in tasks if t['group'] not in rejected]
    final=root/'training'; pending=root/'training.pending'
    if final.exists():
        for name in ('overview-objects','detail-objects','detail-defects'): validate_dataset(final/name)
        return read(final/'export-report.json')
    pending.mkdir(exist_ok=False)
    results={}
    for name,domain,role in (('overview-objects','overview','object_detector'),('detail-objects','detail','object_detector'),
                             ('detail-defects','detail','known_defect_detector')):
        target=pending/name; target.mkdir()
        meta=dict(schema_version=1,kind='synthetic',role=role,product_id='ASH',camera_target=plan['camera_target'],
            generator_plan_sha256=ph,cad_lineage=plan['cad_lineage'],split_unit=plan['split_unit'],cad_split=plan['cad_split'],
            evaluation_scope=plan['evaluation_scope'],specimen_ids_meaning='synthetic scene instances, not physical specimen identities',
            real_validation_required=True,generation_gates=GATES)
        write(target/'dataset.json',meta)
        for split in ('train','valid','test'):
            folder=target/split; folder.mkdir(); images=[]; annotations=[]
            codes=['ASH'] if role=='object_detector' else CODES[1:]
            categories=[dict(id=i+1,name=code,supercategory='object' if role=='object_detector' else 'defect') for i,code in enumerate(codes)]
            for task in (t for t in accepted if t['domain']==domain and t['split']==split):
                identity=len(images)+1; filename=task['id']+'.png'
                shutil.copyfile(root/'raw'/filename,folder/filename)
                if sha(folder/filename)!=qualities[task['id']]['image_sha256']: raise ValueError('Copy mismatch')
                images.append(dict(id=identity,file_name=filename,width=1280,height=720,
                    specimen_ids=[f"synthetic:{task['id']}:{i}" for i in range(len(task['objects']))],
                    capture_session_id=task['group'],cad_source_ids=[o['code'] for o in task['objects']],
                    render_scene_id=task['id'],synthetic=True))
                labels=qualities[task['id']]['objects' if role=='object_detector' else 'defects']
                for v in labels:
                    b=v['bbox']; code='ASH' if role=='object_detector' else v['code']
                    annotations.append(dict(id=len(annotations)+1,image_id=identity,category_id=codes.index(code)+1,
                        bbox=b,area=b[2]*b[3],iscrowd=0))
            write(folder/'_annotations.coco.json',dict(info=dict(description=plan['evaluation_scope']),licenses=[],
                images=images,annotations=annotations,categories=categories))
        validation=validate_dataset(target); results[name]=dict(fingerprint=validation['fingerprint'],splits=validation['splits'])
    report=dict(status='validated',planned_rgb=len(tasks),accepted_rgb=len(accepted),rejected_rgb=len(tasks)-len(accepted),
        rejected_groups=rejected,groups=results,shared_cad_across_splits=True,evaluation_scope=plan['evaluation_scope'],
        model_trained=False,physical_camera_calibrated=False,thresholds=GATES,
        exporter_sha256=sha(Path(__file__)),plan_sha256=ph)
    write(pending/'export-report.json',report); pending.rename(final)
    write(root/'status.json',dict(status='dataset_ready',accepted_rgb=len(accepted),trained=False))
    return report

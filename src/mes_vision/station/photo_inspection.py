"""Saved photograph inference using production adapters, with no device authority."""
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json, time, os
from uuid import uuid4

from mes_vision.inputs import ImageSource
from mes_vision.inspection import InspectionPipeline, Mode
from mes_vision.inspection.model_registry import default_registry
from mes_vision.decision import load_policy, apply_policy
from mes_vision.training.data import read_json, write_json, sha256, require
from mes_vision.vlm.snapshots import save_snapshot, load_snapshot
from mes_vision.anomaly.features import fingerprint
from .vision import checked_batch, _Detection


def load_bundle(path, root):
    bundle=read_json(Path(path)); root=Path(root).resolve()
    require(bundle['schema_version']==1 and bundle['purpose']=='saved_photo_inspection', 'Unsupported photo model bundle')
    require(bundle['production_ready'] is False and bundle['product_id'], 'Photo bundle cannot authorize production')
    require(set(bundle['models'])=={'overview','objects','defects'}, 'Three separate model roles required')
    for role,spec in bundle['models'].items():
        require(spec['backend']=='rfdetr-small','Unsupported model provider')
        require(spec['inference_profile']=='standard' and spec['max_batch_size']==1,'Use the evaluated execution profile')
        require(type(spec['threshold']) in (int,float) and 0<=spec['threshold']<=1,'Invalid candidate threshold')
        path=Path(spec['weights']); path=path if path.is_absolute() else root/path
        require(sha256(path)==spec['sha256'],role+' model changed')
        spec['weights']=str(path.resolve())
        expected=[f'NG{i:02d}' for i in range(1,7)] if role=='defects' else [bundle['product_id']]
        require(spec['class_names']==expected,'Model class order changed')
        if role=='defects':
            require(spec['class_codes']=={str(i):n for i,n in enumerate(expected)},'Defect code mapping changed')
    return bundle


def atomic_json(path, value):
    path=Path(path); temporary=path.with_name('.'+path.name+'-'+uuid4().hex+'.tmp')
    try:
        write_json(temporary,value); temporary.replace(path)
    finally:
        if temporary.exists(): temporary.unlink()


class ResidentPhotos:
    """Own all three model roles and the GPU reservation for one worker lifetime."""
    def __init__(self, bundle_path, root, runtime, registry=None, *, allow_background=False):
        self.allow_background = allow_background
        self.allow_vlm = False
        self.root=Path(root); self.runtime=Path(runtime); self.registry=registry or default_registry()
        self.source_bundle=read_json(Path(bundle_path))
        self.bundle=load_bundle(bundle_path,root); self.handles={}; self.stack=ExitStack()

    def __enter__(self):
        from filelock import FileLock
        from mes_vision.vlm.gpu import GpuCoordinator
        self.gpu=GpuCoordinator(self.runtime/'vlm/gpu-coordination')
        try:
            self.stack.enter_context(FileLock(str(self.gpu.root/'resident.lock'),timeout=0))
            atomic_json(self.gpu.root/'resident.json',dict(pid=os.getpid(),active=True,allow_vlm=False))
            with ExitStack() as loading:
                (loading if self.allow_background else self.stack).enter_context(self.gpu.foreground(timeout=60))
                for role in ('overview','objects','defects'):
                    handle=self.registry.create(self.bundle['models'][role],'objects' if role=='overview' else role)
                    self.stack.callback(handle.close); handle.load(); self.handles[role]=handle
                if self.allow_background:
                    import torch
                    if torch.cuda.is_available():
                        free,total=torch.cuda.mem_get_info()
                        self.allow_vlm=free>=12*1024**3 and total>=20*1024**3
                    atomic_json(self.gpu.root/'resident.json',dict(pid=os.getpid(),active=True,allow_vlm=self.allow_vlm))
            return self
        except BaseException:
            self.stack.close(); raise

    def __exit__(self,*args): self.stack.close()


def run_photos(request, root, *, registry=None, resident=None):
    """All outputs retain file input identity; no frame is relabelled as live."""
    root=Path(root); output=Path(request['output']).resolve()
    require(not output.exists(),'Choose a new result directory')
    jobs=request['images']; require(isinstance(jobs,list) and 1<=len(jobs)<=200,'Select 1 to 200 photographs')
    require(request['image_kind'] in ('real','synthetic'),'Image origin required')
    require(all(j['role'] in ('overview','detail') for j in jobs),'Unknown capture domain')
    require(len({str(Path(j['path']).resolve()) for j in jobs})==len(jobs),'Duplicate photograph')
    if resident is None: bundle=load_bundle(request['bundle'],root)
    else:
        require(Path(root).resolve()==resident.root.resolve() and Path(request['runtime']).resolve()==resident.runtime.resolve(),'Resident environment changed')
        require(read_json(Path(request['bundle']))==resident.source_bundle,'Resident model bundle changed; reload models')
        bundle=resident.bundle
    output.mkdir(parents=True); atomic_json(output/'status.json',{'state':'LOADING','completed':0,'total':len(jobs)})
    write_json(output/'bundle.json',bundle)
    records=[]; registry=registry or default_registry(); began=time.perf_counter()
    from filelock import FileLock
    from mes_vision.vlm.gpu import GpuCoordinator
    coordinator=GpuCoordinator(Path(request['runtime'])/'vlm/gpu-coordination')
    try:
        with ExitStack() as stack:
            # Do not compete with loaded live models, even when the station is paused.
            if resident is None: stack.enter_context(FileLock(str(coordinator.root/'resident.lock'),timeout=0))
            if resident is None or resident.allow_background: stack.enter_context(coordinator.foreground(timeout=60))
            handles={} if resident is None else resident.handles
            needed=set()
            for job in jobs: needed.update(('overview',) if job['role']=='overview' else ('objects','defects'))
            for role in ('overview','objects','defects'):
                if resident is not None: continue
                if role not in needed: continue
                handle=registry.create(bundle['models'][role],'objects' if role=='overview' else role)
                stack.callback(handle.close); handle.load(); handles[role]=handle
            policy=load_policy(root/'configs/decision/unconfigured.json')
            for index,job in enumerate(jobs):
                started=time.perf_counter(); path=Path(job['path'])
                require(sha256(path)==job['sha256'],'Photograph changed before inspection')
                with ImageSource(path) as source:
                    event=source.read(); require(event.frame is not None,event.message); frame=event.frame
                require(not frame.is_live and sha256(path)==job['sha256'],'Photograph changed during reading')
                detector=handles['overview' if job['role']=='overview' else 'objects'].adapter
                batch=checked_batch(detector,frame,100)
                dedup_audit=None
                if job['role']=='detail':
                    from .detail_dedup import deduplicate_detail
                    batch,dedup_audit=deduplicate_detail(batch)
                associated=False
                if job['role']=='detail' and len(batch.detections)==1:
                    d=batch.detections[0]; b=d.box
                    associated=(d.label==bundle['product_id'] and abs((b.x1+b.x2)/2/frame.width-.5)<=.25
                                and abs((b.y1+b.y2)/2/frame.height-.5)<=.25)
                inspectors=(handles['defects'].adapter,) if associated else ()
                pipeline=InspectionPipeline(_Detection(batch),inspectors,mode=Mode.MODEL_FILE,
                    product_id=bundle['product_id'],expected_count=1 if job['role']=='detail' else None,
                    inspection_batch_size=1)
                raw=pipeline.run(frame)
                from mes_vision.inspection.finding_scores import RULE
                raw.config['finding_presentation']=dict(RULE)
                if dedup_audit is not None: raw.config['detail_deduplication']=dedup_audit
                raw.config['saved_photo']={'role':job['role'],'source_sha256':job['sha256'],
                    'image_kind':request['image_kind'],'model_bundle_digest':fingerprint(bundle),
                    'detail_association_confirmed':associated,'production_ready':False}
                # Keep existing incomplete-criteria policy semantics; no synthetic OK or NG approval.
                result=apply_policy(raw,policy)
                snapshot=output/f'photo-{index:04d}'
                manifest=save_snapshot(snapshot,frame,result,kind=request['image_kind'])
                require(not result.robot_commands_enabled and all(o.final_decision=='REVIEW' for o in result.objects),
                        'Unvalidated photo inspection cannot approve products')
                record={'id':str(index),'name':path.name,'source_path':str(path.resolve()),'source_sha256':job['sha256'],
                    'role':job['role'],'snapshot':snapshot.name,'snapshot_digest':fingerprint(manifest),
                    'association_confirmed':associated,'objects':len(result.objects),
                    'elapsed_ms':(time.perf_counter()-started)*1000}
                records.append(record)
                atomic_json(output/'status.json',{'state':'INSPECTING','completed':len(records),'total':len(jobs)})
        report={'schema_version':1,'bundle_digest':fingerprint(bundle),'model_bundle':'bundle.json',
            'records':records,'image_kind':request['image_kind'],'production_ready':False,
            'robot_commands_enabled':False,'vlm_enqueued':False,'total_ms':(time.perf_counter()-began)*1000}
        write_json(output/'results.json',report)
        atomic_json(output/'status.json',{'state':'COMPLETED','completed':len(records),'total':len(jobs)})
        return report
    except BaseException as exc:
        atomic_json(output/'status.json',{'state':'FAILED','completed':len(records),'total':len(jobs),'error':str(exc)})
        raise


def open_results(path):
    path=Path(path).resolve(); root=path.parent; report=read_json(path)
    require(report['schema_version']==1 and report['production_ready'] is False and report['robot_commands_enabled'] is False,
            'Unsupported saved photo results')
    require(report['bundle_digest']==fingerprint(read_json(root/'bundle.json')),'Model bundle changed')
    require(len({r['id'] for r in report['records']})==len(report['records']),'Duplicate photo identity')
    for record in report['records']:
        snapshot=(root/record['snapshot']).resolve()
        require(snapshot.parent==root,'Snapshot escaped the result directory')
        manifest,result,_=load_snapshot(snapshot,expected_digest=record['snapshot_digest'])
        meta=result['config']['saved_photo']
        require(meta['role']==record['role'] and meta['source_sha256']==record['source_sha256']
                and meta['model_bundle_digest']==report['bundle_digest']
                and meta['image_kind']==report['image_kind']==manifest['kind']
                and meta['detail_association_confirmed']==record['association_confirmed']
                and meta['production_ready'] is False and result['mode']=='model_file'
                and record['objects']==len(result['objects'])
                and all(o['final_decision']=='REVIEW' for o in result['objects'])
                and result['robot_commands_enabled'] is False and not result['frame']['is_live'],
                'Photo identity, model or capture domain changed')
    return root,report

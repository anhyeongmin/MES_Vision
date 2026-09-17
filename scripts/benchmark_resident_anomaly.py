"""Compare step-2 transfers against engine-owned GPU references, using local example inputs."""
import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--reference-bank',type=Path,required=True,help='Existing synthetic DINO bank for integration parity')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=15)
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageEnhance,ImageOps
    from filelock import FileLock
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import read_json,write_json,sha256,require
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.anomaly.features import DinoFeatures
    from mes_vision.anomaly.scoring import nearest_neighbors,AnomalyEngine,Criteria
    from mes_vision.anomaly.resident import ResidentNeighbors
    from mes_vision.validation.metrics import distribution
    from mes_vision.vlm.gpu import GpuCoordinator
    require(5<=args.samples<=200,'Invalid sample count')
    require(torch.cuda.is_available(),'CUDA required')
    meta=read_json(args.reference_bank/'bank.json')
    require(meta['kind']=='synthetic','Use a synthetic bank for benchmark integration checks')
    args.output.mkdir(parents=True,exist_ok=False)
    with Image.open(args.image) as opened:
        image=opened.convert('RGB'); image.thumbnail((640,480))
    variants=[image,ImageOps.mirror(image),ImageEnhance.Brightness(image).enhance(.7),ImageEnhance.Brightness(image).enhance(1.2)]
    crops=[np.asarray(variants[i%4].resize(((200,200),(240,160),(160,240),(320,200))[i%4])).copy() for i in range(8)]
    rgb=np.asarray(image).copy(); rng=np.random.default_rng(17)
    memory=rng.normal(size=(2048,384)).astype(np.float32); memory/=np.linalg.norm(memory,axis=1,keepdims=True)
    config=JobConfig.load(args.project_root/'configs/training/objects.json')
    detector=RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.05,training_scope='coco_general',max_batch_size=1)
    defects=RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.05,training_scope='coco_general',max_batch_size=4)
    features=DinoFeatures(args.project_root,max_batch_size=4)
    search=None; engines=[]
    report={'status':'RUNNING','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'scope':'Full-frame RF-DETR + max-four crop RF-DETR/DINO/exact L2. No camera, tracking, saving, database, UI, robot or VLM.',
        'weights_sha256':config.initial_sha256,'image_sha256':sha256(args.image),'bank_vectors':2048,
        'synthetic_memory':True,'generic_detection_weights':True,'product_accuracy_validated':False,
        'samples':[],'input_digests':[hashlib.sha256(a.tobytes()).hexdigest() for a in crops]}
    gate=GpuCoordinator(args.project_root/'artifacts/operation/vlm/gpu-coordination')
    def timed(fn):
        torch.cuda.synchronize(); began=time.perf_counter(); value=fn(); torch.cuda.synchronize()
        return value,(time.perf_counter()-began)*1000
    def cycle(count,mode,retain=False):
        began=time.perf_counter()
        row={'count':count,'mode':mode,'defects_ms':0.,'features_ms':0.,'distance_ms':0.}
        _,row['detect_ms']=timed(lambda:detector.predict_rgb(rgb))
        results=[]
        for start in range(0,count,4):
            group=crops[start:min(start+4,count)]
            _,ms=timed(lambda:defects.predict_many_rgb(group)); row['defects_ms']+=ms
            extract=features.extract_many_device if mode=='resident' else features.extract_many
            embeddings,ms=timed(lambda:extract(group)); row['features_ms']+=ms
            for embedding in embeddings:
                flat=embedding.reshape(-1,384)
                (distances,indices),ms=timed(lambda:search.search(flat) if mode=='resident' else nearest_neighbors(flat,memory,device='cuda'))
                row['distance_ms']+=ms
                if retain:
                    host=embedding.cpu().numpy().copy() if mode=='resident' else embedding
                    results.append((host,distances,indices))
        row['total_ms']=(time.perf_counter()-began)*1000
        return row,results
    try:
        with FileLock(str(gate.root/'resident.lock'),timeout=0),gate.foreground(timeout=10):
            began=time.perf_counter(); detector.load(); defects.load(); features.load()
            search=ResidentNeighbors(memory)
            pointer=search._memory.data_ptr()
            report['reference_resident_bytes']=search.resident_bytes
            # Warm every actual batch shape before randomized timing samples.
            for _ in range(2):
                for mode in ('legacy','resident'):
                    for count in (1,2,3,4): cycle(count,mode)
            report['prepare_seconds']=time.perf_counter()-began
            report['resident_allocated_mib']=torch.cuda.memory_allocated()/2**20
            _,baseline=cycle(8,'legacy',True); _,candidate=cycle(8,'resident',True)
            max_deltas=[max(float(np.max(np.abs(a[i]-b[i]))) for a,b in zip(baseline,candidate)) for i in (0,1)]
            index_equal=all(np.array_equal(a[2],b[2]) for a,b in zip(baseline,candidate))
            require(max_deltas==[0.,0.] and index_equal,'Feature/distance/reference parity failed')
            report['parity']={'max_feature_abs_delta':max_deltas[0],'max_distance_abs_delta':max_deltas[1],'nearest_indices_equal':index_equal}
            # Real CUDA engine path against a previously saved bank, including threshold equality.
            fixture_features=DinoFeatures(args.project_root,image_size=meta['feature_signature']['preprocessing']['size'],
                max_batch_size=meta['feature_signature'].get('batch_limit',1))
            fixture_features.model=features.model
            for mode in ('legacy','resident'):
                engine=AnomalyEngine(args.reference_bank,fixture_features,product_id=meta['product_id'],allow_synthetic=True,distance_execution=mode)
                engines.append(engine); engine.prepare()
            value=engines[0].score(crops[0])[0]['raw_score']
            require(.02<value<1.98,'Fixture score unsuitable for boundary checks')
            criteria=[None]+[Criteria('synthetic-boundary',engines[0].bank.digest,meta['product_id'],lo,hi,pixel,
                True,'synthetic GPU comparison only','synthetic') for lo,hi,pixel in
                ((value,value+.01,value+.005),(value-.01,value,value),(value-.01,value+.01,value))]
            cases=[]
            for criterion in criteria:
                for engine in engines: engine.criteria=criterion
                first,second=(engine.score_many(crops[:3]) for engine in engines)
                for a,b in zip(first,second):
                    details=lambda r:{k:v for k,v in r[0].items() if k not in ('elapsed_ms','distance_execution')}
                    require(details(a)==details(b) and np.array_equal(a[1],b[1]) and np.array_equal(a[2],b[2]),'Engine verdict/evidence parity failed')
                cases.append(first[0][0]['status'])
            require(cases==['UNCERTAIN','PASS','FAIL','UNCERTAIN'],'Threshold boundary coverage failed')
            report['engine_parity']={'passed':True,'bank_digest':engines[0].bank.digest,'cases':12,'boundary_statuses':cases}
            for engine in engines: engine.close()
            engines.clear(); fixture_features.model=None
            # Boundary ties and a bank spanning several chunks exercise the actual CUDA search.
            wide=rng.normal(size=(4099,384)).astype(np.float32); wide/=np.linalg.norm(wide,axis=1,keepdims=True)
            wide[2048]=wide[0]; wide[4098]=wide[0]
            query=wide[[0,10,2047,2048,4098]].copy()
            check=ResidentNeighbors(wide)
            try:
                expected=nearest_neighbors(query,wide,device='cuda'); actual=check.search(torch.from_numpy(query).cuda())
                require(all(np.array_equal(a,b) for a,b in zip(expected,actual)),'CUDA cross-chunk/tie parity failed')
                require(actual[1][0]==actual[1][3]==actual[1][4]==0,'Nearest tie order changed')
                report['cuda_chunk_and_tie_parity']=True
            finally: check.close()
            gc.collect(); torch.cuda.reset_peak_memory_stats()
            schedule=[(count,mode) for _ in range(args.samples) for count in (1,4,8) for mode in ('legacy','resident')]
            random.Random(29).shuffle(schedule)
            for count,mode in schedule: report['samples'].append(cycle(count,mode)[0])
            report['inference_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            report['timings']={str(count):{mode:{key:distribution([r[key] for r in report['samples'] if r['count']==count and r['mode']==mode])
                for key in ('total_ms','detect_ms','defects_ms','features_ms','distance_ms')} for mode in ('legacy','resident')} for count in (1,4,8)}
            require(pointer==search._memory.data_ptr(),'Reference buffer was replaced during inspection')
            report['reference_buffer_reused']=True; report['status']='COMPLETED'
    except Exception as exc:
        report['status']='FAILED'; report['error']=f'{type(exc).__name__}: {exc}'; raise
    finally:
        for engine in engines: engine.close()
        if search: search.close()
        detector.close(); defects.close(); features.close(); gc.collect(); torch.cuda.empty_cache()
        report['reference_released']=search is None or search.resident_bytes==0
        write_json(args.output/'report.json',report)
        if report['samples']:
            with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(report['samples'][0])); writer.writeheader(); writer.writerows(report['samples'])
        print(json.dumps({k:v for k,v in report.items() if k not in ('samples','timings')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

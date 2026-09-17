"""Compare 1/2/4-crop execution using resident real models and identical prepared inputs."""
import argparse
import csv
from dataclasses import asdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project-root',type=Path,default=ROOT); p.add_argument('--image',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True); p.add_argument('--samples',type=int,default=12)
    args=p.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageEnhance,ImageOps
    from filelock import FileLock
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import write_json,sha256,require
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.anomaly.features import DinoFeatures
    from mes_vision.anomaly.scoring import nearest_neighbors
    from mes_vision.validation.metrics import distribution
    from mes_vision.validation.inference_comparison import compare_predictions
    from mes_vision.vlm.gpu import GpuCoordinator
    require(5<=args.samples<=200,'Invalid sample count'); require(torch.cuda.is_available(),'CUDA required')
    args.output.mkdir(parents=True,exist_ok=False)
    with Image.open(args.image) as opened: image=opened.convert('RGB'); image.thumbnail((640,480))
    variants=[image,ImageOps.mirror(image),ImageEnhance.Brightness(image).enhance(.7),ImageEnhance.Brightness(image).enhance(1.2)]
    crops=[np.asarray(variants[i%4].resize(((200,200),(240,160),(160,240),(320,200))[i%4])).copy() for i in range(8)]
    rgb=np.asarray(image).copy(); rng=np.random.default_rng(17)
    memory=rng.normal(size=(2048,384)).astype(np.float32); memory/=np.linalg.norm(memory,axis=1,keepdims=True)
    config=JobConfig.load(args.project_root/'configs/training/objects.json')
    detector=RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.05,training_scope='coco_general',max_batch_size=1)
    defects=RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.05,training_scope='coco_general',max_batch_size=4)
    features=DinoFeatures(args.project_root,max_batch_size=4)
    report={'status':'RUNNING','gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'rfdetr':'1.9.4',
        'scope':'Full-frame RF-DETR + crop RF-DETR/DINO/unchanged nearest-neighbor search. No camera, tracking, saving, database, UI, robot or VLM.',
        'weights_sha256':config.initial_sha256,'image_sha256':sha256(args.image),'bank_vectors':2048,
        'synthetic_memory':True,'generic_detection_weights':True,'product_accuracy_validated':False,
        'samples':[],'parity':{},'input_digests':[hashlib.sha256(a.tobytes()).hexdigest() for a in crops]}
    gate=GpuCoordinator(args.project_root/'artifacts/operation/vlm/gpu-coordination')
    def timed(fn):
        torch.cuda.synchronize(); began=time.perf_counter(); value=fn(); torch.cuda.synchronize()
        return value,(time.perf_counter()-began)*1000
    def cycle(count,cap,retain=False):
        began=time.perf_counter(); row={'count':count,'batch_limit':cap,'defects_ms':0.,'features_ms':0.,'distance_ms':0.}
        _,row['detect_ms']=timed(lambda:detector.predict_rgb(rgb))
        predicted={}; patches=[]; grids=[]
        for start in range(0,count,cap):
            group=crops[start:min(start+cap,count)]
            values,ms=timed(lambda:defects.predict_many_rgb(group)); row['defects_ms']+=ms
            embeddings,ms=timed(lambda:features.extract_many(group)); row['features_ms']+=ms
            for index,(value,embedding) in enumerate(zip(values,embeddings,strict=True)):
                (distance,nearest),ms=timed(lambda:nearest_neighbors(embedding.reshape(-1,384),memory,device='cuda')); row['distance_ms']+=ms
                if retain:
                    predicted[str(start+index)]=[asdict(d) for d in value]; patches.append(embedding); grids.append(distance)
        row['total_ms']=(time.perf_counter()-began)*1000
        return row,(predicted,patches,grids)
    try:
        with FileLock(str(gate.root/'resident.lock'),timeout=0),gate.foreground(timeout=10):
            began=time.perf_counter(); detector.load(); defects.load(); features.load()
            for cap in (1,2,4): cycle(4,cap)
            report['prepare_seconds']=time.perf_counter()-began
            report['resident_allocated_mib']=torch.cuda.memory_allocated()/2**20
            report['compiled_sizes']=sorted(defects._batch_networks)
            before_ids={size:id(net.model.inference_model) for size,net in defects._batch_networks.items()}
            _,baseline=cycle(8,1,True)
            for cap in (2,4):
                _,candidate=cycle(8,cap,True)
                report['parity'][str(cap)]={'detections':[compare_predictions(baseline[0],candidate[0],threshold=t) for t in (.05,.4,.7)],
                    'max_feature_abs_delta':max(float(np.max(np.abs(a-b))) for a,b in zip(baseline[1],candidate[1],strict=True)),
                    'max_distance_abs_delta':max(float(np.max(np.abs(a-b))) for a,b in zip(baseline[2],candidate[2],strict=True))}
                write_json(args.output/f'predictions-batch{cap}.json',candidate[0])
            write_json(args.output/'predictions-serial.json',baseline[0])
            torch.cuda.reset_peak_memory_stats()
            schedule=[(count,cap) for _ in range(args.samples) for count in (1,2,3,4,8) for cap in (1,2,4)]
            random.Random(29).shuffle(schedule)
            for count,cap in schedule: report['samples'].append(cycle(count,cap)[0])
            report['inference_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            report['timings']={str(count):{str(cap):{key:distribution([r[key] for r in report['samples'] if r['count']==count and r['batch_limit']==cap])
                for key in ('total_ms','detect_ms','defects_ms','features_ms','distance_ms')} for cap in (1,2,4)} for count in (1,2,3,4,8)}
            require(before_ids=={size:id(net.model.inference_model) for size,net in defects._batch_networks.items()},'Recompiled during inspection')
            report['no_runtime_recompile']=True; report['status']='COMPLETED'
    except Exception as exc: report['status']='FAILED'; report['error']=f'{type(exc).__name__}: {exc}'; raise
    finally:
        detector.close(); defects.close(); features.close(); gc.collect(); torch.cuda.empty_cache()
        write_json(args.output/'report.json',report)
        if report['samples']:
            with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(report['samples'][0])); writer.writeheader(); writer.writerows(report['samples'])
        print(json.dumps({k:v for k,v in report.items() if k not in ('samples','timings','parity')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

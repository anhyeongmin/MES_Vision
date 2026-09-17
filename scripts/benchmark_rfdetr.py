"""Compare RF-DETR execution profiles with identical local inputs; no camera, robot or training."""
import argparse
import csv
from dataclasses import asdict
import gc
import importlib.metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--images',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profile',choices=('standard','fp32_jit','fp16','fp16_jit'),required=True)
    parser.add_argument('--repeats',type=int,default=10)
    args=parser.parse_args()
    import numpy as np
    import torch
    from PIL import Image,ImageEnhance,ImageOps
    from filelock import FileLock
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import write_json,sha256,require
    from mes_vision.validation.metrics import distribution
    from mes_vision.vlm.gpu import GpuCoordinator
    require(5<=args.repeats<=1000,'Invalid repeat count')
    require(torch.cuda.is_available(),'CUDA required')
    images={}; sources=[]
    for path in sorted(args.images.iterdir()):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png'}: continue
        with Image.open(path) as opened: image=opened.convert('RGB'); image.thumbnail((960,720))
        sources.append({'file':path.name,'sha256':sha256(path),'size':list(image.size)})
        variants={'original':image,'mirror':ImageOps.mirror(image),'dark':ImageEnhance.Brightness(image).enhance(.7),
            'bright':ImageEnhance.Brightness(image).enhance(1.25),'small':image.resize((200,200)),
            'wide':ImageOps.pad(image,(800,400)), 'portrait':ImageOps.pad(image,(400,800))}
        tile=image.resize((256,256)); mosaic=Image.new('RGB',(512,512))
        for xy in ((0,0),(256,0),(0,256),(256,256)): mosaic.paste(tile,xy)
        variants['multiple']=mosaic
        for name,variant in variants.items(): images[path.stem+'-'+name]=np.asarray(variant).copy()
    require(sources,'At least one local image required')
    images['blank']=np.full((480,640,3),128,np.uint8)
    images['noise']=np.random.default_rng(17).integers(70,190,(480,640,3),dtype=np.uint8)
    args.output.mkdir(parents=True,exist_ok=False)
    config=JobConfig.load(args.project_root/'configs/training/objects.json')
    gate=GpuCoordinator(args.project_root/'artifacts/operation/vlm/gpu-coordination')
    backend=RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.05,training_scope='coco_general',inference_profile=args.profile)
    report={'status':'RUNNING','profile':args.profile,'sources':sources,'samples':[],
        'scope':'Single-image adapter call including CPU preprocessing/transfer/postprocessing; synchronized CUDA completion; no camera/tracking/other inspectors/storage/UI/VLM',
        'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'cuda':torch.version.cuda,'rfdetr':importlib.metadata.version('rfdetr'),
        'platform':platform.platform(),'weights_sha256':config.initial_sha256,'product_accuracy_validated':False,
        'threshold':.05,'tf32_matmul':torch.backends.cuda.matmul.allow_tf32,'tf32_cudnn':torch.backends.cudnn.allow_tf32}
    try:
        with FileLock(str(gate.root/'resident.lock'),timeout=0),gate.foreground(timeout=10):
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter(); backend.load(); torch.cuda.synchronize()
            report['load_optimize_seconds']=time.perf_counter()-began
            report['load_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            began=time.perf_counter()
            for _ in range(2):
                for rgb in images.values(): backend.predict_rgb(rgb)
            torch.cuda.synchronize(); report['warmup_seconds']=time.perf_counter()-began
            report['resident_allocated_mib']=torch.cuda.memory_allocated()/2**20
            torch.cuda.reset_peak_memory_stats()
            predictions={}; sequence=list(images)*args.repeats; random.Random(18).shuffle(sequence)
            for name in sequence:
                torch.cuda.synchronize(); began=time.perf_counter(); detections=backend.predict_rgb(images[name]); torch.cuda.synchronize()
                report['samples'].append({'image':name,'ms':(time.perf_counter()-began)*1000})
                if name not in predictions: predictions[name]=[asdict(d) for d in detections]
            report['timing_ms']=distribution([row['ms'] for row in report['samples']])
            report['per_image_ms']={name:distribution([r['ms'] for r in report['samples'] if r['image']==name]) for name in images}
            report['inference_peak_allocated_mib']=torch.cuda.max_memory_allocated()/2**20
            report['model_reference']=asdict(backend.model)
            report['predictions']=predictions
            report['input_shapes']={name:list(rgb.shape) for name,rgb in images.items()}
            report['status']='COMPLETED'
    except Exception as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        backend.close(); gc.collect(); torch.cuda.empty_cache()
        write_json(args.output/'report.json',report)
        with (args.output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=('image','ms')); writer.writeheader(); writer.writerows(report['samples'])
        print(json.dumps({k:v for k,v in report.items() if k not in ('predictions','samples','per_image_ms')},ensure_ascii=False),flush=True)


if __name__=='__main__': main()

"""Internal resident GPU runtime benchmark. Generic weights do not validate products."""
from pathlib import Path
import argparse
import sys
import time
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--project-root",type=Path,default=ROOT); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args()
    import torch
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.anomaly.features import DinoFeatures
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import write_json
    config=JobConfig.load(args.project_root/"configs/training/objects.json")
    backends=[RFDETRBackend(config.initial_weights,config.initial_sha256,threshold=.4,training_scope="coco_general") for _ in range(2)]
    features=None
    try:
        start=time.perf_counter()
        for b in backends: b.load()
        features=DinoFeatures(args.project_root); features.load()
        rgb=np.full((480,640,3),128,np.uint8); crop=rgb[100:300,200:400].copy()
        for b in backends: b.predict_rgb(rgb)
        features.extract(crop); torch.cuda.synchronize(); warmup=time.perf_counter()-start
        identities=[id(b._network) for b in backends]+[id(features.model)]; samples=[]
        for _ in range(10):
            row={}
            for key,fn in (("object_detector_ms",lambda:backends[0].predict_rgb(rgb)),("defect_crop_ms",lambda:backends[1].predict_rgb(crop)),("normal_features_ms",lambda:features.extract(crop))):
                start=time.perf_counter(); fn(); torch.cuda.synchronize(); row[key]=(time.perf_counter()-start)*1000
            samples.append(row)
        assert identities==[id(b._network) for b in backends]+[id(features.model)]
        free,total=torch.cuda.mem_get_info(); report={"gpu":torch.cuda.get_device_name(),"warmup_seconds":warmup,"reloaded_between_frames":False,
            "generic_weights_runtime_only":True,"product_accuracy_validated":False,"includes_vlm":False,"includes_camera_or_disk":False,
            "median_ms":{k:statistics.median(r[k] for r in samples) for k in samples[0]},"samples":samples,
            "free_gib":free/1024**3,"total_gib":total/1024**3}
        args.output.parent.mkdir(parents=True,exist_ok=True); write_json(args.output,report); print(report)
    finally:
        for b in backends: b.close()
        if features: features.close()

if __name__=="__main__": main()

"""Check compiled inference against existing synthetic-trained custom heads, without retraining or registering products."""
import argparse
import gc
from pathlib import Path
import sys
import time
from dataclasses import asdict
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--training-check',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from filelock import FileLock
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.training.data import sha256,write_json,require
    from mes_vision.validation.inference_comparison import compare_predictions
    from mes_vision.vlm.gpu import GpuCoordinator
    require(not args.output.exists(),'Refusing to overwrite verification output')
    gate=GpuCoordinator(args.project_root/'artifacts/operation/vlm/gpu-coordination')
    report={'passed':False,'synthetic_trained_heads_only':True,'product_accuracy_validated':False,'roles':{}}
    inputs={'square':np.full((200,200,3),120,np.uint8),'wide':np.full((320,640,3),120,np.uint8),
        'noise':np.random.default_rng(8).integers(0,256,(480,320,3),dtype=np.uint8)}
    with FileLock(str(gate.root/'resident.lock'),timeout=0),gate.foreground(timeout=10):
        for role,folder in (('objects','objects-resumed'),('defects','defects-trained')):
            weights=args.training_check/folder/'inference.pth'; digest=sha256(weights)
            saved=torch.load(weights,map_location='cpu',weights_only=True,mmap=True)
            names=tuple(saved['args']['class_names']); head_shape=list(saved['model']['class_embed.weight'].shape); del saved
            results={}; references={}
            for profile in ('standard','fp32_jit'):
                backend=RFDETRBackend(weights,digest,threshold=.05,training_scope='product_'+role,class_names=names,inference_profile=profile)
                try:
                    backend.load(); backend.load()
                    require(list(backend._network.model.model.class_embed.weight.shape)==head_shape,'Registered head changed')
                    results[profile]={name:[asdict(d) for d in backend.predict_rgb(rgb)] for name,rgb in inputs.items()}
                    references[profile]=asdict(backend.model)
                    require(all(d['label']==names[d['class_id']] for detections in results[profile].values() for d in detections),'Class ordering changed')
                    batched=backend.predict_many_rgb(list(inputs.values()))
                    batch_results={name:[asdict(d) for d in rows] for name,rows in zip(inputs,batched,strict=True)}
                    require(all(d['label']==names[d['class_id']] for rows in batch_results.values() for d in rows),'Batched class ordering changed')
                    report.setdefault('batch_comparisons',{})[role+'-'+profile]=compare_predictions(results[profile],batch_results,threshold=.05)
                    require(set(backend.actual_batch_sizes)<=set((1,2,4)),'Unexpected batch size')
                finally: backend.close(); gc.collect(); torch.cuda.empty_cache()
            require(sha256(weights)==digest,'Weight file changed')
            require(references['standard']!=references['fp32_jit'],'Policy identity not separated')
            report['roles'][role]={'classes':names,'head_shape':head_shape,'weights_sha256':digest,'model_references':references,
                'comparison':compare_predictions(results['standard'],results['fp32_jit'],threshold=.05),'outputs':results}
            print(role+' custom head verified',flush=True)
    report['passed']=True; write_json(args.output,report)


if __name__=='__main__': main()

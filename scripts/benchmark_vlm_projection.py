"""Compare full Qwen advisory responses with unchanged inputs and projection weights."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=2)
    args=parser.parse_args()
    import torch
    from filelock import FileLock
    from mes_vision.vlm.backend import QwenBackend,GenerationConfig,parse_response
    from mes_vision.vlm.optimization import configure_patch_projection
    from mes_vision.vlm.fixtures import make_vlm_fixture
    from mes_vision.vlm.snapshots import save_snapshot
    from mes_vision.vlm.gpu import GpuCoordinator
    from mes_vision.training.data import write_json,sha256,require
    from mes_vision.validation.metrics import distribution
    require(1<=args.repeats<=10,'Invalid repeat count')
    args.output.mkdir(parents=True,exist_ok=False)
    fixture=make_vlm_fixture(args.output/'fixture',enabled=True)
    queue=fixture['queue']; jobs=[queue.get(fixture['job_id'])]
    jobs.append(queue.get(queue.enqueue(fixture['snapshot'],fixture['result'].objects[0].object_id,asdict(GenerationConfig()))))
    jobs.append(queue.get(queue.enqueue(fixture['snapshot'],fixture['result'].objects[1].object_id,asdict(GenerationConfig(language='en')))))
    missing=args.output/'without-reference'
    save_snapshot(missing,fixture['frame'],fixture['result'],kind='synthetic')
    jobs.append(queue.get(queue.enqueue(missing,fixture['result'].objects[1].object_id,asdict(GenerationConfig()))))
    before={str(p):sha256(p) for root in (fixture['snapshot'],missing) for p in root.rglob('*') if p.is_file()}
    report={'status':'RUNNING','gpu':torch.cuda.get_device_name(),'scope':'Same Qwen BF16/SDPA, prompt/images/tokens; full model.generate plus input preparation. No model loading, queue wait, camera, robot or parallel inspection.',
            'synthetic_inputs':True,'product_accuracy_validated':False,'rows':[]}
    backend=None;gate=GpuCoordinator(args.project_root/'artifacts/operation/vlm/gpu-coordination')
    try:
        with FileLock(str(gate.root/'resident.lock'),timeout=0),gate.foreground(timeout=10):
            backend=QwenBackend(args.project_root,inference_profile='standard')
            report['model']=dict(backend.provenance)
            patch=backend.model.model.visual.patch_embed
            weight=patch.proj.weight.detach().clone(); bias=patch.proj.bias.detach().clone()
            keys=list(backend.model.state_dict())
            # Record first complete requests separately; timed rows have warm libraries.
            report['warmup']=[]
            for mode in ('standard','linear_patch_v1'):
                backend.provenance['execution']=configure_patch_projection(backend.model,mode)
                response=backend.generate(jobs[0]); parse_response(response['raw_text'])
                report['warmup'].append({'mode':mode,'metrics':response['metrics'],'raw_text':response['raw_text']})
            for repeat in range(args.repeats):
                for index,job in enumerate(jobs):
                    modes=('standard','linear_patch_v1') if (repeat+index)%2==0 else ('linear_patch_v1','standard')
                    for mode in modes:
                        backend.provenance['execution']=configure_patch_projection(backend.model,mode)
                        began=time.perf_counter();response=backend.generate(job);wall=time.perf_counter()-began
                        require(not response['truncated'] and not response['deadline_expired'],'Incomplete benchmark response')
                        parsed=parse_response(response['raw_text'])
                        row={'repeat':repeat,'case':index,'mode':mode,'wall_seconds':wall,
                            'metrics':response['metrics'],'analysis':parsed,'raw_text':response['raw_text'],'prompt':response['prompt']}
                        report['rows'].append(row);write_json(args.output/'progress.json',{'completed':len(report['rows'])})
                        print(json.dumps({k:row[k] for k in ('repeat','case','mode','wall_seconds','analysis')},ensure_ascii=False),flush=True)
            pairs=[]
            for repeat in range(args.repeats):
                for index in range(len(jobs)):
                    a=next(r for r in report['rows'] if r['repeat']==repeat and r['case']==index and r['mode']=='standard')
                    b=next(r for r in report['rows'] if r['repeat']==repeat and r['case']==index and r['mode']=='linear_patch_v1')
                    require(a['prompt']==b['prompt'],'Prompt/input provenance changed')
                    pairs.append({'repeat':repeat,'case':index,'raw_text_equal':a['raw_text']==b['raw_text'],
                        'needs_review_equal':a['analysis']['needs_review']==b['analysis']['needs_review']})
            report['response_comparison']=pairs
            require(torch.equal(weight,patch.proj.weight) and torch.equal(bias,patch.proj.bias)
                    and keys==list(backend.model.state_dict()),'Model parameters changed')
            report['parameters_unchanged']=True
            report['snapshots_unchanged']=all(sha256(Path(p))==digest for p,digest in before.items())
            require(report['snapshots_unchanged'],'Saved inspection mutated')
            report['timings']={mode:{'wall_seconds':distribution([r['wall_seconds'] for r in report['rows'] if r['mode']==mode]),
                'first_token_seconds':distribution([r['metrics']['first_token_seconds'] for r in report['rows'] if r['mode']==mode])}
                for mode in ('standard','linear_patch_v1')}
            report['status']='COMPLETED'
    except Exception as exc:
        report['status']='FAILED';report['error']=f'{type(exc).__name__}: {exc}';raise
    finally:
        if backend is not None: backend.close()
        write_json(args.output/'report.json',report)
        print(json.dumps({k:v for k,v in report.items() if k in ('status','timings','response_comparison','error')},ensure_ascii=False),flush=True)


if __name__=='__main__':main()

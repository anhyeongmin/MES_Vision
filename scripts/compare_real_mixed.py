"""Offline training-set diagnostics. Does not train, deploy, or claim validation accuracy."""
from pathlib import Path
import argparse, csv, gc, hashlib, html, json, os, sys, time
from contextlib import ExitStack
from dataclasses import asdict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
for key, folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch'}.items():
    os.environ[key] = str(ROOT / '.cache' / folder)
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''): h.update(block)
    return h.hexdigest()

def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)

def variants():
    # Four orientations, each original/dark/bright/blurred = 16 conditions.
    return [(r, effect) for r in range(4) for effect in ['original','dark','bright','blur']]

def transform(rgb, boxes, turns, effect):
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
    h, w = rgb.shape[:2]
    mapped = [list(b) for b in boxes]
    for _ in range(turns):
        mapped = [[y1, w-x2, y2, w-x1] for x1,y1,x2,y2 in mapped]
        w, h = h, w
    im = Image.fromarray(np.rot90(rgb, turns).copy())
    if effect == 'dark': im = ImageEnhance.Brightness(im).enhance(.7)
    if effect == 'bright': im = ImageEnhance.Brightness(im).enhance(1.3)
    if effect == 'blur': im = im.filter(ImageFilter.GaussianBlur(1.2))
    return np.array(im), mapped

def iou(a, b):
    area = max(0,min(a[2],b[2])-max(a[0],b[0])) * max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area
    return area/union if union else 0.

def assess(preds, targets, threshold):
    ds = [d for d in preds if d['score'] >= threshold]
    used = set(); hits = 0
    for target in targets:
        candidates = []
        for i,d in enumerate(ds):
            if i in used or d['label'] != target['label']: continue
            b=d['box']; score=iou(target['box'],[b[k] for k in ['x1','y1','x2','y2']])
            if score >= .5: candidates.append((score,i))
        if candidates:
            _,i=max(candidates);used.add(i);hits+=1
    expected = {t['label'] for t in targets}
    return dict(normal=not targets, false_normal=not targets and bool(ds),
                missed_class=bool(targets) and not expected.issubset({d['label'] for d in ds}),
                other_class=bool(targets) and any(d['label'] not in expected for d in ds),
                target_count=len(targets), matched=hits, unmatched_predictions=len(ds)-hits,
                localization_miss=hits<len(targets))

def export_checkpoint(source, final, dest, epoch):
    import torch
    template=torch.load(final,map_location='cpu',weights_only=False)
    checkpoint=torch.load(source,map_location='cpu',weights_only=False)
    assert checkpoint['epoch']+1 == epoch
    state={k[len('model.'):]:v for k,v in checkpoint['state_dict'].items() if k.startswith('model.')}
    assert state.keys()==template['model'].keys()
    assert all(v.shape==template['model'][k].shape for k,v in state.items())
    assert all(torch.isfinite(v).all() for v in state.values() if v.is_floating_point())
    template['model']=state
    torch.save(template,dest)
    del template,checkpoint,state
    gc.collect()

def report(out, all_results, jobs):
    import numpy as np
    from PIL import Image, ImageDraw
    rows=[];pictures=[]
    (out/'examples').mkdir(exist_ok=True)
    for model, results in all_results.items():
        for condition in ['ALL'] + [f'rot{r*90}_{e}' for r,e in variants()]:
            selected=[v for v in results if condition=='ALL' or v['condition']==condition]
            for threshold in [.1,.2,.3,.5]:
                metrics=[assess(v['predictions'],v['targets'],threshold) for v in selected]
                rows.append(dict(model=model,condition=condition,threshold=threshold,images=len(metrics),
                    normal_images=sum(m['normal'] for m in metrics),
                    normal_false_images=sum(m['false_normal'] for m in metrics),
                    defect_images=sum(not m['normal'] for m in metrics),
                    class_miss_images=sum(m['missed_class'] for m in metrics),
                    other_class_images=sum(m['other_class'] for m in metrics),
                    target_boxes=sum(m['target_count'] for m in metrics),
                    matched_iou50=sum(m['matched'] for m in metrics),
                    unmatched_predictions=sum(m['unmatched_predictions'] for m in metrics)))
        failures=[v for v in results if any(assess(v['predictions'],v['targets'],.1)[k] for k in ['false_normal','missed_class','other_class','localization_miss'])]
        # Show original failures first, then varied conditions; retain every result in JSON.
        failures.sort(key=lambda v:(v['condition']!='rot0_original',v['image']))
        for i,v in enumerate(failures[:24]):
            job=jobs[v['image']];rgb=np.array(Image.open(job['path']).convert('RGB'))
            rgb,_=transform(rgb,[],v['turns'],v['effect']);im=Image.fromarray(rgb);d=ImageDraw.Draw(im)
            for t in v['targets']: d.rectangle(t['box'],outline='lime',width=3)
            for p in v['predictions']:
                b=p['box'];box=[b[k] for k in ['x1','y1','x2','y2']]
                d.rectangle(box,outline='red',width=2);d.text((box[0],max(0,box[1]-12)),f"{p['label']} {p['score']:.2f}",fill='red')
            name=f'examples/{model}-{i:02d}.jpg';im.save(out/name)
            pictures.append(f'<figure><img loading="lazy" src="{name}"><figcaption>{html.escape(model+" / "+v["image"]+" / "+v["condition"])}</figcaption></figure>')
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    base=[r for r in rows if r['condition']=='rot0_original' and r['threshold']==.1]
    table=''.join(f'<tr><td>{r["model"]}</td><td>{r["normal_false_images"]}/{r["normal_images"]}</td><td>{r["class_miss_images"]}/{r["defect_images"]}</td><td>{r["matched_iou50"]}/{r["target_boxes"]}</td></tr>' for r in base)
    page='''<!doctype html><meta charset="utf-8"><title>모델 비교 결과</title><style>body{font-family:sans-serif;max-width:1100px;margin:35px auto;padding:15px}td,th{padding:12px;border:1px solid #ccc}table{border-collapse:collapse}figure{display:inline-block;width:300px;vertical-align:top;margin:10px}img{max-width:100%;max-height:320px}figcaption{overflow-wrap:anywhere}</style><h1>실제 사진 모델 vs 실제+합성 혼합 모델</h1><p>real_rotation_100: 실제 사진 회전 보강 100회(700 업데이트). mixed_cad_050: 실제+합성 50회(1600 업데이트). 학습량도 달라 합성 데이터만의 인과적 효과를 분리한 실험은 아닙니다.</p><p><strong>독립적인 성능 검증이 아닙니다.</strong> 55장과 그 변형으로 학습 상태와 취약점을 점검했습니다. 변형 이미지는 독립 표본이 아니며 실제 반사·원근 변화 전체를 재현하지 않습니다. 별도 촬영 없이 최적 모델이나 배포 가능 여부를 결정할 수 없습니다.</p><p>회전 4종 × 원본·어두움·밝음·흐림 = 모델마다 880회. 종합 수치는 같은 사진의 반복 조건을 포함합니다. 임계값 비교는 진단용이며 프로그램 설정은 바꾸지 않습니다.</p><h2>원래 사진, 임계값 0.10</h2><table><tr><th>모델</th><th>정상 오탐 장수</th><th>불량 종류 미검출 장수</th><th>정답 박스 일치 IoU≥0.5</th></tr>'''+table+'''</table><p>박스 정답에는 초안과 동그라미 기반 근사 영역이 포함돼 위치 지표도 참고용입니다. 전체 조건·임계값 결과는 summary.csv, 모든 예측은 모델별 JSON에 저장됩니다.</p><h2>오류 예시: 모델당 최대 24장</h2><p>초록=정답 영역, 빨강=모델 예측. 정상 사진에는 초록 박스가 없습니다.</p>'''+''.join(pictures)
    (out/'report.html').write_text(page,encoding='utf-8')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',default='artifacts/training/ash-mixed-cad-20260914-181529-815363/detail-defects')
    parser.add_argument('--output',default='artifacts/real-vs-mixed-comparison')
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--check-only',action='store_true',help='Verify source files without inference')
    args=parser.parse_args()
    import numpy as np
    if args.self_test:
        a=np.zeros((10,20,3),dtype=np.uint8)
        for r,expected in [(1,[2,15,6,19]),(2,[15,4,19,8]),(3,[4,1,8,5])]:
            _,boxes=transform(a,[[1,2,5,6]],r,'original');assert boxes==[expected]
        assert iou([0,0,2,2],[0,0,2,2])==1
        p=dict(label='NG01',score=.9,box=dict(x1=0,y1=0,x2=2,y2=2))
        assert assess([p],[dict(label='NG01',box=[0,0,2,2])],.1)['matched']==1
        assert assess([p],[],.1)['false_normal']
        print('SELF TEST PASSED');return
    import torch
    from PIL import Image
    from filelock import FileLock
    from mes_vision.vlm.gpu import GpuCoordinator
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    run=ROOT/args.run;out=ROOT/args.output;out.mkdir(parents=True,exist_ok=True)
    coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(out/'comparison.lock'),timeout=0))
        meta=json.loads((run/'run.json').read_text());assert meta['status']=='COMPLETED_UNVALIDATED' and meta['epoch']==50
        final=run/'inference-unvalidated.pth';assert sha(final)==meta['weights_sha256']
        dataset=ROOT/'datasets/ash-real-reviewed-v3';coco=json.loads((dataset/'train/_annotations.coco.json').read_text())
        provenance=json.loads((dataset/'provenance.json').read_text(encoding='utf-8'))
        jobs={}
        assert len(coco['images'])==len(provenance['items'])==55
        for im,source in zip(coco['images'],provenance['items']):
            p=dataset/'train'/im['file_name'];assert sha(p)==source['image_sha256']
            targets=[]
            for a in coco['annotations']:
                if a['image_id']==im['id']:
                    x,y,w,h=a['bbox'];targets.append(dict(label=f"NG{a['category_id']:02d}",box=[x,y,x+w,y+h]))
            jobs[im['file_name']]=dict(path=str(p),targets=targets)
        real_run=ROOT/'artifacts/training/ash-rotation-20260914-140119-337220/detail-defects'
        real_meta=json.loads((real_run/'run.json').read_text())
        real_weights=real_run/'inference-unvalidated.pth'
        assert sha(real_weights)==real_meta['weights_sha256']=='6806b694d99b6bd77f76832832a72ad8851d67d5e45a4fb82715854c677303c9'
        assert sha(final)=='38ae34e8a2965273e6e1734b38a9c51526900aa705df7cb4e82f812c318eb1a7'
        sources={'real_rotation_100':real_weights,'mixed_cad_050':final}
        spec=dict(models={k:sha(p) for k,p in sources.items()},dataset=sha(dataset/'train/_annotations.coco.json'),provenance=sha(dataset/'provenance.json'),variants=variants(),threshold=.1,independent_evaluation=False,script_sha256=sha(Path(__file__)))
        if args.check_only:
            print('CHECK PASSED: two model hashes and 55 real images verified; no inference or training');return
        # JSON normalizes tuples; compare serialized forms for repeat runs.
        if (out/'manifest.json').exists():assert json.loads((out/'manifest.json').read_text())==json.loads(json.dumps(spec)), 'Inputs changed; use a new --output folder'
        else:write_json(out/'manifest.json',spec)
        stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
        total=55*16*2;start=time.time();all_results={}
        print(f'OUTPUT: {out}\nModels: 2 | Images per model: 880 | Total: {total}',flush=True)
        for model,source in sources.items():
            dest=out/(model+'-predictions.json')
            results=json.loads(dest.read_text()) if dest.exists() else []
            done={(v['image'],v['condition']) for v in results}
            assert len(done)==len(results) and len(results)<=880
            if len(results)==880:
                all_results[model]=results;print(model,'already complete',flush=True);continue
            weights=source
            if source.suffix=='.ckpt':
                weights=out/(model+'.pth');export_checkpoint(source,final,weights,int(model[-3:]))
            backend=RFDETRBackend(weights,sha(weights),threshold=.1,training_scope='product_defects',class_names=tuple(f'NG{i:02}' for i in range(1,7)),inference_profile='standard',max_batch_size=1)
            try:
                backend.load()
                for name,job in jobs.items():
                    raw=np.array(Image.open(job['path']).convert('RGB'))
                    for turns,effect in variants():
                        condition=f'rot{turns*90}_{effect}'
                        if (name,condition) in done:continue
                        rgb,boxes=transform(raw,[t['box'] for t in job['targets']],turns,effect)
                        preds=[asdict(d) for d in backend.predict_rgb(rgb)]
                        results.append(dict(image=name,condition=condition,turns=turns,effect=effect,targets=[dict(label=t['label'],box=b) for t,b in zip(job['targets'],boxes)],predictions=preds))
                    write_json(dest,results)
                    print(f'{model}: {len(results)}/880 | elapsed {(time.time()-start)/60:.1f} min',flush=True)
            finally:
                write_json(dest,results);backend.close();del backend;gc.collect();torch.cuda.empty_cache()
            all_results[model]=results
        report(out,all_results,jobs)
        write_json(out/'completed.json',dict(status='COMPLETED',predictions=sum(len(r) for r in all_results.values()),elapsed_seconds=time.time()-start,independent_evaluation=False))
        print('COMPARISON FINISHED:',out/'report.html',flush=True)

if __name__=='__main__':main()

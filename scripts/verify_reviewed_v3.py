from compare_undistortion import *
import torch
out=ROOT/'artifacts/training/ash-real-reviewed-v3/detail-defects'
run=json.loads((out/'run.json').read_text());assert run['status']=='COMPLETED_UNVALIDATED'
p=Path(run['weights']);assert sha256(p)==run['weights_sha256']
a=torch.load(p,map_location='cpu',weights_only=False)['model']
b=torch.load(ROOT/'artifacts/training/ash-real-rectified-v2/detail-defects/inference-unvalidated.pth',map_location='cpu',weights_only=False)['model']
assert all(torch.isfinite(t).all() for t in a.values() if t.is_floating_point())
changed=sum(not torch.equal(t,b[k]) for k,t in a.items());assert changed>0
del a,b;gc.collect()
base=ROOT/'datasets/ash-real-reviewed-v3';coco=json.loads((base/'train/_annotations.coco.json').read_text());results=[]
coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
with ExitStack() as stack:
 stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
 backend=RFDETRBackend(p,run['weights_sha256'],threshold=.1,training_scope='product_defects',class_names=tuple(f'NG{i:02}' for i in range(1,7)),inference_profile='standard',max_batch_size=1);backend.load()
 for im in coco['images']:
  ds=[asdict(d) for d in backend.predict_rgb(np.array(Image.open(base/'train'/im['file_name']).convert('RGB')))];results.append(dict(file_name=im['file_name'],predictions=ds))
 backend.close()
(out/'train-inference-check.json').write_text(json.dumps(results,indent=2))
report=dict(weights_sha256=run['weights_sha256'],all_weights_finite=True,changed_tensors=changed,inference_images=len(results),independent_evaluation=False,scope='Load and inference smoke check on training images; not generalization evidence')
(out/'verification.json').write_text(json.dumps(report,indent=2));print(report)

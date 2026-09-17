"""Offline SAM mask proposals. No decisions, automatic reviews or live inference."""
from contextlib import ExitStack
from pathlib import Path
import json,time
import numpy as np
from PIL import Image
from filelock import FileLock
from mes_vision.training.data import require,sha256,write_json


def mask_box(mask):
    require(mask.ndim==2 and mask.dtype==np.bool_, 'Expected a binary 2D mask')
    yy,xx=np.nonzero(mask)
    require(len(xx)>0, 'SAM mask is empty')
    return [int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1]


def validate_prompt(request,width,height):
    points=request.get('points',[]);labels=request.get('labels',[]);box=request.get('box')
    require(len(points)==len(labels) and len(points)<=64,'Invalid SAM points')
    for point,label in zip(points,labels):
        require(len(point)==2 and np.isfinite(point).all() and 0<=point[0]<width and 0<=point[1]<height,'Point outside image')
        require(type(label) is int and label in (0,1),'Invalid point label')
    if box is not None:
        require(len(box)==4 and np.isfinite(box).all() and 0<=box[0]<box[2]<=width and 0<=box[1]<box[3]<=height,'Box outside image')
    require(box is not None or 1 in labels,'Add an inclusion point or bounding box')


def propose(request,root):
    import torch
    from transformers import Sam2Model,Sam2Processor
    from mes_vision.vlm.gpu import GpuCoordinator
    root=Path(root);directory=root/'models/sam2.1-hiera-small'
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    require(manifest['repository']=='facebook/sam2.1-hiera-small','Unexpected SAM repository')
    for name,entry in manifest['files'].items():
        path=(directory/name).resolve();require(path.is_relative_to(directory.resolve()),'Invalid model path')
        require(sha256(path)==entry['sha256'],'SAM file changed: '+name)
    image_path=Path(request['image']);require(sha256(image_path)==request['image_sha256'],'Source image changed')
    with Image.open(image_path) as source:
        require(source.mode=='RGB','Use the collection-normalized RGB image')
        picture=source.copy()
    width,height=picture.size;validate_prompt(request,width,height)
    require(sha256(image_path)==request['image_sha256'],'Source image changed during reading')
    require(torch.cuda.is_available(),'SAM annotation requires the configured CUDA GPU')
    output=Path(request['output']);output.mkdir(parents=True,exist_ok=False)
    coordinator=GpuCoordinator(Path(request['runtime'])/'vlm/gpu-coordination')
    began=time.perf_counter()
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(coordinator.root/'resident.lock'),timeout=0))
        stack.enter_context(coordinator.foreground(timeout=10))
        model=Sam2Model.from_pretrained(directory,local_files_only=True).to('cuda').eval()
        processor=Sam2Processor.from_pretrained(directory,local_files_only=True)
        prompts={}
        if request.get('points'):
            prompts.update(input_points=[[request['points']]],input_labels=[[request['labels']]])
        if request.get('box') is not None:prompts['input_boxes']=[[request['box']]]
        inputs=processor(images=picture,return_tensors='pt',**prompts).to('cuda')
        with torch.inference_mode():
            prediction=model(**inputs,multimask_output=True)
        masks=processor.post_process_masks(prediction.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][0].numpy().astype(bool)
        scores=prediction.iou_scores[0,0].detach().cpu().float().numpy()
        require(masks.shape[1:]==(height,width) and np.isfinite(scores).all(),'Invalid SAM output')
        candidates=[]
        for index in np.argsort(-scores):
            mask=masks[index]
            if not mask.any():continue
            name=f'mask-{int(index)}.png';Image.fromarray(mask.astype(np.uint8)*255).save(output/name)
            candidates.append({'mask':name,'sha256':sha256(output/name),'box':mask_box(mask),
                               'model_score':float(scores[index]),'area_pixels':int(mask.sum())})
        require(candidates,'SAM returned no nonempty mask')
        write_json(output/'result.json',{'schema_version':1,'kind':'unreviewed_sam_proposal',
            'image_sha256':request['image_sha256'],'image_size':[width,height],
            'prompt':{k:request.get(k) for k in ('points','labels','box')},
            'model_repository':manifest['repository'],'model_revision':manifest['revision'],
            'model_sha256':manifest['files']['model.safetensors']['sha256'],
            'candidates':candidates,'seconds':time.perf_counter()-began})
    return output/'result.json'

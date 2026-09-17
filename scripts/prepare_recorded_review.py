"""Recorded clips -> conservative candidates -> SAM/DETR drafts; never train."""
from pathlib import Path
import os,sys,json,argparse,math
from collections import deque,Counter
from dataclasses import asdict
from contextlib import ExitStack
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key,folder in {'HF_HOME':'huggingface','TORCH_HOME':'torch'}.items():os.environ[key]=str(ROOT/'.cache'/folder)
os.environ['HF_HUB_OFFLINE']='1'
import cv2,numpy as np
from PIL import Image,ImageDraw
from mes_vision.training.data import sha256
OUT=ROOT/'artifacts/recorded-label-review-20260914-v3'
CAL=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json'

def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def candidate(rgb,previous):
    a=cv2.GaussianBlur(rgb,(5,5),0).astype(np.float32);r,g,b=a.transpose(2,0,1)
    white=((b>=g*.85)&(r>=g*.9)&(a.mean(2)>105)).astype('uint8')
    white=cv2.morphologyEx(white,cv2.MORPH_OPEN,np.ones((5,5),np.uint8))
    white=cv2.morphologyEx(white,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    _,_,stats,_=cv2.connectedComponentsWithStats(white)
    objects=[]
    for x,y,w,h,area in stats[1:]:
        if 3500<area<130000 and 100<w<550 and 100<h<550 and .55<w/h<1.8 and area/(w*h)>.35 and x>35 and y>10 and x+w<1245 and y+h<710:
            objects.append((int(area),[int(x),int(y),int(x+w),int(y+h)]))
    if not objects:return None,'no_isolated_object'
    objects.sort(reverse=True)
    if len(objects)>1 and objects[1][0]>.45*objects[0][0]:return None,'ambiguous_object'
    box=objects[0][1];x,y,x2,y2=box
    crop=a[max(0,y-15):min(720,y2+15),max(0,x-15):min(1280,x2+15)]
    cr,cg,cb=crop.transpose(2,0,1)
    skin=(cr>cg*1.14)&(cr>cb*1.24)&(cr-cg>15)&(cr>65)
    skin_fraction=float(skin.mean())
    if skin_fraction>.015:return None,'hand_near_object'
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    sharpness=float(cv2.Laplacian(cv2.resize(gray[y:y2,x:x2],(160,160)),cv2.CV_64F).var())
    if sharpness<40:return None,'low_sharpness_candidate'
    small=cv2.GaussianBlur(cv2.resize(gray,(320,180)),(5,5),0);motion=0.
    if previous is not None:
        diff=cv2.absdiff(small,previous)[y//4:max(y//4+1,y2//4),x//4:max(x//4+1,x2//4)]
        motion=float(diff.mean())
        # A short exposure can freeze a moving part sharply. Motion is a
        # ranking penalty, not proof that this individual frame is blurred.
        if motion>25:return None,'large_recent_motion'
    thumb=cv2.resize(gray[y:y2,x:x2],(32,32)).astype('float32');dct=cv2.dct(thumb)[:8,:8]
    bits=(dct>np.median(dct)).reshape(-1)
    return dict(raw_box=box,sharpness=sharpness,skin_fraction=skin_fraction,motion=motion,phash=''.join('1' if v else '0' for v in bits)),None

def extract():
    import av
    if (OUT/'selection.json').exists():raise RuntimeError('Selection already exists; use propose or a new version.')
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ('candidates','raw','masks','crops','overlays'):(OUT/name).mkdir(exist_ok=True)
    selected=[];sources=[]
    for manifest in sorted((ROOT/'collections/recordings').glob('*/recording.json')):
        meta=json.loads(manifest.read_text(encoding='utf-8'));video=manifest.parent/meta['video']
        source={'folder':manifest.parent.name,'metadata':meta,'manifest_sha256':sha256(manifest)}
        if not video.exists() or meta['status']!='COMPLETED':
            source['excluded']='missing_or_incomplete_video';sources.append(source);continue
        assert meta['purpose']=='train' and meta['capture_profile']=='detail'
        assert meta['camera_settings']['width']==1280 and meta['camera_settings']['height']==720
        source['video']=str(video);source['video_sha256']=sha256(video)
        pool=[];reasons=Counter();past=deque(maxlen=4)
        with av.open(str(video)) as container:
            for index,frame in enumerate(container.decode(video=0)):
                rgb=frame.to_ndarray(format='rgb24');gray=cv2.GaussianBlur(cv2.resize(cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY),(320,180)),(5,5),0)
                if index%15==0 and index>10 and index<meta['frames']-10:
                    row,reason=candidate(rgb,past[0] if len(past)==4 else None)
                    if row:
                        identity=f"{meta['label']}_{manifest.parent.name.split('_')[1]}_f{index:04}"
                        path=OUT/'candidates'/(identity+'.png');Image.fromarray(rgb).save(path)
                        row.update(id=identity,label=meta['label'],source_folder=manifest.parent.name,frame_index=index,
                            time_seconds=float(frame.pts*frame.time_base),candidate_image=str(path),raw_sha256=sha256(path))
                        pool.append(row)
                    else:reasons[reason]+=1
                past.append(gray)
        source['decoded_frames']=index+1;assert index+1==meta['frames']
        # Pick sharp, temporally separated, visually distinct poses; no adjacent-frame train/test split.
        picks=[]
        for row in sorted(pool,key=lambda r:r['sharpness']/(1+.15*r['motion']),reverse=True):
            if any(abs(row['time_seconds']-p['time_seconds'])<1 for p in picks):continue
            if any(sum(a!=b for a,b in zip(row['phash'],p['phash']))<5 for p in picks):continue
            picks.append(row)
            if len(picks)==16:break
        picks.sort(key=lambda r:r['time_seconds'])
        for row in picks:
            path=OUT/'raw'/(row['id']+'.png');path.write_bytes(Path(row['candidate_image']).read_bytes());row['raw_image']=str(path)
        source.update(candidate_count=len(pool),selected_count=len(picks),rejected=dict(reasons))
        selected.extend(picks);sources.append(source);print(meta['label'],len(pool),'candidates ->',len(picks),'selected',dict(reasons),flush=True)
    write(OUT/'selection.json',dict(status='CANDIDATES_NOT_LABELLED',independent_evaluation=False,
        source_script_sha256=sha256(Path(__file__)),sources=sources,items=selected,
        filtering='heuristic neutral-object isolation, skin proximity, short-term motion, normalized sharpness, temporal and pHash dedup; visual review still required'))
    sheets(selected,'selection',raw=True)

def sheets(items,prefix,raw=False):
    for label in ('OK',*[f'NG{i:02}' for i in range(1,7)]):
        rows=[r for r in items if r['label']==label];sheet=Image.new('RGB',(1400,300*max(1,math.ceil(len(rows)/4))),'#edf0f5');d=ImageDraw.Draw(sheet)
        for i,row in enumerate(rows):
            path=row.get('overlay') or row.get('crop') or row['raw_image'];im=Image.open(path).convert('RGB')
            if raw:im=Image.open(row['raw_image']).crop(tuple(row['raw_box']))
            im.thumbnail((340,260));x=i%4*350;y=i//4*300;sheet.paste(im,(x,y))
            d.text((x+4,y+262),row['id']+f" {row['time_seconds']:.1f}s",fill='black')
            if 'review_status' in row:d.text((x+4,y+278),row['review_status'],fill='black')
        sheet.save(OUT/(prefix+'-'+label+'.jpg'),quality=92)

def propose():
    import torch
    from transformers import Sam2Model,Sam2Processor
    from mes_vision.data_management.sam_assist import mask_box
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    from mes_vision.vlm.gpu import GpuCoordinator
    from filelock import FileLock
    from compare_undistortion import corrected_crop
    plan=json.loads((OUT/'selection.json').read_text(encoding='utf-8'));items=plan['items']
    if (OUT/'drafts.json').exists():items=json.loads((OUT/'drafts.json').read_text(encoding='utf-8'))['items']
    cal=json.loads(CAL.read_text());K=np.array(cal['K']);D=np.array(cal['D'])
    assert sha256(CAL)=='f695d2dcb95ece6f879d3de0a3cfceb02792e9c7d1be52564dba83234820d79b'
    directory=ROOT/'models/sam2.1-hiera-small';manifest=json.loads((directory/'manifest.json').read_text())
    for name,entry in manifest['files'].items():assert sha256(directory/name)==entry['sha256']
    coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination')
    with ExitStack() as stack:
        stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
        sam=Sam2Model.from_pretrained(directory,local_files_only=True).to('cuda').eval()
        proc=Sam2Processor.from_pretrained(directory,local_files_only=True)
        for i,row in enumerate(items):
            assert sha256(Path(row['raw_image']))==row['raw_sha256']
            if 'crop' in row:
                assert sha256(Path(row['crop']))==row['crop_sha256'];continue
            im=Image.open(row['raw_image']).convert('RGB');box=row['raw_box']
            expanded=[max(0,box[0]-45),max(0,box[1]-45),min(im.width,box[2]+45),min(im.height,box[3]+45)]
            inputs=proc(images=im,input_boxes=[[expanded]],return_tensors='pt').to('cuda')
            with torch.inference_mode():pred=sam(**inputs,multimask_output=True)
            masks=proc.post_process_masks(pred.pred_masks.cpu(),inputs['original_sizes'].cpu())[0][0].numpy().astype(bool)
            scores=pred.iou_scores[0,0].detach().cpu().numpy();eligible=[]
            for j,mask in enumerate(masks):
                if not mask.any():continue
                b=mask_box(mask);x,y,x2,y2=b
                if x>=expanded[0]-5 and y>=expanded[1]-5 and x2<=expanded[2]+5 and y2<=expanded[3]+5 and (x2-x)*(y2-y)>.7*(box[2]-box[0])*(box[3]-box[1]):eligible.append(j)
            if not eligible:row['review_status']='SAM_OBJECT_REVIEW';continue
            index=max(eligible,key=lambda j:scores[j]);mask=masks[index];b=mask_box(mask)
            maskpath=OUT/'masks'/(row['id']+'.png');Image.fromarray(mask.astype('uint8')*255).save(maskpath)
            padded=[max(0,b[0]-12),max(0,b[1]-12),min(im.width,b[2]+12),min(im.height,b[3]+12)]
            crop,info=corrected_crop(np.array(im),padded,K,D)
            dest=OUT/'crops'/(row['id']+'.png');Image.fromarray(crop).save(dest)
            row.update(mask=str(maskpath),mask_sha256=sha256(maskpath),sam_box=b,sam_score=float(scores[index]),
                crop=str(dest),crop_sha256=sha256(dest),crop_raw_bounds=padded,rectification=info)
            if (i+1)%10==0:print('SAM',i+1,'/',len(items),flush=True)
        del sam,proc;torch.cuda.empty_cache()
        weights=ROOT/'artifacts/training/ratio-ablation-v1/real/inference-unvalidated.pth'
        digest='cab34ffefbb4d6befc68412be12f21a729ab8b10ebc02c58a4175053afb22323'
        backend=RFDETRBackend(weights,digest,threshold=.1,training_scope='product_defects',class_names=tuple(f'NG{i:02}' for i in range(1,7)),inference_profile='standard',max_batch_size=1)
        backend.load()
        try:
            for row in items:
                if 'crop' not in row:continue
                rgb=np.array(Image.open(row['crop']).convert('RGB'));ds=[asdict(v) for v in backend.predict_rgb(rgb)];row['predictions']=ds
                expected=[v for v in ds if v['label']==row['label'] and v['score']>=.25]
                other=[v for v in ds if v['label']!=row['label'] and v['score']>=.25]
                row['review_status']='OK_DRAFT' if row['label']=='OK' and not other else ('DEFECT_DRAFT' if expected and not other else 'NEEDS_REVIEW')
                row['draft_defects']=sorted(expected,key=lambda v:-v['score'])[:1] if row['label']!='OK' else []
                row['annotation_reviewed']=False
                im=Image.fromarray(rgb);draw=ImageDraw.Draw(im)
                for pred in ds:
                    if pred['score']<.25:continue
                    b=pred['box'];xy=[b[k] for k in ('x1','y1','x2','y2')]
                    draw.rectangle(xy,outline='red' if pred['label']!=row['label'] else '#00b36b',width=3)
                    draw.text((xy[0],max(0,xy[1]-12)),pred['label']+f" {pred['score']:.2f}",fill='red')
                path=OUT/'overlays'/(row['id']+'.jpg');im.save(path,quality=95);row['overlay']=str(path)
                print(row['id'],row['review_status'],flush=True)
        finally:backend.close()
    write(OUT/'drafts.json',dict(status='UNREVIEWED_LABEL_DRAFTS',training_performed=False,independent_evaluation=False,
        calibration_sha256=sha256(CAL),calibration_status='existing_exploratory_calibration_not_robot_validation',model_sha256=digest,
        sam_model_sha256=manifest['files']['model.safetensors']['sha256'],items=items))
    sheets(items,'drafts')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['extract','propose']);args=p.parse_args()
    extract() if args.stage=='extract' else propose()

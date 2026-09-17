"""Prepare experimental real/CAD box dataset; never learn from hidden NG as OK."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'datasets/ash-real-cad-mixed-v1'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def prepare():
    real=ROOT/'datasets/ash-real-reviewed-v3';cad=ROOT/'datasets/ash-cad-today-parallel-v1'
    provenance=json.loads((real/'provenance.json').read_text(encoding='utf-8'))
    source=json.loads((real/'train/_annotations.coco.json').read_text());assert json.loads((cad/'completed.json').read_text())['images']==1008
    if OUT.exists():
        meta=json.loads((OUT/'provenance.json').read_text(encoding='utf-8'))
        assert meta['preparer_sha256']==sha(Path(__file__))
        assert meta['source_coco_sha256']==sha(real/'train/_annotations.coco.json')
        coco=json.loads((OUT/'train/_annotations.coco.json').read_text())
        assert len(coco['images'])==len(meta['items'])
        for im,item in zip(coco['images'],meta['items']):assert sha(OUT/'train'/im['file_name'])==item['image_sha256']
        return OUT
    OUT.mkdir();(OUT/'train').mkdir();(OUT/'valid').mkdir()
    images=[];anns=[];items=[];excluded=[];receipts=sorted(cad.glob('worker-*/receipts/**/*.json'));assert len(receipts)==1008
    def add(im,name,targets,info):
        dest=OUT/'train'/name;im.save(dest);iid=len(images)+1
        images.append(dict(id=iid,file_name=name,width=im.width,height=im.height))
        for code,b in targets:
            x,y,x2,y2=b;assert 0<=x<x2<=im.width and 0<=y<y2<=im.height
            anns.append(dict(id=len(anns)+1,image_id=iid,category_id=int(code[2:]),bbox=[x,y,x2-x,y2-y],area=(x2-x)*(y2-y),iscrowd=0))
        items.append(dict(info,image_sha256=sha(dest)))
    for im,item in zip(source['images'],provenance['items']):
        p=real/'train'/im['file_name'];assert sha(p)==item['image_sha256']
        targets=[]
        for a in source['annotations']:
            if a['image_id']==im['id']:
                x,y,w,h=a['bbox'];targets.append((f"NG{a['category_id']:02}",[x,y,x+w,y+h]))
        add(Image.open(p).convert('RGB'),'real--'+im['file_name'],targets,dict(domain='real',sampling_weight=1,source=str(p),source_sha256=sha(p)))
    for rp in receipts:
        r=json.loads(rp.read_text());worker=rp.parents[2];rgb=worker/r['relative_rgb'];assert sha(rgb)==r['rgb_sha256']
        maskroot=worker/'masks'/rp.parent.name
        objpath=maskroot/(r['id']+'-object.png');diffpath=maskroot/(r['id']+'-difference.png')
        assert sha(objpath)==r['object_mask_sha256'] and sha(diffpath)==r['difference_mask_sha256']
        obj=np.array(Image.open(objpath).convert('L'))>127;diff=np.array(Image.open(diffpath).convert('L'))>127
        assert diff.shape==obj.shape==(720,1280) and int(diff.sum())==r['difference_pixels']
        if r['code']!='OK' and diff.sum()<16:
            excluded.append(dict(id=r['id'],reason='no_visible_geometry' if not diff.any() else 'tiny_geometry_under_16_pixels',pixels=int(diff.sum())));continue
        if r['code']=='OK':assert not diff.any()
        # Include expected missing feature location in crop, not just actual object pixels.
        yy,xx=np.where(obj|diff);assert len(xx)
        if obj[0].any() or obj[-1].any() or obj[:,0].any() or obj[:,-1].any():
            excluded.append(dict(id=r['id'],reason='object_touches_frame_edge'));continue
        x0=max(0,int(xx.min())-16);y0=max(0,int(yy.min())-16);x1=min(1280,int(xx.max())+17);y1=min(720,int(yy.max())+17)
        targets=[]
        if r['code']!='OK':
            yy,xx=np.where(diff);targets=[(r['code'],[int(xx.min())-x0,int(yy.min())-y0,int(xx.max())+1-x0,int(yy.max())+1-y0])]
        im=Image.open(rgb).convert('RGB').crop((x0,y0,x1,y1))
        add(im,'cad--'+r['id']+'.png',targets,dict(domain='synthetic',sampling_weight=r.get('proposed_training_weight',1),source=str(rgb),source_sha256=r['rgb_sha256'],receipt=str(rp),crop=[x0,y0,x1,y1],label_status='EXPERIMENTAL_GEOMETRY_BOX_NOT_HUMAN_APPROVED'))
    assert len(images)>55
    for split,ii,aa in [('train',images,anns),('valid',[],[])]:
        (OUT/split/'_annotations.coco.json').write_text(json.dumps(dict(images=ii,annotations=aa,categories=source['categories']),indent=2))
    meta=dict(scope='train_only_unvalidated',preparer_sha256=sha(Path(__file__)),source_coco_sha256=sha(real/'train/_annotations.coco.json'),calibration_sha256=provenance['calibration_sha256'],images=len(images),real_images=55,synthetic_images=len(images)-55,negative_images=sum(not any(a['image_id']==im['id'] for a in anns) for im in images),items=items,excluded=excluded,note='Real already rectified; synthetic ideal pinhole is not rectified again. Geometry labels remain experimental. Brightness visibility is not guaranteed by a nonempty geometry mask.')
    (OUT/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    # Four deterministic examples per class spread across views for review.
    groups={}
    for im in images[55:]:
        code=next((f"NG{a['category_id']:02}" for a in anns if a['image_id']==im['id']),'OK');groups.setdefault(code,[]).append(im)
    selected=[]
    for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
        group=groups[code];selected += [group[int(i)] for i in np.linspace(0,len(group)-1,4)]
    sheet=Image.new('RGB',(1400,1200),'white');d=ImageDraw.Draw(sheet)
    for n,im in enumerate(selected):
        pic=Image.open(OUT/'train'/im['file_name']);dr=ImageDraw.Draw(pic)
        for a in anns:
            if a['image_id']==im['id']:
                x,y,w,h=a['bbox'];dr.rectangle((x,y,x+w,y+h),outline='red',width=2)
        pic.thumbnail((195,255));x=n//4*200;y=n%4*300;sheet.paste(pic,(x,y));d.text((x,y+260),im['file_name'][:30],fill='black')
    sheet.save(OUT/'label-review.jpg')
    print('DATASET:',len(images),'images;',len(anns),'boxes;',len(excluded),'excluded',flush=True)
    return OUT
if __name__=='__main__':prepare()

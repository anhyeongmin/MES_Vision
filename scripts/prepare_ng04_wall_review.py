"""Versioned manual box review; image pixels and unrelated labels stay intact."""
from pathlib import Path
import json, hashlib, shutil, html
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'datasets/ash-real-recorded-merged-v1'
DEST = ROOT/'datasets/ash-real-wall-reviewed-v2'
REPORT = ROOT/'artifacts/ng04-wall-review-v1'
# Individually inspected native image boxes, expressed as normalized XYXY.
# One NG04 region spans the affected wall rim/inner edge and its junction.
BOXES = {
    18:(.025,.06,.34,.95), 19:(.025,.055,.34,.95),
    20:(.025,.05,.34,.955), 21:(.025,.06,.34,.95),
    22:(.025,.06,.345,.95), 44:(.065,.775,.91,.965),
    54:(.065,.11,.325,.955), 55:(.20,.07,.89,.35),
    97:(.605,.04,.925,.91), 98:(.595,.035,.93,.915),
    99:(.64,.135,.925,.94), 100:(.565,.035,.855,.87),
    137:(.64,.12,.91,.94), 138:(.575,.025,.86,.88),
    139:(.085,.06,.895,.36), 140:(.08,.065,.915,.36),
}
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))
def write(p,v): p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')

def prepare():
    coco=read(SOURCE/'train/_annotations.coco.json')
    meta=read(SOURCE/'provenance.json')
    if DEST.exists():
        saved=read(DEST/'provenance.json')
        assert saved['source_coco_sha256']==digest(SOURCE/'train/_annotations.coco.json')
        assert saved['review_script_sha256']==digest(Path(__file__))
        assert saved['coco_sha256']==digest(DEST/'train/_annotations.coco.json')
        for item in saved['items']:
            assert digest(DEST/'train'/item['file_name'])==item['image_sha256']
        return DEST
    assert {a['image_id'] for a in coco['annotations'] if a['category_id']==4}==set(BOXES)
    assert len(coco['images'])==141 and len(coco['annotations'])==118
    changes=[]; pages=[]
    REPORT.mkdir(exist_ok=True,parents=True)
    for im in coco['images']:
        iid=im['id']; original=SOURCE/'train'/im['file_name']
        item=next(r for r in meta['items'] if r['file_name']==im['file_name'])
        assert digest(original)==item['image_sha256']
        if iid not in BOXES: continue
        ann=next(a for a in coco['annotations'] if a['image_id']==iid and a['category_id']==4)
        old=list(ann['bbox']); x1,y1,x2,y2=BOXES[iid]
        x1,y1,x2,y2=round(x1*im['width']),round(y1*im['height']),round(x2*im['width']),round(y2*im['height'])
        assert 0<=x1<x2<=im['width'] and 0<=y1<y2<=im['height']
        ann.update(bbox=[x1,y1,x2-x1,y2-y1],area=(x2-x1)*(y2-y1))
        # This dataset uses boxes, not segmentation masks.
        assert not ann.get('segmentation')
        change=dict(image_id=iid,file_name=im['file_name'],old_bbox=old,new_bbox=ann['bbox'],
                    review_status='assistant_visual_review_not_user_confirmed',
                    rationale='Affected wall rim/inner edge and cylinder junction; no metric thickness inferred')
        changes.append(change); item['wall_review']=change
        pic=Image.open(original).convert('RGB'); dr=ImageDraw.Draw(pic)
        ox,oy,ow,oh=old; dr.rectangle((ox,oy,ox+ow,oy+oh),outline='#ff5656',width=2)
        dr.rectangle((x1,y1,x2,y2),outline='#00ff80',width=2)
        name=f'wall-{iid}.png';pic.save(REPORT/name)
        pages.append(f'<figure><img src="{name}"><figcaption>ID {iid}: {html.escape(im["file_name"])}</figcaption></figure>')
    negatives=[im['id'] for im in coco['images'] if not any(a['image_id']==im['id'] for a in coco['annotations'])]
    assert len(negatives)==23
    staging=DEST.with_name(DEST.name+'.preparing');staging.mkdir(exist_ok=False)
    for split in ('train','valid'): (staging/split).mkdir()
    for im in coco['images']: shutil.copy2(SOURCE/'train'/im['file_name'],staging/'train'/im['file_name'])
    write(staging/'train/_annotations.coco.json',coco)
    shutil.copy2(SOURCE/'valid/_annotations.coco.json',staging/'valid/_annotations.coco.json')
    meta.update(source_dataset=str(SOURCE),source_coco_sha256=digest(SOURCE/'train/_annotations.coco.json'),
                review_script_sha256=digest(Path(__file__)),coco_sha256=digest(staging/'train/_annotations.coco.json'),
                annotation_status='16 NG04 wall regions visually revised by assistant; other labels inherited; not user-confirmed pixel truth',
                wall_review=changes,normal_review={'count':23,'action':'retained_empty_annotations','thickness_measurement':False},
                independent_evaluation=False,production_ready=False)
    write(staging/'provenance.json',meta);staging.rename(DEST)
    write(REPORT/'changes.json',changes)
    (REPORT/'report.html').write_text('<meta charset="utf-8"><style>body{background:#111;color:#eee;font-family:sans-serif}main{display:flex;flex-wrap:wrap}figure{width:360px}img{max-width:360px;max-height:420px}figcaption{overflow-wrap:anywhere}</style><h1>NG04 벽 라벨 검토</h1><p>빨강: 기존 / 초록: 수정. NG04 16장 수정, OK 23장 무불량 라벨 유지. 기존 사진이며 독립 평가 아님. 픽셀 라벨은 AI 검토안이며 실제 두께 측정값이 아님.</p><main>'+''.join(pages)+'</main>',encoding='utf-8')
    return DEST

if __name__=='__main__': print(prepare())

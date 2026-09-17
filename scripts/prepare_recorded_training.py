from pathlib import Path
import json,hashlib,shutil
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
SOURCES=[ROOT/'datasets/ash-real-reviewed-v3',ROOT/'datasets/ash-recorded-low-exposure-reviewed-v2']
DEST=ROOT/'datasets/ash-real-recorded-merged-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def prepare():
 images=[];anns=[];items=[];seen=set();calibration=None;fingerprints=[]
 for source in SOURCES:
  meta=read(source/'provenance.json');coco=read(source/'train/_annotations.coco.json')
  fingerprints.append({'dataset':str(source),'provenance_sha256':sha(source/'provenance.json'),'coco_sha256':sha(source/'train/_annotations.coco.json')})
  if calibration is None:calibration=meta['calibration_sha256']
  assert calibration==meta['calibration_sha256']
  assert [(c['id'],c['name']) for c in coco['categories']]==[(i,f'NG{i:02}') for i in range(1,7)]
  assert not read(source/'valid/_annotations.coco.json')['images']
  by_name={r['id']:r for r in meta['items']}
  for im in coco['images']:
   p=source/'train'/im['file_name'];matches=[r for key,r in by_name.items() if Path(im['file_name']).stem==key or Path(im['file_name']).stem.endswith('--'+key)];assert len(matches)==1,im['file_name'];original=matches[0];digest=sha(p)
   assert digest==original.get('image_sha256',original.get('crop_sha256'))
   with Image.open(p) as pic: assert pic.size==(im['width'],im['height']);pic.verify()
   assert digest not in seen,'Duplicate source image: resolve labels before merging';seen.add(digest)
   iid=len(images)+1;name=f'{len(fingerprints)}_'+im['file_name'];images.append({**im,'id':iid,'file_name':name})
   items.append({'id':Path(name).stem,'file_name':name,'image_sha256':digest,'source_image':str(p),'source_dataset':str(source),'source_provenance':original})
   for a in coco['annotations']:
    if a['image_id']==im['id']:
     x,y,w,h=a['bbox'];assert 0<=x and 0<=y and w>0 and h>0 and x+w<=im['width'] and y+h<=im['height']
     anns.append({**a,'id':len(anns)+1,'image_id':iid})
 assert len(images)==141 and len(anns)==118
 manifest={'scope':'train_only_unvalidated','images':141,'annotations':118,'negative_images':23,'calibration_sha256':calibration,'sources':fingerprints,'sampling':'each image once per epoch with online augmentation','annotation_status':'prior real labels plus 76 assistant drafts and 10 user-circled coarse regions','independent_evaluation':False,'production_ready':False,'items':items}
 if DEST.exists():
  assert read(DEST/'provenance.json')==manifest,'Existing dataset differs; refusing overwrite'
  assert read(DEST/'train/_annotations.coco.json')==dict(images=images,annotations=anns,categories=coco['categories'])
  assert not read(DEST/'valid/_annotations.coco.json')['images']
  for r in items:assert sha(DEST/'train'/r['file_name'])==r['image_sha256']
  return DEST
 staging=DEST.with_name(DEST.name+'.preparing');staging.mkdir(exist_ok=False)
 for split in ['train','valid']:(staging/split).mkdir()
 for r in items:shutil.copyfile(r['source_image'],staging/'train'/r['file_name'])
 for split,ii,aa in [('train',images,anns),('valid',[],[])]:
  (staging/split/'_annotations.coco.json').write_text(json.dumps(dict(images=ii,annotations=aa,categories=coco['categories']),ensure_ascii=False,indent=2),encoding='utf-8')
 (staging/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
 staging.rename(DEST)
 return DEST
if __name__=='__main__':print(prepare())

"""Import ten user-circled, already rectified crops without changing prior datasets."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'artifacts/recorded-label-review-20260914-v3'
OUT=BASE/'returned-review'
DST=ROOT/'datasets/ash-recorded-low-exposure-reviewed-v2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 assert not DST.exists(),'Do not overwrite an existing dataset'
 mapping=json.loads((BASE/'user-review/review-map.json').read_text(encoding='utf-8'))
 visual={r['id']:r for r in json.loads((BASE/'visual-review.json').read_text(encoding='utf-8'))['items']}
 additions=[];sheet=Image.new('RGB',(1500,1000),'#dddddd');draw=ImageDraw.Draw(sheet)
 for i,r in enumerate(mapping):
  clean=Path(r['source_crop']);marked=OUT/r['file']
  assert sha(clean)==r['clean_sha256']
  im=Image.open(clean).convert('RGB');a=np.array(im).astype(np.int16);b=np.array(Image.open(marked).convert('RGB')).astype(np.int16)
  assert a.shape==b.shape and im.size==(r['width'],r['height'])
  mask=(b[:,:,0]>150)&(b[:,:,0]>b[:,:,1]+65)&(b[:,:,0]>b[:,:,2]+65)&(np.max(abs(b-a),axis=2)>50)
  yy,xx=np.where(mask);assert len(xx)>30,r['id']
  x1,y1,x2,y2=int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)
  box=[x1,y1,x2-x1,y2-y1]
  row=dict(visual[r['id']]);row.update(annotation_reviewed=True,annotation_status='user_circled_coarse_box',preparation_status='USER_REGION_IMPORTED',user_marked_sha256=sha(marked),user_marked_file=str(marked),bbox=box,box_method='bounding extent of changed red ink; visually inspected; coarse region, not segmentation',selected_defects=[{'label':r['expected_code'],'box':dict(x1=x1,y1=y1,x2=x2,y2=y2),'source':'user_circle'}])
  additions.append(row)
  ImageDraw.Draw(im).rectangle((x1,y1,x2-1,y2-1),outline='#00ff00',width=2)
  im.save(OUT/(r['id']+'-box.png'));im.thumbnail((290,450));x=i%5*300;y=i//5*500;sheet.paste(im,(x,y+30));draw.text((x+5,y+5),r['id'],fill='black')
 src=ROOT/'datasets/ash-recorded-low-exposure-draft-v1';shutil.copytree(src,DST)
 coco=json.loads((DST/'train/_annotations.coco.json').read_text(encoding='utf-8'))
 provenance=json.loads((DST/'provenance.json').read_text(encoding='utf-8'))
 for row in additions:
  name=row['id']+'.png';p=Path(row['crop']);shutil.copyfile(p,DST/'train'/name);w,h=Image.open(p).size
  iid=len(coco['images'])+1;coco['images'].append(dict(id=iid,file_name=name,width=w,height=h))
  b=row['bbox'];coco['annotations'].append(dict(id=len(coco['annotations'])+1,image_id=iid,category_id=int(row['label'][2:]),bbox=b,area=b[2]*b[3],iscrowd=0));provenance['items'].append(row)
 assert len(coco['images'])==86 and len(coco['annotations'])==72
 ids={r['id']:r for r in coco['images']};assert len(ids)==86
 for an in coco['annotations']:
  im=ids[an['image_id']];x,y,w,h=an['bbox'];assert w>0 and h>0 and x>=0 and y>=0 and x+w<=im['width'] and y+h<=im['height']
 for row in provenance['items']:assert sha(DST/'train'/(row['id']+'.png'))==row['crop_sha256']
 provenance.update(status='PREPARED_MIXED_REVIEW_LABELS',annotation_status='76_assistant_drafts_plus_10_user_circled_regions',train_images=86,positive_images=72,negative_images=14,pending_user_regions=0,user_circled_images=10,assistant_draft_images=76,training_performed=False,production_ready=False,validation_images=0,source_review_zip_sha256=sha(BASE/'표시필요_10장/01_여기에_표시.zip'))
 for p,data in [(DST/'train/_annotations.coco.json',coco),(DST/'provenance.json',provenance),(OUT/'import.json',dict(dataset=str(DST),items=additions,train_images=86,annotations=72,validation_images=0))]:p.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 (DST/'README.md').write_text('# New recording labels\n\n86 clean rectified crops: 14 OK, 72 NG boxes. 76 assistant-reviewed drafts and 10 user-circled coarse regions. 8 visibility exclusions retained outside training. No independent validation; no training performed. Original source images and earlier datasets retained. Circle extents are coarse bounding boxes, not segmentation masks. No second undistortion applied.\n',encoding='utf-8')
 sheet.save(OUT/'imported-boxes.jpg')
 print(json.dumps(dict(dataset=str(DST),images=86,boxes=72,user_regions=10,checks='clean hashes, dimensions, bounding boxes, unique IDs passed')))
if __name__=='__main__':main()

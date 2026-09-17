"""Versioned rectified training crops and transformed annotation boxes."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import cv2,numpy as np
from PIL import Image,ImageDraw
from compare_undistortion import corrected_crop
from mes_vision.data_management.collection import Collection,validate_record
from mes_vision.training.data import sha256

def map_box(box,K,D,origin):
 x,y,x2,y2=box;t=np.linspace(0,1,257)
 edges=np.concatenate([np.c_[x+(x2-x)*t,np.full_like(t,y)],np.c_[x+(x2-x)*t,np.full_like(t,y2)],np.c_[np.full_like(t,x),y+(y2-y)*t],np.c_[np.full_like(t,x2),y+(y2-y)*t]])
 p=cv2.undistortPoints(edges.reshape(-1,1,2),K,D,P=K,criteria=(3,200,1e-10)).reshape(-1,2)-origin
 return [*np.floor(p.min(0)).astype(int).tolist(),*np.ceil(p.max(0)).astype(int).tolist()]

def main():
 out=ROOT/'datasets/ash-real-rectified-bootstrap-v2';out.mkdir(exist_ok=False)
 for s in ['train','valid']:(out/s).mkdir()
 calpath=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json';cal=json.loads(calpath.read_text());K=np.array(cal['K']);D=np.array(cal['D'])
 c=Collection(ROOT/'collections/ash-real-detail-videos-20260914');items=[]
 for r in c.data['records']:
  if r['excluded'] or not r['objects']:continue
  validate_record(r,complete=True);assert sha256(c.image_path(r))==r['image_sha256']
  o=r['objects'][0];items.append({'id':r['id'],'image':str(c.image_path(r)),'raw_box':o['bbox'],'defects':o['defects'],'source_record':r,'origin':'initial_training_collection'})
 evalroot=ROOT/'artifacts/real-eval-20260914';jobs=json.loads((evalroot/'frozen-selection.json').read_text())
 for j in jobs:
  if j['expected_code']!='OK':continue
  assert sha256(Path(j['crop']))==j['crop_sha256']
  items.append({'id':j['name'],'image':str(evalroot/'frames'/(j['name']+'.png')),'raw_box':j['box'],'defects':[],'source_record':j,'origin':'previous_eval_normal_now_training'})
 assert len(items)==35
 images=[];annotations=[];provenance=[];cats=[{'id':i,'name':f'NG{i:02}','supercategory':'defect'} for i in range(1,7)]
 sheet=Image.new('RGB',(1400,240*5),'white');dr=ImageDraw.Draw(sheet)
 for i,item in enumerate(items,1):
  rgb=np.array(Image.open(item['image']).convert('RGB'));crop,info=corrected_crop(rgb,item['raw_box'],K,D);name=item['id']+'.png';Image.fromarray(crop).save(out/'train'/name);h,w=crop.shape[:2]
  images.append({'id':i,'file_name':name,'width':w,'height':h});boxes=[]
  for de in item['defects']:
   b=map_box(de['bbox'],K,D,np.array(info['rectified_box'][:2]));assert 0<=b[0]<b[2]<=w and 0<=b[1]<b[3]<=h
   annotations.append({'id':len(annotations)+1,'image_id':i,'category_id':int(de['code'][2:]),'bbox':[b[0],b[1],b[2]-b[0],b[3]-b[1]],'area':(b[2]-b[0])*(b[3]-b[1]),'iscrowd':0});boxes.append(b)
  provenance.append(dict(item,**info,rectified_defect_boxes=boxes,image_sha256=sha256(out/'train'/name)))
  preview=Image.fromarray(crop);d=ImageDraw.Draw(preview)
  for b in boxes:d.rectangle(b,outline='red',width=2)
  preview.thumbnail((195,210));x=(i-1)%7*200;y=(i-1)//7*240;sheet.paste(preview,(x,y));dr.text((x+2,y+214),f'{i}: '+(item['defects'][0]['code'] if item['defects'] else 'OK'),fill='black')
 for s,ims,anns in [('train',images,annotations),('valid',[],[])]:
  (out/s/'_annotations.coco.json').write_text(json.dumps({'images':ims,'annotations':anns,'categories':cats}),encoding='utf-8')
 meta={'scope':'train_only_unvalidated','annotation_status':'visually inspected drafts','calibration_sha256':sha256(calpath),'calibration':cal,'images':35,'negative_images':6,'positive_images':29,'preprocessing':'full-frame undistortion then transformed-ROI crop','independent_evaluation':False,'items':provenance}
 (out/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');sheet.save(out/'label-review.jpg')
 (evalroot/'training-reuse-v2.json').write_text(json.dumps({'status':'PARTIALLY_REUSED_FOR_TRAINING','source_video':'OK_1.mp4','image_ids':[j['name'] for j in jobs if j['expected_code']=='OK'],'dataset':str(out),'note':'Original reports are historical. Normal frames/video no longer independent evaluation for v2. Same-specimen and previously-inspected NG footage remains development comparison, not final acceptance.'},indent=2))
 print(out)

if __name__=='__main__':main()

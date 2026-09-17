from prepare_rectified_real import *
import hashlib

def main():
 base=ROOT/'artifacts/error-review-20260914';returned=base/'returned-review'
 manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
 out=ROOT/'datasets/ash-user-reviewed-corrections-v1';out.mkdir(exist_ok=False);(out/'train').mkdir()
 calpath=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json';cal=json.loads(calpath.read_text());K=np.array(cal['K']);D=np.array(cal['D'])
 images=[];annotations=[];records=[];sheet=Image.new('RGB',(1200,1200),'white');sd=ImageDraw.Draw(sheet)
 for i,j in enumerate(manifest['items']):
  ident=j['id'];full=base/'03_전체화면_참고'/(ident+'.png');assert sha256(full)==j['source_frame_sha256']
  rgb=np.array(Image.open(full).convert('RGB'));x,y,x2,y2=j['raw_roi'];raw=rgb[y:y2,x:x2];marked=np.array(Image.open(returned/(ident+'.png')).convert('RGB'));assert raw.shape==marked.shape
  # Recover clean images from archived full frames, never from user-edited crops.
  rec=dict(id=ident,code=j['expected_code'],source_frame=str(full),source_frame_sha256=sha256(full),marked_sha256=sha256(returned/(ident+'.png')),raw_roi=j['raw_roi'])
  if ident.startswith('12_'):
   rec.update(status='HOLD',reason='User marked a question mark; not a defect region');records.append(rec);continue
  crop,info=corrected_crop(rgb,j['raw_roi'],K,D);h,w=crop.shape[:2];box=None
  if j['expected_code']!='OK':
   a=marked.astype(int);delta=np.max(np.abs(a-raw.astype(int)),axis=2)
   ink=(a[:,:,0]>140)&(a[:,:,0]-a[:,:,1]>55)&(a[:,:,0]-a[:,:,2]>45)&(delta>30)
   yy,xx=np.where(ink);assert len(xx)>20
   local=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
   absolute=[local[0]+x,local[1]+y,local[2]+x,local[3]+y]
   box=map_box(absolute,K,D,np.array(info['rectified_box'][:2]));box=[max(0,box[0]),max(0,box[1]),min(w,box[2]),min(h,box[3])];assert box[2]>box[0] and box[3]>box[1]
   rec.update(user_region_raw_crop=local,status='USER_REGION_ASSISTANT_BOX',note='Bounding rectangle around user circle; region guidance, not pixel-accurate mask')
  else:rec.update(status='USER_CONFIRMED_NORMAL')
  path=out/'train'/(ident+'.png');Image.fromarray(crop).save(path);imageid=len(images)+1
  images.append(dict(id=imageid,file_name=path.name,width=w,height=h))
  if box:
   bx,by,bx2,by2=box;annotations.append(dict(id=len(annotations)+1,image_id=imageid,category_id=int(j['expected_code'][2:]),bbox=[bx,by,bx2-bx,by2-by],area=(bx2-bx)*(by2-by),iscrowd=0))
  rec.update(**info,rectified_defect_box=box,image_sha256=sha256(path));records.append(rec)
  preview=Image.fromarray(crop);d=ImageDraw.Draw(preview)
  if box:d.rectangle(box,outline='lime',width=3)
  preview.thumbnail((195,260));sx=i%6*200;sy=i//6*300;sheet.paste(preview,(sx,sy));sd.text((sx,sy+270),ident,fill='black')
 cats=[dict(id=i,name=f'NG{i:02}',supercategory='defect') for i in range(1,7)]
 (out/'train/_annotations.coco.json').write_text(json.dumps(dict(images=images,annotations=annotations,categories=cats),indent=2))
 meta=dict(training_performed=False,production_ready=False,scope='User-reviewed correction pool; staged for future training, not a validation set',calibration_sha256=sha256(calpath),archive_sha256=sha256(base/'01_여기에_표시.zip'),records=records)
 (out/'provenance.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8');sheet.save(out/'review.jpg')
 (out/'README.md').write_text('User corrections: 6 confirmed normal images, 17 user-indicated defect regions, 1 question-mark image held out. Clean source frames were hash-checked and rectified before cropping. Red marks are never used as training pixels. Boxes enclose user circles and are assistant-generated region approximations, not segmentation masks. No training has run. These reviewed error frames are development/training candidates and must not be claimed as independent evaluation. Merge with existing data by source frame hash to avoid duplicate OK_1 frames. Returned and editable source crops were identical because the user edited the source folder in place; full-frame archive supplies clean pixels.\n',encoding='utf-8')
 print(json.dumps(dict(images=len(images),annotations=len(annotations),hold=1,path=str(out))))

if __name__=='__main__':main()

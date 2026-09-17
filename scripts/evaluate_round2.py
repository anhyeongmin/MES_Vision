from compare_undistortion import *
import shutil

OUT=ROOT/'artifacts/real-eval-round2'
def prepare():
 calpath=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json'
 cal=json.loads(calpath.read_text());K=np.array(cal['K']);D=np.array(cal['D'])
 # Coordinates selected from contact sheets before inference, at 320x180 preview scale.
 selections={'OK':[(32,[112,5,203,113]),(95,[185,34,273,137]),(475,[218,104,318,179]),(728,[111,20,224,115])],
             'NG04':[(94,[202,25,293,137]),(406,[117,81,214,179]),(718,[13,96,111,179])]}
 jobs=[]
 for code,frames in selections.items():
  source=Path('C:/Users/alexi/OneDrive/사진/카메라 앨범')/(code+'_2.mp4')
  dest=OUT/(code+'_2.mp4')
  if not dest.exists():shutil.copy2(source,dest)
  digest=sha256(source);assert digest==sha256(dest)
  cap=cv2.VideoCapture(str(dest))
  for index,box in frames:
   cap.set(cv2.CAP_PROP_POS_FRAMES,index);ok,bgr=cap.read();assert ok
   rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);assert rgb.shape==(720,1280,3)
   box=[v*4 for v in box];name=f'{code}-{index}'
   Image.fromarray(rgb).save(OUT/(name+'-full.png'))
   crop,info=corrected_crop(rgb,box,K,D);path=OUT/(name+'.png');Image.fromarray(crop).save(path)
   jobs.append(dict(code=code,name=name,frame=index,box=box,crop=str(path),crop_sha256=sha256(path),source_sha256=digest,**info))
  cap.release()
 manifest=dict(training_performed=False,threshold=.1,calibration_sha256=sha256(calpath),jobs=jobs,scope='Manually selected whole-object crops; fresh captures of existing specimens; not end-to-end or independent specimen validation')
 (OUT/'selection.json').write_text(json.dumps(manifest,indent=2))
 sheet=Image.new('RGB',(1000,520),'#dddddd');draw=ImageDraw.Draw(sheet)
 for i,j in enumerate(jobs):
  im=Image.open(j['crop']);im.thumbnail((240,225));x=(i%4)*250;y=(i//4)*260;sheet.paste(im,(x,y));draw.text((x+4,y+230),j['name'],fill='black')
 sheet.save(OUT/'selected-crops.jpg')

def infer():
 import torch
 jobs=json.loads((OUT/'selection.json').read_text())['jobs']
 models={'v1':('artifacts/training/ash-real-bootstrap-v1/detail-defects/inference-unvalidated.pth','b865c613dc43140c97deefef64c5325d6709150200ddaab34034561a887b1975'),
 'v2':('artifacts/training/ash-real-rectified-v2/detail-defects/inference-unvalidated.pth','d976bbd04d91b73bf3a01f0ed32c5b563aab00124454102b9e3e56de7b6a7097')}
 coord=GpuCoordinator(ROOT/'.cache/real-labeling/vlm/gpu-coordination');summary={}
 with ExitStack() as stack:
  stack.enter_context(FileLock(str(coord.root/'resident.lock'),timeout=0));stack.enter_context(coord.foreground(timeout=10))
  for model,(path,digest) in models.items():
   backend=RFDETRBackend(ROOT/path,digest,threshold=.1,training_scope='product_defects',class_names=tuple(f'NG{i:02}' for i in range(1,7)),inference_profile='standard',max_batch_size=1);backend.load();results=[]
   for j in jobs:
    assert sha256(Path(j['crop']))==j['crop_sha256']
    ds=[asdict(d) for d in backend.predict_rgb(np.array(Image.open(j['crop']).convert('RGB')))];results.append(dict(j,predictions=ds))
   (OUT/(model+'-predictions.json')).write_text(json.dumps(results,indent=2))
   summary[model]={'OK_false_candidate':sum(bool(j['predictions']) for j in results if j['code']=='OK'),'OK_total':4,'NG04_present':sum(any(d['label']=='NG04' for d in j['predictions']) for j in results if j['code']=='NG04'),'NG04_total':3,'NG04_other_class':sum(any(d['label']!='NG04' for d in j['predictions']) for j in results if j['code']=='NG04')}
   backend.close();del backend;gc.collect();torch.cuda.empty_cache();print(model,summary[model],flush=True)
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))

def render():
 data={m:json.loads((OUT/(m+'-predictions.json')).read_text()) for m in ['v1','v2']}
 for code in ['OK','NG04']:
  rows=[j for j in data['v2'] if j['code']==code];sheet=Image.new('RGB',(900,len(rows)*340),'#eeeeee');draw=ImageDraw.Draw(sheet)
  for r,j in enumerate(rows):
   for c,m in enumerate(['v1','v2']):
    result=next(v for v in data[m] if v['name']==j['name']);im=Image.open(j['crop']).convert('RGB');d=ImageDraw.Draw(im)
    for p in result['predictions']:
     b=p['box'];d.rectangle([b['x1'],b['y1'],b['x2'],b['y2']],outline='red',width=3)
     d.text((b['x1'],max(0,b['y1']-12)),f"{p['label']} {p['score']:.2f}",fill='red')
    im.thumbnail((440,285));sheet.paste(im,(c*450,r*340+25));draw.text((c*450+5,r*340+5),f"{j['name']} | {m}",fill='black')
    labels=', '.join(f"{p['label']}:{p['score']:.2f}" for p in result['predictions']) or 'No defect candidate'
    draw.text((c*450+5,r*340+312),labels,fill='black')
    print(j['name'],m,labels)
  sheet.save(OUT/(code+'-comparison.jpg'))

if __name__=='__main__':
 if '--render' in sys.argv:render()
 elif '--infer' in sys.argv:infer()
 else:prepare()

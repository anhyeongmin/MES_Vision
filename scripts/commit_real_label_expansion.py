"""Save visually selected annotations as drafts; never auto-approve ground truth."""
from pathlib import Path
import sys,json,shutil
from uuid import uuid4
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mes_vision.data_management.collection import Collection,new_object
from mes_vision.training.data import sha256

SELECT={'OK':[3,15,19],'NG01':[3,19],'NG02':[0,3,4,5,18,19],
        'NG03':[1,4,5,17,18,19],'NG04':[1,3,4,5,19],
        'NG05':[3,4,17,18,19],'NG06':[0,3,4,18,19]}
REGIONS={'NG01':(.12,.65,.43,.90),'NG02':(.12,.13,.36,.42),'NG03':(.20,.49,.56,.66),
         'NG05':(.56,.08,.89,.37),'NG06':(.34,.40,.69,.72)}

def main():
 out=ROOT/'artifacts/real-videos-20260914/expanded';c=Collection(ROOT/'collections/ash-real-detail-videos-20260914')
 if (out/'selection.json').exists():
  raise RuntimeError('Initial drafts already committed; preserve subsequent manual and user corrections.')
 proposals=[json.loads(p.read_text()) for p in out.glob('*.json') if len(p.stem)==32]
 assert len(proposals)==140
 dispositions=[]
 for r in proposals:
  chosen=r['index'] in SELECT[r['code']]
  dispositions.append(dict(record_id=r['record_id'],code=r['code'],index=r['index'],selected=chosen,
      reason='Initial near-top-view label set' if chosen else 'Not selected for initial set after contact-sheet review; raw frame retained; no ground truth assigned'))
  if not chosen:continue
  rec=c.record(r['record_id']);assert not rec['reviewed'],'Do not overwrite approved annotation'
  assert sha256(Path(r['image']))==r['image_sha256']
  cand=r['candidates'][0];assert sha256(Path(cand['mask']))==cand['sha256']
  folder=c.root/'assists'/('expanded-'+r['record_id']);folder.mkdir(parents=True,exist_ok=True)
  shutil.copyfile(cand['mask'],folder/'mask.png');(folder/'proposal.json').write_text(json.dumps(r,indent=2))
  b=cand['box'];obj=new_object('ASH-20260914-'+r['code']+'-01',b)
  obj['annotation_assist']={'kind':'sam_mask_to_box','mask':(folder/'mask.png').relative_to(c.root).as_posix(),'mask_sha256':cand['sha256'],'proposal':(folder/'proposal.json').relative_to(c.root).as_posix(),'original_box':b,'reviewed_automatically':False}
  code=r['code'];obj['condition']='NORMAL' if code=='OK' else 'UNCERTAIN' if code=='NG04' else 'KNOWN_NG'
  obj['note']='SAM 물체 후보와 시각 확인 기반 라벨 초안. 최종 정답 검토 전.'
  if code=='NG04':obj['note']+=' CAD 변경은 왼쪽 내부 벽이나 실사진 경계가 불명확하여 보류. 불량 없음으로 취급하지 않음.'
  if code in REGIONS:
   u,v,s,t=REGIONS[code];x,y,x2,y2=b;box=[round(x+(x2-x)*u),round(y+(y2-y)*v),round(x+(x2-x)*s),round(y+(y2-y)*t)]
   obj['defects']=[{'id':uuid4().hex,'code':code,'bbox':box,'note':'시각 검토 기반 문맥 영역 초안; SAM 불량 마스크 아님.'+(' 누락된 돌기가 있어야 할 자리.' if code=='NG01' else '')}]
  rec['objects']=[obj];rec['excluded']=False;rec['exclude_reason']='';c.save_record(rec)
 (out/'selection.json').write_text(json.dumps(dispositions,indent=2))
 for code in SELECT:
  records=[r for r in c.data['records'] if r['objects'] and r['capture_session_id'].endswith(code)]
  sheet=Image.new('RGB',(1200,320*((len(records)+3)//4)),'white');d=ImageDraw.Draw(sheet)
  for i,r in enumerate(records):
   im=Image.open(c.image_path(r)).convert('RGB');o=r['objects'][0];dr=ImageDraw.Draw(im);dr.rectangle(o['bbox'],outline='cyan',width=3)
   for defect in o['defects']:dr.rectangle(defect['bbox'],outline='red',width=3)
   x,y,x2,y2=o['bbox'];crop=im.crop((max(0,x-10),max(0,y-10),min(im.width,x2+10),min(im.height,y2+10)));crop.thumbnail((295,285))
   px=i%4*300;py=i//4*320;sheet.paste(crop,(px,py));d.text((px+3,py+290),f"{i}: {r['video_source']['estimated_seconds']:.1f}s {o['condition']}",fill='black')
  sheet.save(out/(code+'-labels.jpg'))
 print(c.inventory())

if __name__=='__main__':main()

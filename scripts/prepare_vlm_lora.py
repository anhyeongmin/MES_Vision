"""Prepare existing real photos for a small advisory VLM LoRA experiment."""
import json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SYSTEM='사진 속 흰색 부품의 관찰 결과를 짧게 설명하세요. 정상은 원통 두 개와 열린 구멍 두 개, 곧은 벽입니다. 적층 무늬 자체는 불량이 아닙니다. NG01=원통 누락, NG02=추가 돌출물, NG03=균열, NG04=벽 변형, NG05=구멍 막힘, NG06=표면 패임. 확인이 어려우면 UNCERTAIN으로 답하세요. 원인이나 치수를 추측하지 마세요. 출력은 {"code":"코드","observation":"위치와 관찰 특징 한 문장"} JSON만 사용하세요. 이 결과는 보조 설명입니다.'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def prepare():
 src=ROOT/'datasets/ash-real-recorded-merged-v1';out=ROOT/'datasets/ash-vlm-lora-v1';out.mkdir(exist_ok=True)
 coco=json.loads((src/'train/_annotations.coco.json').read_text());meta=json.loads((src/'provenance.json').read_text(encoding='utf-8'));rows=[]
 features={1:'원통이 있어야 할 자리가 비어 있습니다.',2:'벽 안쪽에 사각형 돌출물이 있습니다.',3:'바닥에 선 모양의 균열이 보입니다.',4:'벽의 형상이 변형되어 보입니다.',5:'원통 윗면의 구멍이 막혀 있습니다.',6:'바닥 표면에 원형 패임이 보입니다.'}
 for im,r in zip(coco['images'],meta['items']):
  p=src/'train'/im['file_name'];assert sha(p)==r['image_sha256'];anns=[a for a in coco['annotations'] if a['image_id']==im['id']];assert len(anns)<=1
  if not anns:code='OK';text='두 원통과 열린 구멍이 보이며 뚜렷한 불량은 보이지 않습니다.';region=None
  else:
   a=anns[0];x,y,w,h=a['bbox'];cx=(x+w/2)/im['width'];cy=(y+h/2)/im['height'];code=f"NG{a['category_id']:02}";region=a['bbox']
   vertical='위쪽' if cy<.4 else '아래쪽' if cy>.6 else '중앙';horizontal='왼쪽' if cx<.4 else '오른쪽' if cx>.6 else ''
   text='사진 '+(' '.join([horizontal,vertical]).strip())+'에서 '+features[a['category_id']]
  split='train' if im['file_name'].startswith('2_') else 'eval'
  rows.append(dict(id=r['id'],image=str(p),image_sha256=r['image_sha256'],split=split,target={'code':code,'observation':text},source_region=region,source_provenance=r['source_provenance'],caption_status='assistant_visual_reviewed_template_and_region_draft_not_user_confirmed'))
 data=dict(system=SYSTEM,scope='pilot only; new recording sessions train86, prior sessions eval55; same physical specimens may recur; not independent product validation; no unknown-defect examples',source_sha256=sha(src/'provenance.json'),items=rows)
 dest=out/'dataset.json'
 if dest.exists():assert json.loads(dest.read_text(encoding='utf-8'))==data
 else:dest.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 assert sum(r['split']=='train' for r in rows)==86 and sum(r['split']=='eval' for r in rows)==55
 assert len({r['image_sha256'] for r in rows})==141
 return dest
if __name__=='__main__':print(prepare())

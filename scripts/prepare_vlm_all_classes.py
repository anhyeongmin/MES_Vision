"""Local, provenance-preserving all-class region/whole-object caption dataset."""
import json, sys, hashlib, math
from pathlib import Path
from collections import Counter
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
OUT=ROOT/'artifacts/vlm-all-classes-v1'
OLD=Path('C:/Users/alexi/Documents/ChatGPT/MES 비전/artifacts/vlm-region-abstain-v1')
QUESTIONS={
 'OK':'물체 전체에서 보이는 원통, 구멍, 바닥 표면의 특징을 간단히 설명하세요.',
 'NG01':'물체 내부 원통의 개수와 보이는 배치를 설명하세요.',
 'NG02':'확대 영역의 벽 안쪽에 사각 돌출물이 보이는지 설명하세요.',
 'NG03':'확대 영역의 바닥에 선 모양 균열이 보이는지 설명하세요.',
 'NG04':'확대 영역의 원통과 벽 접합부에 뚜렷한 틈이 보이는지 설명하세요.',
 'NG05':'확대 영역의 원통 구멍이 열려 있는지 설명하세요.',
 'NG06':'확대 영역의 바닥에 원형 표면 흔적이 보이는지 설명하세요.'}
TEXT={
 'OK':'두 원통과 열린 구멍이 보이며 뚜렷한 이상은 관찰되지 않습니다. 사진만으로 정상 여부를 확정할 수는 없습니다.',
 'NG01':'내부 원통이 하나만 보이며 다른 원통 자리가 비어 있습니다.',
 'NG02':'벽 안쪽에 사각형 돌출물이 보입니다.',
 'NG03':'바닥에 선 모양의 균열이 보입니다.',
 'NG04':'벽 형상과 원통 접합부의 상태를 이 사진만으로 명확히 확인하기 어렵습니다.',
 'NG05':'원통 윗면이 채워져 구멍이 막혀 있습니다.',
 'NG06':'바닥에 원형 표면 흔적이 보입니다. 깊이는 사진만으로 확인하기 어렵습니다.'}
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True); (OUT/'crops').mkdir(exist_ok=True)
 original=read(ROOT/'datasets/ash-vlm-caption-v3/dataset.json')
 old=read(OLD/'dataset.json'); rows=[]; oldparents={r['parent_sha256'] for r in old['items'] if 'parent_sha256' in r}
 def add(r,code,label,text,box=None):
  image=Path(r['image']); assert sha(image)==r['image_sha256']
  with Image.open(image) as im:
   im=im.convert('RGB')
   if box: im=im.crop(box)
   dest=OUT/'crops'/(r['id']+'--'+label+'.png');im.save(dest)
  rows.append(dict(id=r['id']+'--'+label,image=str(dest),image_sha256=sha(dest),parent_image=str(image),
   parent_sha256=r['image_sha256'],split=r['split'],label=label,code=code,question_text=QUESTIONS[code],
   target={'observation':text},preprocessing='already rectified object crop; no second undistortion; optional region crop',
   roi_xyxy=box,review_status='existing assistant region drafts and user coarse circles; not independent ground truth'))
 for r in original['items']:
  if r['split']=='eval' and r['image_sha256'] in oldparents:continue
  code=r['target']['code']
  # Whole-object question is trained across conditions, not just normal objects.
  add(r,'OK','whole_'+code,TEXT[code])
  add(r,'NG01','one_cylinder' if code=='NG01' else 'two_cylinders',TEXT['NG01'] if code=='NG01' else '내부에 원통 두 개가 보입니다.')
  if code in {'NG02','NG03','NG06'}:
   x,y,w,h=r['source_region']
   with Image.open(r['image']) as im: W,H=im.size
   box=[max(0,math.floor(x-w*.25)),max(0,math.floor(y-h*.25)),min(W,math.ceil(x+w*1.25)),min(H,math.ceil(y+h*1.25))]
   add(r,code,code+'_present',TEXT[code],box)
  if code=='OK':
   # A clean interior floor is a negative/visibility example, never fabricated damage.
   with Image.open(r['image']) as im: W,H=im.size
   box=[int(W*.38),int(H*.38),int(W*.62),int(H*.62)]
   for question in ('NG02','NG03','NG06'):
    text={'NG02':'이 영역에서는 벽 안쪽 사각 돌출물을 확인하기 어렵습니다.',
          'NG03':'이 영역에서 뚜렷한 선 모양 균열은 보이지 않습니다.',
          'NG06':'이 영역에서 뚜렷한 원형 표면 흔적은 보이지 않습니다.'}[question]
    add(r,question,question+'_absent',text,box)
 rows+=old['items']
 trainhash={r['image_sha256'] for r in rows if r['split']=='train'}
 assert all(r['image_sha256'] not in trainhash for r in rows if r['split']=='eval')
 data=dict(system=old['system'],items=rows,questions=QUESTIONS,source_sha256=sha(ROOT/'datasets/ash-vlm-caption-v3/dataset.json'),
           replay_sha256=sha(OLD/'dataset.json'),independent_evaluation=False,
           limitation='Prior-session diagnostic split; same specimens and previously used experiments. Not production accuracy.')
 (OUT/'dataset.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
 print(json.dumps(dict(examples=len(rows),counts=dict(Counter(r['split'] for r in rows)),labels=dict(Counter(r['label'] for r in rows))),ensure_ascii=False))
if __name__=='__main__':main()

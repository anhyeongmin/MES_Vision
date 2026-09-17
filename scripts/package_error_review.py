"""Package all retained real-video evaluation failures for human correction."""
from pathlib import Path
import json, shutil, hashlib, csv, zipfile
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/error-review-20260914'
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=False)
 groups={};sources=[]
 for round_name,base in [('R1',ROOT/'artifacts/real-eval-20260914'),('R2',ROOT/'artifacts/real-eval-round2')]:
  paths=list(base.glob('*-predictions.json'))
  if round_name=='R1':paths+=list((base/'undistortion').glob('*-predictions.json'))
  for p in sorted(paths):
   sources.append(dict(path=str(p),sha256=digest(p)))
   for j in json.loads(p.read_text()):
    code=j.get('expected_code',j.get('code'));key=(round_name,j['name'])
    g=groups.setdefault(key,dict(round=round_name,name=j['name'],code=code,job=j,base=base,variants=[]))
    pred=j['predictions'];variants=pred if isinstance(pred,dict) else {'corrected' if round_name=='R2' else 'raw':pred}
    for variant,ds in variants.items():
     reasons=[]
     if code=='OK' and ds:reasons.append('normal_false_candidate')
     if code!='OK':
      if not any(d['label']==code for d in ds):reasons.append('expected_class_missing')
      if any(d['label']!=code for d in ds):reasons.append('other_class_candidate')
     crop=j.get('corrected_crop') if variant=='corrected' else j['crop']
     if round_name=='R2':crop=j['crop']
     g['variants'].append(dict(source=str(p.relative_to(ROOT)),variant=variant,crop=crop,predictions=ds,reasons=reasons))
 selected=[g for g in groups.values() if any(v['reasons'] for v in g['variants'])]
 selected.sort(key=lambda g:(g['round'],g['code']!='OK',g['code'],g['name']))
 for folder in ['01_여기에_표시','02_모델결과_참고','03_전체화면_참고']:(OUT/folder).mkdir()
 rows=[];manifest=[]
 font=ImageFont.truetype('C:/Windows/Fonts/malgun.ttf',17)
 for i,g in enumerate(selected,1):
  ident=f"{i:02d}_{g['round']}_{g['name']}";j=g['job'];base=g['base']
  full=base/'frames'/(g['name']+'.png') if g['round']=='R1' else base/(g['name']+'-full.png')
  im=Image.open(full).convert('RGB');raw=im.crop(tuple(j['box']));mark=OUT/'01_여기에_표시'/(ident+'.png');raw.save(mark)
  shutil.copy2(full,OUT/'03_전체화면_참고'/(ident+'.png'))
  # Deduplicate repeated raw results in the paired ablation while retaining every distinct result.
  vs=[];seen=set()
  for v in g['variants']:
   k=(digest(Path(v['crop'])),json.dumps(v['predictions'],sort_keys=True))
   if k not in seen:vs.append(v);seen.add(k)
  sheet=Image.new('RGB',(600*min(2,len(vs)),430*((len(vs)+1)//2)),'#eeeeee');d=ImageDraw.Draw(sheet)
  for n,v in enumerate(vs):
   pic=Image.open(v['crop']).convert('RGB');scale=min(570/pic.width,320/pic.height);pic=pic.resize((round(pic.width*scale),round(pic.height*scale)));pd=ImageDraw.Draw(pic)
   for det in v['predictions']:
    b=det['box'];xy=[b[k]*scale for k in ['x1','y1','x2','y2']];pd.rectangle(xy,outline='red',width=3);pd.text((xy[0],max(0,xy[1]-18)),f"{det['label']} {det['score']:.2f}",font=font,fill='red')
   x=n%2*600;y=n//2*430;sheet.paste(pic,(x+10,y+55))
   title=Path(v['source']).stem+' / '+v['variant'];d.text((x+10,y+5),ident+' '+g['code'],font=font,fill='black');d.text((x+10,y+28),title,font=font,fill='black')
   reason=' / '.join(v['reasons']) or 'No class-level error';d.text((x+10,y+385),reason,font=font,fill='black')
  sheet.save(OUT/'02_모델결과_참고'/(ident+'.jpg'))
  reasons=sorted({r for v in vs for r in v['reasons']})
  rows.append([ident,g['code'],'정상 확인만 하면 됨' if g['code']=='OK' else '실제 불량 부위를 동그라미로 표시','',''])
  manifest.append(dict(id=ident,expected_code=g['code'],raw_roi=j['box'],source_frame_sha256=digest(full),annotation_image_sha256=digest(mark),source_video_sha256=j.get('source_sha256'),variants=vs,reasons=reasons))
 with (OUT/'확인목록.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(['사진번호','기존 품목','요청','확인한 정답','메모']);w.writerows(rows)
 (OUT/'manifest.json').write_text(json.dumps(dict(scope='All class-level failures from retained real-video evaluation runs; synthetic evaluations and superseded invalid crop runs excluded',sources=sources,items=manifest),ensure_ascii=False,indent=2),encoding='utf-8')
 readme=f'''사진 표시 안내 — 총 {len(rows)}장 (같은 촬영 프레임은 한 번만 포함)

1. 01_여기에_표시 폴더의 사진에 실제 불량 부위를 동그라미로 표시해 주세요.
   파일명은 그대로 유지하고 이 폴더를 다시 압축해서 주시면 됩니다.
2. OK 사진은 정상이라면 표시하지 않고, 'OK 사진 모두 정상'이라고 알려주시면 됩니다.
   정상 사진에서 모델이 표시한 부위는 불량 정답으로 표시하지 마세요.
3. NG 사진은 모델의 빨간 박스가 아니라 실제 불량 전체를 표시해 주세요.
   사진에서 불량을 확인할 수 없으면 '안 보임', 판단이 어려우면 '불확실'이라고 적어 주세요.
4. 설명이 필요하면 확인목록.csv에 사진번호별 메모를 남기거나 채팅으로 알려주세요.
5. 02_모델결과_참고: 이전/새 모델, 보정 전/후의 예측입니다. 여기는 표시하지 않아도 됩니다.
   raw=보정 전, corrected=왜곡 보정 후. 빨간 박스는 모델 예측이며 정답이 아닙니다.
   normal_false_candidate=정상 오탐, expected_class_missing=해당 불량 미검출,
   other_class_candidate=다른 불량 종류로 검출. No class-level error=종류 기준 오류 없음.
6. 03_전체화면_참고: 주변 상황을 확인하기 위한 원본 전체 프레임입니다.

대상: 지금까지 저장한 실제 영상 평가에서 한 모델/조건이라도 오탐·미검출한 사진.
합성 STL 이미지와 잘못 잘라 폐기한 예비 평가 결과는 제외했습니다.
박스 위치 정답은 아직 검증되지 않아, 종류는 맞아도 위치가 틀린 사례까지 자동 판별한 것은 아닙니다.
표시용 사진은 보정 전 원본 잘라내기이며 모델 결과의 보정 후 사진과 모양이 다를 수 있습니다.
원본과 좌표 연결 정보를 보관했으므로 표시를 받으면 필요한 좌표계로 변환합니다.
이번 작업은 검토 자료 포장만 했으며, 추가 학습이나 평가 데이터 재배정은 하지 않았습니다.
'''
 (OUT/'먼저읽기.txt').write_text(readme,encoding='utf-8-sig')
 archive=Path(shutil.make_archive(str(OUT),'zip',root_dir=OUT.parent,base_dir=OUT.name))
 with zipfile.ZipFile(archive) as z:assert z.testzip() is None;assert len(z.namelist())>=len(rows)*3
 print(json.dumps(dict(count=len(rows),normal=sum(g['code']=='OK' for g in selected),ng=sum(g['code']!='OK' for g in selected),zip=str(archive),MB=round(archive.stat().st_size/1e6,1)),ensure_ascii=False))

if __name__=='__main__':main()

"""Assistant visual-review decisions after inspecting all 94 crop sheets."""
from pathlib import Path
import sys,json,zipfile,html
from copy import deepcopy
from collections import Counter
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import sha256
BASE=ROOT/'artifacts/recorded-label-review-20260914-v3'

EXCLUDE={
    'NG02_204657_f0285','NG02_204657_f0315','NG02_204657_f0360','NG02_204657_f0600',
    'NG03_205152_f0180','NG03_205152_f0375','NG03_205152_f0615','NG03_205152_f0885'}
PENDING={
    'NG01_204549_f0420','NG01_204549_f0450','NG01_204549_f0570','NG01_204549_f0885','NG01_204549_f0945',
    'NG04_205354_f0240','NG04_205354_f0345','NG04_205354_f0660','NG04_205354_f0690','NG06_205545_f0180'}
ALTERNATE={'NG01_204549_f0015':1,'NG01_204549_f0795':1}
RESOLVED={'NG01_204549_f0525','NG02_204657_f0660','NG06_205545_f0525','NG06_205545_f0690'}

def main():
    data=json.loads((BASE/'drafts.json').read_text(encoding='utf-8'));rows=deepcopy(data['items'])
    assert len(rows)==94 and all('crop' in r for r in rows)
    package=BASE/'user-review';package.mkdir(exist_ok=True)
    for name in ('01_여기에_표시','02_모델_참고'):(package/name).mkdir(exist_ok=True)
    dataset=ROOT/'datasets/ash-recorded-low-exposure-draft-v1'
    assert not dataset.exists(),'Dataset exists; do not replace prior labels'
    for split in ('train','valid'):(dataset/split).mkdir(parents=True)
    images=[];annotations=[];requests=[];ready=[];cards=[]
    for row in rows:
        assert sha256(Path(row['crop']))==row['crop_sha256']
        identity=row['id'];code=row['label'];preds=row['predictions']
        if identity in EXCLUDE:
            status='EXCLUDED_VISIBILITY';reason='사진에서 해당 불량 부위를 충분히 확인하기 어려워 이번 초안에서 제외';boxes=[]
        elif identity in PENDING:
            status='USER_REGION_NEEDED';reason='누락 위치 또는 불량 영역을 직접 확인해야 함';boxes=[]
        elif code=='OK':
            status='ASSISTANT_CHECKED_DRAFT';reason='사용자가 OK로 수집한 영상. 물체 가림 없이 두 원통 확인; 모델의 NG05 후보는 정답으로 사용하지 않음';boxes=[]
        else:
            status='ASSISTANT_CHECKED_DRAFT';reason='전체 연락판에서 물체 가림 및 표시된 불량 위치를 육안 검토한 초안'
            expected=sorted([p for p in preds if p['label']==code and p['score']>=.25],key=lambda p:-p['score'])
            assert expected,identity
            boxes=[deepcopy(expected[ALTERNATE.get(identity,0)])]
            if identity in ALTERNATE:reason='누락 원통의 반대쪽 정상 위치를 짚은 높은 점수 후보를 제외하고 실제 누락 위치 후보를 선택'
            if identity in RESOLVED:reason='시각적으로 확인한 해당 불량 위치만 채택; 함께 나온 다른 불량 코드 후보 제외'
        row.update(preparation_status=status,assistant_review_note=reason,selected_defects=boxes,
                   annotation_reviewed=False,annotation_status='assistant_checked_draft_not_user_confirmed')
        im=Image.open(row['crop']).convert('RGB');w,h=im.size
        for p in boxes:
            b=p['box'];assert 0<=b['x1']<b['x2']<=w and 0<=b['y1']<b['y2']<=h
        if status=='ASSISTANT_CHECKED_DRAFT':
            image_id=len(images)+1;name=identity+'.png';(dataset/'train'/name).write_bytes(Path(row['crop']).read_bytes())
            images.append({'id':image_id,'file_name':name,'width':w,'height':h})
            for p in boxes:
                b=p['box'];box=[b['x1'],b['y1'],b['x2']-b['x1'],b['y2']-b['y1']]
                annotations.append({'id':len(annotations)+1,'image_id':image_id,'category_id':int(code[2:]),'bbox':box,'area':box[2]*box[3],'iscrowd':0})
            ready.append(row)
        elif status=='USER_REGION_NEEDED':
            name=f'{len(requests)+1:02}_{identity}.png'
            clean=package/'01_여기에_표시'/name;clean.write_bytes(Path(row['crop']).read_bytes())
            reference=package/'02_모델_참고'/(Path(name).stem+'.jpg');reference.write_bytes(Path(row['overlay']).read_bytes())
            requests.append({'file':name,'id':identity,'expected_code':code,'width':w,'height':h,'clean_sha256':sha256(clean),
                'source_crop':row['crop'],'source_folder':row['source_folder'],'frame_index':row['frame_index'],'time_seconds':row['time_seconds'],
                'instructions':'보이는 불량 영역에 빨간 동그라미. 확인할 수 없으면 물음표 또는 제외 표시. NG01은 원통이 있어야 할 누락 위치.'})
        card=f'<article data-status="{status}" data-code="{code}"><h3>{html.escape(identity)}</h3><p>{status} · {row["time_seconds"]:.1f}초</p>'
        card+=f'<div class="pair"><a href="crops/{identity}.png"><img loading="lazy" src="crops/{identity}.png"></a><a href="overlays/{identity}.jpg"><img loading="lazy" src="overlays/{identity}.jpg"></a></div><p>{html.escape(reason)}</p></article>'
        cards.append(card)
    assert len(ready)==76 and len(annotations)==62 and len(requests)==10
    categories=[{'id':i,'name':f'NG{i:02}','supercategory':'defect'} for i in range(1,7)]
    for split,ims,anns in [('train',images,annotations),('valid',[],[])]:
        (dataset/split/'_annotations.coco.json').write_text(json.dumps({'images':ims,'annotations':anns,'categories':categories},ensure_ascii=False,indent=2),encoding='utf-8')
    manifest={**{k:v for k,v in data.items() if k!='items'},'status':'PREPARED_ASSISTANT_CHECKED_DRAFT',
        'annotation_status':'assistant_checked_draft_not_user_confirmed','training_performed':False,'production_ready':False,
        'train_images':76,'negative_images':14,'positive_images':62,'pending_user_regions':10,'excluded_visibility':8,
        'source_selection_sha256':sha256(BASE/'selection.json'),'items':ready}
    (dataset/'provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    (BASE/'visual-review.json').write_text(json.dumps({'items':rows,'counts':dict(Counter(r['preparation_status'] for r in rows))},ensure_ascii=False,indent=2),encoding='utf-8')
    (package/'review-map.json').write_text(json.dumps(requests,ensure_ascii=False,indent=2),encoding='utf-8')
    instructions='''표시가 필요한 10장만 담았습니다.

01_여기에_표시: 원본 크기 그대로 빨간색으로 불량 영역을 동그라미 쳐 주세요.
NG01: 원통이 없어져 있어야 할 위치를 표시해 주세요.
NG04: 변형된 벽의 영역을 표시해 주세요.
NG06: 표면 불량 영역을 표시해 주세요.
부위가 보이지 않거나 판단하기 어려우면 억지로 표시하지 말고 물음표 또는 제외라고 적어 주세요.

02_모델_참고: 기존 모델의 예측입니다. 오답이 포함되어 있으므로 정답으로 따라 그리지 않아도 됩니다.
파일명, 해상도, 캔버스 크기는 유지해 주세요. OK 영상은 이미 정상 후보로 정리해 표시할 필요가 없습니다.
표시한 폴더를 압축해 보내주면 좌표를 원래 사진과 연결할 수 있습니다.
'''
    (package/'읽어주세요.txt').write_text(instructions,encoding='utf-8-sig')
    with zipfile.ZipFile(BASE/'표시필요_10장.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in package.rglob('*'):
            if p.is_file():z.write(p,p.relative_to(package))
    intro='<h1>노출 -9 신규 영상 · 학습 준비 검토</h1><p>7개 영상 / 7,089 프레임 → 대표 94장 → 초안 76장(OK 14 / NG 62), 표시 필요 10장, 가시성 제외 8장.</p><p>왜곡 보정 후 물체 영역 추출. 모든 예측은 새 영상으로 추가 학습하기 전의 기존 실물 모델 결과입니다. 독립 성능 평가가 아니며 아직 학습하지 않았습니다.</p><p>왼쪽은 보정된 사진, 오른쪽은 수정 전 모델 예측입니다. 초안에서 선택한 정답 박스는 provenance.json에 따로 기록했습니다.</p>'
    controls='<label>분류 <select id="code"><option value="">전체</option>'+''.join(f'<option>{c}</option>' for c in ['OK']+[f'NG{i:02}' for i in range(1,7)])+'</select></label> <label>상태 <select id="status"><option value="">전체</option><option value="USER_REGION_NEEDED">표시 필요 10장</option><option value="ASSISTANT_CHECKED_DRAFT">초안 76장</option><option value="EXCLUDED_VISIBILITY">제외 8장</option></select></label>'
    page='<!doctype html><html lang="ko"><meta charset="utf-8"><title>신규 영상 학습 준비</title><style>body{font:16px system-ui;background:#f0f3f7;color:#182535;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(480px,1fr));gap:16px}article{background:white;padding:16px;border-radius:12px}h3{font-size:16px}.pair{display:flex;gap:8px}.pair a{width:50%}img{width:100%;height:280px;object-fit:contain;background:#e9edf2}select{padding:8px;margin:12px}article[hidden]{display:none}</style>'+intro+controls+'<main>'+''.join(cards)+'</main><script>function filter(){document.querySelectorAll("article").forEach(e=>e.hidden=(code.value&&e.dataset.code!==code.value)||(status.value&&e.dataset.status!==status.value))}document.querySelectorAll("select").forEach(e=>e.onchange=filter);const code=document.getElementById("code"),status=document.getElementById("status");</script></html>'
    (BASE/'report.html').write_text(page,encoding='utf-8')
    (dataset/'README.md').write_text('학습 전 검토용 초안: 실물 76장, 정상14장, NG62개 박스. 사용자 확인 전 초안이며 독립 검증 자료가 없습니다. 표시 대기10장과 가시성 제외8장은 학습 이미지에 포함하지 않았습니다. 기존55장/STL 데이터와 합치거나 학습한 상태가 아닙니다.\n',encoding='utf-8')
    print(json.dumps({'dataset':str(dataset),'ready_draft':len(ready),'regions':len(annotations),'user_review':len(requests),'excluded':len(EXCLUDE),'archive_MB':round((BASE/'표시필요_10장.zip').stat().st_size/1e6,2)},ensure_ascii=False))

if __name__=='__main__':main()

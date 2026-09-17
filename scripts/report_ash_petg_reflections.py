"""Summarize measured software outputs without claiming measured filament optics."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.station.photo_inspection import open_results
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.synthetic.detection_review import summarize
OUT=ROOT/'artifacts/preprint-petg-reflections-v1'; BASE=ROOT/'artifacts/preprint-olive-white-v1'


def main():
    plan=read_json(OUT/'plan.json'); tasks={t['id']:t for t in plan['tasks']}
    root,report=open_results(OUT/'inference/results.json'); rows=[]; errors=[]
    for record in report['records']:
        identity=Path(record['name']).stem; task=tasks[identity]; q=read_json(OUT/'quality'/(identity+'.json'))
        _,result,_=load_snapshot(root/record['snapshot'],expected_digest=record['snapshot_digest'])
        require(not result['robot_commands_enabled'] and not result['frame']['is_live'],'Unexpected live output')
        def xyxy(box):
            x,y,w,h=box; return [x,y,x+w,y+h]
        object_preds=[{'label':'ASH','score':o['detection']['score'],'box':[o['effective_box'][k] for k in ('x1','y1','x2','y2')]} for o in result['objects']]
        findings=[]
        for obj in result['objects']:
            require(obj['final_decision']=='REVIEW','Unexpected product approval')
            for check in obj['checks']:
                if check['status']=='ERROR': errors.append({'id':identity,'check':check['check_id']})
                for f in check['findings']:
                    c=f['crop_box']; b=obj['crop']['bounds_original_xyxy']; original=f['original_box']
                    require(original==dict(x1=c['x1']+b['x1'],y1=c['y1']+b['y1'],x2=c['x2']+b['x1'],y2=c['y2']+b['y1']),'Wrong original defect coordinates')
                    findings.append({'label':f['defect_code'],'score':f['score'],'box':[original[k] for k in ('x1','y1','x2','y2')]})
        rows.append({'file_name':identity,'domain':task['domain'],'finish':task['finish'],'lighting':task['lighting'],
            'targets':[{'label':f['code'],'box':xyxy(f['bbox'])} for f in q['defects']],
            'predictions':findings,'object_targets':[{'label':'ASH','box':xyxy(o['bbox'])} for o in q['objects']],
            'object_predictions':object_preds,'association_confirmed':record['association_confirmed']})
    summaries={}
    for finish in plan['finishes']:
        subset=[r for r in rows if r['finish']==finish]; details=[r for r in subset if r['domain']=='detail']
        overview=[dict(file_name=r['file_name'],targets=r['object_targets'],predictions=r['object_predictions']) for r in subset if r['domain']=='overview']
        summaries[finish]={'defects':summarize(details,.1,[f'NG{i:02d}' for i in range(1,7)]),
            'overview':summarize(overview,.05,['ASH']),'unconfirmed_details':sum(not r['association_confirmed'] for r in details),
            'by_lighting':{light:summarize([r for r in details if r['lighting']==light],.1,[f'NG{i:02d}' for i in range(1,7)]) for light in ('side','near-axis')}}
    write_json(OUT/'evaluation.json',{'finishes':summaries,'errors':errors,'photos':len(rows),'physical_optics_measured':False,'retrained':False})
    write_json(OUT/'predictions.json',rows); require(not errors,'Model errors detected')
    font=lambda size:ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    board=Image.new('RGB',(1200,880),'#e9edf1'); draw=ImageDraw.Draw(board)
    draw.text((10,10),'SYNTHETIC roughness sensitivity | NG03 | GREEN: CAD / ORANGE: AI | NOT calibrated JAMG HE PETG',font=font(17),fill='black')
    for col,finish in enumerate(plan['finishes']):
        for row,light in enumerate(('side','near-axis')):
            identity=f'{finish}-{light}-NG03'; info=next(r for r in rows if r['file_name']==identity)
            img=Image.open(OUT/'raw'/(identity+'.png')).convert('RGB'); d=ImageDraw.Draw(img)
            for f in info['targets']: d.rectangle(f['box'],outline='#14a470',width=3)
            for f in info['predictions']:
                d.rectangle(f['box'],outline='#db8200',width=3)
                d.text((f['box'][0],f['box'][1]-18),f"{f['label']} {f['score']:.2f}",font=font(16),fill='#a65f00')
            q=read_json(OUT/'quality'/(identity+'.json')); x,y,w,h=q['objects'][0]['bbox']
            img=img.crop((x-20,y-20,x+w+20,y+h+20)); img.thumbnail((385,365))
            bx=col*400; by=45+row*410
            draw.text((bx+10,by+4),finish+' / '+light,font=font(16),fill='black')
            board.paste(img,(bx+(400-img.width)//2,by+37+(365-img.height)//2))
    board.save(OUT/'review/reflection-comparison.png')
    lines=['# 잠허 PETG 올리브그린·화이트 출력 전 검토','',
        '**사용자 지정 재료: 바닥 JAMG HE PETG Olive Green, OK·NG 물체 JAMG HE PETG White.**','',
        'STL 7종 점검, 바닥색 비교92장, 추가 반사·조명 조건45장까지 총137장의 합성 사진을 생성하고 기존 학습 모델로 점검했다. 실제 필라멘트의 광학 측정이나 출력 시험은 아니다.','',
        '## 확인한 내용','',
        '- 원본 STL 7종은 기존 등록본과 일치한다. 기본 메시 검사에서 열린 모서리·면적 0 삼각형은 발견되지 않았다. 모든 NG가 같은 +Y 검사면에서 형상 차이를 보인다.',
        '- 좌표 단위를 mm로 가져오면 X×Y×Z는16×14×20이다. 검사면은16×20, 높이는14이다. 슬라이서 단위와 100% 배율을 실제로 확인해야 한다.',
        '- 바닥색 비교에서는 4가지 가상 바닥 각각의 7개 물체 전체사진에서 위치7개를 검출했고 빈 전체사진에는 물체를 검출하지 않았다. 한 배치의 반복이므로 일반적인 실물 위치 정확도를 의미하지 않는다.',
        '- 중간 올리브그린의 상세21장에서는 불량 정답18개를 모두 찾았지만 추가 후보6개가 있었다. 그중 정상3장 중1장에도 후보가 발생했다. 비교용 차콜도 추가 후보5개가 있어 색만으로 해결되는 문제로 볼 수 없다.',
        '- 색 비교의 일부 조건에서 정상 테두리를 NG04로 오검출했다. 그림에서 그림자·테두리와 후보가 겹치는 것은 관찰 결과이며, 원인을 확정한 실험은 아니다.','',
        '## 반사와 조명 변화 결과','',
        '중간 올리브그린 한 색에 재질 거칠기3조건과 조명2방향을 짝지어 적용했다. 조건별7종×2방향=14장, 전체사진1장이다. 거칠기 숫자는 임의의 Blender 재질 매개변수이며 잠허 PETG의 실측값이 아니다. 투과·실제 적층선·출력 결함은 재현하지 않았다.','',
        '| 반사 가정 | 불량 일치 / 정답12 | 추가 후보 | 놓침 | 정상2장 중 후보 발생 | 상세 물체 확인 실패 | 전체 위치 |',
        '|---|---|---|---|---|---|---|']
    for name,s in summaries.items():
        f=s['defects']['total']; o=s['overview']['total']
        lines.append(f"| {name} | {f['tp']}/12 | {f['fp']} | {f['fn']} | {s['defects']['false_alarm_images']}/2 | {s['unconfirmed_details']} | {o['tp']}/7, 추가{o['fp']} |")
    lines+=['','세 모델의 가중치와 후보 기준 .05/.20/.10을 그대로 유지했다. 결과를 보고 기준을 올려 오검출을 숨기거나 재학습하지 않았다. 정상 참조와 실물 판정 기준이 없어 최종 판정은 모두 보류다.','',
        '## 출력 전에 정할 것','',
        '이 결과만으로 올리브그린·화이트 조합을 폐기할 근거는 없다. 다만 흰 PETG의 실제 반사와 출력 적층선은 출력 후 확인해야 한다. 프린터·노즐·레이어 높이가 확인되지 않아 슬라이싱이나 출력 가능성 승인은 하지 않았다.',
        '소량 출력은 OK·NG03·NG04·NG06을 먼저 확인 대상으로 삼을 수 있다. 균열의 가는 끝과 좁은 벽 변형이 출력 경로에서 남는지, 표면 함몰과 적층 무늬가 구분되는지 확인한다. 검사면에 지지대 접촉이나 제거 흔적이 남는지도 살펴야 한다.','',
        '실물 촬영 후 정상 테두리·반사·적층 무늬를 포함한 사진으로 보완 학습하고, 다른 실물과 촬영 회차로 평가해야 한다. 바닥색이나 출력 품목별 색으로 정답을 구분하게 만들지 않는다.','',
        '## 파일','',
        '- [STL 및 바닥색 상세 결과](../preprint-olive-white-v1/README.md)',
        '- [STL 검사면](../preprint-olive-white-v1/review/cad-face.png)',
        '- [바닥색 비교](../preprint-olive-white-v1/review/palette-comparison.png)',
        '- [색 비교의 실제 모델 예측](../preprint-olive-white-v1/review/defect-predictions.png)',
        '- [반사·조명 비교](review/reflection-comparison.png)',
        '- `inference/results.json`과 `../preprint-olive-white-v1/inference/results.json`은 프로그램의 저장사진 검사 → 저장 결과 열기로 조회한다.',
        '- 제조사 제품 확인: [JAMG HE PETG](https://www.jamghe.com/products/petg-filament-1-75mm-1kg). 이 문서에 사용한 색·거칠기는 제조사에서 받은 측정값이 아니다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({name:s['defects']['total'] for name,s in summaries.items()},flush=True)


if __name__=='__main__': main()

"""Report all preprint paired conditions, retaining failures and uncertain cases."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.station.photo_inspection import open_results
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.synthetic.detection_review import summarize
from mes_vision.synthetic.planning import matrix
from mes_vision.synthetic.projection import raster_depth,world_triangles
OUT=ROOT/'artifacts/preprint-olive-white-v1'


def xyxy(box):
    x,y,w,h=box; return [x,y,x+w,y+h]


def main():
    plan=read_json(OUT/'plan.json'); tasks={t['id']:t for t in plan['tasks']}
    root,report=open_results(OUT/'inference/results.json'); geometry=read_json(OUT/'geometry.json')
    registry=read_json(ROOT/'datasets/cad-sources/ash-v1/cad-sources.json'); checked=[]
    for variant in registry['variants']:
        original=Path(variant['mesh_check']['path']); require(sha256(original)==variant['mesh_check']['sha256'],'Original download STL changed')
        checked.append({'code':variant['code'],'path':str(original),'sha256':sha256(original)})
    write_json(OUT/'original-source-check.json',{'matched':checked,'count':len(checked)})
    records={}; appearance=[]; errors=[]
    for record in report['records']:
        identity=Path(record['name']).stem; task=tasks[identity]; q=read_json(OUT/'quality'/(identity+'.json'))
        manifest,result,_=load_snapshot(root/record['snapshot'],expected_digest=record['snapshot_digest'])
        require(all(o['final_decision']=='REVIEW' for o in result['objects']),'Unexpected product approval')
        detections=[{'label':'ASH','box':[o['effective_box'][k] for k in ('x1','y1','x2','y2')],
                     'score':o['detection']['score']} for o in result['objects']]
        findings=[{'label':f['defect_code'],'score':f['score'],'box':[f['original_box'][k] for k in ('x1','y1','x2','y2')]}
                  for o in result['objects'] for c in o['checks'] for f in c['findings']]
        for obj in result['objects']:
            for check in obj['checks']:
                if check['status']=='ERROR': errors.append({'image':identity,'check':check['check_id'],'messages':check['messages']})
                for finding in check['findings']:
                    crop=finding['crop_box']; original=finding['original_box']; bounds=obj['crop']['bounds_original_xyxy']
                    require(original==dict(x1=crop['x1']+bounds['x1'],y1=crop['y1']+bounds['y1'],
                        x2=crop['x2']+bounds['x1'],y2=crop['y2']+bounds['y1']),'Wrong defect coordinate shift')
        records[identity]={'file_name':identity,'palette':task['palette'],'domain':task['domain'],
            'object_targets':[{'label':'ASH','box':xyxy(o['bbox'])} for o in q['objects']],
            'object_predictions':detections,'targets':[{'label':f['code'],'box':xyxy(f['bbox'])} for f in q['defects']],
            'predictions':findings,'association_confirmed':record['association_confirmed']}
        rgb=np.asarray(Image.open(OUT/'raw'/(identity+'.png')).convert('RGB'),dtype=float)
        if task['domain']=='detail':
            obj=task['objects'][0]; mask=np.isfinite(raster_depth(world_triangles(geometry[obj['code']],matrix(obj)),task['camera']))
            visible=rgb[mask]; stats={'image':identity,'palette':task['palette'],'object_median_rgb':np.median(visible,axis=0).tolist(),
                'near_white_pixel_fraction':float(np.mean(np.all(visible>=250,axis=1))),'pixel_threshold_note':'display RGB only, not sensor saturation measurement'}
            if obj['code']!='OK':
                normal=np.asarray(Image.open(OUT/'raw'/(task['group']+'-OK.png')).convert('RGB'),dtype=float)
                delta=np.asarray(Image.open(OUT/'masks'/(identity+'.png')))>0
                stats['geometry_region_mean_rgb_difference']=float(np.abs(rgb-normal).mean(axis=2)[delta].mean())
            appearance.append(stats)
    summaries={}
    for palette in plan['palettes_srgb']:
        subset=[r for r in records.values() if r['palette']==palette]
        details=[r for r in subset if r['domain']=='detail']
        overview=[dict(file_name=r['file_name'],targets=r['object_targets'],predictions=r['object_predictions']) for r in subset if r['domain']=='overview']
        objects=[dict(file_name=r['file_name'],targets=r['object_targets'],predictions=r['object_predictions']) for r in details]
        summaries[palette]={'overview':summarize(overview,.05,['ASH']),'detail_objects':summarize(objects,.2,['ASH']),
                           'defects':summarize(details,.1,[f'NG{i:02d}' for i in range(1,7)]),
                           'unconfirmed_details':sum(not r['association_confirmed'] for r in details)}
    write_json(OUT/'evaluation.json',{'palette_results':summaries,'errors':errors,'photographs':len(records),
        'appearance':appearance,'scope':'Known CAD; three matched camera/light conditions repeated across four hypothetical palettes.',
        'not_independent_specimens':True,'thresholds_retuned':False,'training_performed':False,'physical_performance_confirmed':False})
    write_json(OUT/'predictions.json',list(records.values()))
    require(not errors,'Inference errors present')
    def font(size): return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size)
    def tile(task,width,height,annotated=False):
        identity=task['id']; img=Image.open(OUT/'raw'/(identity+'.png')).convert('RGB'); q=read_json(OUT/'quality'/(identity+'.json'))
        if annotated:
            draw=ImageDraw.Draw(img)
            for row in records[identity]['targets']:
                draw.rectangle(row['box'],outline='#14a470',width=3)
            for row in records[identity]['predictions']:
                draw.rectangle(row['box'],outline='#db8200',width=3)
                draw.text((row['box'][0],max(0,row['box'][1]-18)),f"{row['label']} {row['score']:.2f}",font=font(15),fill='#a65f00')
        if task['domain']=='detail':
            x,y,w,h=q['objects'][0]['bbox']; img=img.crop((max(0,x-30),max(0,y-30),min(1280,x+w+30),min(720,y+h+30)))
        img.thumbnail((width,height)); return img
    palettes=list(plan['palettes_srgb'])
    board=Image.new('RGB',(1440,780),'#e9edf1'); draw=ImageDraw.Draw(board)
    draw.text((12,10),'SYNTHETIC white parts | hypothetical olive shades and charcoal control | identical lights per row',font=font(18),fill='black')
    for col,palette in enumerate(palettes):
        for row,key in enumerate((f'{palette}-o7',f'{palette}-d1-OK')):
            x=360*col; y=45+row*360
            draw.text((x+8,y+4),palette+' '+plan['palettes_srgb'][palette],font=font(16),fill='black')
            img=tile(tasks[key],350,315); board.paste(img,(x+(360-img.width)//2,y+35+(315-img.height)//2))
    board.save(OUT/'review/palette-comparison.png')
    board=Image.new('RGB',(7*280,3*320+48),'#e9edf1'); draw=ImageDraw.Draw(board)
    draw.text((12,10),'SYNTHETIC olive-medium | 3 camera/light conditions | GREEN: CAD difference / ORANGE: AI candidate',font=font(19),fill='black')
    for row in range(3):
        for col,code in enumerate(['OK',*[f'NG{i:02d}' for i in range(1,7)]]):
            task=tasks[f'olive-medium-d{row}-{code}']; x=280*col; y=48+row*320
            draw.text((x+8,y+3),f'Condition {row+1} / {code}',font=font(17),fill='black')
            img=tile(task,268,279,True); board.paste(img,(x+(280-img.width)//2,y+32+(279-img.height)//2))
    board.save(OUT/'review/defect-predictions.png')
    lines=['# 출력 전 STL·올리브그린 바닥 점검','',
        '흰 물체와 올리브그린 바닥을 가정한 합성 검토다. 필라멘트를 측정한 색이 아니며 출력·실물 성능을 승인한 결과가 아니다.','',
        '## STL 확인','',
        '- Downloads/OKNG의 원본 7개와 등록한 STL의 해시가 모두 일치한다. 원본과 학습 모델은 수정하지 않았다.',
        '- 7종 모두 좌표 범위 X×Y×Z=16×14×20이다. 슬라이서에서 단위를 mm로 해석하고 100%로 불러올 경우의 치수이며, 검사면은 X×Z=16×20, 높이는 Y=14이다.',
        '- +Y에서 내려다보는 동일 검사면에 NG01~NG06의 형상 차이가 모두 있다. 기본 모서리 연결 검사와 면적 0 삼각형 검사는 통과했다. 자기 교차, 최소 벽 두께, 슬라이서 경로 및 출력 재현성은 확인하지 않았다.','',
        '| 종류 | 검사면에서 보이는 변화 | 변화 영역의 투영 범위 X×Z (좌표 단위, 근사) |','|---|---|---|']
    names={'NG01':'아래쪽 원형 보스 누락','NG02':'왼쪽 위 돌출부 추가','NG03':'바닥면 균열 모양 홈','NG04':'왼쪽 벽 변형','NG05':'위쪽 원형 구멍 막힘','NG06':'바닥면 원형 함몰'}
    for asset in read_json(OUT/'cad-audit.json')['assets']:
        if asset['code']=='OK': continue
        a,b=asset['inspection_face_positive_Y']['projected_change_extent_XZ_coordinate_units']
        lines.append(f"| {asset['code']} | {names[asset['code']]} | {a:.2f} × {b:.2f} |")
    lines += ['', '위 범위는 변경된 부분 전체의 바운딩 박스다. 균열의 최소 폭이나 실제 벽 두께가 아니다. NG03의 홈과 NG06 함몰은 이 방향의 표면 높이 차이가 약 1좌표 단위다. 균열 끝의 가늘어지는 부분과 좁은 NG04 변형은 출력 후 별도 확인해야 한다.','',
        '## 합성 촬영과 기존 모델 결과','',
        '올리브 3색과 비교용 차콜에 동일한 흰 물체를 놓았다. 색별 상세 21장(7종×촬영 조건3개), 전체 2장(빈 장면/7개 배치), 총92장이다. 동일 CAD와 조건을 색만 바꿔 반복하므로 독립 실물 표본92개가 아니다.',
        '전체 위치·상세 물체·불량 모델은 기존 것을 그대로 사용했다. 후보 기준 .05/.20/.10과 IoU 0.5를 유지하고 이 결과로 재선정하지 않았다. 최종 판정은 실물 기준이 없어 보류이며 아래 숫자는 불량 후보 검출 결과다.','',
        '| 가상 바닥 | 전체 물체 검출 | 상세 물체 검출 | 불량 일치 / 정답18 | 추가 후보 | 놓침 | 정상3장 중 후보 발생 |','|---|---|---|---|---|---|---|']
    for palette,s in summaries.items():
        o=s['overview']['total']; d=s['detail_objects']['total']; f=s['defects']['total']
        lines.append(f"| {palette} {plan['palettes_srgb'][palette]} | {o['tp']}/7 (추가{o['fp']}) | {d['tp']}/21 (추가{d['fp']}) | {f['tp']}/18 | {f['fp']} | {f['fn']} | {s['defects']['false_alarm_images']}/3 |")
    lines += ['', '모든 생성 사진을 평가에 포함했다. 보이지 않거나 대비가 낮은 사례를 골라 제외하지 않았다. 실제 필라멘트의 적층 결, 광택, 치수 오차, 카메라 초점·왜곡 및 실제 조명은 아직 재현·확인되지 않았다.', '',
        '## 결과 파일','', '- `review/cad-face.png`: 검사면과 형상 차이. 파란색은 CAD 진단 표시이며 제품 색이 아니다.',
        '- `review/palette-comparison.png`: 바닥색 비교. 16진수는 가정한 sRGB 재질색이며 조명 아래 렌더 픽셀색과 다르다.',
        '- `review/defect-predictions.png`: 올리브 중간색의 3개 촬영 조건. 초록은 CAD 형상 차이 정답, 주황은 모델 후보.',
        '- `inference/results.json`: 프로그램의 품목 설정 → 저장사진 검사 → 저장 결과 열기에서 조회 가능.',
        '- `evaluation.json` 및 `predictions.json`: 누락·오검출 포함 전체 수치.', '',
        '프린터·노즐·레이어 높이가 미정이면 출력 가능 여부의 최종 판단은 보류한다. 먼저 OK, NG03, NG04, NG06을 소량 출력하고 실제 U20CAM 근접 촬영에서 확인하는 것이 다음 단계다.']
    (OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print({p:s['defects']['total'] for p,s in summaries.items()},flush=True)


if __name__=='__main__': main()

"""Per-class paired diagnostics; all images overlap training, not validation."""
from pathlib import Path
import json, html
from compare_recorded_models import assess
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'artifacts/wall-model-comparison-v1'

def main():
    models={k:json.loads((OUT/(k+'-predictions.json')).read_text()) for k in ('before_wall','after_wall')}
    assert all(len(rows)==2256 for rows in models.values())
    rows=[]
    for name,results in models.items():
        for scope in ('original','all_transforms'):
            selected=[r for r in results if scope!='original' or r['condition']=='rot0_original']
            for threshold in (.1,.2,.3,.5):
                for code in ('OK',*[f'NG{i:02}' for i in range(1,7)]):
                    group=[r for r in selected if (not r['targets'] if code=='OK' else code in {t['label'] for t in r['targets']})]
                    metric=dict(model=name,scope=scope,threshold=threshold,code=code,images=len(group),
                                any_false_on_ok=0,ng04_false_on_ok=0,class_missed=0,other_class=0,localization_miss=0)
                    for r in group:
                        pred=[p for p in r['predictions'] if p['score']>=threshold]
                        labels={p['label'] for p in pred}
                        if code=='OK':
                            metric['any_false_on_ok']+=bool(pred)
                            metric['ng04_false_on_ok']+='NG04' in labels
                        else:
                            metric['class_missed']+=code not in labels
                            metric['other_class']+=any(label!=code for label in labels)
                            metric['localization_miss']+=assess(pred,r['targets'],threshold)['localization_miss']
                    rows.append(metric)
    (OUT/'class-summary.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    headings='<tr><th>조건</th><th>임계값</th><th>종류</th><th>모델</th><th>사진 수</th><th>OK 전체 오탐</th><th>OK→NG04</th><th>정답 종류 미검출</th><th>다른 종류 검출</th><th>위치 미일치</th></tr>'
    body=''.join('<tr>'+''.join(f'<td>{html.escape(str(r[k]))}</td>' for k in ('scope','threshold','code','model','images','any_false_on_ok','ng04_false_on_ok','class_missed','other_class','localization_miss'))+'</tr>' for r in sorted(rows,key=lambda r:(r['scope'],r['threshold'],r['code'],r['model'])))
    (OUT/'class-report.html').write_text('<meta charset="utf-8"><style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse}th,td{border:1px solid #bbb;padding:6px}th{position:sticky;top:0;background:#ddd}</style><h1>NG04 벽 라벨 보완 전후 · 종류별 비교</h1><p>전부 학습에 사용한 사진입니다. 변형 조건도 독립 표본이 아니며 실제 촬영 정확도를 뜻하지 않습니다. 동일한 수정 라벨을 두 모델에 사용했습니다. 위치 미일치는 IoU 0.5 기준이며 라벨 자체도 시각 검토 근사 박스입니다. NG04 미검출이 곧 프로그램의 최종 OK 판정을 뜻하지는 않습니다.</p><p><a href="report.html">오류 이미지 및 종합 결과</a></p><table>'+headings+body+'</table>',encoding='utf-8')
    for r in rows:
        if r['scope']=='original' and r['threshold']==.1: print(r)

if __name__=='__main__':main()

"""Evaluate fixed unseen clip samples through manual-inspection pipeline. No training."""
from pathlib import Path
import sys,os,json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1'
from mes_vision.station.photo_inspection import run_photos
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.training.data import sha256
OUT=ROOT/'artifacts/heldout-ok-ng04-20260916'

def main():
    selection=json.loads((OUT/'selection.json').read_text(encoding='utf-8'))
    # Visual exclusions fixed before reading model results; retain all frames in raw report.
    exclusions={'OK':{1:'no object',13:'object truncated',17:'hand',23:'hand',31:'hand'},
                'NG04':{1:'object truncated',15:'object truncated',19:'hand occlusion',27:'hand occlusion',31:'hand'}}
    rows=[]
    for name,bundle in [('before','ash-recorded-baseline-v1'),('after','ash-wall-reviewed-v2')]:
        dest=OUT/name
        request=dict(output=str(dest),images=[{k:r[k] for k in ('path','sha256','role')} for r in selection['images']],
                     image_kind='real',bundle=str(ROOT/'configs/inspection'/f'{bundle}.json'),runtime=str(ROOT/'artifacts/operation'))
        if not (dest/'results.json').exists():run_photos(request,ROOT)
        report=json.loads((dest/'results.json').read_text())
        for record,source in zip(report['records'],selection['images']):
            assert record['source_sha256']==source['sha256']
            _,result,_=load_snapshot(dest/record['snapshot'],expected_digest=record['snapshot_digest'])
            findings=[f for o in result['objects'] for c in o['checks'] for f in c['findings']]
            row=dict(model=name,**source,association=record['association_confirmed'],objects=record['objects'],
                     exclusion=exclusions[source['label']].get(round(source['time'])),findings=findings)
            rows.append(row)
    (OUT/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
    summaries=[]
    for name in ('before','after'):
        for label in ('OK','NG04'):
            group=[r for r in rows if r['model']==name and r['label']==label and not r['exclusion']]
            for threshold in (.1,.2,.3,.5):
                item=dict(model=name,label=label,threshold=threshold,frames=len(group),association_failures=0,any_defect=0,ng04_detected=0,other_defect=0)
                for r in group:
                    item['association_failures']+=not r['association']
                    codes={f['defect_code'] for f in r['findings'] if f.get('score',0)>=threshold}
                    item['any_defect']+=bool(codes);item['ng04_detected']+='NG04' in codes
                    item['other_defect']+=bool(codes-{'NG04'})
                summaries.append(item)
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
    for row in summaries: print(row)

if __name__=='__main__':main()

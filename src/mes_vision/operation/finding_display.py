"""One presentation contract for original-image and cropped defect evidence."""
from mes_vision.i18n import tr, trf

DEFECT_NAMES = {'NG01':'누락', 'NG02':'돌출·버', 'NG03':'균열', 'NG04':'변형',
                'NG05':'구멍 불량', 'NG06':'표면 불량', 'NG_UNKNOWN':'미지 불량'}


def finding_caption(finding):
    code=finding.get('defect_code') or ''
    name=tr(DEFECT_NAMES[code]) if code in DEFECT_NAMES else finding['label']
    label=code+' · '+name if code else name
    if finding.get('score') is not None:
        label=label+trf(' · 점수 {score:.3f}',score=finding['score'])
    return label


def object_overlays(obj, *, identity=None, crop=False):
    """Findings have parent selection IDs; clicking one still selects its object."""
    identity=identity or obj['object_id']; offset=(0,0)
    if crop:
        b=obj['crop']['bounds_original_xyxy']; offset=(b['x1'],b['y1'])
    def coords(b):
        return [b['x1']-offset[0],b['y1']-offset[1],b['x2']-offset[0],b['y2']-offset[1]]
    rows=[{'track_id':identity,'status':obj.get('final_decision') or 'REVIEW',
           'box':coords(obj['effective_box'])}]
    for check in obj['checks']:
        assessment=next((r for r in obj.get('decision_details',{}).get('checks',[])
                         if r['check_id']==check['check_id']),{})
        for finding in check['findings']:
            if finding.get('original_box'):
                confirmed=(check['status']=='FAIL' and finding.get('defect_code') in assessment.get('defect_codes',[]))
                rows.append({'track_id':identity,'box':coords(finding['original_box']),
                             'status':'NG' if confirmed else 'REVIEW',
                             'label':finding_caption(finding), 'finding':True})
    return rows

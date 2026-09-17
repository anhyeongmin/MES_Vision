"""One presentation contract for original-image and cropped defect evidence."""
from mes_vision.i18n import tr, trf, text_join
from mes_vision.inspection.finding_scores import score_band, visible_findings, hidden_count
from html import escape

BAND_LABELS={'DETECTED':'검출','SUSPECT':'의심'}
BAND_COLORS={'DETECTED':'#e45b65','SUSPECT':'#d6a52b'}

DEFECT_NAMES = {'NG01':'누락', 'NG02':'돌출·버', 'NG03':'균열', 'NG04':'변형',
                'NG05':'구멍 불량', 'NG06':'표면 불량', 'NG_UNKNOWN':'미지 불량'}


def finding_caption(finding):
    code=finding.get('defect_code') or ''
    name=tr(DEFECT_NAMES[code]) if code in DEFECT_NAMES else finding['label']
    label=code+' · '+name if code else name
    if finding.get('score') is not None:
        label=label+trf(' · 점수 {score:.3f}',score=finding['score'])
    band=score_band(finding)
    if band in BAND_LABELS:label='['+tr(BAND_LABELS[band])+'] '+label
    return label


def finding_summary(objects):
    groups={'DETECTED':set(),'SUSPECT':set(),'UNSCORED':set()}
    for obj in objects:
        for finding in visible_findings(obj):
            if finding.get('defect_code'):groups[score_band(finding)].add(finding['defect_code'])
    return text_join(' · ',((tr(BAND_LABELS[band])+': ' if band in BAND_LABELS else '')+', '.join(sorted(codes))
                      for band,codes in groups.items() if codes))


def finding_details_text(lines, objects):
    result=list(lines)
    findings=[f for obj in objects for f in visible_findings(obj)]
    for band in ('DETECTED','SUSPECT','UNSCORED'):
        result.extend(finding_caption(f) for f in findings if score_band(f)==band)
    if not findings:result.append(tr('표시할 불량 없음 · 정상 확정은 아닙니다.'))
    count=sum(hidden_count(obj) for obj in objects)
    if count:result.append(trf('표시 기준 미만 {count}건 · 원본 기록 보존',count=count))
    result.append(tr('검출 ≥ 0.5 · 의심 0.2~0.5 미만 · 최종 판정과 별도'))
    return text_join('\n',result)


def finding_details_html(lines, objects):
    html=[escape(str(line)) for line in lines]
    findings=[f for obj in objects for f in visible_findings(obj)]
    for band in ('DETECTED','SUSPECT','UNSCORED'):
        for f in findings:
            if score_band(f)==band:
                html.append('<span style="color:'+BAND_COLORS.get(band,'inherit')+'">'+escape(str(finding_caption(f)))+'</span>')
    if not findings:html.append(escape(str(tr('표시할 불량 없음 · 정상 확정은 아닙니다.'))))
    count=sum(hidden_count(obj) for obj in objects)
    if count:html.append(escape(str(trf('표시 기준 미만 {count}건 · 원본 기록 보존',count=count))))
    html.append(escape(str(tr('검출 ≥ 0.5 · 의심 0.2~0.5 미만 · 최종 판정과 별도'))))
    return '<br>'.join(html)


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
            if score_band(finding)=='HIDDEN':continue
            if finding.get('original_box'):
                confirmed=(check['status']=='FAIL' and finding.get('defect_code') in assessment.get('defect_codes',[]))
                rows.append({'track_id':identity,'box':coords(finding['original_box']),
                             'status':'NG' if confirmed else 'REVIEW',
                             'label':finding_caption(finding), 'finding':True,
                             'display_color':BAND_COLORS.get(score_band(finding))})
    return rows

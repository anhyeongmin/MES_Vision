"""Presentation/VLM bands, not calibrated probabilities or verdict thresholds."""
import math

RULE = {'version':'finding-display-v1', 'visible_min':0.2, 'detected_min':0.5,
        'validated_for_verdict':False}


def score_band(finding):
    if finding.get('defect_code') not in {f'NG{i:02d}' for i in range(1,7)}:
        return 'UNSCORED'
    score=finding.get('score')
    if score is None:return 'SUSPECT'
    if type(score) not in (int,float) or not math.isfinite(score) or not 0<=score<=1:
        raise ValueError('Invalid defect display score')
    if score < RULE['visible_min']:return 'HIDDEN'
    return 'DETECTED' if score >= RULE['detected_min'] else 'SUSPECT'


def visible_findings(obj):
    return [f for c in obj['checks'] for f in c['findings'] if score_band(f)!='HIDDEN']


def hidden_count(obj):
    return sum(score_band(f)=='HIDDEN' for c in obj['checks'] for f in c['findings'])

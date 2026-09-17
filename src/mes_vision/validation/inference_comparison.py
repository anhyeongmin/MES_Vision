"""Execution parity, not accuracy: class-aware one-to-one matching against baseline outputs."""
from collections import defaultdict
import math
from mes_vision.inspection.contracts import Box
from mes_vision.validation.metrics import match_boxes


def compare_predictions(baseline, candidate, *, threshold=.4, min_iou=.5):
    if set(baseline)!=set(candidate): raise ValueError('Comparison image sets differ')
    if not 0<=threshold<=1: raise ValueError('Invalid score threshold')
    counts={'baseline':0,'candidate':0,'matched':0,'unmatched_baseline':0,'unmatched_candidate':0}
    coordinates=[]; scores=[]; overlaps=[]; per_image={}
    def groups(rows):
        result=defaultdict(list)
        for row in rows:
            score=row['score']; class_id=row['class_id']
            if not math.isfinite(score) or not 0<=score<=1 or type(class_id) is not int: raise ValueError('Invalid prediction')
            box=Box(**row['box'])
            if score>=threshold: result[class_id].append((row,[box.x1,box.y1,box.x2,box.y2]))
        return result
    for image in baseline:
        a,b=groups(baseline[image]),groups(candidate[image]); totals={key:0 for key in counts}
        for class_id in set(a)|set(b):
            left,right=a[class_id],b[class_id]
            matches=match_boxes([box for _,box in left],[box for _,box in right],min_iou)
            totals['baseline']+=len(left); totals['candidate']+=len(right); totals['matched']+=len(matches)
            totals['unmatched_baseline']+=len(left)-len(matches); totals['unmatched_candidate']+=len(right)-len(matches)
            for i,j,overlap in matches:
                coordinates.append(max(abs(x-y) for x,y in zip(left[i][1],right[j][1])))
                scores.append(abs(left[i][0]['score']-right[j][0]['score'])); overlaps.append(overlap)
        for key,value in totals.items(): counts[key]+=value
        per_image[image]=totals
    return dict(counts,threshold=threshold,min_matching_iou=min_iou,
        max_coordinate_delta_px=max(coordinates,default=None),max_score_delta=max(scores,default=None),
        minimum_matched_iou=min(overlaps,default=None),per_image=per_image,
        accuracy_assessed=False)

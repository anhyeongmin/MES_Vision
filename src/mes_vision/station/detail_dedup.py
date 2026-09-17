"""Conservative, audited suppression of near-identical detail object boxes."""
from dataclasses import asdict, replace


def deduplicate_detail(batch):
    kept=[]; removed=[]
    for index,d in sorted(enumerate(batch.detections),key=lambda v:(-v[1].score,v[0])):
        match=None
        for kept_index,k in kept:
            if (d.class_id,d.label)!=(k.class_id,k.label):continue
            a,b=d.box,k.box
            intersection=max(0,min(a.x2,b.x2)-max(a.x1,b.x1))*max(0,min(a.y2,b.y2)-max(a.y1,b.y1))
            aa=(a.x2-a.x1)*(a.y2-a.y1);bb=(b.x2-b.x1)*(b.y2-b.y1)
            iou=intersection/(aa+bb-intersection)
            if iou>=.90 and min(aa,bb)/max(aa,bb)>=.90:
                match=(kept_index,iou);break
        if match is None:kept.append((index,d))
        else:removed.append(dict(index=index,kept_index=match[0],iou=match[1],detection=asdict(d)))
    audit=dict(method='detail_near_duplicate_v1',iou_threshold=.90,area_ratio_min=.90,
               original_count=len(batch.detections),kept_indices=[i for i,_ in sorted(kept)],removed=removed)
    return replace(batch,detections=tuple(d for _,d in sorted(kept))),audit

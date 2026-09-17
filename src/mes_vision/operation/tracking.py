from dataclasses import dataclass, field
import math
from uuid import uuid4
import cv2
import numpy as np


def overlap(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    return area / max(1., (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def signature(rgb, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    crop = rgb[max(0,y1):min(rgb.shape[0],y2), max(0,x1):min(rgb.shape[1],x2)]
    if crop.size == 0: return np.zeros((12, 12, 3), np.float32)
    return cv2.resize(crop, (12,12), interpolation=cv2.INTER_AREA).astype(np.float32)/255.


@dataclass(eq=False)
class Track:
    id: str
    box: tuple
    class_id: int
    anchor: tuple
    appearance: np.ndarray
    stable_since: float
    last_seen: float
    revision: int = 1
    inspected_revision: int = 0
    status: str = "WAITING"
    result: dict | None = None
    missing: bool = False
    observed_frames: int = 1

    def invalidate(self, now, state="WAITING"):
        self.revision += 1; self.inspected_revision = 0; self.result = None
        self.stable_since = now; self.observed_frames = 1; self.status = state

    def data(self):
        return {"track_id": self.id, "box": list(self.box), "class_id": self.class_id, "revision": self.revision,
                "status": self.status, "result": self.result, "missing": self.missing,
                "last_seen": self.last_seen, "stable_since": self.stable_since}


class Tracker:
    """Conservative fixed-camera association; ambiguity never transfers an inspection."""
    def __init__(self, *, session=None, stable_seconds=.6, max_gap=.5, motion_px=4., appearance_delta=.10):
        self.session = session or uuid4().hex; self.sequence = 0; self.tracks = {}
        self.stable_seconds=stable_seconds; self.max_gap=max_gap; self.motion_px=motion_px
        self.appearance_delta=appearance_delta; self.events=[]; self.last_time=None

    def update(self, detections, rgb, now):
        if self.last_time is not None and now < self.last_time: raise ValueError("추적 시각이 역전됐습니다.")
        if self.last_time is not None and now-self.last_time > self.max_gap:
            for t in self.tracks.values(): t.invalidate(now, "RECHECK")
        self.last_time=now; self.events=[]
        boxes = [(d.box.x1,d.box.y1,d.box.x2,d.box.y2) for d in detections]
        candidates=[]
        for i, box in enumerate(boxes):
            matches=[t for t in self.tracks.values() if t.class_id==detections[i].class_id
                     and now-t.last_seen <= self.max_gap and overlap(t.box,box) >= .35]
            candidates.append(matches)
        ambiguous={i for i, matches in enumerate(candidates) if len(matches)>1}
        for t in self.tracks.values():
            hits=[i for i,m in enumerate(candidates) if t in m]
            if len(hits)>1: ambiguous.update(hits)
        # Even if assignment looks unique, overlapping objects are not an inspection opportunity.
        occluded={i for i,b in enumerate(boxes) if any(j!=i and overlap(b,c)>0 for j,c in enumerate(boxes))}
        assigned=[]; seen=set()
        for i,(d,b) in enumerate(zip(detections,boxes)):
            feature=signature(rgb,b)
            if i in ambiguous:
                for old in candidates[i]:
                    old.invalidate(now,"UNCERTAIN"); old.missing=True
                assigned.append(None); continue
            t=candidates[i][0] if candidates[i] else None
            if t is None:
                self.sequence+=1
                t=Track(f"{self.session}:T{self.sequence:06d}",b,d.class_id,b,feature,now,now)
                self.tracks[t.id]=t
            else:
                displacement=max(abs(a-c) for a,c in zip(b,t.anchor))
                changed=float(np.mean(np.abs(feature-t.appearance))) > self.appearance_delta
                if t.missing or displacement>self.motion_px or changed:
                    t.invalidate(now,"RECHECK"); t.anchor=b; t.appearance=feature
                else: t.observed_frames+=1
                t.box=b; t.last_seen=now
            t.missing=False
            if i in occluded:
                t.invalidate(now,"OCCLUDED")
            elif t.inspected_revision != t.revision:
                t.status="STABLE" if now-t.stable_since>=self.stable_seconds and t.observed_frames>=3 else "WAITING"
            seen.add(t.id); assigned.append(t)
        for identity,t in list(self.tracks.items()):
            if identity not in seen:
                if not t.missing: t.invalidate(now,"LOST"); t.missing=True
                if now-t.last_seen > self.max_gap:
                    self.events.append({"track_id":identity,"event":"REMOVED"}); del self.tracks[identity]
        return assigned

    def invalidate_all(self, now):
        for t in self.tracks.values(): t.invalidate(now,"RECHECK")

    def accept(self, identity, revision, result):
        t=self.tracks.get(identity)
        if not t or t.missing or t.revision!=revision or t.status not in {"STABLE","INSPECTING"}: return False
        t.inspected_revision=revision; t.result=result; t.status=result["decision"]
        return True

    def recheck(self, identity, now):
        if identity not in self.tracks: raise ValueError("현재 추적 중인 물체가 아닙니다.")
        self.tracks[identity].invalidate(now,"RECHECK")

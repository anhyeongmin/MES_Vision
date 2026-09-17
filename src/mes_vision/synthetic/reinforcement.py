"""Paired CAD conditions and label-preserving crop perturbations."""
import math
import random
from .planning import CODES, camera, condition, fingerprint

SEED = 2026091203
FOCUS = {"OK", "NG01", "NG04"}


def make_plan(groups=(48, 18, 18), seed=SEED):
    if len(groups) != 3 or any(type(n) is not int or n < 1 for n in groups):
        raise ValueError("Three positive split sizes required")
    tasks = []
    for split, count in zip(("train", "valid", "test"), groups):
        rng = random.Random(f"ash-reinforcement:{seed}:{split}")
        for i in range(count):
            identity = f"reinforce-{split}-d{i:04d}"
            z = rng.uniform(.038, .062)
            c = condition(rng, identity, z)
            c["angle"] = (i % 16)*22.5 + rng.uniform(-8, 8)
            stratum = ("light_neutral", "dark_neutral", "broad_color")[i % 3]
            if stratum != "broad_color":
                value = rng.uniform(.42, .72) if stratum == "light_neutral" else rng.uniform(.035, .13)
                c["color"] = [value]*3 + [1]
                background = rng.uniform(.16, .38) if stratum == "light_neutral" else rng.uniform(.025, .16)
                c["background"] = [background]*3 + [1]
            cam = camera([rng.uniform(-.0025, .0025), rng.uniform(-.0025, .0025), z])
            for code in CODES:
                tasks.append(dict(id=identity+"-"+code, group=identity, split=split, domain="detail",
                    condition=c, camera=cam, condition_stratum=stratum,
                    objects=[dict(code=code, x=0, y=0, angle=c["angle"])]))
    return dict(schema_version=1, seed=seed, tasks=tasks, split_unit="render_condition_group",
        cad_split="all seven CAD identities shared across splits",
        evaluation_scope="new rendering conditions of known CAD only; never independent physical specimens",
        camera_target="INNOMAKER U20CAM-720P", camera_approximation="1280x720 nominal HFOV102 pinhole",
        units="original coordinates assumed mm; physical scale and focus unconfirmed",
        previous_test_policy="old validation/test images excluded; old test is now development evidence",
        focus_codes=sorted(FOCUS), new_groups=dict(zip(("train", "valid", "test"), groups)))


def xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]


def union_box(boxes):
    if not boxes:
        raise ValueError("Object boxes required")
    b = [xyxy(v) for v in boxes]
    return [min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b)]


def acceptable(bounds, object_boxes, defect_boxes, width=1280, height=720):
    x1, y1, x2, y2 = bounds
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        return False
    for box in object_boxes:
        a, b, c, d = xyxy(box)
        area = max(0, min(x2,c)-max(x1,a))*max(0, min(y2,d)-max(y1,b))
        if area / (box[2]*box[3]) < .95:
            return False
    # Check all six counterpart defect locations, including when producing OK crops.
    for box in defect_boxes:
        a, b, c, d = xyxy(box)
        if not (a >= x1+4 and b >= y1+4 and c <= x2-4 and d <= y2-4):
            return False
    return True


def crop_variant(group_id, index, object_boxes, defect_boxes, *, stress=False):
    base = union_box(object_boxes)
    cx, cy = (base[0]+base[2])/2, (base[1]+base[3])/2
    w, h = base[2]-base[0], base[3]-base[1]
    rng = random.Random(f"{SEED}:{group_id}:{index}:{stress}")
    for attempt in range(200):
        if stress:
            dx, dy, scale = ((-.04,.04,1.16),(.04,-.04,1.16),(0,0,1.25))[index]
        else:
            shift = (.04, .08, .08, .10)[index]
            dx, dy = rng.uniform(-shift,shift), rng.uniform(-shift,shift)
            scale = rng.uniform(.96, 1.12) if index == 0 else rng.uniform(.94, 1.24) if index == 1 else rng.uniform(1.0,1.30)
        bounds = [math.floor(cx+dx*w-w*scale/2), math.floor(cy+dy*h-h*scale/2),
                  math.ceil(cx+dx*w+w*scale/2), math.ceil(cy+dy*h+h*scale/2)]
        if acceptable(bounds, object_boxes, defect_boxes):
            return dict(bounds=bounds, dx_fraction=dx, dy_fraction=dy, scale=scale, attempt=attempt)
        if stress:
            break
    raise ValueError("No crop preserves paired defect context and object coverage")

"""Deterministic scene groups with honest shared-CAD lineage and split isolation."""
import hashlib
import json
import math
import random
import numpy as np
from .projection import world_triangles,raster_depth,difference_mask,bbox

CODES=['OK']+[f'NG{i:02d}' for i in range(1,7)]


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


def matrix(obj):
    a=math.radians(obj['angle']); c,s=math.cos(a),math.sin(a)
    return [[c,-s,0,obj['x']],[s,c,0,obj['y']],[0,0,1,0],[0,0,0,1]]


def camera(position):
    f=1280/(2*math.tan(math.radians(102)/2))
    return dict(width=1280,height=720,fx=f,fy=f,cx=640,cy=360,position=position,
        orientation='vertical_down_world_minus_Z')


def condition(rng,identity,z):
    hue=[rng.uniform(.035,.6) for _ in range(3)]
    if rng.random()<.5: hue=[rng.uniform(.07,.6)]*3
    bg=[rng.uniform(.025,.3) for _ in range(3)]
    if rng.random()<.65: bg=[rng.uniform(.03,.27)]*3
    a=rng.uniform(0,2*math.pi); radius=rng.uniform(.025,.065)
    return dict(id=identity,angle=rng.uniform(-180,180),camera_z=z,color=hue+[1],background=bg+[1],
        light=[radius*math.cos(a),radius*math.sin(a),rng.uniform(.05,.085)],
        energy=rng.uniform(.18,.75),roughness=rng.uniform(.45,.9),seed=rng.randrange(1,2**30))


def make_plan(seed=20260911,groups=(42,9,9),overviews=(84,18,18)):
    if len(groups)!=3 or len(overviews)!=3 or any(type(n)!=int or n<1 for n in (*groups,*overviews)):
        raise ValueError('Three positive split sizes required')
    rng=random.Random(seed); tasks=[]
    for split,n_detail,n_overview in zip(('train','valid','test'),groups,overviews):
        for i in range(n_detail):
            group=f'{split}-d{i:04d}'; z=rng.uniform(.036,.060)
            c=condition(rng,group,z); x,y=rng.uniform(-.0015,.0015),rng.uniform(-.0015,.0015)
            for code in CODES:
                tasks.append(dict(id=group+'-'+code,group=group,split=split,domain='detail',condition=c,
                    camera=camera([x,y,z]),objects=[dict(code=code,x=0,y=0,angle=c['angle'])]))
        for i in range(n_overview):
            group=f'{split}-o{i:04d}'; z=rng.uniform(.085,.13)
            c=condition(rng,group,z); c['energy']*=2
            # Reserved cell centres keep geometry apart; shuffled occupancy removes label/position shortcuts.
            slots=[((col-1.5)*.029,(.5-row)*.031) for row in range(2) for col in range(4)]
            rng.shuffle(slots); count=0 if i%10==0 else rng.randint(1,7)
            objects=[dict(code=rng.choice(CODES),x=x+rng.uniform(-.001,.001),y=y+rng.uniform(-.001,.001),
                          angle=rng.uniform(-180,180)) for x,y in slots[:count]]
            tasks.append(dict(id=group,group=group,split=split,domain='overview',condition=c,
                camera=camera([rng.uniform(-.002,.002),rng.uniform(-.002,.002),z]),objects=objects))
    return dict(schema_version=1,seed=seed,tasks=tasks,
        split_unit='render_condition_group',cad_split='all seven CAD identities shared across splits',
        evaluation_scope='new render conditions of known CAD only, never independent physical specimens',
        camera_target='INNOMAKER U20CAM-720P',camera_approximation='1280x720 / nominal horizontal FOV102 / undistorted pinhole',
        units='original coordinates assumed mm; physical size and focus unconfirmed')


def geometry_labels(task,geometry):
    depths=[raster_depth(world_triangles(geometry[o['code']],matrix(o)),task['camera']) for o in task['objects']]
    labels=[]; defects=[]
    if not depths: return labels,defects,None
    stack=np.stack(depths); nearest=np.argmin(stack,axis=0); seen=np.isfinite(np.min(stack,axis=0))
    for i,(obj,depth) in enumerate(zip(task['objects'],depths)):
        full=np.isfinite(depth); visible=seen&(nearest==i)
        box=bbox(visible)
        if box is None or min(box[2:])<30 or box[0]<4 or box[1]<4 or box[0]+box[2]>1276 or box[1]+box[3]>716:
            raise ValueError('Object too small or clipped')
        if visible.sum()/max(1,full.sum())<.995: raise ValueError('Object occluded')
        labels.append(dict(bbox=box,code=obj['code'],instance=i,visible_pixels=int(visible.sum())))
    mask=None
    if task['domain']=='detail':
        obj=task['objects'][0]
        normal=raster_depth(world_triangles(geometry['OK'],matrix(obj)),task['camera'])
        mask=difference_mask(depths[0],normal)
        if obj['code']=='OK':
            if mask.any(): raise ValueError('Normal has a defect mask')
        else:
            box=bbox(mask,2)
            if int(mask.sum())<80 or box is None or min(box[2:])<5: raise ValueError('Defect too small or invisible')
            defects=[dict(bbox=box,code=obj['code'],pixels=int(mask.sum()))]
    return labels,defects,mask

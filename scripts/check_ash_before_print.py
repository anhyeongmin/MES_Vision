"""Read-only CAD audit and paired olive-table/white-part rendering evaluation."""
from pathlib import Path
import argparse,copy,json,subprocess,sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src')); sys.path.insert(0,str(ROOT/'scripts'))
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.synthetic.planning import camera,CODES,geometry_labels
OUT=ROOT/'artifacts/preprint-olive-white-v1'
PALETTES={'olive-dark':'#343B24','olive-medium':'#4B532D','olive-light':'#657046','charcoal-control':'#343434'}


def linear(hex_color):
    channels=[int(hex_color[i:i+2],16)/255 for i in (1,3,5)]
    return [x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in channels]+[1]


def prepare():
    from inspect_ash_cad import mesh,render,VIEWS
    registry_path=ROOT/'datasets/cad-sources/ash-v1/cad-sources.json'; registry=read_json(registry_path)
    OUT.mkdir(parents=True,exist_ok=False)
    for name in ('raw','receipts','quality','masks','review'): (OUT/name).mkdir()
    geometry={}; originals={}; audit=[]
    for variant in registry['variants']:
        ref=next(f for f in variant['files'] if f['path'].endswith('.stl'))
        source=registry_path.parent/ref['path']; require(sha256(source)==ref['sha256'],'Registered STL changed')
        original=Path(registry['source'])/('ASH_'+variant['code'])/('ASH_'+variant['code']+'.stl')
        # The registration remains authoritative if the original download was moved.
        if original.exists(): require(sha256(original)==ref['sha256'],'Original and registered STL differ')
        triangles,record=mesh(source); code=variant['code']; record['code']=code; audit.append(record)
        originals[code]=triangles; geometry[code]=(triangles[:,:,[0,2,1]]*np.array([.001,-.001,.001])).tolist()
    points=np.concatenate([t.reshape(-1,3) for t in originals.values()]); center=(points.min(0)+points.max(0))/2
    span=float(np.ptp(points,axis=0).max()*1.18); size=768; basis=VIEWS[2][1:]
    _,normal=render(originals['OK'],basis,center,span,size=size)
    from PIL import Image,ImageDraw,ImageFont
    board=Image.new('RGB',(7*256,590),'white'); draw=ImageDraw.Draw(board)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17)
    draw.text((10,10),'CAD +Y FACE | TOP: depth shading | BOTTOM: geometric difference (red) | NOT camera images',font=font,fill='black')
    for index,record in enumerate(audit):
        code=record['code']; rgb,depth=render(originals[code],basis,center,span,size=size)
        both=np.isfinite(depth)&np.isfinite(normal); mask=np.isfinite(depth)^np.isfinite(normal)
        mask[both]|=np.abs(depth[both]-normal[both])>1e-4
        ys,xs=np.nonzero(mask); delta=np.abs(depth[both&mask]-normal[both&mask])
        record['inspection_face_positive_Y']={'changed_pixels':int(mask.sum()),'pixel_pitch_coordinate_units':span/size,
            'projected_change_extent_XZ_coordinate_units':[(xs.max()-xs.min()+1)*span/size,(ys.max()-ys.min()+1)*span/size] if len(xs) else [0,0],
            'visible_depth_difference_coordinate_units_percentiles':np.percentile(delta,[5,50,95,100]).tolist() if len(delta) else [],
            'note':'Projected change extent is not minimum wall thickness, crack width, or slicer validation.'}
        record['face_visible']=code=='OK' or bool(mask.any())
        # Depth shading distinguishes coplanar normals at different heights.
        visible=np.isfinite(depth); lo,hi=np.min(depth[visible]),np.max(depth[visible])
        rgb[visible]=(rgb[visible]*(.45+.55*(depth[visible]-lo)/max(hi-lo,1e-9))[:,None]).astype('uint8')
        changed=rgb.copy(); changed[mask]=[224,56,55]
        draw.text((index*256+8,42),code,font=font,fill='black')
        board.paste(Image.fromarray(rgb).resize((256,256)),(index*256,68))
        board.paste(Image.fromarray(changed).resize((256,256)),(index*256,326))
    board.save(OUT/'review/cad-face.png')
    write_json(OUT/'cad-audit.json',{'registry_sha256':sha256(registry_path),'coordinate_scale':registry['scale'],
        'inspection_axis':'+Y','assets':audit,'printability_confirmed':False,'mesh_modified':False})
    tasks=[]
    for palette,hex_color in PALETTES.items():
        for index,(z,angle,light,energy) in enumerate([
            (.036,0,[.034,-.025,.065],.48),(.045,35,[-.036,.012,.060],.55),(.055,95,[.015,.040,.072],.72)]):
            group=f'{palette}-d{index}'
            condition=dict(id=group,angle=angle,camera_z=z,color=[.80,.80,.80,1],background=linear(hex_color),
                light=light,energy=energy,roughness=.75,seed=202609120+index)
            for code in CODES:
                tasks.append(dict(id=group+'-'+code,group=group,split='diagnostic',domain='detail',palette=palette,
                    condition=copy.deepcopy(condition),camera=camera([0,0,z]),objects=[dict(code=code,x=0,y=0,angle=angle)]))
        for count in (0,7):
            group=f'{palette}-o{count}'; condition=copy.deepcopy(condition)
            condition.update(id=group,camera_z=.1,energy=1.1,light=[-.036,.012,.060],seed=202609130)
            objects=[dict(code=code,x=(i%4-1.5)*.029,y=(.5-i//4)*.031,angle=[0,35,80,125,-45,-95,160][i])
                     for i,code in enumerate(CODES[:count])]
            tasks.append(dict(id=group,group=group,split='diagnostic',domain='overview',palette=palette,
                condition=condition,camera=camera([0,0,.1]),objects=objects))
    plan={'schema_version':1,'tasks':tasks,'palettes_srgb':PALETTES,'purpose':'paired appearance diagnostic only; no training',
        'color_measurement':'hypothetical material colors, not measured filament','part_linear_color':[.8,.8,.8],
        'pairing':'identical geometry, camera, lights and seed across table palettes',
        'camera':'nominal U20CAM 1280x720 / HFOV102 pinhole; no focus, lens distortion or real layer-line validation'}
    write_json(OUT/'geometry.json',geometry); write_json(OUT/'plan.json',plan)
    sources=['scripts/check_ash_before_print.py','scripts/render_ash_samples.py','scripts/render_ash_dataset.py',
             'src/mes_vision/synthetic/planning.py','src/mes_vision/synthetic/projection.py']
    write_json(OUT/'contract.json',{'plan_sha256':sha256(OUT/'plan.json'),'geometry_sha256':sha256(OUT/'geometry.json'),
        'samples':32,'source_hashes':{p:sha256(ROOT/p) for p in sources}})
    print('PREPARED',len(tasks),'paired diagnostic images',flush=True)


def render_all():
    subprocess.run([str(ROOT/'.tools/blender-4.5.7-windows-x64/blender.exe'),'--background','--factory-startup',
        '--python-exit-code','1','--python',str(ROOT/'scripts/render_ash_dataset.py'),'--','--root',str(OUT)],check=True,cwd=ROOT)


def label_and_inspect():
    from PIL import Image
    from mes_vision.station.photo_inspection import run_photos
    geometry=read_json(OUT/'geometry.json'); plan=read_json(OUT/'plan.json'); images=[]
    for task in plan['tasks']:
        path=OUT/'raw'/(task['id']+'.png'); receipt=read_json(OUT/'receipts'/(task['id']+'.json'))
        require(sha256(path)==receipt['sha256'],'Rendered photograph changed')
        require(receipt['camera']['projection_verified_against_blender'],'Projection not verified')
        objects,defects,mask=geometry_labels(task,geometry)
        if mask is not None: Image.fromarray(mask.astype('uint8')*255).save(OUT/'masks'/(task['id']+'.png'))
        write_json(OUT/'quality'/(task['id']+'.json'),{'objects':objects,'defects':defects,'image_sha256':receipt['sha256']})
        images.append({'role':task['domain'],'path':str(path),'sha256':receipt['sha256']})
    request={'output':str(OUT/'inference'),'runtime':str(ROOT/'artifacts/operation'),
        'bundle':str(ROOT/'configs/inspection/ash-trained-v3.json'),'image_kind':'synthetic','images':images}
    write_json(OUT/'inference-request.json',request); run_photos(request,ROOT)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('action',choices=['prepare','render','inspect'])
    action=parser.parse_args().action
    {'prepare':prepare,'render':render_all,'inspect':label_and_inspect}[action]()


if __name__=='__main__': main()

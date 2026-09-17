"""Blender pilot: doubled STL, near-overhead viewpoints and PETG-like material variation."""
from pathlib import Path
import sys,json,math,hashlib,argparse,shutil
import numpy as np
import bpy
from mathutils import Vector, Euler
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import render_ash_samples as base
from mes_vision.synthetic.projection import raster_depth,difference_mask,bbox

OUT=None
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8');tmp.replace(p)
def save_mask(path,mask):
    # Blender's image API consumes bottom-up RGBA; masks stay lossless and binary.
    h,w=mask.shape;rgba=np.ones((h,w,4),np.float32);rgba[:,:,:3]=mask[:,:,None]
    im=bpy.data.images.new('MaskExport',w,h,alpha=True);im.pixels.foreach_set(rgba[::-1].ravel());im.filepath_raw=str(path);im.file_format='PNG';im.save();bpy.data.images.remove(im)

def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int,default=0);parser.add_argument('--start',type=int,default=0);parser.add_argument('--output',type=Path,default=ROOT/'datasets/ash-cad-today-v1')
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT=args.output.resolve()
    OUT.mkdir(parents=True,exist_ok=True)
    for f in ['rgb','masks','sources','receipts']:(OUT/f).mkdir(exist_ok=True)
    calibration=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json'
    cal=json.loads(calibration.read_text());focal=(cal['K'][0][0]+cal['K'][1][1])/2
    base.HFOV=math.degrees(2*math.atan(base.WIDTH/(2*focal)))
    files=sorted((OUT/'sources').glob('*.stl')) or sorted(Path('C:/Users/alexi/Downloads/OKNG').rglob('*.stl'));assert len(files)==7
    geometry={};sources={}
    for p in files:
        code='OK' if p.stem=='ASH_OK' else p.stem.split('_')[1]
        geometry[code]=base.stl(p)*2
        (OUT/'sources'/p.name).write_bytes(p.read_bytes());sources[code]=dict(file=p.name,sha256=sha(p),scaled_extents_mm=(np.ptp(geometry[code].reshape(-1,3),axis=0)*1000).tolist())
    lights=[([-.085,-.04,.17],1.0,.10),([.07,.055,.14],.8,.07),([-.04,.065,.12],1.0,.035),([.02,-.08,.16],.65,.12)]
    grid=dict(roll=list(range(-45,46,5)),pitch=list(range(-45,46,5)),yaw=list(range(0,360,5)),distances_mm=[110,130,150,170],lights=4,classes=['OK']+[f'NG{i:02}' for i in range(1,7)])
    plan=json.loads((OUT/'plan.json').read_text(encoding='utf-8'));total=len(plan['conditions'])*7;assert total==1008
    pose_pairs=sorted([(r,p) for r in grid['roll'] for p in grid['pitch']],key=lambda rp:(-math.cos(math.radians(rp[0]))*math.cos(math.radians(rp[1])),abs(rp[0])+abs(rp[1]),rp))
    sampling_policy=dict(status='PROPOSED_FOR_FUTURE_TRAINING_NOT_APPLIED',basis='actual optical-axis tilt from vertical',weights={'0_to_15_degrees':4,'over15_to30_degrees':2,'over30_degrees':1},note='All generated views retained; no training run by generator')
    contract=dict(plan_sha256=sha(OUT/'plan.json'),generation_order='balanced_grid_coverage_then_topdown',sampling_policy=sampling_policy,script=sha(Path(__file__)),base=sha(ROOT/'scripts/render_ash_samples.py'),projection=sha(ROOT/'src/mes_vision/synthetic/projection.py'),sources=sources,calibration=sha(calibration),grid=grid,lights=lights,convention='Camera Euler XYZ=(roll,pitch,0), camera at target + R*(0,0,distance_mm/1000); distance is optical center to part center at Z=14mm; grounded object yaw about world Z. Camera local -Z looks at target. Not rotation of floating object.',scale=2)
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text(encoding='utf-8'))==json.loads(json.dumps(contract)), 'Source/settings changed; use a new output'
    else:write(OUT/'contract.json',contract)
    manifest=dict(status='RENDERING',planned_images=total,grid=grid,labels_approved=False,training_performed=False,completed_prefix=0,receipts='Per-frame JSON, sharded by roll/pitch; authoritative completion records',camera_note='Ideal rectified pinhole, centered principal point, mean measured focal')
    def conditions():
        for item in plan['conditions']:
             roll,pitch,angle,distance_mm,light=[item[k] for k in ['roll','pitch','yaw','distance_mm','light']]
             location,power,size=lights[light]
             yield dict(id=f'R{roll:+03d}_P{pitch:+03d}_Y{angle:03d}_L{light}_D{distance_mm:03d}',distance_mm=distance_mm,roll=roll,pitch=pitch,angle=angle,camera_z=.145,color=[.65,.65,.63,1],background=[.038,.045,.023,1],light=location,energy=power,roughness=[.32,.4,.25,.48][light],table_roughness=[.4,.5,.3,.6][light],light_size=size,seed=11000+(roll+45)*100000+(pitch+45)*1000+angle*4+light+distance_mm*10000000)
    processed=0
    for condition in conditions():
      for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
        if args.limit and processed>=args.limit:
            manifest.update(status='PARTIAL_LIMIT',completed_prefix=processed);write(OUT/'manifest.json',manifest);return
        processed+=1
        if processed<=args.start:continue
        name=condition['id']+'_'+code
        shard=f"R{condition['roll']:+03d}_P{condition['pitch']:+03d}_D{condition['distance_mm']:03d}"
        for folder in ['rgb','masks','receipts']:(OUT/folder/shard).mkdir(exist_ok=True)
        rgbdir=OUT/'rgb'/shard;maskdir=OUT/'masks'/shard
        receipt=OUT/'receipts'/shard/(name+'.json')
        if receipt.exists():
            rec=json.loads(receipt.read_text());assert sha(rgbdir/(name+'.png'))==rec['rgb_sha256']
            for kind in ['object','difference']:assert sha(maskdir/(name+'-'+kind+'.png'))==rec[kind+'_mask_sha256']
            manifest['completed_prefix']=processed
            if processed%100==0:write(OUT/'manifest.json',manifest);print('SKIP',processed,'/',total,flush=True)
            continue
        if shutil.disk_usage(OUT).free < 512*1024**2:
            manifest.update(status='PAUSED_LOW_DISK',completed_prefix=processed-1);write(OUT/'manifest.json',manifest);raise RuntimeError('Low disk space; completed images retained. Free space and rerun.')
        scene,mat,devices=base.scene_setup(condition,False,24)
        scene.world.node_tree.nodes['Background'].inputs[1].default_value=.20
        table=bpy.data.materials['TableMat'].node_tree.nodes.get('Principled BSDF');table.inputs['Roughness'].default_value=condition['table_roughness']
        bpy.data.lights['Key'].size=condition['light_size']
        bpy.data.objects['Fill'].location=(-.1,-.09,.20);bpy.data.lights['Fill'].energy=.35;bpy.data.lights['Fill'].size=.15
        # Approximate 0.2mm FDM layer texture; a material cue, not exact printed geometry.
        nodes=mat.node_tree.nodes;links=mat.node_tree.links;bsdf=nodes.get('Principled BSDF')
        geo=nodes.new('ShaderNodeNewGeometry');sep=nodes.new('ShaderNodeSeparateXYZ');mul=nodes.new('ShaderNodeMath');mul.operation='MULTIPLY';mul.inputs[1].default_value=2*math.pi/.0002
        sine=nodes.new('ShaderNodeMath');sine.operation='SINE';bump=nodes.new('ShaderNodeBump');bump.inputs['Strength'].default_value=.12;bump.inputs['Distance'].default_value=.00001
        links.new(geo.outputs['Position'],sep.inputs[0]);links.new(sep.outputs['Z'],mul.inputs[0]);links.new(mul.outputs[0],sine.inputs[0]);links.new(sine.outputs[0],bump.inputs['Height']);links.new(bump.outputs['Normal'],bsdf.inputs['Normal'])
        target=Vector((0,0,.014));rotation=Euler((math.radians(condition['roll']),math.radians(condition['pitch']),0),'XYZ')
        scene.camera.rotation_euler=rotation
        scene.camera.location=target+rotation.to_matrix()@Vector((0,0,condition['distance_mm']/1000))
        obj=base.add_part(code,geometry[code],mat,angle=condition['angle']);bpy.context.view_layer.update()
        world=np.array(obj.matrix_world);view=np.array(scene.camera.matrix_world.inverted())
        def in_camera(tris):
            m=view@world;return tris@m[:3,:3].T+m[:3,3]
        camera=dict(width=base.WIDTH,height=base.HEIGHT,fx=focal,fy=focal,cx=base.WIDTH/2,cy=base.HEIGHT/2,position=[0,0,0],orientation='vertical_down_world_minus_Z')
        # Verify arbitrary-pose camera projections against Blender before making geometry masks.
        from bpy_extras.object_utils import world_to_camera_view
        for p in geometry[code].reshape(-1,3)[::max(1,len(geometry[code])//8)]:
            v=world@np.r_[p,1];q=view@v;ndc=world_to_camera_view(scene,scene.camera,Vector(v[:3]))
            actual=[ndc.x*base.WIDTH,(1-ndc.y)*base.HEIGHT];expected=[base.WIDTH/2+focal*q[0]/-q[2],base.HEIGHT/2-focal*q[1]/-q[2]]
            assert np.allclose(actual,expected,atol=.01),(actual,expected)
        actual=raster_depth(in_camera(geometry[code]),camera);reference=raster_depth(in_camera(geometry['OK']),camera)
        difference=difference_mask(actual,reference,tolerance=.00002) if code!='OK' else np.zeros_like(actual,dtype=bool)
        save_mask(maskdir/(name+'-object.png'),np.isfinite(actual));save_mask(maskdir/(name+'-difference.png'),difference)
        scene.render.filepath=str(rgbdir/(name+'.png'));bpy.ops.render.render(write_still=True)
        rec=dict(id=name,code=code,condition=condition['id'],camera_to_world=np.array(scene.camera.matrix_world).tolist(),object_to_world=world.tolist(),focal_px=focal,object_box_xywh=bbox(np.isfinite(actual)),difference_box_xywh=bbox(difference),difference_pixels=int(difference.sum()),mask_note='Visible-surface depth difference vs OK, including removed features. Geometry proposal, not guaranteed photometric visibility or semantic defect mask.',rgb_sha256=sha(rgbdir/(name+'.png')),devices=devices)
        actual_tilt=math.degrees(math.acos(max(-1,min(1,math.cos(math.radians(condition['roll']))*math.cos(math.radians(condition['pitch']))))))
        rec.update(camera_to_part_center_mm=condition['distance_mm'],actual_view_tilt_deg=actual_tilt,proposed_training_weight=4 if actual_tilt<=15.000001 else 2 if actual_tilt<=30.000001 else 1,object_mask_sha256=sha(maskdir/(name+'-object.png')),difference_mask_sha256=sha(maskdir/(name+'-difference.png')),label_status='GEOMETRY_PROPOSAL_NOT_APPROVED',angles=dict(roll=condition['roll'],pitch=condition['pitch'],yaw=condition['angle']),relative_rgb=str((rgbdir/(name+'.png')).relative_to(OUT)),difference_visible_geometry=bool(difference.any()))
        write(receipt,rec);manifest['completed_prefix']=processed;write(OUT/'manifest.json',manifest);print('RENDER',processed,'/',total,name,flush=True)
    manifest['status']='COMPLETED';write(OUT/'manifest.json',manifest)

if __name__=='__main__':main()

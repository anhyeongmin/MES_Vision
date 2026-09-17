"""Blender pilot: doubled STL, near-overhead viewpoints and PETG-like material variation."""
from pathlib import Path
import sys,json,math,hashlib,argparse
import numpy as np
import bpy
from mathutils import Vector
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
    parser=argparse.ArgumentParser();parser.add_argument('--mode',choices=['pilot','full'],default='pilot');parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:] if '--' in sys.argv else [])
    OUT=ROOT/('artifacts/cad-material-pilot-v2' if args.mode=='pilot' else 'datasets/ash-cad-5deg-v1')
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
    conditions=[]
    lights=[([-.085,-.04,.17],1.0,.10),([.07,.055,.14],.8,.07),([-.04,.065,.12],1.0,.035),([.02,-.08,.16],.65,.12)]
    combinations=[(a,v,l) for a in range(0,360,5) for v in range(3) for l in range(4)] if args.mode=='full' else [(0,0,0),(45,1,1),(95,2,2),(215,0,3)]
    for angle,view,light in combinations:
        location,power,size=lights[light]
        conditions.append(dict(id=f'r{angle:03d}_v{view}_l{light}',angle=angle,camera_z=[.145,.16,.15][view],tilt=[0,4,7][view],azimuth=[0,35,135][view],color=[.65,.65,.63,1],background=[.038,.045,.023,1],light=location,energy=power,roughness=[.32,.4,.25,.48][light],table_roughness=[.4,.5,.3,.6][light],light_size=size,seed=11000+angle*12+view*4+light))
    manifest=dict(status='RENDERING',scale=2,units='metres; source STL units interpreted as mm',source_hashes=sources,script_sha256=sha(Path(__file__)),base_script_sha256=sha(ROOT/'scripts/render_ash_samples.py'),calibration_sha256=sha(calibration),camera_note='Ideal rectified pinhole using mean measured fx/fy; centered principal point; not an exact hardware simulation',material_note='Approximate white PETG/olive green; not measured spectral or print-layer material',training_performed=False,labels_approved=False,conditions=conditions,frames=[])
    manifest['mode']=args.mode;manifest['planned_images']=len(conditions)*7
    contract=dict(script=sha(Path(__file__)),base=sha(ROOT/'scripts/render_ash_samples.py'),projection=sha(ROOT/'src/mes_vision/synthetic/projection.py'),sources=sources,calibration=sha(calibration),conditions=conditions)
    if (OUT/'contract.json').exists():assert json.loads((OUT/'contract.json').read_text(encoding='utf-8'))==contract, 'Source/settings changed; do not resume this dataset'
    else:write(OUT/'contract.json',contract)
    write(OUT/'manifest.json',manifest)
    processed=0
    for condition in conditions:
      for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
        if args.limit and processed>=args.limit: return
        processed+=1
        name=condition['id']+'_'+code;receipt=OUT/'receipts'/(name+'.json')
        if receipt.exists():
            rec=json.loads(receipt.read_text());assert sha(OUT/'rgb'/(name+'.png'))==rec['rgb_sha256']
            for kind in ['object','difference']:assert sha(OUT/'masks'/(name+'-'+kind+'.png'))==rec[kind+'_mask_sha256']
            manifest['frames'].append(rec);print('SKIP',processed,'/',manifest['planned_images'],name,flush=True);continue
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
        target=Vector((0,0,.014));tilt=math.radians(condition['tilt']);az=math.radians(condition['azimuth']);height=condition['camera_z']
        offset=(height-target.z)*math.tan(tilt);scene.camera.location=(offset*math.cos(az),offset*math.sin(az),height)
        scene.camera.rotation_euler=(target-scene.camera.location).to_track_quat('-Z','Y').to_euler()
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
        save_mask(OUT/'masks'/(name+'-object.png'),np.isfinite(actual));save_mask(OUT/'masks'/(name+'-difference.png'),difference)
        scene.render.filepath=str(OUT/'rgb'/(name+'.png'));bpy.ops.render.render(write_still=True)
        rec=dict(id=name,code=code,condition=condition['id'],camera_to_world=np.array(scene.camera.matrix_world).tolist(),object_to_world=world.tolist(),focal_px=focal,object_box_xywh=bbox(np.isfinite(actual)),difference_box_xywh=bbox(difference),difference_pixels=int(difference.sum()),mask_note='Visible-surface depth difference vs OK, including removed features. Geometry proposal, not guaranteed photometric visibility or semantic defect mask.',rgb_sha256=sha(OUT/'rgb'/(name+'.png')),devices=devices)
        rec.update(object_mask_sha256=sha(OUT/'masks'/(name+'-object.png')),difference_mask_sha256=sha(OUT/'masks'/(name+'-difference.png')),label_status='GEOMETRY_PROPOSAL_NOT_APPROVED')
        write(receipt,rec);manifest['frames'].append(rec);write(OUT/'manifest.json',manifest);print('RENDER',processed,'/',manifest['planned_images'],name,flush=True)
    manifest['status']='COMPLETED';write(OUT/'manifest.json',manifest)

if __name__=='__main__':main()

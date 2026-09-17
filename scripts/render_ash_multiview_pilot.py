"""Blender pilot: doubled STL, near-overhead viewpoints and PETG-like material variation."""
from pathlib import Path
import sys,json,math,hashlib
import numpy as np
import bpy
from mathutils import Vector
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'src'))
import render_ash_samples as base
from mes_vision.synthetic.projection import raster_depth,difference_mask,bbox

OUT=ROOT/'artifacts/cad-multiview-pilot-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False),encoding='utf-8')
def save_mask(path,mask):
    # Blender's image API consumes bottom-up RGBA; masks stay lossless and binary.
    h,w=mask.shape;rgba=np.ones((h,w,4),np.float32);rgba[:,:,:3]=mask[:,:,None]
    im=bpy.data.images.new('MaskExport',w,h,alpha=True);im.pixels.foreach_set(rgba[::-1].ravel());im.filepath_raw=str(path);im.file_format='PNG';im.save();bpy.data.images.remove(im)

def main():
    OUT.mkdir(exist_ok=False)
    for f in ['rgb','masks','sources']:(OUT/f).mkdir()
    calibration=ROOT/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json'
    cal=json.loads(calibration.read_text());focal=(cal['K'][0][0]+cal['K'][1][1])/2
    base.HFOV=math.degrees(2*math.atan(base.WIDTH/(2*focal)))
    files=sorted(Path('C:/Users/alexi/Downloads/OKNG').rglob('*.stl'));assert len(files)==7
    geometry={};sources={}
    for p in files:
        code='OK' if p.stem=='ASH_OK' else p.stem.split('_')[1]
        geometry[code]=base.stl(p)*2
        (OUT/'sources'/p.name).write_bytes(p.read_bytes());sources[code]=dict(path=str(p),sha256=sha(p),scaled_extents_mm=(np.ptp(geometry[code].reshape(-1,3),axis=0)*1000).tolist())
    conditions=[
      dict(id='A',angle=0,camera_z=.145,tilt=0,azimuth=0,color=[.72,.72,.70,1],background=[.14,.16,.075,1],light=[-.085,-.04,.17],energy=3.0,roughness=.38,table_roughness=.5,light_size=.10,seed=9301),
      dict(id='B',angle=90,camera_z=.165,tilt=4,azimuth=35,color=[.72,.72,.70,1],background=[.14,.16,.075,1],light=[.07,.055,.14],energy=2.6,roughness=.27,table_roughness=.32,light_size=.055,seed=9302),
      dict(id='C',angle=213,camera_z=.15,tilt=7,azimuth=135,color=[.72,.72,.70,1],background=[.14,.16,.075,1],light=[-.04,.065,.12],energy=4.0,roughness=.2,table_roughness=.25,light_size=.025,seed=9303)]
    manifest=dict(status='RENDERING',scale=2,units='metres; source STL units interpreted as mm',source_hashes=sources,script_sha256=sha(Path(__file__)),base_script_sha256=sha(ROOT/'scripts/render_ash_samples.py'),calibration_sha256=sha(calibration),camera_note='Ideal rectified pinhole using mean measured fx/fy; centered principal point; not an exact hardware simulation',material_note='Approximate white PETG/olive green; not measured spectral or print-layer material',training_performed=False,labels_approved=False,conditions=conditions,frames=[])
    write(OUT/'manifest.json',manifest)
    for condition in conditions:
      for code in ['OK']+[f'NG{i:02}' for i in range(1,7)]:
        scene,mat,devices=base.scene_setup(condition,False,24)
        scene.world.node_tree.nodes['Background'].inputs[1].default_value=.35
        table=bpy.data.materials['TableMat'].node_tree.nodes.get('Principled BSDF');table.inputs['Roughness'].default_value=condition['table_roughness']
        bpy.data.lights['Key'].size=condition['light_size']
        bpy.data.objects['Fill'].location=(-.1,-.09,.20);bpy.data.lights['Fill'].energy=1.0;bpy.data.lights['Fill'].size=.15
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
        name=condition['id']+'_'+code;save_mask(OUT/'masks'/(name+'-object.png'),np.isfinite(actual));save_mask(OUT/'masks'/(name+'-difference.png'),difference)
        scene.render.filepath=str(OUT/'rgb'/(name+'.png'));bpy.ops.render.render(write_still=True)
        rec=dict(id=name,code=code,condition=condition['id'],camera_to_world=np.array(scene.camera.matrix_world).tolist(),object_to_world=world.tolist(),focal_px=focal,object_box_xywh=bbox(np.isfinite(actual)),difference_box_xywh=bbox(difference),difference_pixels=int(difference.sum()),mask_note='Visible-surface depth difference vs OK, including removed features. Geometry proposal, not guaranteed photometric visibility or semantic defect mask.',rgb_sha256=sha(OUT/'rgb'/(name+'.png')),devices=devices)
        manifest['frames'].append(rec);write(OUT/'manifest.json',manifest);print('PILOT',len(manifest['frames']),'/21',name,flush=True)
    manifest['status']='COMPLETED';write(OUT/'manifest.json',manifest)

if __name__=='__main__':main()

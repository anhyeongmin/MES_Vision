"""Run with Blender 4.5.7 --background --factory-startup --python this_file -- ...

Cycles RGB samples only. Camera is a nominal pinhole approximation, not calibrated U20CAM.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
import sys
import numpy as np
import bpy
from mathutils import Vector

WIDTH, HEIGHT = 1280, 720
HFOV = 102.0
CONDITIONS = [
    dict(id='A', angle=0, camera_z=.036, color=[.20,.36,.52,1], background=[.075,.085,.10,1],
         light=[.034,-.025,.065], energy=.48, roughness=.65, seed=731),
    dict(id='B', angle=32, camera_z=.045, color=[.48,.47,.40,1], background=[.09,.16,.13,1],
         light=[-.036,.012,.06], energy=.55, roughness=.72, seed=732),
    dict(id='C', angle=-27, camera_z=.052, color=[.65,.65,.65,1], background=[.045,.055,.07,1],
         light=[.015,.04,.072], energy=.72, roughness=.54, seed=733),
]


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def stl(path):
    data = path.read_bytes()
    dtype = np.dtype([('normal','<f4',(3,)),('v','<f4',(3,3)),('attr','<u2')])
    count = int.from_bytes(data[80:84], 'little')
    if len(data) != 84+50*count:
        raise ValueError('Incomplete STL')
    tri = np.frombuffer(data, dtype=dtype, offset=84)['v'].astype(float)
    # Convert coordinate units to metres under the explicitly unconfirmed mm assumption.
    # A rigid basis change places the original +Y inspection face upwards (+Z).
    return tri[:, :, [0,2,1]] * np.array([.001,-.001,.001])


def material(name, color, roughness, texture_scale=4500):
    mat = bpy.data.materials.new(name); mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get('Principled BSDF')
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = roughness
    noise = nodes.new('ShaderNodeTexNoise'); noise.inputs['Scale'].default_value = texture_scale
    noise.inputs['Detail'].default_value = 2
    coord = nodes.new('ShaderNodeNewGeometry')
    links.new(coord.outputs['Position'], noise.inputs['Vector'])
    bump = nodes.new('ShaderNodeBump'); bump.inputs['Strength'].default_value = .13
    bump.inputs['Distance'].default_value = .000012
    links.new(noise.outputs['Fac'], bump.inputs['Height'])
    links.new(bump.outputs['Normal'], bsdf.inputs['Normal'])
    return mat


def area(name, location, power, size, color):
    data = bpy.data.lights.new(name, 'AREA'); data.energy = power; data.shape='DISK'; data.size=size; data.color=color
    obj = bpy.data.objects.new(name, data); bpy.context.collection.objects.link(obj); obj.location=location
    obj.rotation_euler=(Vector((0,0,.006))-obj.location).to_track_quat('-Z','Y').to_euler()


def scene_setup(condition, overview, samples):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine='CYCLES'; scene.cycles.samples=samples; scene.cycles.use_denoising=True
    scene.cycles.seed=condition['seed']; scene.cycles.max_bounces=6
    devices=[]
    try:
        prefs=bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type='CUDA'; prefs.get_devices()
        for device in prefs.devices:
            device.use=device.type=='CUDA'
            if device.use: devices.append(device.name)
        scene.cycles.device='GPU' if devices else 'CPU'
    except Exception:
        scene.cycles.device='CPU'
    scene.render.resolution_x=WIDTH; scene.render.resolution_y=HEIGHT; scene.render.resolution_percentage=100
    scene.render.image_settings.file_format='PNG'; scene.render.image_settings.color_mode='RGB'
    scene.render.image_settings.color_depth='8'; scene.render.film_transparent=False
    scene.view_settings.view_transform='AgX'
    world=bpy.data.worlds.new('World'); world.use_nodes=True
    world.node_tree.nodes['Background'].inputs[0].default_value=(.62,.69,.80,1)
    world.node_tree.nodes['Background'].inputs[1].default_value=.24; scene.world=world
    camera_data=bpy.data.cameras.new('Camera'); camera_data.type='PERSP'; camera_data.sensor_fit='HORIZONTAL'
    camera_data.sensor_width=36; camera_data.lens=36/(2*math.tan(math.radians(HFOV)/2))
    camera_data.clip_start=.0001; camera_data.clip_end=10; camera_data.dof.use_dof=False
    camera=bpy.data.objects.new('Camera',camera_data); bpy.context.collection.objects.link(camera)
    camera.location=(0,0,.082+(.006*'ABC'.index(condition['id'])) if overview else condition['camera_z'])
    scene.camera=camera  # Identity rotation: local -Z looks vertically down, local +Y is image up.
    bpy.ops.mesh.primitive_plane_add(size=.6, location=(0,0,-.00005))
    bpy.context.object.name='Table'
    bpy.context.object.data.materials.append(material('TableMat',condition['background'],.9,1200))
    factor=2.4 if overview else 1
    area('Key',condition['light'],condition['energy']*factor,.035,(1,.93,.84))
    area('Fill',(-.03,-.035,.075),.12*factor,.065,(.82,.9,1))
    part_material=material('PartMat',condition['color'],condition['roughness'])
    return scene, part_material, devices


def add_part(code, triangles, mat, x=0, y=0, angle=0):
    vertices, inverse=np.unique(triangles.reshape(-1,3),axis=0,return_inverse=True)
    data=bpy.data.meshes.new(code)
    data.from_pydata(vertices.tolist(), [], inverse.reshape(-1,3).tolist()); data.update()
    obj=bpy.data.objects.new(code,data); bpy.context.collection.objects.link(obj)
    obj.data.materials.append(mat); obj.location=(x,y,0); obj.rotation_euler.z=math.radians(angle)
    return obj


def camera_record(scene):
    from bpy_extras.object_utils import world_to_camera_view
    fx=WIDTH/(2*math.tan(math.radians(HFOV)/2))
    scene.view_layers.update()
    position=list(scene.camera.location)
    # Independently verify exported intrinsics against Blender's own projection.
    for point in ((.003,.002,.002),(-.005,.004,.012),(0,0,0)):
        ndc=world_to_camera_view(scene,scene.camera,Vector(point))
        actual=np.array([ndc.x*WIDTH,(1-ndc.y)*HEIGHT])
        depth=position[2]-point[2]
        expected=np.array([WIDTH/2+fx*(point[0]-position[0])/depth,HEIGHT/2-fx*(point[1]-position[1])/depth])
        if not np.allclose(actual,expected,atol=.002):
            raise ValueError('Camera projection mismatch: '+str((actual,expected)))
    return dict(position=position, fx=fx, fy=fx, cx=WIDTH/2, cy=HEIGHT/2,
        width=WIDTH,height=HEIGHT,hfov_deg=HFOV,orientation='vertical_down_world_minus_Z',
        projection_verified_against_blender=True, distortion='not_simulated', depth_of_field='not_simulated')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sources',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=48); parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(sys.argv[sys.argv.index('--')+1:])
    registry=json.loads(args.sources.read_text(encoding='utf-8'))
    variants={}
    for v in registry['variants']:
        source=next(p for p in v['files'] if p['path'].endswith('.stl'))
        path=args.sources.parent/source['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=source['sha256']: raise ValueError('CAD source changed')
        variants[v['code']]=stl(path)
    output=args.output.resolve(); output.mkdir(parents=True,exist_ok=False)
    for folder in ('rgb','scenes'): (output/folder).mkdir()
    write(output/'geometry.json',{code:tri.tolist() for code,tri in variants.items()})
    report=dict(schema_version=1,kind='synthetic_cad_samples',status='rendering',
        created_utc=datetime.now(timezone.utc).isoformat(),camera_target='INNOMAKER U20CAM-720P',
        camera_source='https://www.inno-maker.com/product/u20cam-720p/',
        camera_match='nominal resolution and horizontal FOV only; not real-device calibration',
        units='source coordinates assumed mm for scene construction; no original STL scaling applied',
        geometry_sha256=hashlib.sha256((output/'geometry.json').read_bytes()).hexdigest(),
        source_registry_sha256=hashlib.sha256(args.sources.read_bytes()).hexdigest(),
        blender=bpy.app.version_string,samples=args.samples,frames=[],
        scope='24 small review samples, not an accuracy dataset. No model training or device connection.')
    write(output/'render-manifest.json',report)
    tasks=[(cond,False,code) for cond in CONDITIONS for code in variants]
    tasks += [(cond,True,None) for cond in CONDITIONS]
    if args.limit: tasks=tasks[:args.limit]
    for condition,overview,code in tasks:
        identity=condition['id']+('_overview' if overview else '_'+code)
        scene,mat,devices=scene_setup(condition,overview,args.samples)
        objects=[]
        if overview:
            for i,(variant,tri) in enumerate(variants.items()):
                x=(i%4-1.5)*.027; y=(.5-i//4)*.028
                obj=add_part(variant,tri,mat,x,y,condition['angle']+((i%3)-1)*9)
                objects.append((variant,obj))
        else:
            objects=[(code,add_part(code,variants[code],mat,angle=condition['angle']))]
        bpy.context.view_layer.update()
        frame=dict(id=identity,domain='overview' if overview else 'detail',condition=condition,
            camera=camera_record(scene),devices=devices,objects=[dict(code=k,matrix_world=[list(r) for r in o.matrix_world]) for k,o in objects],
            file_name='rgb/'+identity+'.png',normal_pair=None if overview else 'rgb/'+condition['id']+'_OK.png')
        scene.render.filepath=str(output/frame['file_name'])
        print('ASH_RENDER_BEGIN '+identity,flush=True)
        bpy.ops.render.render(write_still=True)
        frame['image_sha256']=hashlib.sha256((output/frame['file_name']).read_bytes()).hexdigest()
        write(output/'scenes'/(identity+'.json'),frame)
        report['frames'].append(frame); write(output/'render-manifest.json',report)
        print('ASH_RENDER_DONE '+identity,flush=True)
    report['status']='rendered'; write(output/'render-manifest.json',report)


if __name__=='__main__': main()

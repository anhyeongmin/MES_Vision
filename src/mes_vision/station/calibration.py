"""Measured overview-pixel -> capture TCP and pick TCP maps at fixed poses.

No constant camera/gripper offset or Magician mounting kinematics are assumed.
Both empirical planar maps must pass independent checks over their application
region; a mount whose motion cannot meet those limits needs another model.
"""
from copy import deepcopy
from pathlib import Path
import json,math
from mes_vision.calibration.core import load,spec_from_dict
from mes_vision.anomaly.features import fingerprint
from mes_vision.training.data import require,read_json,sha256
from .sequence import valid_pose


def pose(value):
    require(isinstance(value,dict) and set(value)=={'x','y','z','r'} and
            all(type(v) in (int,float) and math.isfinite(v) for v in value.values()),'Complete finite robot pose required')


def asset(path):
    path=Path(path).resolve(); return {'path':str(path),'sha256':sha256(path)}


def build_bundle(capture_path,pick_path,*,version,overview_pose,detail_z,detail_r,pick_z,pick_r,
                 travel_z,grasp_uv,grasp_version,mount_link,validation_reference):
    value={'schema_version':1,'kind':'robot_mounted_overview_detail','version':version,
        'overview_pose':deepcopy(overview_pose),'detail_z':detail_z,'detail_r':detail_r,
        'pick_z':pick_z,'pick_r':pick_r,'travel_z':travel_z,'grasp_uv':list(grasp_uv),
        'grasp_version':grasp_version,'mount_link':mount_link,'validation_reference':validation_reference,
        'capture_map':asset(capture_path),'pick_map':asset(pick_path)}
    MountedCalibration(value)
    return value


def save_bundle(path,value):
    MountedCalibration(value)
    with Path(path).open('x',encoding='utf-8') as stream:
        json.dump({'data':value,'sha256':fingerprint(value)},stream,ensure_ascii=False,indent=2,allow_nan=False)


class MountedCalibration:
    def __init__(self,value):
        self.value=deepcopy(value); v=self.value
        require(v.get('schema_version')==1 and v.get('kind')=='robot_mounted_overview_detail','Moving-camera calibration bundle required')
        for key in ('version','grasp_version','mount_link','validation_reference'):
            require(isinstance(v.get(key),str) and v[key].strip(),'Missing calibration identity: '+key)
        pose(v['overview_pose'])
        require(all(type(v.get(k)) in (int,float) and math.isfinite(v[k]) for k in
            ('detail_z','detail_r','pick_z','pick_r','travel_z')),'Measured capture and pick heights required')
        require(v['travel_z']>=v['overview_pose']['z'] and v['travel_z']>max(v['detail_z'],v['pick_z']), 'Travel height must clear capture and pick heights')
        require(isinstance(v['grasp_uv'],list) and len(v['grasp_uv'])==2 and
            all(type(x) in (int,float) and math.isfinite(x) and 0<x<1 for x in v['grasp_uv']),'Validated interior grasp point required')
        self.maps={}
        for key in ('capture_map','pick_map'):
            ref=v[key]; require(sha256(Path(ref['path']))==ref['sha256'],'Calibration asset changed')
            calibration=load(Path(ref['path'])); spec=spec_from_dict(calibration.data['specification'])
            require(calibration.ready and spec.kind=='real','Accepted measured calibration required')
            self.maps[key]=calibration
        a,b=(spec_from_dict(self.maps[k].data['specification']) for k in ('capture_map','pick_map'))
        require(a.context==b.context and a.product_id==b.product_id and a.plane_z_mm==b.plane_z_mm
                and a.plane_tolerance_mm==b.plane_tolerance_mm and a.lens==b.lens,'Capture/pick maps have different camera, plane, product or lens conditions')
        require({p.pixel for p in a.pairs}=={p.pixel for p in b.pairs},'Capture and pick maps must refer to the same overview reference points')
        self.context=a.context; self.product_id=a.product_id; self.plane_z=a.plane_z_mm
        require(not self.context.transformations,'Moving-camera maps must use original overview pixels')
        self.digest=fingerprint(v); self.identity=v['version']+':'+self.digest

    @classmethod
    def load(cls,path):
        envelope=read_json(Path(path))
        require(set(envelope)=={'data','sha256'} and fingerprint(envelope['data'])==envelope['sha256'],'Moving-camera bundle integrity mismatch')
        return cls(envelope['data'])

    def verify_context(self,profile,equipment):
        v=self.value
        from mes_vision.operation.acquisition import inspection_camera, processing
        c=inspection_camera(equipment['camera'])
        if processing(equipment['camera']):
            require(all(spec_from_dict(m.data['specification']).lens.mode=='none_verified' for m in self.maps.values()),
                    'Rectified acquisition must not apply a second lens correction')
        require(profile['calibration_digest']==self.digest and profile['product_id']==self.product_id,'Scan calibration/product changed')
        require(profile['overview_pose']==v['overview_pose'],'Overview camera pose changed')
        expected=self.context
        require((c['serial'],c['mount_revision'],c['acquisition_revision'],c['robot_base_id'],c['tool_frame_id'],(c['width'],c['height']))==
            (expected.camera_id,expected.mount_revision,expected.acquisition_revision,expected.robot_base_id,expected.tool_frame_id,expected.image_size),'Camera/mount/acquisition/base/tool context changed')
        require(profile['camera_serial']==c['serial'] and profile['mount_revision']==c['mount_revision']
            and list(profile['image_size'])==list(expected.image_size),'Scan camera context changed')
        for ref in (v['capture_map'],v['pick_map']):
            require(sha256(Path(ref['path']))==ref['sha256'],'Calibration changed after loading')

    def map_targets(self,request,profile,equipment):
        require(request['kind']=='map_targets' and request['payload']['calibration_digest']==self.digest,'Wrong mapping request')
        self.verify_context(profile,equipment); v=self.value; output=[]
        for target in request['payload']['targets']:
            box=target['overview']['box']; x1,y1,x2,y2=(box[k] for k in ('x1','y1','x2','y2'))
            require(x1<x2 and y1<y2,'Invalid overview object box')
            center=((x1+x2)/2,(y1+y2)/2); grasp=(x1+(x2-x1)*v['grasp_uv'][0],y1+(y2-y1)*v['grasp_uv'][1])
            mapped={}
            for name,pixel,z,r in (('capture',center,v['detail_z'],v['detail_r']),('pick',grasp,v['pick_z'],v['pick_r'])):
                x,y=self.maps[name+'_map'].map_xy(pixel,self.context,plane_z_mm=self.plane_z,kind='real')
                mapped[name+'_pose']={'x':x,'y':y,'z':z,'r':r}; valid_pose(mapped[name+'_pose'],profile)
            output.append({'id':target['id'],**mapped})
        return {'calibration_digest':self.digest,'targets':output}

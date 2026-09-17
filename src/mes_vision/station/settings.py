"""Versioned moving-camera setup; missing physical values remain unconfigured."""
from copy import deepcopy
from pathlib import Path
import json,os
from uuid import uuid4
from filelock import FileLock
from mes_vision.training.data import require,sha256
from .calibration import MountedCalibration


def defaults():
    return {'schema_version':1,'version':0,'calibration_bundle':None,'calibration_sha256':None,
            'settle_seconds':None,'action_timeout_seconds':None,'validation_reference':'',
            'max_frame_age_seconds':None,'timing_validation_reference':''}


class StationSettings:
    def __init__(self,runtime):
        self.path=Path(runtime)/'station-settings.json'
        self.value=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else defaults()
        self.original=deepcopy(self.value)
        require(set(self.value)==set(defaults()) and self.value['schema_version']==1
                and type(self.value['version']) is int and self.value['version']>=0,'Invalid moving-camera settings')

    def save(self,value):
        value=deepcopy(value)
        require(set(value)==set(defaults()) and value['schema_version']==1 and value['version']==self.original['version'],'Invalid or stale settings')
        for key,low,high in (('settle_seconds',.1,10),('action_timeout_seconds',1,300),('max_frame_age_seconds',.001,2)):
            item=value[key]; require(item is None or type(item) in (int,float) and low<=item<=high,'Invalid capture timing')
        require(isinstance(value['validation_reference'],str),'Validation reference must be text')
        require(isinstance(value['timing_validation_reference'],str),'Timing validation reference must be text')
        if value['calibration_bundle']:
            path=Path(value['calibration_bundle']).resolve(); MountedCalibration.load(path)
            value['calibration_bundle']=str(path); value['calibration_sha256']=sha256(path)
        else: value['calibration_bundle']=value['calibration_sha256']=None
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with FileLock(str(self.path)+'.lock',timeout=0):
            current=json.loads(self.path.read_text(encoding='utf-8')) if self.path.exists() else defaults()
            require(current==self.original,'Settings changed in another window; reopen settings')
            value['version']+=1; temporary=self.path.with_name('.station-settings-'+uuid4().hex+'.tmp')
            try:
                with temporary.open('x',encoding='utf-8') as stream:
                    json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
                temporary.replace(self.path)
            finally:
                if temporary.exists(): temporary.unlink()
        self.value=value; self.original=deepcopy(value)

    def calibrated(self):
        v=self.value
        require(v['calibration_bundle'] and v['calibration_sha256'] and v['validation_reference'].strip()
                and v['settle_seconds'] is not None and v['action_timeout_seconds'] is not None
                and v['max_frame_age_seconds'] is not None and v['timing_validation_reference'].strip(),'Moving-camera capture setup is incomplete')
        require(sha256(Path(v['calibration_bundle']))==v['calibration_sha256'],'Registered moving-camera calibration changed')
        return MountedCalibration.load(v['calibration_bundle'])

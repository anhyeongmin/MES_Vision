"""Explicit same-optics reattachment; preserve packaged calibration and guards."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from filelock import FileLock
from mes_vision.training.data import read_json, write_json, sha256, require
from .acquisition import acquisition_config_path, configure_equipment


def plan_binding(root, camera, *, confirmed=False):
    require(confirmed, '동일한 카메라·렌즈·초점임을 확인해야 합니다.')
    root=Path(root).resolve(); source=acquisition_config_path(root)
    config=read_json(source); spec=deepcopy(config['processing'])
    require(config['schema_version']==1 and spec['kind']=='undistorted_center_square_v1','보정 형식 오류')
    require(camera.get('driver')=='uvc' and camera.get('serial','').startswith('uvc:'),'U20CAM을 먼저 선택하세요.')
    require([camera['width'],camera['height']]==spec['sensor_size'],'기존 보정과 같은 해상도를 선택하세요.')
    lens=(root/spec['calibration_file']).resolve()
    require(sha256(lens)==spec['calibration_sha256'],'보정 파일이 변경됐습니다. 재보정이 필요합니다.')
    profile=read_json(lens)
    require(profile['image_size']==spec['sensor_size'],'보정 해상도 불일치')
    require(spec.get('camera_serial') and spec['camera_serial']!=camera['serial'],'이미 연결된 카메라입니다.')
    previous=spec['camera_serial']; spec['camera_serial']=camera['serial']
    spec['calibration_file']=str(lens)
    return dict(spec=spec, previous_serial=previous, source=str(source), source_sha256=sha256(source),
                acquisition_revision='acq-rebind-'+uuid4().hex[:16])


def bound_equipment(root, equipment, plan):
    require(equipment['camera']['serial']==plan['spec']['camera_serial'],'선택한 카메라가 변경됐습니다. 다시 연결 확인하세요.')
    e=deepcopy(equipment)
    e['camera']['acquisition_revision']=plan['acquisition_revision']
    e['calibration']=None
    e['workspace'].update(roi=[],excluded=[],validation_reference='')
    e['workspace'].pop('acquisition_identity',None)
    return configure_equipment(root,e,spec_override=plan['spec'])


def persist_binding(root, equipment, plan, store):
    root=Path(root).resolve(); folder=root/'artifacts/operation'; folder.mkdir(parents=True,exist_ok=True)
    path=folder/'camera-acquisition.json'
    with FileLock(str(folder/'camera-acquisition.lock'),timeout=0):
        source=acquisition_config_path(root)
        require(str(source.resolve())==str(Path(plan['source']).resolve()) and sha256(source)==plan['source_sha256'],
                '보정 설정이 변경됐습니다. 장비 설정을 다시 여세요.')
        require(sha256(Path(plan['spec']['calibration_file']))==plan['spec']['calibration_sha256'],'보정 파일이 변경됐습니다.')
        e=bound_equipment(root,equipment,plan)
        from .quality import validate_equipment
        validate_equipment(e)
        backup=folder/'camera-binding-history'/uuid4().hex; backup.mkdir(parents=True)
        old=path.read_bytes() if path.exists() else None
        (backup/'previous-acquisition.json').write_bytes(source.read_bytes())
        write_json(backup/'previous-equipment.json',store.equipment())
        record=dict(schema_version=1,processing=plan['spec'],binding=dict(
            confirmed_same_camera_lens_focus=True,previous_serial=plan['previous_serial'],
            new_serial=plan['spec']['camera_serial'],utc=datetime.now(timezone.utc).isoformat(),
            optical_recalibration_performed=False,robot_coordinates_validated=False))
        write_json(backup/'requested-acquisition.json',record)
        temp=path.with_suffix('.new'); write_json(temp,record); temp.replace(path)
        try:
            saved=store.save_equipment(e)
        except Exception:
            if old is None:path.unlink(missing_ok=True)
            else:
                temp.write_bytes(old);temp.replace(path)
            raise
        return saved

"""Explicit sensor -> inspection pixel contract; raw recordings remain raw.

This is a new acquisition coordinate system, not a transform of a robot map.
Its fingerprint must be part of newly measured calibration and ROI identities.
"""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from mes_vision.anomaly.features import fingerprint
from mes_vision.training.data import read_json, require
from mes_vision.inputs.camera_identity import camera_driver


def processing(camera):
    spec = camera.get('inspection_acquisition')
    if spec is None: return None
    require(camera_driver(camera) == 'uvc', 'Rectification is registered for UVC only')
    require(not spec.get('camera_serial') or spec['camera_serial']==camera.get('serial'), 'Lens calibration belongs to another camera')
    require(spec['kind'] == 'undistorted_center_square_v1', 'Unknown acquisition pipeline')
    require(list(spec['sensor_size']) == [camera['width'], camera['height']], 'Acquisition sensor resolution changed')
    return spec


def identity(camera):
    spec = processing(camera)
    if not spec: return None
    return fingerprint({k: spec[k] for k in ('kind', 'sensor_size', 'calibration_sha256')})


def inspection_camera(camera):
    c = deepcopy(camera)
    if processing(c):
        c['width'] = c['height'] = min(camera['width'], camera['height'])
        c['acquisition_revision'] = camera['acquisition_revision'] + ':rectified-square:' + identity(camera)
    return c


def acquisition_config_path(root):
    local = Path(root)/'artifacts/operation/camera-acquisition.json'
    return local if local.exists() else Path(root)/'configs/camera/acquisition.json'


def configure_equipment(root, equipment, *, spec_override=None):
    """Attach the installed lens profile; invalidate old ROI approval in memory.

    Persisting/confirming the new ROI still uses the existing equipment dialog.
    Never translate old polygons or robot maps by a guessed crop offset.
    """
    e = deepcopy(equipment); c = e['camera']
    path = acquisition_config_path(root)
    if camera_driver(c) != 'uvc':
        if c.pop('inspection_acquisition', None):
            w=e['workspace']; w.pop('acquisition_identity',None)
            w.update(roi=[],excluded=[],validation_reference='')
        return e
    # An unregistered camera must be configurable before a lens profile can bind.
    # Registered cameras still pass through processing() and its identity guards.
    if not c.get('serial'):
        c.pop('inspection_acquisition', None)
        e['calibration'] = None
        e['workspace'].update(roi=[], excluded=[], validation_reference='')
        e['workspace'].pop('acquisition_identity', None)
        return e
    if not path.exists() and spec_override is None: return e
    config = read_json(path) if spec_override is None else {'schema_version':1,'processing':spec_override}
    require(config['schema_version'] == 1, 'Acquisition configuration version changed')
    spec = deepcopy(config['processing'])
    lens = (Path(root)/spec['calibration_file']).resolve()
    spec['calibration_file'] = str(lens)
    c['inspection_acquisition'] = spec
    key = identity(c); w = e['workspace']
    if w.get('acquisition_identity') != key:
        side = min(c['width'], c['height'])
        w.update(roi=[[0,0],[side,0],[side,side],[0,side]], excluded=[],
                 validation_reference='', acquisition_identity=key)
        # This legacy fixed-camera map cannot describe the new pixel domain.
        # Retain its on-disk asset; the station's measured bundle remains guarded.
        e['calibration'] = None
    return e


def verify_frame(frame, camera):
    c = inspection_camera(camera)
    require((frame.width, frame.height) == (c['width'], c['height']), 'Inspection frame dimensions changed')
    require(frame.acquisition_identity == identity(camera), 'Inspection acquisition identity changed')
    require(not frame.transformations and frame.coordinate_space=='input_rgb_pixels', 'Inspection input was transformed again')
    spec=processing(camera)
    if spec:
        w,h=spec['sensor_size']; side=min(w,h); source=frame.acquisition_source or {}
        require(source.get('sensor_size')==[w,h] and source.get('crop_xyxy')==[(w-side)//2,(h-side)//2,(w+side)//2,(h+side)//2]
                and source.get('calibration_sha256')==spec['calibration_sha256'], 'Inspection acquisition provenance changed')
        require(frame.frame_id==source.get('raw_frame_id','')+':acq:'+identity(camera), 'Inspection/raw frame identity changed')


class Acquisition:
    def __init__(self, camera):
        from mes_vision.station.manual_capture import Rectifier
        self.camera = deepcopy(camera); spec = processing(camera)
        self.key = identity(camera)
        self.rectifier = Rectifier(spec['calibration_file']) if spec else None
        if spec:
            require(self.rectifier.digest == spec['calibration_sha256'], 'Lens calibration file changed')
            require(list(self.rectifier.size) == spec['sensor_size'], 'Lens sensor resolution changed')

    def apply(self, raw):
        require(not raw.transformations and raw.acquisition_identity is None,
                'Acquisition requires an unprocessed sensor frame')
        if self.rectifier is None: return raw
        require(raw.source_kind.value == 'uvc', 'Wrong camera for lens calibration')
        _, rgb, box = self.rectifier.apply(raw.rgb, square=True)
        rgb.setflags(write=False)
        # Distinct IDs avoid aliasing raw and processed pixels in saved evidence.
        return replace(raw, frame_id=raw.frame_id+':acq:'+self.key,
                       rgb=rgb, encoded_size=(rgb.shape[1],rgb.shape[0]),
                       acquisition_identity=self.key,
                       acquisition_source={'raw_frame_id':raw.frame_id,
                           'sensor_size':list(self.rectifier.size), 'crop_xyxy':list(box),
                           'calibration_sha256':self.rectifier.digest})

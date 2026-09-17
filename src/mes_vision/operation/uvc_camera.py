"""UVC RGB capture. No device exposure timestamp or lens calibration is invented."""
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import hashlib
import math
import platform
import time
import numpy as np
from mes_vision.inputs import Frame, SourceKind
from mes_vision.inputs.camera_identity import camera_uri
from mes_vision.training.data import require


def device_identity(backend, path):
    require(isinstance(path, str) and bool(path.strip()), 'Camera has no stable device path')
    return 'uvc:' + str(backend) + ':' + hashlib.sha256(path.encode('utf-8')).hexdigest()


class UVCCamera:
    def __init__(self, settings, *, cv=None, enumerate_fn=None, clock=time.monotonic):
        if cv is None:
            import cv2 as cv
        self.cv=cv; self.settings=dict(settings); self.enumerate_fn=enumerate_fn; self.clock=clock
        self.settings.update(settings.get('capture_profiles',{}).get('overview',{}))
        self.capture=None; self.sequence=0; self.session=uuid4().hex; self.info={}; self.last_read=None

    @staticmethod
    def devices(*, cv=None, enumerate_fn=None):
        if cv is None:
            import cv2 as cv
        if enumerate_fn is None:
            from cv2_enumerate_cameras import enumerate_cameras
            enumerate_fn=enumerate_cameras
        system=platform.system()
        require(system in ('Windows','Linux'), 'UVC capture supports Windows and Linux')
        backend=cv.CAP_DSHOW if system=='Windows' else cv.CAP_V4L2
        rows=[]
        for item in enumerate_fn(backend):
            path=item.path
            if not path: continue
            if system=='Linux':
                # /dev/videoN can refer to a different camera after replugging.
                aliases=[p for directory in ('/dev/v4l/by-id','/dev/v4l/by-path')
                         for p in sorted(Path(directory).glob('*')) if p.resolve()==Path(path).resolve()]
                if not aliases: continue
                path=str(aliases[0])
            rows.append({'serial':device_identity(backend,path),'name':item.name,'path':path,
                'index':item.index,'backend':backend,'driver':'uvc','profiles':None,'profile_error':None})
        require(len({r['serial'] for r in rows})==len(rows), 'Ambiguous camera device identities')
        return sorted(rows,key=lambda r:(r['name'],r['serial']))

    def _devices(self): return self.devices(cv=self.cv,enumerate_fn=self.enumerate_fn)

    def open(self):
        cv=self.cv; c=self.settings
        require(c.get('serial','').startswith('uvc:'), 'USB 카메라를 검색하고 선택하세요.')
        selected=[r for r in self._devices() if r['serial']==c['serial']]
        require(len(selected)==1, '선택한 USB 카메라를 찾을 수 없습니다. 다시 검색하세요.')
        device=selected[0]; pixel_format=c.get('pixel_format','MJPG')
        from .camera_modes import UVC_PRESETS
        require(c['fps'] in UVC_PRESETS.get(pixel_format,{}).get((c['width'],c['height']),()), 'U20CAM 촬영 모드를 확인하세요.')
        try:
            target=device['index'] if platform.system()=='Windows' else device['path']
            self.capture=cv.VideoCapture(target,device['backend'])
            cap=self.capture
            require(cap.isOpened(), '카메라를 열 수 없습니다. 다른 프로그램의 카메라 사용 여부를 확인하세요.')
            require(cap.getBackendName()==('DSHOW' if platform.system()=='Windows' else 'V4L2'), 'Camera backend changed')
            again=[r for r in self._devices() if r['serial']==c['serial']]
            require(len(again)==1 and again[0]['index']==device['index'] and again[0]['path']==device['path'], 'Camera list changed while connecting; reconnect')
            cap.set(cv.CAP_PROP_FPS,c['fps'])
            cap.set(cv.CAP_PROP_FRAME_WIDTH,c['width']); cap.set(cv.CAP_PROP_FRAME_HEIGHT,c['height'])
            # DirectShow rebuilds its graph on resolution changes and can reset
            # the subtype to YUY2. Negotiate the requested subtype LAST.
            cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*pixel_format))
            cap.set(cv.CAP_PROP_BUFFERSIZE,1)
            auto=float(c['auto_exposure']) if platform.system()=='Windows' else (.75 if c['auto_exposure'] else .25)
            require(cap.set(cv.CAP_PROP_AUTO_EXPOSURE,auto), '자동/수동 노출 설정을 적용하지 못했습니다.')
            actual_auto=cap.get(cv.CAP_PROP_AUTO_EXPOSURE)
            # DirectShow supports setting this property but does not implement its getter.
            if platform.system()!='Windows':
                require(math.isfinite(actual_auto) and abs(actual_auto-auto)<.01, '노출 설정이 요청값과 다릅니다.')
            if not c['auto_exposure']:
                require(type(c['exposure']) in (int,float) and math.isfinite(c['exposure']), '수동 노출값을 입력하세요.')
                require(cap.set(cv.CAP_PROP_EXPOSURE,c['exposure']), '수동 노출 설정을 적용하지 못했습니다.')
                require(abs(cap.get(cv.CAP_PROP_EXPOSURE)-c['exposure'])<.01, '수동 노출값이 요청값과 다릅니다.')
            self._verify_mode()
            self.info={'driver':'uvc','serial':c['serial'],'name':device['name'],'device_path':device['path'],
                'backend':cap.getBackendName(),'width':c['width'],'height':c['height'],'fps':cap.get(cv.CAP_PROP_FPS),
                'pixel_format':pixel_format,'settings':c,'distortion':'unverified','firmware':'unavailable',
                'timing_basis':'host_read_completion','device_timestamp_available':False,
                'auto_exposure_readback_available':platform.system()!='Windows'}
            if c.get('capture_profiles'):
                self.apply_controls(c['capture_profiles']['overview'])
        except BaseException:
            self.close(); raise

    def controls_info(self):
        from .capture_controls import directshow_ranges
        return directshow_ranges(self.info['device_path'])

    def apply_controls(self, values):
        from .capture_controls import validate_controls
        ranges=self.controls_info(); validate_controls(values,ranges)
        cv=self.cv; cap=self.capture
        auto=float(values['auto_exposure']) if platform.system()=='Windows' else (.75 if values['auto_exposure'] else .25)
        require(cap.set(cv.CAP_PROP_AUTO_EXPOSURE,auto),'자동 노출 설정 실패')
        for key,prop in (('exposure',cv.CAP_PROP_EXPOSURE),('gain',cv.CAP_PROP_GAIN)):
            value=values.get(key)
            if value is None or key=='exposure' and values['auto_exposure']: continue
            require(cap.set(prop,value),'카메라 '+key+' 적용 실패')
            require(abs(cap.get(prop)-value)<.01,'카메라 '+key+' 확인값 불일치')
        self.settings.update(values); self.info['settings']=dict(self.settings)
        return {'values':dict(values),'ranges':ranges,'applied_at':self.clock()}

    def _verify_mode(self):
        cv=self.cv; cap=self.capture; c=self.settings
        require(all(math.isfinite(cap.get(prop)) and abs(cap.get(prop)-wanted)<.5 for prop,wanted in (
            (cv.CAP_PROP_FRAME_WIDTH,c['width']),(cv.CAP_PROP_FRAME_HEIGHT,c['height']),(cv.CAP_PROP_FPS,c['fps']))),
            '카메라가 요청한 해상도·프레임 속도로 연결되지 않았습니다.')
        fourcc=int(cap.get(cv.CAP_PROP_FOURCC))
        require(fourcc==cv.VideoWriter_fourcc(*c.get('pixel_format','MJPG')), '카메라 전송 형식이 요청값과 다릅니다.')

    def read(self):
        require(self.capture is not None, '카메라가 연결되지 않았습니다.')
        ok,bgr=self.capture.read(); completed=self.clock()
        require(ok and bgr is not None, '카메라 영상 수신이 중단됐습니다. 다시 연결하세요.')
        require(math.isfinite(completed), 'Camera host clock is not finite')
        # Windows Python 3.12 monotonic uses GetTickCount64 (15.625 ms).
        # A second read in the same tick is not clock reversal. Discard it:
        # published frames must still have strictly increasing host timestamps.
        if self.last_read is not None and completed == self.last_read:
            return None
        require(self.last_read is None or completed>self.last_read,
                f'Camera host clock moved backward (previous={self.last_read!r}, current={completed!r})')
        self.last_read=completed
        require(bgr.dtype==np.uint8 and bgr.ndim==3 and bgr.shape[2]==3, 'Expected 8-bit camera color image')
        require((bgr.shape[1],bgr.shape[0])==(self.settings['width'],self.settings['height']), '카메라 영상 크기가 변경됐습니다.')
        self._verify_mode()
        rgb=np.ascontiguousarray(bgr[:,:,::-1]); rgb.setflags(write=False)
        frame=Frame(f'{self.session}:{self.sequence}',self.session,self.sequence,SourceKind.UVC,
            camera_uri('uvc',self.settings['serial']),datetime.now(timezone.utc),rgb,
            encoded_size=(rgb.shape[1],rgb.shape[0]),is_live=True,host_read_completed_monotonic=completed)
        self.sequence+=1
        return frame

    def close(self):
        if self.capture is not None:
            try: self.capture.release()
            finally: self.capture=None

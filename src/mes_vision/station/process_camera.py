"""Fresh camera selection on the UI; disk writes in the bounded model process."""
from copy import deepcopy
from mes_vision.training.data import require
from .camera_port import CaptureGate


class ProcessCameraPort:
    def __init__(self,camera,engine,settings):
        self.camera=camera; self.engine=engine; self.settings=deepcopy(settings); self.gate=None; self.pending=None; self.closed=False; self.switching=None; self.controls=None
    @property
    def busy(self): return self.switching is not None or self.gate is not None or self.pending is not None and self.engine.process.is_alive()
    def submit(self,request,profile):
        require(not self.busy and not self.closed,'Capture is busy or closed')
        camera_settings=getattr(self.camera,'settings',{})
        if isinstance(camera_settings,dict) and camera_settings.get('capture_profiles'):
            name='overview' if request['kind']=='capture_overview' else 'detail'
            self.switching=(self.camera.command('profile',{'name':name}),deepcopy(request),deepcopy(profile))
        else: self.gate=CaptureGate(request,profile,self.settings['max_frame_age_seconds'],self.settings['timing_validation_reference'])
    def tick(self,now):
        if self.switching:
            token,request,profile=self.switching
            require(now<request['deadline'],'촬영 조건 적용 시간이 초과됐습니다.')
            reply=self.camera.replies.pop(token,None)
            if reply is None: return None
            require('error' not in reply,reply.get('error','촬영 조건 적용 실패'))
            self.controls=deepcopy(reply['result'])
            self.switching=None
            self.gate=CaptureGate(request,profile,self.settings['max_frame_age_seconds'],self.settings['timing_validation_reference'])
            self.gate.not_before=max(request['issued_at'],reply['result']['applied_at'])
        if self.gate:
            gate=self.gate; frame,received=self.camera.get_latest(); timing=gate.observe(frame,received,now)
            if timing:
                if self.controls: timing['capture_controls']=deepcopy(self.controls)
                self.engine.submit(gate.request,frame=frame,timing=timing,camera_serial=gate.profile['camera_serial'],camera_session=gate.profile['camera_session'])
                self.pending=gate.request; self.gate=None
        return None
    def finished(self,request):
        if request==self.pending: self.pending=None
    def cancel(self):
        self.gate=None; self.switching=None
        if self.pending: self.engine.stop()
    def close(self): self.closed=True; self.cancel()

"""Fresh one-shot captures from the existing live CameraProcess.

The installation must validate a worst-case exposure-to-host latency bound for
its selected camera profile. Device timestamps are used to advance beyond a
post-request anchor for RealSense. UVC uses the host read completion minus the
validated complete exposure-to-host pipeline bound. Neither host timestamp is
relabelled as a device exposure timestamp.
"""
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import math
from mes_vision.training.data import require
from .evidence import save_capture
from mes_vision.inputs.camera_identity import matches_camera


class CaptureGate:
    def __init__(self,request,profile,max_frame_age_seconds,validation_reference):
        require(request['kind'] in {'capture_overview','capture_detail'},'Capture request required')
        require(type(max_frame_age_seconds) in (int,float) and math.isfinite(max_frame_age_seconds)
                and 0<max_frame_age_seconds<=2 and validation_reference,'Validated exposure-to-host latency bound required')
        self.request=deepcopy(request); self.profile=deepcopy(profile); self.bound=max_frame_age_seconds
        self.reference=validation_reference; self.anchor=None; self.last_sequence=-1; self.last_device=None
        self.not_before=request['issued_at']
        self.last_received=None

    def observe(self,frame,received,now):
        require(self.request['issued_at']<=now<self.request['deadline'],'Capture request expired')
        if frame is None: return None
        p=self.profile
        require(matches_camera(frame,p['camera_serial'],p.get('camera_driver','d405'))
                and frame.session_id==p['camera_session'],'Camera session/device changed')
        require([frame.width,frame.height]==list(p['image_size']) and not frame.transformations
                and frame.coordinate_space=='input_rgb_pixels','Camera geometry changed')
        require(frame.acquisition_identity==p.get('acquisition_identity'),'Camera acquisition changed')
        require(type(received) in (int,float) and math.isfinite(received) and received<=now,'Invalid camera receive time')
        if received<self.not_before: return None
        if frame.sequence==self.last_sequence: return None
        require(frame.sequence>self.last_sequence,'Camera sequence moved backward')
        if frame.source_kind.value=='uvc':
            stamp=frame.host_read_completed_monotonic
            require(type(stamp) in (int,float) and math.isfinite(stamp) and stamp<=received,
                    'UVC host read-completion evidence unavailable')
            require(frame.media_time_seconds is None and frame.captured_at_utc is None,
                    'Host time must not be relabelled as camera exposure time')
            require(self.last_received is None or received>=self.last_received, 'Host receipt moved backward')
            require(self.last_device is None or stamp>self.last_device, 'UVC read time moved backward')
            self.last_sequence=frame.sequence; self.last_device=stamp; self.last_received=received
            require(now-stamp<=self.bound, 'Live preview frame is stale')
            earliest=stamp-self.bound
            if earliest<self.not_before: return None
            return {'acquired_at':earliest,'timing_basis':'validated_pipeline_bound_and_host_read_completion',
                    'timing_validation_reference':self.reference,'max_frame_age_seconds':self.bound,
                    'host_read_completed_monotonic':stamp,'received_monotonic':received}
        stamp=frame.media_time_seconds
        require(type(stamp) in (int,float) and math.isfinite(stamp),'Device timestamp unavailable; capture timing cannot be established')
        if self.last_device is not None:
            require(stamp>self.last_device and received>=self.last_received,'Camera time moved backward')
            require(stamp-self.last_device<=(received-self.last_received)*1.01+self.bound,'Camera time jumped outside the validated latency bound')
        self.last_sequence=frame.sequence; self.last_device=stamp; self.last_received=received
        if self.anchor is None:
            self.anchor=(stamp,received); return None
        device_elapsed=(stamp-self.anchor[0])/1.01  # Conservative 1% device-rate allowance.
        earliest=self.anchor[1]-self.bound+device_elapsed
        require(earliest<=received,'Camera timing evidence is inconsistent')
        if earliest<self.not_before: return None
        require(now-received<=self.bound,'Live preview frame is stale')
        return {'acquired_at':earliest,'timing_basis':'validated_exposure_to_host_bound_and_device_clock',
                'timing_validation_reference':self.reference,'max_frame_age_seconds':self.bound,
                'anchor_device_seconds':self.anchor[0],'anchor_received_monotonic':self.anchor[1],
                'device_seconds':stamp,'received_monotonic':received}


class StationCameraPort:
    """Uses a connected CameraProcess; stopping a scan leaves preview connected."""
    def __init__(self,camera,capture_root,settings):
        self.camera=camera; self.root=capture_root; self.settings=deepcopy(settings)
        self.writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix='station-capture')
        self.gate=None; self.saving=None; self.closed=False; self.discard=False

    @property
    def busy(self): return self.gate is not None or self.saving is not None

    def submit(self,request,profile):
        require(not self.closed and not self.busy,'Capture port is busy or closed')
        self.gate=CaptureGate(request,profile,self.settings['max_frame_age_seconds'],self.settings['timing_validation_reference'])

    def tick(self,now):
        if self.saving:
            request,future=self.saving
            if not future.done(): return None
            self.saving=None
            if self.discard:
                self.discard=False
                # Drain even failed/cancelled work without publishing a stale completion.
                if not future.cancelled(): future.exception()
                return None
            return {'request':request,'payload':future.result()}
        if self.gate is None: return None
        gate=self.gate; frame,received=self.camera.get_latest(); timing=gate.observe(frame,received,now)
        if timing is None: return None
        request=gate.request; self.gate=None
        def save():
            receipt=save_capture(self.root,frame,camera_serial=gate.profile['camera_serial'],acquired_at=timing['acquired_at'])
            receipt['timing']=timing; return receipt
        self.saving=(request,self.writer.submit(save)); return None

    def cancel(self):
        self.gate=None
        if self.saving:
            # Already-written original files remain unlinked evidence; no late callback.
            self.discard=True; self.saving[1].cancel()

    def close(self):
        self.closed=True; self.cancel(); self.writer.shutdown(wait=False,cancel_futures=True)
        # A running disk write cannot be killed safely. Retain it in busy until tick()
        # drains it; callers must finish draining before terminating the application.

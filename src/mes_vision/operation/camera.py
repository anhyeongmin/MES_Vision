from dataclasses import replace
from datetime import datetime,timezone
import threading
import multiprocessing as mp
import os
import queue
import math
import time
from uuid import uuid4
import numpy as np
from PySide6.QtCore import QObject,QTimer,QThread,Signal
from mes_vision.inputs import Frame,SourceKind


class D405Camera:
    def __init__(self,settings,*,sdk=None):
        if sdk is None:
            import pyrealsense2 as sdk
        self.rs=sdk; self.settings=dict(settings); self.pipeline=None; self.session=uuid4().hex; self.sequence=0; self.info={}; self.last_number=None; self.last_timestamp=None
    @staticmethod
    def devices(*,sdk=None):
        """Enumerate RGB8 capabilities without starting a stream or changing options."""
        if sdk is None:
            import pyrealsense2 as sdk
        rs=sdk; found=[]
        for device in rs.context().query_devices():
            name=device.get_info(rs.camera_info.name)
            if "D405" not in name: continue
            entry={"serial":device.get_info(rs.camera_info.serial_number),"name":name,"profiles":None,"profile_error":None}
            try:
                modes=set()
                for sensor in device.query_sensors():
                    for profile in sensor.get_stream_profiles():
                        if profile.stream_type()!=rs.stream.color or profile.format()!=rs.format.rgb8: continue
                        video=profile.as_video_stream_profile()
                        modes.add((video.width(),video.height(),video.fps()))
                entry["profiles"]=[{"width":w,"height":h,"fps":fps} for w,h,fps in sorted(modes,reverse=True)]
            except Exception as exc: entry["profile_error"]=str(exc)
            found.append(entry)
        return sorted(found,key=lambda item:item["serial"])
    def open(self):
        rs=self.rs; c=self.settings
        if not c["serial"]: raise ValueError("D405 장치를 선택하세요.")
        device=next((d for d in rs.context().query_devices() if d.get_info(rs.camera_info.serial_number)==c["serial"]),None)
        if device is None or "D405" not in device.get_info(rs.camera_info.name): raise ValueError("선택한 D405를 찾을 수 없습니다.")
        pipeline=rs.pipeline(); config=rs.config(); config.enable_device(c["serial"])
        config.enable_stream(rs.stream.color,c["width"],c["height"],rs.format.rgb8,c["fps"])
        try:
            profile=pipeline.start(config); self.pipeline=pipeline
            video=profile.get_stream(rs.stream.color).as_video_stream_profile(); intrinsics=video.get_intrinsics()
            for sensor in profile.get_device().query_sensors():
                profiles=sensor.get_stream_profiles()
                if not any(p.stream_type()==rs.stream.color for p in profiles): continue
                if sensor.supports(rs.option.enable_auto_exposure): sensor.set_option(rs.option.enable_auto_exposure,float(c["auto_exposure"]))
                if not c["auto_exposure"]:
                    if c["exposure"] is None or not sensor.supports(rs.option.exposure): raise ValueError("수동 노출값 또는 해당 장치 지원을 확인하세요.")
                    bounds=sensor.get_option_range(rs.option.exposure)
                    if not bounds.min<=c["exposure"]<=bounds.max: raise ValueError("노출값이 장치 지원 범위를 벗어났습니다.")
                    sensor.set_option(rs.option.exposure,c["exposure"])
            self.info={"serial":c["serial"],"width":intrinsics.width,"height":intrinsics.height,"fx":intrinsics.fx,"fy":intrinsics.fy,
                "ppx":intrinsics.ppx,"ppy":intrinsics.ppy,"coeffs":list(intrinsics.coeffs),"distortion":str(intrinsics.model),
                "firmware":device.get_info(rs.camera_info.firmware_version),"settings":c}
        except Exception:
            self.close(); raise
    def read(self):
        if self.pipeline is None: raise ValueError("카메라가 연결되지 않았습니다.")
        ok,frames=self.pipeline.try_wait_for_frames(150)
        if not ok: return None
        color=frames.get_color_frame()
        if not color: return None
        number=color.get_frame_number(); timestamp=color.get_timestamp()
        if not math.isfinite(timestamp): raise RuntimeError("카메라 시간 정보가 유효하지 않습니다.")
        if self.last_number is not None:
            if number==self.last_number: return None
            if number<self.last_number or timestamp<=self.last_timestamp: raise RuntimeError("카메라 프레임 순서가 초기화됐습니다. 다시 연결하세요.")
        self.last_number=number; self.last_timestamp=timestamp
        rgb=np.array(np.asanyarray(color.get_data()),dtype=np.uint8,order="C",copy=True); rgb.setflags(write=False)
        c=self.settings
        if (rgb.shape[1],rgb.shape[0])!=(c["width"],c["height"]): raise ValueError("카메라 영상 크기가 변경됐습니다.")
        frame=Frame(f"{self.session}:{self.sequence}",self.session,self.sequence,SourceKind.D405,
            "realsense://"+c["serial"],datetime.now(timezone.utc),rgb,media_time_seconds=timestamp/1000.,
            encoded_size=(rgb.shape[1],rgb.shape[0]),is_live=True)
        self.sequence+=1; return frame
    def close(self):
        if self.pipeline is not None:
            try: self.pipeline.stop()
            finally: self.pipeline=None


def create_camera(settings):
    from mes_vision.inputs.camera_identity import camera_driver
    if camera_driver(settings)=='uvc':
        from .uvc_camera import UVCCamera
        return UVCCamera(settings)
    return D405Camera(settings)


class CameraThread(QThread):
    connected=Signal(object); failed=Signal(str)
    def __init__(self,settings,*,factory=create_camera):
        super().__init__(); self.settings=settings; self.factory=factory; self.stopping=threading.Event()
        self.lock=threading.Lock(); self.latest=None; self.raw_latest=None; self.received_at=0.
    def get_latest(self):
        with self.lock: return self.latest,self.received_at
    def get_latest_raw(self):
        with self.lock: return self.raw_latest,self.received_at
    def run(self):
        camera=None
        try:
            from .acquisition import Acquisition
            acquisition=Acquisition(self.settings)
            camera=self.factory(self.settings); camera.open(); self.connected.emit(camera.info); last=time.monotonic()
            while not self.stopping.is_set():
                frame=camera.read()
                if frame is not None:
                    last=time.monotonic()
                    view=acquisition.apply(frame)
                    with self.lock: self.latest=view; self.raw_latest=frame; self.received_at=last
                elif time.monotonic()-last>2.: raise RuntimeError("카메라 영상 수신이 중단됐습니다. 연결을 확인하고 다시 연결하세요.")
        except Exception as exc: self.failed.emit(str(exc))
        finally:
            if camera:
                try: camera.close()
                except Exception: pass
            with self.lock: self.latest=None; self.raw_latest=None; self.received_at=0.


def camera_process(settings,frames,events,cancel,parent_pid,factory,commands=None):
    from mes_vision.vlm.worker import watch_parent
    from .engine import latest_put
    watch_parent(parent_pid); camera=None; recorder=None
    try:
        from .acquisition import Acquisition
        acquisition=Acquisition(settings)
        camera=factory(settings); camera.open(); events.put({"type":"connected","info":camera.info},timeout=1)
        last=time.monotonic()
        while not cancel.is_set():
            try: command=commands.get_nowait() if commands is not None else None
            except queue.Empty: command=None
            if command:
                try:
                    kind=command['kind']; payload=command.get('payload',{})
                    if kind=='ranges': result=camera.controls_info()
                    elif kind in ('controls','profile'):
                        if recorder: raise ValueError('녹화 중에는 촬영 조건을 바꿀 수 없습니다.')
                        values=payload if kind=='controls' else settings.get('capture_profiles',{}).get(payload['name'])
                        result=camera.apply_controls(values) if values else {'applied_at':time.monotonic(),'values':{}}
                    elif kind=='record_start':
                        if recorder: raise ValueError('이미 녹화 중입니다.')
                        from .training_recorder import TrainingRecorder
                        recorder=TrainingRecorder(payload['root'],payload['metadata'],dict(camera.settings))
                        result={'folder':str(recorder.folder)}
                    elif kind=='record_stop':
                        if not recorder: raise ValueError('녹화 중이 아닙니다.')
                        active=recorder; recorder=None; result={'folder':active.close()}
                    else: raise ValueError('Unknown camera command')
                    events.put({'type':'reply','id':command['id'],'result':result},timeout=1)
                except Exception as exc:
                    events.put({'type':'reply','id':command['id'],'error':str(exc)},timeout=1)
            frame=camera.read()
            if frame is not None:
                if recorder: recorder.add(frame)
                last=time.monotonic(); latest_put(frames,(acquisition.apply(frame),last,frame))
            elif time.monotonic()-last>2: raise TimeoutError("카메라 영상 수신이 중단됐습니다. 다시 연결하세요.")
    except Exception as exc:
        try: events.put({"type":"error","message":str(exc)},timeout=.2)
        except Exception: pass
    finally:
        if recorder:
            try: recorder.close('INTERRUPTED')
            except Exception: pass
        if camera:
            try: camera.close()
            except Exception: pass
        frames.cancel_join_thread()


class CameraProcess(QObject):
    """Isolate native SDK calls so a blocked USB driver cannot trap the UI on close."""
    connected=Signal(object); failed=Signal(str); finished=Signal(); command_finished=Signal(object)
    def __init__(self,settings,*,factory=create_camera,connect_timeout=15.,frame_timeout=2.,stop_timeout=2.):
        super().__init__(); ctx=mp.get_context("spawn"); self.frames=ctx.Queue(1); self.events=ctx.Queue(4); self.cancel=ctx.Event()
        self.commands=ctx.Queue(8); self.replies={}; self.settings=dict(settings)
        self.process=ctx.Process(target=camera_process,args=(settings,self.frames,self.events,self.cancel,os.getpid(),factory,self.commands),daemon=True)
        self.stopping=threading.Event(); self.latest=None; self.raw_latest=None; self.received_at=0.; self.connected_at=None; self.stop_at=None; self.disposed=False
        self.connect_timeout=connect_timeout; self.frame_timeout=frame_timeout; self.stop_timeout=stop_timeout
        self.timer=QTimer(self); self.timer.timeout.connect(self.poll)
    def start(self):
        self.started=time.monotonic()
        try: self.process.start()
        except Exception:
            self.disposed=True
            for q in (self.frames,self.events,self.commands): q.cancel_join_thread(); q.close()
            raise
        self.timer.start(20)
    def get_latest(self): return self.latest,self.received_at
    def get_latest_raw(self): return self.raw_latest,self.received_at
    def command(self,kind,payload=None):
        token=uuid4().hex
        if self.disposed or self.stopping.is_set(): raise RuntimeError('카메라 연결이 종료됐습니다.')
        self.commands.put_nowait({'id':token,'kind':kind,'payload':payload or {}})
        return token
    def isRunning(self): return not self.disposed and self.process.is_alive()
    def fail(self,message):
        if self.stopping.is_set(): return
        self.stopping.set(); self.latest=None; self.raw_latest=None; self.received_at=0.; self.failed.emit(message)
    def poll(self):
        if self.disposed: return
        now=time.monotonic()
        if not self.stopping.is_set():
            try:
                for _ in range(8):
                    try: event=self.events.get_nowait()
                    except queue.Empty: break
                    if event["type"]=="connected": self.connected_at=now; self.connected.emit(event["info"])
                    elif event['type']=='reply':
                        self.replies[event['id']]=event
                        while len(self.replies)>32: self.replies.pop(next(iter(self.replies)))
                        self.command_finished.emit(event)
                    else: self.fail(event["message"])
                for _ in range(4):
                    try:
                        packet=self.frames.get_nowait()
                        frame,received=packet[:2]; raw=packet[2] if len(packet)==3 else frame
                    except queue.Empty: break
                    if not self.stopping.is_set() and now-received<=self.frame_timeout: self.latest=frame; self.raw_latest=raw; self.received_at=received
                if not self.connected_at and now-self.started>self.connect_timeout: self.fail("카메라 연결 시간이 초과됐습니다. USB 연결과 드라이버를 확인하세요.")
                elif self.connected_at and now-max(self.received_at,self.connected_at)>self.frame_timeout: self.fail("카메라 영상 응답이 멈췄습니다. 연결 해제 후 다시 연결하세요.")
            except Exception as exc: self.fail("카메라 처리 연결 오류: "+str(exc))
        if not self.process.is_alive():
            if not self.stopping.is_set(): self.fail("카메라 처리 프로세스가 종료됐습니다. 다시 연결하세요.")
            self.timer.stop(); self.process.join(timeout=0); self.latest=None; self.raw_latest=None; self.received_at=0.; self.disposed=True
            for q in (self.frames,self.events,self.commands): q.cancel_join_thread(); q.close()
            self.finished.emit(); return
        if self.stopping.is_set():
            if self.stop_at is None: self.stop_at=now; self.cancel.set()
            elif now-self.stop_at>self.stop_timeout: self.process.terminate()

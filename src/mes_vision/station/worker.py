"""Resident model process: one saved frame request at a time, no robot access."""
from pathlib import Path
from datetime import datetime
from dataclasses import asdict
from contextlib import nullcontext
import multiprocessing as mp
import os,queue,time,hashlib
import numpy as np
from PIL import Image
from filelock import FileLock
from mes_vision.training.data import require,sha256,write_json
from mes_vision.inputs.sources import Frame,SourceKind
from mes_vision.inputs.camera_identity import camera_uri,matches_camera
from mes_vision.anomaly.features import fingerprint
from mes_vision.vlm.gpu import GpuCoordinator
from .models import StationModels
from .evidence import save_detail,save_capture


def verify_acquisition(frame,equipment):
    # The saved-photo worker protocol also serves inputs without device settings.
    # Only legacy unprocessed captures may use that path; rectified inputs must
    # always have an explicitly registered acquisition contract.
    camera=equipment.get('camera')
    if camera is None:
        require(frame.acquisition_identity is None,'Rectified capture requires registered acquisition settings')
    else:
        from mes_vision.operation.acquisition import verify_frame
        verify_frame(frame,camera)


def read_capture(receipt):
    path=Path(receipt['path']); require(sha256(path)==receipt['sha256'],'Saved capture changed')
    with Image.open(path) as image:
        require(image.mode=='RGB' and list(image.size)==receipt['image_size'],'Capture image format changed')
        rgb=np.array(image,copy=True)
    require(hashlib.sha256(rgb.tobytes()).hexdigest()==receipt['rgb_sha256'],'Capture pixels changed')
    m=receipt['frame_metadata']; require(m['frame_id']==receipt['frame_id'] and m['session_id']==receipt['camera_session']
        and m['source_uri']==camera_uri(m['source_kind'],receipt['camera_serial']) and m['is_live'] is True,'Capture metadata changed')
    rgb.setflags(write=False)
    frame=Frame(m['frame_id'],m['session_id'],m['sequence'],SourceKind(m['source_kind']),m['source_uri'],
        datetime.fromisoformat(m['read_at_utc']),rgb,datetime.fromisoformat(m['captured_at_utc']) if m['captured_at_utc'] else None,
        m['media_time_seconds'],tuple(m['encoded_size']) if m['encoded_size'] else None,tuple(m['transformations']),m['coordinate_space'],m['is_live'],m.get('host_read_completed_monotonic'),m.get('acquisition_identity'),m.get('acquisition_source'))
    require(frame.sequence==receipt['sequence'] and [frame.width,frame.height]==receipt['image_size']
            and matches_camera(frame,receipt['camera_serial']) and not frame.transformations,'Capture geometry changed')
    return frame


def process_main(root,runtime,product,equipment,recipe,commands,events,stop,parent_pid,model_factory=None):
    from mes_vision.vlm.worker import watch_parent
    import threading
    threading.Thread(target=watch_parent,args=(parent_pid,),daemon=True).start()
    models=None; gpu=GpuCoordinator(Path(runtime)/'vlm'/'gpu-coordination'); owner=FileLock(str(gpu.root/'resident.lock'),timeout=0)
    try:
        with owner:
            try:
                models=(model_factory or StationModels)(Path(root),recipe['overview_asset'],product,capture_domains=recipe['capture_domains'])
                with gpu.foreground(timeout=60): models.load()
                service=models.service(overview_workspace=equipment['workspace'],detail_workspace=recipe['detail_workspace'])
                import torch
                concurrent=False
                if torch.cuda.is_available():
                    free,total=torch.cuda.mem_get_info(); concurrent=free>=12*1024**3 and total>=20*1024**3
                write_json(gpu.root/'resident.json',{'pid':os.getpid(),'allow_vlm':concurrent,'active':True})
                events.put({'type':'ready','overview_model_digest':service.overview_detector.model.weights_sha256,
                    'detail_policy_digest':fingerprint(asdict(service.policy)),'concurrent_vlm':concurrent},timeout=1)
                while not stop.is_set():
                    try: message=commands.get(timeout=.1)
                    except queue.Empty: continue
                    request=message['request']
                    try:
                        require(time.monotonic()<request['deadline'],'Inspection request expired')
                        if request['kind'] in {'capture_overview','capture_detail'}:
                            frame=message['frame']; timing=message['timing']
                            verify_acquisition(frame,equipment)
                            require(matches_camera(frame,message['camera_serial']) and frame.session_id==message['camera_session']
                                and request['issued_at']<=timing['acquired_at']<=time.monotonic(),'Fresh capture evidence required')
                            from mes_vision.operation.storage import check_free_space
                            from mes_vision.operation.catalog import OperationStore
                            check_free_space(OperationStore(runtime),frame.rgb.nbytes*3)
                            payload=save_capture(Path(runtime)/'station-captures',frame,camera_serial=message['camera_serial'],acquired_at=timing['acquired_at'])
                            payload['timing']=timing
                            if not stop.is_set(): events.put({'type':'completed','request':request,'payload':payload},timeout=1)
                            continue
                        frame=read_capture(request['payload']['capture'])
                        verify_acquisition(frame,equipment)
                        with nullcontext() if concurrent else gpu.foreground(timeout=10):
                            if request['kind']=='detect_overview':
                                payload={'frame_id':frame.frame_id,'model_digest':service.overview_detector.model.weights_sha256,'objects':service.locate(frame)}
                            else:
                                require(request['kind']=='inspect_detail','Unknown model request')
                                output=service.inspect_detail(frame,cycle_id=request['cycle_id'],target_id=request['target_id'],
                                    overview_frame_id=request['payload']['overview_frame_id'],expected_label=request['payload']['expected_label'])
                        if request['kind']=='inspect_detail':
                            from mes_vision.operation.storage import check_free_space
                            from mes_vision.operation.catalog import OperationStore
                            check_free_space(OperationStore(runtime),frame.rgb.nbytes*3)
                            payload=save_detail(Path(runtime)/'station-evidence',frame,output,normal_export=product['normal_reference'],
                                criteria={'product':product['name'],'instructions':product['vlm_criteria']} if product['vlm_criteria'].strip() else None)
                        if not stop.is_set(): events.put({'type':'completed','request':request,'payload':payload},timeout=1)
                    except Exception as exc:
                        if not stop.is_set(): events.put({'type':'completed','request':request,'error':str(exc)},timeout=1)
            finally:
                if models: models.close()
                write_json(gpu.root/'resident.json',{'pid':os.getpid(),'allow_vlm':True,'active':False})
    except Exception as exc:
        try: events.put({'type':'error','message':str(exc)},timeout=1)
        except Exception: pass


class StationProcess:
    def __init__(self,root,runtime,product,equipment,recipe,*,model_factory=None):
        ctx=mp.get_context('spawn'); self.commands=ctx.Queue(1); self.events=ctx.Queue(4); self.stop_event=ctx.Event()
        self.process=ctx.Process(target=process_main,args=(str(root),str(runtime),product,equipment,recipe,self.commands,
            self.events,self.stop_event,os.getpid(),model_factory),daemon=True)
        self.stopping=False; self.pending=None; self.ready=False; self.started=time.monotonic(); self.stop_at=None
        try: self.process.start()
        except BaseException:
            for q in (self.commands,self.events): q.cancel_join_thread(); q.close()
            raise
    def submit(self,request,**capture):
        require(self.ready and not self.stopping and self.pending is None,'Model process is busy or not ready')
        self.commands.put_nowait({'request':request,**capture}); self.pending=request
    def poll(self):
        result=[]
        for _ in range(8):
            try: event=self.events.get_nowait()
            except queue.Empty: break
            if event['type']=='ready': self.ready=True
            if event['type']=='completed' and self.pending and event['request']==self.pending: self.pending=None
            result.append(event)
        return result
    def watchdog_error(self,now):
        if self.stopping: return None
        if not self.process.is_alive(): return 'Model process exited'
        if not self.ready and now-self.started>180: return 'Model preparation timed out'
        if self.pending and now>=self.pending['deadline']: return 'Inspection request timed out'
        return None
    def stop(self):
        if not self.stopping: self.stopping=True; self.stop_at=time.monotonic(); self.stop_event.set()
    def dispose(self):
        if self.process.is_alive(): self.process.terminate()
        self.process.join(1)
        if self.process.is_alive(): self.process.kill(); self.process.join(1)
        require(not self.process.is_alive(),'Model process did not terminate')
        for q in (self.commands,self.events): q.cancel_join_thread(); q.close()

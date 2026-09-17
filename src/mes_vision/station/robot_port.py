"""Exclusive serial worker for journal-authorized camera moves and sorting."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import queue,threading,time
from PySide6.QtCore import QThread,Signal
from mes_vision.robot import RobotController,load_profile
from mes_vision.robot.magician import Magician
from mes_vision.training.data import require,sha256
from .sequence import ScanJournal
from .settings import StationSettings
from .motion import CaptureMotion
from .sorting import SortMotion,SortPreparation,current_sort


def authorize_move(journal,request,calibration,equipment):
    cycle=journal.read(request['cycle_id']); p=cycle['profile']
    require(cycle['pending']==request and cycle['generation']==request['generation'],'Stale or unissued camera motion request')
    calibration.verify_context(p,equipment)
    if request['kind']=='move_detail':
        require(cycle['coordinate_valid'],'Overview coordinates were invalidated')
        target=next((x for x in cycle['targets'] if x['id']==request['target_id']),None)
        require(target is not None and target['status']=='MOVING','Detail target is no longer assigned for capture motion')
        mapped=calibration.map_targets({'kind':'map_targets','payload':{
            'calibration_digest':calibration.digest,'targets':[target]}},p,equipment)['targets'][0]
        require(mapped['capture_pose']==request['payload']['pose'] and mapped['pick_pose']==target['pick_pose'],
                'Detail motion differs from saved overview calibration')
    else: require(request['kind']=='move_overview','Capture move request required')
    return p


class StationRobotPort(QThread):
    completed=Signal(object); failed=Signal(str); changed=Signal(object)
    def __init__(self,runtime,equipment,*,adapter_factory=Magician):
        super().__init__(); self.runtime=Path(runtime); self.equipment=deepcopy(equipment); self.factory=adapter_factory
        self.commands=queue.Queue(1); self.stopping=threading.Event(); self.stop_requested=threading.Event()
        self.lock=threading.Lock(); self.generation=0; self.state='DISCONNECTED'

    def request(self,kind,**payload):
        require(kind in {'recover','move','sort'},'Unsupported station robot request')
        with self.lock:
            require(not self.stopping.is_set() and not self.stop_requested.is_set(),'Robot is stopping')
            self.commands.put_nowait({'kind':kind,'generation':self.generation,**deepcopy(payload)})

    def stop_motion(self):
        with self.lock:
            self.generation+=1; self.stop_requested.set()
            while True:
                try: self.commands.get_nowait()
                except queue.Empty: break

    def current(self,message):
        with self.lock:
            return message['generation']==self.generation and not self.stop_requested.is_set() and not self.stopping.is_set()

    def run(self):
        adapter=None; controller=None; motion=None; ready_digest=None; last_status=0; preparation=None
        try:
            # Acquire shared controller ownership before opening serial. Creating a
            # second owner must not stop an already operating Magician on connect.
            from filelock import FileLock
            self.runtime.joinpath('robot').mkdir(parents=True,exist_ok=True)
            connection_owner=FileLock(str(self.runtime/'robot'/'connection.lock'),timeout=0)
            with connection_owner:
                from .owner import OwnedAdapter
                with OwnedAdapter(self.runtime/'robot',lambda:self.factory(self.equipment['robot'])) as owned:
                    adapter=owned.adapter; controller=owned.controller; motion=CaptureMotion(controller)
                    journal=ScanJournal(self.runtime)
                    sorter=SortMotion(controller,journal,is_authorized=lambda:not self.stop_requested.is_set() and not self.stopping.is_set())
                    controller.transition('RECOVERY','STATION_CONNECTED_REQUIRES_CONFIRMATION')
                    while not self.stopping.is_set():
                        if self.stop_requested.is_set():
                            active=motion.active or sorter.active
                            if preparation: preparation.cancelled=True
                            motion.active=None; sorter.stop(); self.stop_requested.clear()
                            if active: self.completed.emit({'request':active['request'],'error':'STATION_STOP'})
                        try: message=self.commands.get_nowait()
                        except queue.Empty: message=None
                        if message and self.current(message):
                            try:
                                profile_path=Path(self.equipment['robot']['profile']); profile=load_profile(profile_path)
                                if message['kind']=='recover':
                                    require(motion.active is None and sorter.active is None and preparation is None,'Robot action is active')
                                    adapter.configure(profile); controller.recover(confirmation_reference=message['reference'])
                                    ready_digest=sha256(profile_path)
                                else:
                                    require(ready_digest is not None and ready_digest==sha256(profile_path),'Robot profile changed; readiness must be reconfirmed')
                                    settings=StationSettings(self.runtime); calibration=settings.calibrated()
                                    request=message['request']
                                    require(motion.active is None and sorter.active is None and preparation is None,'Robot action is already active')
                                    p=(current_sort(journal,request)[0]['profile'] if message['kind']=='sort'
                                       else authorize_move(journal,request,calibration,self.equipment))
                                    require(p['robot_connection_epoch']==adapter.epoch,'Robot reconnected or readiness changed; start a new overview cycle')
                                    require(p['settle_seconds']==settings.value['settle_seconds']
                                        and p['action_timeout_seconds']==settings.value['action_timeout_seconds']
                                        and p['station_settings_version']==settings.value['version']
                                        and p['robot_profile_sha256']==ready_digest,'Capture settings or robot profile changed')
                                    if self.current(message):
                                        if message['kind']=='sort': preparation=SortPreparation(journal,request,calibration,self.equipment,profile,adapter.status().pose)
                                        else: motion.start(request,p,calibration,profile,self.equipment,now=time.monotonic())
                            except Exception as exc:
                                motion.active=None; sorter.stop('STATION_REQUEST_FAILED'); ready_digest=None
                                if message['kind'] in {'move','sort'}: self.completed.emit({'request':message['request'],'error':str(exc)})
                                self.failed.emit(str(exc))
                        if preparation:
                            if time.monotonic()>=preparation.request['deadline'] and not preparation.cancelled:
                                preparation.cancelled=True; sorter.stop('SORT_PREPARATION_TIMEOUT')
                                self.completed.emit({'request':preparation.request,'error':'SORT_PREPARATION_TIMEOUT'})
                            result=preparation.poll()
                            if result is not None:
                                task=preparation; preparation=None
                                if not result.get('cancelled') and not self.stop_requested.is_set() and not self.stopping.is_set():
                                    try:
                                        require('error' not in result,result.get('error',''))
                                        require(sha256(Path(self.equipment['robot']['profile']))==ready_digest,'Robot profile changed during sorting preparation')
                                        settings=StationSettings(self.runtime)
                                        require(settings.value['version']==result['validated'][2]['station_settings_version'],'Station settings changed during sorting preparation')
                                        settings.calibrated().verify_context(result['validated'][2],self.equipment)
                                        sorter.start(task.request,task.calibration,self.equipment,task.profile,now=time.monotonic(),prepared=result)
                                    except Exception as exc:
                                        sorter.stop('SORT_PREPARATION_FAILED'); ready_digest=None
                                        self.completed.emit({'request':task.request,'error':str(exc)}); self.failed.emit(str(exc))
                        if not self.stop_requested.is_set() and not self.stopping.is_set():
                            event=motion.tick(now=time.monotonic())
                            if event: self.completed.emit(event)
                            event=sorter.tick(now=time.monotonic())
                            if event: self.completed.emit(event)
                        if time.monotonic()-last_status>=.3:
                            last_status=time.monotonic(); self.state=controller.state
                            self.changed.emit({'state':self.state,'status':asdict(adapter.status()),'busy':motion.active is not None or sorter.active is not None or preparation is not None})
                        self.stopping.wait(.02)
        except Exception as exc: self.failed.emit(str(exc))
        finally:
            if preparation: preparation.cancelled=True
            self.state='DISCONNECTED'; self.changed.emit({'state':'DISCONNECTED','busy':False})

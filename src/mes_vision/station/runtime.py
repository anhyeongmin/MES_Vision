"""Single owner dispatcher. Stop is sent before any fallible persistence work."""
from copy import deepcopy
from .sequence import ScanSequence,TERMINAL
from mes_vision.training.data import require


class ScanRuntime:
    def __init__(self,journal,camera,robot,vision,calibration,equipment):
        self.sequence=ScanSequence(journal); self.camera=camera; self.robot=robot; self.vision=vision
        self.calibration=calibration; self.equipment=deepcopy(equipment); self.authorized=False; self.dispatched=set()
    @property
    def active(self): return self.authorized and self.sequence.data is not None and self.sequence.data['state'] not in TERMINAL
    def start(self,profile,now):
        require(not self.active and not self.camera.busy,'Previous cycle or capture write is still active')
        self.authorized=True
        try: self.dispatch(self.sequence.start(profile,now),now)
        except BaseException: self.stop('START_FAILED'); raise
    def stop(self,reason='OPERATOR_STOP'):
        self.authorized=False
        try: self.robot.stop_motion()
        finally:
            self.camera.cancel()
            try: self.vision.stop()
            finally: self.sequence.stop(reason)
    def dispatch(self,request,now):
        if request is None or not self.active: return
        require(request['id'] not in self.dispatched,'Request already dispatched')
        self.dispatched.add(request['id']); kind=request['kind']
        if kind.startswith('move_'): self.robot.request('move',request=request)
        elif kind=='sort_object': self.robot.request('sort',request=request)
        elif kind.startswith('capture_'): self.camera.submit(request,self.sequence.data['profile'])
        elif kind=='map_targets': self.complete({'request':request,'payload':self.calibration.map_targets(request,self.sequence.data['profile'],self.equipment)},now)
        elif kind in {'detect_overview','inspect_detail'}: self.vision.submit(request)
        else: raise ValueError('Unsupported station request: '+kind)
    def complete(self,event,now):
        if not self.active: return False
        r=event['request']; current=self.sequence.data['pending']
        if current!=r: return False
        try:
            next_request=self.sequence.complete(r['id'],r['cycle_id'],r['generation'],event.get('payload'),now,error=event.get('error'))
            if self.sequence.data['state']=='FAILED':
                self.stop(self.sequence.data['error']); return False
            self.dispatch(next_request,now); return True
        except BaseException: self.stop('COMPLETION_FAILED'); raise
    def tick(self,now):
        try:
            event=self.camera.tick(now)
            if event: self.complete(event,now)
            if self.active:
                self.dispatch(self.sequence.tick(now),now)
                if self.sequence.data['state']=='FAILED': self.stop(self.sequence.data['error'])
        except BaseException: self.stop('DISPATCH_FAILED'); raise

"""Tick-driven capture moves using the same durable robot owner as pick/place.

Runs exclusively on a device worker, never the GUI or inference worker. No
automatic recovery or retransmission after an uncertain physical outcome.
"""
from copy import deepcopy
from dataclasses import asdict
import json,math
from mes_vision.training.data import require
from mes_vision.robot.contracts import Pose,Command,Completion
from .sequence import valid_profile,valid_pose


def xyzr(p): return {k:getattr(p,k) for k in ('x','y','z','r')}


def within(actual,target,profile):
    return (all(abs(getattr(actual,k)-target[k])<=profile['pose_tolerance_mm'] for k in ('x','y','z'))
            and abs(actual.r-target['r'])<=profile['rotation_tolerance_deg'])


def capture_path(current,target,profile,calibration,robot_profile,equipment):
    valid_profile(profile); calibration.verify_context(profile,equipment)
    v=calibration.value; rp=robot_profile
    require(rp.kind=='real' and rp.validated and rp.product_id==profile['product_id']
            and rp.calibration_version==calibration.identity and rp.grasp_policy_version==v['grasp_version']
            and rp.travel_z_mm==v['travel_z'] and rp.pick_z_mm==v['pick_z'], 'Robot profile differs from measured camera/pick calibration')
    require(profile['equipment_version']==equipment['version'],'Equipment version changed')
    for key in ('pose_tolerance_mm','rotation_tolerance_deg'):
        require(profile[key]==equipment['robot'][key],'Robot arrival tolerance changed')
    require(isinstance(current,Pose),'Current robot pose is unconfirmed')
    valid_pose(xyzr(current),profile); valid_pose(target,profile)
    require(current.z<=v['travel_z'],'Robot is above the validated travel plane; reconcile position first')
    poses=[Pose(current.x,current.y,v['travel_z'],current.r),
           Pose(target['x'],target['y'],v['travel_z'],target['r']),Pose(**target)]
    for item in [current,*poses]:
        valid_pose(xyzr(item),profile)
        require(rp.workspace.contains(item),'Capture path leaves validated robot workspace')
    return poses


class CaptureMotion:
    def __init__(self,controller):
        self.controller=controller; self.adapter=controller.adapter; self.active=None
        with controller.journal.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS capture_requests(id TEXT PRIMARY KEY,payload TEXT NOT NULL)')

    def start(self,request,profile,calibration,robot_profile,equipment,*,now):
        c=self.controller
        require(self.active is None and not c.closed and c.state in {'IDLE','RECAPTURE'},'Capture motion is not ready')
        require(self.adapter.kind=='real' and request['kind'] in {'move_overview','move_detail'},'Capture move request required')
        require(all(isinstance(request.get(k),str) and request[k] for k in ('id','cycle_id'))
                and type(request.get('generation')) is int and request['generation']>=0,'Invalid capture request identity')
        require(all(type(request.get(k)) in (int,float) and math.isfinite(request[k]) for k in ('issued_at','deadline'))
                and request['issued_at']<=now<request['deadline'],'Capture move request expired')
        require(request['deadline']-request['issued_at']<=profile['action_timeout_seconds']+1e-6,'Capture deadline changed')
        target=request['payload']['pose']
        if request['kind']=='move_overview': require(target==profile['overview_pose'],'Overview pose changed')
        else:
            require(request.get('target_id') and target['z']==calibration.value['detail_z']
                    and target['r']==calibration.value['detail_r'],'Detail pose must match calibrated height and orientation')
        status=self.adapter.status()
        require(status.connected and status.motion_state=='READY' and status.holding is False,'Robot must be ready with empty tool')
        poses=capture_path(status.pose,target,profile,calibration,robot_profile,equipment)
        commands=[Command(request['id']+':'+str(i),request['cycle_id'],stage,'move',p)
                  for i,(stage,p) in enumerate(zip(('CAPTURE_LIFT','CAPTURE_TRAVEL','CAPTURE_DESCEND'),poses))]
        # This table shares the existing robot ownership lock. Even an acknowledged
        # or interrupted request cannot be retried after reconnect/restart.
        with c.journal.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO capture_requests VALUES(?,?)',(request['id'],json.dumps(request,allow_nan=False)))
            c.journal.write(db,'CAPTURE_MOVE','CAPTURE_RESERVED',None,{'request':request,'path':[asdict(x) for x in commands]})
        c.state='CAPTURE_MOVE'; c.plan=None
        self.active={'request':deepcopy(request),'profile':deepcopy(profile),'commands':commands,'index':0,
                     'pending':None,'epoch':status.connection_epoch,'last_time':now,
                     'command_timeout':robot_profile.command_timeout_seconds,'deadline':None}

    def tick(self,*,now):
        a=self.active
        if a is None: return None
        c=self.controller; req=a['request']; profile=a['profile']
        try:
            require(a['last_time']<=now<req['deadline'],'Capture motion timed out or clock moved backward'); a['last_time']=now
            status=self.adapter.status()
            require(status.connected and status.connection_epoch==a['epoch'] and status.motion_state=='READY'
                    and status.holding is False,'Robot connection/readiness/tool state changed')
            if a['pending'] is None:
                command=a['commands'][a['index']]
                c.transition(command.stage,'CAPTURE_COMMAND_INTENT',asdict(command))
                require(self.adapter.submit(command) is True,'Capture command rejected')
                a['pending']=command; a['deadline']=min(req['deadline'],now+a['command_timeout'])
                c.transition(command.stage,'CAPTURE_COMMAND_ACCEPTED',{'command_id':command.command_id})
                return None
            require(now<a['deadline'],'Capture command timed out')
            command=a['pending']; response=self.adapter.poll(command.command_id)
            require(isinstance(response,Completion) and response.command_id==command.command_id,'Unmatched capture completion')
            if response.state=='PENDING': return None
            require(response.state=='DONE' and response.pose==command.target,'Capture outcome unconfirmed')
            actual=self.adapter.status()
            require(actual.connected and actual.connection_epoch==a['epoch'] and actual.motion_state=='READY'
                    and actual.holding is False and isinstance(actual.pose,Pose)
                    and within(actual.pose,xyzr(command.target),profile),'Measured capture position is unconfirmed')
            c.transition(command.stage,'CAPTURE_COMMAND_COMPLETED',{'command_id':command.command_id,'actual_pose':xyzr(actual.pose)})
            a['index']+=1; a['pending']=None
            if a['index']==len(a['commands']):
                payload={'motion_complete':True,'actual_pose':xyzr(actual.pose),'connection_epoch':a['epoch']}
                c.transition('RECAPTURE','CAPTURE_POSITION_REACHED',{'request_id':req['id'],**payload})
                self.active=None; return {'request':req,'payload':payload}
            return None
        except Exception as exc:
            self.active=None
            try: c.fault(str(exc))
            except Exception: pass  # transition already attempts a physical stop on storage failure.
            return {'request':req,'error':str(exc)}

    def stop(self,reason='STATION_STOP'):
        self.active=None
        self.controller.stop(reason)

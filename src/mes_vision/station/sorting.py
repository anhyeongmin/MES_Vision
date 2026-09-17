"""Two-image evidence validation and durable, non-retryable station sorting."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json,sqlite3,threading,queue,time
from mes_vision.anomaly.features import fingerprint
from mes_vision.training.data import require
from mes_vision.robot.contracts import Pose,Command,Completion
from mes_vision.decision.io import run_from_dict,frame_evidence_from_dict
from mes_vision.decision.policy import apply_policy,load_policy_data
from mes_vision.vlm.snapshots import load_snapshot
from .worker import read_capture
from .motion import capture_path,xyzr,within
from .sequence import valid_pose


def current_sort(journal,request):
    cycle=journal.read(request['cycle_id'])
    require(cycle['pending']==request and cycle['generation']==request['generation']
        and cycle['state']=='SORT_OBJECT' and cycle['coordinate_valid'] and cycle['profile']['auto_sort'],
        'Sorting request is stale, unissued or revoked')
    require(all(t['status'] in {'OK','NG','REVIEW','SORTING','SORTED'} for t in cycle['targets']),
        'All close-up inspections must finish before sorting')
    target=next((t for t in cycle['targets'] if t['id']==request['target_id']),None)
    require(target is not None and target['status']=='SORTING' and target['decision'] in {'OK','NG'},'Target is not eligible for sorting')
    require(request['payload']=={'pick_pose':target['pick_pose'],'decision':target['decision'],'evidence':target['result']},
        'Sort coordinates or evidence changed')
    return cycle,target


def authorize_sort(journal,request,calibration,equipment,robot_profile,current,*,now):
    cycle,target=current_sort(journal,request); p=cycle['profile']; receipt=target['result']; rp=robot_profile
    require(request['issued_at']<=now<request['deadline'],'Sorting request expired')
    calibration.verify_context(p,equipment)
    overview=read_capture(cycle['overview']); detail=read_capture(target['detail'])
    require(overview.session_id==detail.session_id==p['camera_session'] and overview.frame_id!=detail.frame_id,
        'Overview and detail camera identity mismatch')
    require(0<=now-cycle['overview']['acquired_at']<=rp.frame_max_age_seconds,'Overview coordinates expired; a new cycle is required')
    require(target['overview']['localization_status']=='READY','Overview target is not confirmed')
    mapped=calibration.map_targets({'kind':'map_targets','payload':{'calibration_digest':calibration.digest,'targets':[target]}},p,equipment)['targets'][0]
    require(mapped['pick_pose']==target['pick_pose'] and mapped['capture_pose']==target['capture_pose'],'Saved target differs from overview calibration')
    manifest,saved,_=load_snapshot(receipt['snapshot_path'],expected_digest=receipt['snapshot_digest'])
    expected={'cycle_id':cycle['id'],'target_id':target['id'],'overview_frame_id':overview.frame_id,'detail_frame_id':detail.frame_id}
    require(manifest['kind']=='real' and saved['config'].get('station_link')==expected,'Detailed evidence belongs to another object')
    require(saved['frame']['frame_id']==detail.frame_id and saved['frame']['is_live'] is True
        and saved['config'].get('product_id')==p['product_id'] and saved['decision_status']=='EVALUATED','Live product decision is required')
    from PIL import Image
    import hashlib
    with Image.open(Path(receipt['snapshot_path'])/'frame.png') as image:
        require(image.mode=='RGB' and list(image.size)==p['image_size'] and hashlib.sha256(image.tobytes()).hexdigest()==target['detail']['rgb_sha256'],
            'Decision pixels differ from captured detail')
    require(receipt['cycle_id']==cycle['id'] and receipt['target_id']==target['id'] and receipt['detail_frame_id']==detail.frame_id,'Receipt identity changed')
    inspection=run_from_dict(saved)
    require(len(inspection.objects)==1 and inspection.objects[0].object_id==receipt['object_id'],'Detailed object is not unique')
    obj=inspection.objects[0]; evidence=inspection.decision_details
    policy=load_policy_data(evidence['policy'])
    require(fingerprint(asdict(policy))==p['detail_policy_digest'] and obj.decision_details.get('policy_digest')==p['detail_policy_digest'],
        'Decision policy changed')
    replayed=apply_policy(inspection,policy,frame_evidence_from_dict(evidence['frame_evidence']))
    checked=replayed.objects[0]
    require(checked.final_decision==obj.final_decision==target['decision']
        and checked.decision_details==obj.decision_details and replayed.final_decision==inspection.final_decision,
        'Saved decision does not match its evidence')
    require(obj.decision_details.get('identity_valid') is True and not evidence.get('reasons'),'Decision is not ready for motion')
    forbidden={'OBJECT_ISSUE','OBJECT_TOUCHES_IMAGE_EDGE','OBJECT_PARTIALLY_OUTSIDE_IMAGE','OBJECT_OR_CROP_OVERLAP','UNREGISTERED_CHECK'}
    require(not any(r['code'] in forbidden for r in obj.decision_details.get('reasons',[])),'Object is not ready for placement')
    require(not rp.allow_review_move,'Station sorting never moves REVIEW objects')
    pick=Pose(**target['pick_pose']); dest=rp.destinations[target['decision']]
    # Lift vertically from the last close-up pose before travelling to the pick.
    poses=capture_path(current,xyzr(pick),p,calibration,rp,equipment)
    above_pick=poses[1]; above_place=Pose(dest.x,dest.y,rp.travel_z_mm,dest.r)
    for pose in (dest,above_place):
        valid_pose(xyzr(pose),p); require(rp.workspace.contains(pose),'Destination leaves validated workspace')
    actions=[('SORT_LIFT','move',poses[0]),('APPROACH_PICK','move',above_pick),('DESCEND_PICK','move',pick),
        ('PICK','engage',None),('VERIFY_PICK','verify_pick',None),('LIFT_PICK','move',above_pick),
        ('MOVE_PLACE','move',above_place),('DESCEND_PLACE','move',dest),('PLACE','release',None),
        ('VERIFY_PLACE','verify_place',None),('RETRACT','move',above_place)]
    commands=[Command(request['id']+':'+str(i),request['id'],stage,action,pose) for i,(stage,action,pose) in enumerate(actions)]
    plan={'request':request,'profile':p,'robot_profile':asdict(rp),'overview':cycle['overview'],'target':target,
        'commands':[asdict(c) for c in commands]}
    return commands,fingerprint(plan),p,cycle['overview']['acquired_at']


def verify_sort_receipt(journal,request,payload):
    require(isinstance(payload,dict) and payload.get('object_id')==request['target_id']
        and payload.get('placement_confirmed') is True and payload.get('request_id')==request['id'],'Sorting completion is not confirmed')
    path=journal.root/'robot'/'robot.sqlite3'
    require(path.is_file(),'Robot placement record is missing')
    with sqlite3.connect(path.as_uri()+'?mode=ro',uri=True) as db:
        row=db.execute('SELECT request,receipt FROM station_sorts WHERE id=?',(request['id'],)).fetchone()
    require(row is not None and json.loads(row[0])==request and row[1] is not None and json.loads(row[1])==payload,
        'Sorting completion differs from durable robot evidence')


class SortMotion:
    def __init__(self,controller,journal,*,is_authorized=lambda:True):
        self.controller=controller; self.adapter=controller.adapter; self.journal=journal; self.active=None
        self.is_authorized=is_authorized
        with controller.journal.db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS station_sorts(id TEXT PRIMARY KEY,cycle_id TEXT,target_id TEXT,request TEXT,plan TEXT,receipt TEXT,UNIQUE(cycle_id,target_id))')

    def start(self,request,calibration,equipment,robot_profile,*,now,prepared=None):
        c=self.controller; s=self.adapter.status()
        require(self.active is None and not c.closed and c.state in {'IDLE','RECAPTURE'} and self.adapter.kind=='real','Sorting robot is not ready')
        require(s.connected and s.motion_state=='READY' and s.holding is False,'Sorting requires a ready, empty tool')
        if prepared is None:
            prepared={'pose':s.pose,'validated':authorize_sort(self.journal,request,calibration,equipment,robot_profile,s.pose,now=now)}
        commands,digest,p,observed=prepared['validated']
        current_sort(self.journal,request)
        require(s.pose==prepared['pose'],'Robot moved while sorting evidence was being checked')
        require(request['issued_at']<=now<request['deadline'] and 0<=now-observed<=robot_profile.frame_max_age_seconds,'Sorting preparation expired')
        require(s.connection_epoch==p['robot_connection_epoch'],'Robot connection changed')
        with c.journal.db() as db:
            db.execute('BEGIN IMMEDIATE')
            db.execute('INSERT INTO station_sorts VALUES(?,?,?,?,?,NULL)',(request['id'],request['cycle_id'],request['target_id'],
                json.dumps(request,allow_nan=False),json.dumps({'digest':digest,'commands':[asdict(x) for x in commands]},allow_nan=False)))
            c.journal.write(db,'SORT_OBJECT','SORT_RESERVED',None,{'request_id':request['id'],'plan_digest':digest})
        c.state='SORT_OBJECT'; c.plan=None
        self.active={'request':deepcopy(request),'commands':commands,'digest':digest,'profile':p,'robot_profile':robot_profile,
            'observed':observed,'index':0,'pending':None,'last_time':now,'deadline':None,'epoch':s.connection_epoch}

    def stop(self,reason='STATION_STOP'):
        self.active=None
        try: acknowledged=self.adapter.request_stop()
        except Exception: acknowledged=False
        self.controller.pending=None
        self.controller.transition('STOPPED' if acknowledged else 'FAULT','STATION_STOP_RESULT',
            {'reason':reason,'acknowledged':acknowledged,'physical_safety_confirmed':False})

    def tick(self,*,now):
        a=self.active
        if a is None: return None
        c=self.controller; req=a['request']
        try:
            require(self.is_authorized(),'Sorting was stopped')
            require(a['last_time']<=now<req['deadline'],'Sorting timed out or clock moved backward'); a['last_time']=now
            current_sort(self.journal,req)
            s=self.adapter.status()
            require(s.connected and s.connection_epoch==a['epoch'] and s.motion_state=='READY','Robot connection/readiness changed')
            index=a['index']; command=a['commands'][index]
            if index<=3: require(0<=now-a['observed']<=a['robot_profile'].frame_max_age_seconds,'Overview coordinates expired before pick')
            # Tool transitions may take time; confirmation stages poll their own input.
            if index<3 or index==10 or index==3 and a['pending'] is None: require(s.holding is False,'Unexpected object in tool')
            if 5<=index<=7 or index==8 and a['pending'] is None: require(s.holding is True,'Picked object was lost')
            if a['pending'] is None:
                c.transition(command.stage,'SORT_COMMAND_INTENT',asdict(command))
                require(self.is_authorized(),'Sorting was stopped before command submission')
                require(self.adapter.submit(command) is True,'Sorting command rejected')
                a['pending']=command; a['deadline']=min(req['deadline'],now+a['robot_profile'].command_timeout_seconds)
                c.transition(command.stage,'SORT_COMMAND_ACCEPTED',{'command_id':command.command_id}); return None
            require(now<a['deadline'],'Sorting command timed out')
            response=self.adapter.poll(command.command_id)
            require(isinstance(response,Completion) and response.command_id==command.command_id,'Unmatched sorting completion')
            if response.state=='PENDING': return None
            require(response.state=='DONE','Sorting outcome is unconfirmed')
            actual=self.adapter.status()
            require(actual.connected and actual.connection_epoch==a['epoch'] and actual.motion_state=='READY','Robot changed during completion')
            if command.action=='move':
                require(response.pose==command.target and isinstance(actual.pose,Pose) and within(actual.pose,xyzr(command.target),a['profile']),
                    'Measured sorting position is unconfirmed')
            if command.action.startswith('verify_'):
                require(response.verified is True and actual.holding is (command.action=='verify_pick'),'Pick or placement input is unconfirmed')
            c.transition(command.stage,'SORT_COMMAND_COMPLETED',asdict(response)); a['index']+=1; a['pending']=None
            if a['index']==len(a['commands']):
                require(actual.holding is False,'Tool is not empty after placement')
                payload={'object_id':req['target_id'],'request_id':req['id'],'placement_confirmed':True,
                    'plan_digest':a['digest'],'connection_epoch':a['epoch'],'actual_pose':xyzr(actual.pose)}
                with c.journal.db() as db:
                    db.execute('UPDATE station_sorts SET receipt=? WHERE id=? AND receipt IS NULL',(json.dumps(payload,allow_nan=False),req['id']))
                    c.journal.write(db,'RECAPTURE','SORT_VERIFIED',None,payload)
                c.state='RECAPTURE'; self.active=None; return {'request':req,'payload':payload}
        except Exception as exc:
            self.active=None
            try: self.adapter.request_stop()
            except Exception: pass
            try: c.fault(str(exc))
            except Exception: pass
            return {'request':req,'error':str(exc)}


class SortPreparation:
    """One read-only validation thread; device stop/polling never waits on images.

    A cancelled read may finish later. It never reserves, submits or releases
    anything; its result cannot be consumed after cancellation.
    """
    def __init__(self,journal,request,calibration,equipment,profile,pose):
        self.request=deepcopy(request); self.calibration=calibration; self.equipment=deepcopy(equipment)
        self.profile=profile; self.results=queue.Queue(1); self.cancelled=False
        def work():
            try:
                validated=authorize_sort(journal,self.request,calibration,self.equipment,profile,pose,now=time.monotonic())
                result={'pose':pose,'validated':validated}
            except Exception as exc: result={'error':str(exc)}
            self.results.put_nowait(result)
        self.thread=threading.Thread(target=work,daemon=True,name='station-sort-evidence'); self.thread.start()
    def poll(self):
        try: result=self.results.get_nowait()
        except queue.Empty: return None
        return {'cancelled':True} if self.cancelled else result

"""Durable, event-driven overview -> detail -> sorting sequence.

This core produces requests for a dispatcher. It never calls the Magician or
camera directly. A matching completion is required before advancing. A crashed
or cancelled cycle cannot be resumed with old image coordinates.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json,math,sqlite3,time,hashlib
from PIL import Image
from uuid import uuid4
from mes_vision.training.data import require,sha256
from mes_vision.inspection import Box
from mes_vision.vlm.snapshots import load_snapshot

TERMINAL={'COMPLETED','CANCELLED','FAILED','INTERRUPTED'}


def mark_unconfirmed_sort(data):
    for target in data.get('targets',[]):
        if target['status']=='SORTING': target['status']='SORT_UNCONFIRMED'


def finite(value): return type(value) in (int,float) and math.isfinite(value)


def valid_profile(profile):
    require(profile.get('camera_driver','d405') in ('d405','uvc'), 'Invalid camera driver')
    require(profile.get('schema_version')==1 and profile.get('mode')=='overview_detail', 'Robot-mounted scan profile required')
    for key in ('profile_id','validation_reference','camera_serial','mount_revision','camera_session',
                'calibration_digest','overview_model_digest','detail_policy_digest','product_id'):
        require(isinstance(profile.get(key),str) and profile[key].strip(),'Missing scan profile field: '+key)
    for key in ('calibration_digest','overview_model_digest','detail_policy_digest'):
        require(len(profile[key])==64 and all(c in '0123456789abcdef' for c in profile[key]),'Invalid digest')
    require(profile.get('calibration_kind')=='robot_mounted_overview_detail', 'Fixed-camera calibration cannot be reused')
    require(type(profile.get('product_version')) is int and profile['product_version']>0,'Frozen product version required')
    require(type(profile.get('equipment_version')) is int and profile['equipment_version']>0,'Frozen equipment version required')
    size=profile.get('image_size')
    require(isinstance(size,(list,tuple)) and len(size)==2 and all(type(v) is int and v>0 for v in size),'Original image size required')
    for key,low,high in (('settle_seconds',.1,10),('action_timeout_seconds',1,300),('pose_tolerance_mm',.001,10),('rotation_tolerance_deg',.001,10)):
        require(finite(profile.get(key)) and low<=profile[key]<=high,'Invalid scan timing/tolerance: '+key)
    require(type(profile.get('max_objects')) is int and 1<=profile['max_objects']<=100,'Invalid object capacity')
    count=profile.get('expected_count')
    require(count is None or type(count) is int and 0<=count<=profile['max_objects'],'Invalid expected object count')
    require(type(profile.get('auto_sort')) is bool,'Explicit sorting preference required')
    bounds=profile.get('motion_bounds')
    require(isinstance(bounds,dict) and set(bounds)=={'x','y','z','r'},'Registered motion bounds required')
    for limits in bounds.values():
        require(isinstance(limits,(list,tuple)) and len(limits)==2 and all(finite(v) for v in limits) and limits[0]<limits[1],'Invalid motion bounds')
    valid_pose(profile.get('overview_pose'),profile)
    json.dumps(profile,allow_nan=False)


def valid_pose(pose,profile):
    require(isinstance(pose,dict) and set(pose)=={'x','y','z','r'},'Complete robot pose required')
    for key,value in pose.items():
        require(finite(value) and profile['motion_bounds'][key][0]<=value<=profile['motion_bounds'][key][1], 'Pose is outside registered motion bounds')


class ScanJournal:
    def __init__(self,runtime):
        self.root=Path(runtime).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'station.sqlite3'
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS cycles(id TEXT PRIMARY KEY,revision INTEGER NOT NULL,state TEXT NOT NULL,data TEXT NOT NULL);
              CREATE TABLE IF NOT EXISTS cycle_events(id INTEGER PRIMARY KEY,cycle_id TEXT,revision INTEGER,event TEXT,data TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS cycle_vlm(cycle_id TEXT,target_id TEXT,data TEXT,PRIMARY KEY(cycle_id,target_id));
            ''')

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.path,timeout=5)
        try: yield db; db.commit()
        except BaseException: db.rollback(); raise
        finally: db.close()

    def save(self,data,previous_revision,event):
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            old=db.execute('SELECT revision FROM cycles WHERE id=?',(data['id'],)).fetchone()
            require((old[0] if old else -1)==previous_revision,'Scan cycle changed in another owner')
            if previous_revision==-1:
                busy=db.execute("SELECT id FROM cycles WHERE state NOT IN ('COMPLETED','CANCELLED','FAILED','INTERRUPTED')").fetchone()
                require(busy is None,'An unfinished scan cycle already exists')
            encoded=json.dumps(data,ensure_ascii=False,allow_nan=False)
            db.execute('INSERT OR REPLACE INTO cycles VALUES(?,?,?,?)',(data['id'],data['revision'],data['state'],encoded))
            db.execute('INSERT INTO cycle_events(cycle_id,revision,event,data,created) VALUES(?,?,?,?,?)',
                       (data['id'],data['revision'],event,encoded,time.time()))

    def read(self,cycle_id):
        with self.connect() as db: row=db.execute('SELECT data FROM cycles WHERE id=?',(cycle_id,)).fetchone()
        require(row is not None,'Scan cycle not found'); return json.loads(row[0])

    def recover_interrupted(self):
        """Call only after exclusive application ownership is acquired."""
        recovered=[]
        with self.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            rows=db.execute("SELECT data FROM cycles WHERE state NOT IN ('COMPLETED','CANCELLED','FAILED','INTERRUPTED')").fetchall()
            for row in rows:
                data=json.loads(row[0]); data.update(state='INTERRUPTED',pending=None,coordinate_valid=False,
                    generation=data['generation']+1,revision=data['revision']+1,error='PROCESS_INTERRUPTED')
                mark_unconfirmed_sort(data)
                encoded=json.dumps(data,ensure_ascii=False,allow_nan=False)
                db.execute('UPDATE cycles SET revision=?,state=?,data=? WHERE id=?',(data['revision'],data['state'],encoded,data['id']))
                db.execute('INSERT INTO cycle_events(cycle_id,revision,event,data,created) VALUES(?,?,?,?,?)',
                           (data['id'],data['revision'],'INTERRUPTED',encoded,time.time()))
                recovered.append(data['id'])
        return recovered

    def attach_vlm(self,cycle_id,target_id,detail_frame_id,explanation):
        # Separate table: advisory text has no write access to baseline decisions.
        require(isinstance(explanation,str),'VLM explanation must be text')
        data=self.read(cycle_id)
        target=next((t for t in data['targets'] if t['id']==target_id),None)
        require(target and target.get('detail') and target['detail']['frame_id']==detail_frame_id,'VLM image/target mismatch')
        payload={'detail_frame_id':detail_frame_id,'explanation':explanation}
        with self.connect() as db:
            db.execute('INSERT OR REPLACE INTO cycle_vlm VALUES(?,?,?)',(cycle_id,target_id,json.dumps(payload,ensure_ascii=False)))


class ScanSequence:
    """One serialized owner; adapters return completions through the same owner.

    tick/complete return at most one request. The caller dispatches it once after
    the state commit. No automatic resend is permitted after ambiguous motion.
    """
    def __init__(self,journal): self.journal=journal; self.data=None

    def snapshot(self): return deepcopy(self.data)

    def _persist(self,event,change):
        before=deepcopy(self.data); revision=self.data['revision']
        try:
            request=change(); self.data['revision']+=1
            self.data=json.loads(json.dumps(self.data,ensure_ascii=False,allow_nan=False))
            self.journal.save(self.data,revision,event)
            # The delivered request must be byte-equivalent JSON data to the
            # journal request, including camera tuple/list metadata normalization.
            return json.loads(json.dumps(request,ensure_ascii=False,allow_nan=False))
        except BaseException:
            self.data=before
            raise

    def _request(self,kind,now,*,target=None,**payload):
        request={'id':uuid4().hex,'cycle_id':self.data['id'],'generation':self.data['generation'],
            'kind':kind,'target_id':target,'issued_at':now,
            'deadline':now+self.data['profile']['action_timeout_seconds'],'payload':payload}
        self.data['state']=kind.upper(); self.data['pending']=request
        return request

    def start(self,profile,now):
        valid_profile(profile); require(finite(now),'Invalid monotonic clock')
        require(self.data is None or self.data['state'] in TERMINAL,'Scan is already active')
        previous=self.data
        self.data={'id':uuid4().hex,'revision':-1,'generation':1,'state':'CREATED','profile':deepcopy(profile),
            'overview':None,'targets':[],'pending':None,'coordinate_valid':True,'last_frame_sequence':-1,
            'seen_frame_ids':[],'error':None,'settle_until':None,'phase':None}
        try: return self._persist('START',lambda:self._request('move_overview',now,pose=profile['overview_pose']))
        except BaseException: self.data=previous; raise

    def stop(self,reason='OPERATOR_STOP'):
        if self.data is None or self.data['state'] in TERMINAL: return False
        def change():
            mark_unconfirmed_sort(self.data)
            self.data.update(state='CANCELLED',pending=None,coordinate_valid=False,
                generation=self.data['generation']+1,error=reason)
            # Actual stop must be sent immediately by the dispatcher even if persistence fails.
            return True
        return self._persist('CANCELLED',change)

    def scene_changed(self): return self.stop('SCENE_CHANGED_RECAPTURE_REQUIRED')

    def _failed(self,reason):
        mark_unconfirmed_sort(self.data)
        self.data.update(state='FAILED',pending=None,coordinate_valid=False,
                         generation=self.data['generation']+1,error=str(reason))

    def tick(self,now):
        require(finite(now),'Invalid monotonic clock')
        if self.data is None or self.data['state'] in TERMINAL: return None
        pending=self.data['pending']
        if pending and now>=pending['deadline']:
            return self._persist('TIMEOUT',lambda:self._failed('ACTION_TIMEOUT: '+pending['kind']))
        if self.data['state']=='SETTLING' and now>=self.data['settle_until']:
            phase=self.data['phase']; target=self.data.get('active_target')
            return self._persist('CAPTURE_REQUEST',lambda:self._request('capture_'+phase,now,target=target))
        return None

    def complete(self,request_id,cycle_id,generation,payload,now,*,error=None):
        require(finite(now),'Invalid monotonic clock')
        if self.data is None or self.data['state'] in TERMINAL: return None
        pending=self.data['pending']
        if not pending or (request_id,cycle_id,generation)!=(pending['id'],pending['cycle_id'],pending['generation']): return None
        def change():
            try:
                require(pending['issued_at']<=now<pending['deadline'],'Completion arrived outside its request interval')
                if error is not None: raise ValueError(str(error))
                self.data['pending']=None
                return self._advance(pending,payload,now)
            except Exception as exc:
                self._failed(exc)
        return self._persist('COMPLETE_'+pending['kind'].upper(),change)

    def _target(self,identity):
        return next(t for t in self.data['targets'] if t['id']==identity)

    def _capture(self,payload,request,now):
        p=self.data['profile']
        require(isinstance(payload,dict),'Capture proof required')
        require(payload.get('camera_session')==p['camera_session'] and payload.get('camera_serial')==p['camera_serial'], 'Camera changed during scan')
        require(payload.get('image_size')==list(p['image_size']),'Capture geometry changed')
        require(payload.get('frame_metadata',{}).get('acquisition_identity')==p.get('acquisition_identity'),
                'Capture acquisition changed during scan')
        require(finite(payload.get('acquired_at')) and request['issued_at']<=payload['acquired_at']<=now,
                'Pre-move or stale camera frame cannot be inspected')
        require(type(payload.get('sequence')) is int and payload['sequence']>self.data['last_frame_sequence'],'Camera frame was reused')
        require(isinstance(payload.get('frame_id'),str) and payload['frame_id'] and payload['frame_id'] not in self.data['seen_frame_ids'], 'Frame identity reused')
        require(sha256(Path(payload['path']))==payload['sha256'],'Capture was not saved or its content changed')
        with Image.open(payload['path']) as image:
            require(image.mode=='RGB' and list(image.size)==list(p['image_size']),'Saved capture dimensions or color format differ')
            require(hashlib.sha256(image.tobytes()).hexdigest()==payload['rgb_sha256'],'Saved capture pixels differ')
        self.data['last_frame_sequence']=payload['sequence']; self.data['seen_frame_ids'].append(payload['frame_id'])
        return deepcopy(payload)

    def _advance(self,request,payload,now):
        kind=request['kind']; p=self.data['profile']; identity=request['target_id']
        if kind.startswith('move_'):
            actual=payload['actual_pose']; desired=request['payload']['pose']; valid_pose(actual,p)
            require(payload.get('motion_complete') is True,'Motion completion is not confirmed')
            require(all(abs(actual[k]-desired[k])<=p['pose_tolerance_mm'] for k in ('x','y','z'))
                and abs(actual['r']-desired['r'])<=p['rotation_tolerance_deg'],'Robot did not reach capture pose')
            self.data.update(state='SETTLING',phase='overview' if kind=='move_overview' else 'detail',
                active_target=identity,settle_until=now+p['settle_seconds'])
        elif kind=='capture_overview':
            self.data['overview']=self._capture(payload,request,now)
            return self._request('detect_overview',now,capture=self.data['overview'])
        elif kind=='detect_overview':
            require(payload['frame_id']==self.data['overview']['frame_id'],'Overview detector frame mismatch')
            require(payload['model_digest']==p['overview_model_digest'],'Overview model changed')
            items=payload['objects']; require(isinstance(items,list) and len(items)<=p['max_objects'],'Object capacity exceeded')
            require(p.get('expected_count') is None or len(items)==p['expected_count'],'Overview object count mismatch')
            require(items or p.get('expected_count')==0,'Empty overview requires an explicit empty-scene rule')
            for index,item in enumerate(items):
                box=Box(**item['box']); require(box.clip(*p['image_size'])==box,'Overview box outside original image')
                require(type(item['class_id']) is int and item['class_id']>=0 and finite(item['score']) and 0<=item['score']<=1,'Invalid overview detection')
                require(isinstance(item['label'],str) and item['label'].strip(),'Canonical product label required')
                require(item['localization_status'] in {'READY','REVIEW'},'Invalid localization status')
                self.data['targets'].append(dict(id=self.data['id']+f':OBJ{index+1:04d}',overview=deepcopy(item),
                    detail=None,result=None,decision='REVIEW' if item['localization_status']=='REVIEW' else None,
                    status='REVIEW' if item['localization_status']=='REVIEW' else 'WAITING'))
            ready=[t for t in self.data['targets'] if t['status']=='WAITING']
            if ready: return self._request('map_targets',now,targets=ready,calibration_digest=p['calibration_digest'])
            return self._next(now)
        elif kind=='map_targets':
            require(payload['calibration_digest']==p['calibration_digest'],'Camera/robot calibration changed')
            ready=[t for t in self.data['targets'] if t['status']=='WAITING']; mapped=payload['targets']
            require(len(mapped)==len(ready) and {v['id'] for v in mapped}=={t['id'] for t in ready},'Mapped target set mismatch')
            for item in mapped:
                valid_pose(item['capture_pose'],p); valid_pose(item['pick_pose'],p)
                self._target(item['id']).update(capture_pose=deepcopy(item['capture_pose']),pick_pose=deepcopy(item['pick_pose']))
            return self._next(now)
        elif kind=='capture_detail':
            target=self._target(identity); target['detail']=self._capture(payload,request,now); target['status']='INSPECTING'
            return self._request('inspect_detail',now,target=identity,capture=target['detail'],
                overview_frame_id=self.data['overview']['frame_id'],expected_label=target['overview']['label'])
        elif kind=='inspect_detail':
            target=self._target(identity)
            require(payload['detail_frame_id']==target['detail']['frame_id'] and payload['target_id']==identity
                and payload['cycle_id']==self.data['id'],'Detail result belongs to another object or frame')
            if payload.get('review_reason'):
                require(payload.get('decision')=='REVIEW','Unconfirmed detail cannot pass')
                decision='REVIEW'; result=deepcopy(payload)
            else:
                manifest,inspection,_=load_snapshot(payload['snapshot_path'],expected_digest=payload['snapshot_digest'])
                link=inspection['config'].get('station_link',{})
                require(link=={'cycle_id':self.data['id'],'target_id':identity,'overview_frame_id':self.data['overview']['frame_id'],
                               'detail_frame_id':target['detail']['frame_id']},'Saved snapshot target link mismatch')
                require(manifest['kind']=='real' and inspection['frame']['frame_id']==target['detail']['frame_id'], 'Saved inspection frame mismatch')
                require(inspection['config'].get('product_id')==p['product_id'],'Saved inspection product mismatch')
                require(len(inspection['objects'])==1 and inspection['objects'][0]['object_id']==payload['object_id'],'Saved detail object mismatch')
                details=inspection['objects'][0].get('decision_details',{})
                require(details.get('policy_digest')==p['detail_policy_digest'],'Saved detail decision policy changed')
                with Image.open(Path(payload['snapshot_path'])/'frame.png') as image:
                    require(image.mode=='RGB' and list(image.size)==list(p['image_size'])
                        and hashlib.sha256(image.tobytes()).hexdigest()==target['detail']['rgb_sha256'],'Inspection used different detail pixels')
                decision=inspection['objects'][0]['final_decision']; result=deepcopy(payload)
                require(decision in {'OK','NG','REVIEW'},'Basic decision is unavailable')
            target.update(result=result,decision=decision,status=decision)
            return self._next(now)
        elif kind=='sort_object':
            from .sorting import verify_sort_receipt
            verify_sort_receipt(self.journal,request,payload)
            self._target(identity).update(status='SORTED',placement=deepcopy(payload)); return self._next(now)
        else: raise ValueError('Unknown scan request')

    def _next(self,now):
        target=next((t for t in self.data['targets'] if t['status']=='WAITING'),None)
        if target:
            target['status']='MOVING'
            return self._request('move_detail',now,target=target['id'],pose=target['capture_pose'])
        if self.data['profile']['auto_sort']:
            require(self.data['coordinate_valid'],'Coordinates were invalidated; recapture required')
            target=next((t for t in self.data['targets'] if t['status'] in {'OK','NG'}),None)
            if target:
                target['status']='SORTING'
                return self._request('sort_object',now,target=target['id'],pick_pose=target['pick_pose'],
                    decision=target['decision'],evidence=target['result'])
        self.data.update(state='COMPLETED',pending=None)

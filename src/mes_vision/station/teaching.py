"""Draft hand teaching and explicitly requested empty-tool test motion.

No automatic calibration/profile promotion. Uses StationRobotPort ownership.
"""
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
import json
import math
import os
import queue
import time

from mes_vision.robot.contracts import Pose, JointPose, Command
from mes_vision.training.data import require

ROLES = {
    'home':'HOME (대기 자세)', 'overview':'풀뷰 촬영 위치',
    'bottom_left':'작업판 좌하단', 'bottom_right':'작업판 우하단',
    'top_right':'작업판 우상단', 'top_left':'작업판 좌상단',
    'detail':'상세 촬영 위치 / 높이', 'pick':'집기 위치 / 높이',
    'travel':'이동 높이', 'place_ok':'OK 놓을 위치', 'place_ng':'NG 놓을 위치',
    'camera_reference':'오프셋: 카메라 중심 맞춤', 'tool_reference':'오프셋: 집기점 맞춤',
}
CORNERS = ('bottom_left','bottom_right','top_right','top_left')

def stamp():
    return datetime.now(timezone.utc).isoformat()

class TeachingStore:
    def __init__(self, runtime, context):
        self.folder=Path(runtime)/'robot-teaching'
        self.path=self.folder/'draft.json'
        self.context=deepcopy(context)
        self.value={'schema':1,'state':'draft','context':self.context,'points':{},'offset':None}
        if self.path.exists():
            value=json.loads(self.path.read_text(encoding='utf-8'))
            require(value.get('schema')==1 and value.get('state')=='draft','티칭 파일 형식 오류')
            self.value=value

    def commit(self,value):
        self.folder.mkdir(parents=True,exist_ok=True)
        value=deepcopy(value); value['updated_at']=stamp()
        temp=self.folder/(uuid4().hex+'.tmp')
        try:
            with temp.open('w',encoding='utf-8') as f:
                json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
                f.flush(); os.fsync(f.fileno())
            # Each successful edit also has an immutable revision.
            revision=self.folder/(uuid4().hex+'.json')
            revision.write_text(temp.read_text(encoding='utf-8'),encoding='utf-8')
            os.replace(temp,self.path)
        finally:
            temp.unlink(missing_ok=True)
        self.value=value

    def save(self,role,pose,epoch,overwrite=False):
        require(role in ROLES,'알 수 없는 티칭 위치')
        pose=Pose(**asdict(pose))
        value=deepcopy(self.value)
        require(role not in value['points'] or overwrite,'기존 위치입니다. 덮어쓰기를 선택하세요.')
        value['points'][role]={'pose':asdict(pose),'observed_at':stamp(),'epoch':epoch,
                               'context':self.context,'state':'draft'}
        value['context']=self.context
        if role in ('camera_reference','tool_reference'): value['offset']=None
        self.commit(value)

    def delete(self,role):
        value=deepcopy(self.value); require(role in value['points'],'저장된 위치가 없습니다.')
        del value['points'][role]
        if role in ('camera_reference','tool_reference'): value['offset']=None
        self.commit(value)

    def point(self,role,epoch):
        p=self.value['points'].get(role)
        require(p is not None,'먼저 해당 위치를 저장하세요.')
        require(p['context']==self.context and p['epoch']==epoch,
                '이전 연결 또는 다른 설치의 위치입니다. 현재 연결에서 다시 티칭하여 저장하세요.')
        return Pose(**p['pose'])

    def offset(self,epoch):
        camera=self.point('camera_reference',epoch); tool=self.point('tool_reference',epoch)
        require(abs(camera.r-tool.r)<=1,'오프셋 두 자세의 R을 동일하게 맞추세요.')
        value=deepcopy(self.value)
        value['offset']={'dx':tool.x-camera.x,'dy':tool.y-camera.y,'dz':tool.z-camera.z,
                         'reference_r':camera.r,'camera_rotates_with_r':False,
                         'meaning':'tool_reference_pose minus camera_reference_pose',
                         'state':'draft','measured_at':stamp(),
                         'note':'같은 물리 기준점의 두 로봇 자세 차이. dz는 촬영/집기 높이 차이를 포함. 자동 좌표 보정 아님.'}
        self.commit(value)
        return value['offset']

    def board(self,epoch):
        points=[self.point(r,epoch) for r in CORNERS]
        cross=[]
        for i in range(4):
            a,b,c=points[i],points[(i+1)%4],points[(i+2)%4]
            cross.append((b.x-a.x)*(c.y-b.y)-(b.y-a.y)*(c.x-b.x))
        require(all(x>1e-6 for x in cross) or all(x< -1e-6 for x in cross),
                '작업판 모서리 순서가 교차하거나 영역이 퇴화했습니다.')
        return [asdict(p) for p in points]

    def route(self,role,current,epoch,bounds):
        target=self.point(role,epoch); travel=self.point('travel',epoch)
        require(role not in CORNERS and role!='travel','모서리/이동 높이는 이동 목적지가 아닙니다.')
        require(abs(current.r-target.r)<=1,'첫 시험은 현재 자세와 목적지의 R을 동일하게 맞추세요.')
        require(travel.z>=max(current.z,target.z),'저장된 이동 높이가 현재 또는 목표 높이보다 낮습니다.')
        path=[Pose(current.x,current.y,travel.z,current.r),Pose(target.x,target.y,travel.z,current.r),target]
        for axis in 'xyzr':
            lo,hi=bounds[axis]
            require(all(type(v) in (int,float) and math.isfinite(v) for v in (lo,hi)) and lo<hi,
                    'X/Y/Z/R 허용 범위를 입력하세요.')
            require(all(lo<=getattr(p,axis)<=hi for p in [current,*path]),'현재 자세 또는 경로가 입력한 허용 범위 밖입니다.')
        return path

def stable_pose(adapter,port):
    first=adapter.status()
    first_joints=getattr(adapter,'actual_joints',None)
    require(first.connected and first.pose is not None and first.motion_state in {'STOPPED','READY'},'정지 상태가 아닙니다.')
    require(not port.stopping.wait(.2) and not port.stop_requested.is_set(),'정지 요청됨')
    second=adapter.status()
    require(second.connected and second.connection_epoch==first.connection_epoch and second.pose is not None
            and second.motion_state in {'STOPPED','READY'},'연결 또는 상태가 바뀌었습니다.')
    require(all(abs(getattr(first.pose,k)-getattr(second.pose,k))<=.2 for k in 'xyzr'),
            '위치가 변하고 있습니다. 손과 Unlock 버튼을 놓고 다시 저장하세요.')
    if first_joints is not None:
        require(all(abs(getattr(first_joints,k)-getattr(adapter.actual_joints,k))<=.2 for k in ('j1','j2','j3','j4')),
                '조인트가 움직이고 있습니다. 손과 Unlock 버튼을 놓으세요.')
    return second.pose

def axes_target(message,pose,joints,epoch,now):
    """Resolve increments from fresh measurements, never from previous targets."""
    require(message.get('empty_tool_confirmed') is True and message.get('path_confirmed') is True,
            '빈 집게·손/Unlock 해제·이동 경로를 확인하세요.')
    require(message.get('epoch')==epoch and 0<=now-message['issued_at']<=2,
            '이전 연결 또는 오래된 축 이동 요청입니다.')
    joint=message['space']=='joint'
    require(message['space'] in ('joint','cartesian'),'좌표 종류 오류')
    current=joints if joint else pose
    require(current is not None,'실측 조인트 각도를 읽을 수 없습니다.')
    keys=('j1','j2','j3','j4') if joint else tuple('xyzr')
    expected=message['expected']
    require(all(type(expected[k]) in (int,float) and math.isfinite(expected[k])
                and abs(getattr(current,k)-expected[k])<=.5 for k in keys),'화면 확인 후 위치가 바뀌었습니다. 다시 요청하세요.')
    values={k:getattr(current,k) for k in keys}
    if message['mode']=='step':
        axis=message['axis']; delta=message['delta']
        require(axis in keys and type(delta) in (float,int) and math.isfinite(delta) and 0<abs(delta)<=5,
                '버튼 이동량은 0 초과 5 mm/도 이하입니다.')
        values[axis]+=delta
    else:
        require(message['mode']=='absolute','이동 방식 오류')
        requested=message['target']
        require(bool(requested) and set(requested)<=set(keys),'목표 축 오류')
        values.update(requested)
    target=JointPose(**values) if joint else Pose(**values)
    for k in keys:
        lo,hi=message['bounds'][k]
        require(all(type(v) in (float,int) and math.isfinite(v) for v in (lo,hi)) and lo<hi,
                k.upper()+' 허용 범위를 입력하세요.')
        require(lo<=getattr(current,k)<=hi and lo<=getattr(target,k)<=hi,k.upper()+' 현재/목표 값이 허용 범위 밖입니다.')
    return target

def run_teaching(port):
    from filelock import FileLock
    from .owner import OwnedAdapter
    active=None
    try:
        folder=port.runtime/'robot'; folder.mkdir(parents=True,exist_ok=True)
        with FileLock(str(folder/'connection.lock'),timeout=0):
            with OwnedAdapter(folder,lambda:port.factory(port.equipment['robot'])) as owned:
                adapter=owned.adapter; controller=owned.controller
                controller.transition('RECOVERY','TEACHING_ENTERED_NEW_OVERVIEW_REQUIRED')
                port.state='TEACHING'
                store=TeachingStore(port.runtime,port.teaching_context)
                last_status=0
                while not port.stopping.is_set():
                    if port.stop_requested.is_set():
                        controller.stop('TEACHING_USER_STOP'); active=None
                        port.stop_requested.clear()
                    try: message=port.commands.get_nowait()
                    except queue.Empty: message=None
                    if message and port.current(message):
                        try:
                            require(active is None,'이동 중에는 저장/설정을 변경할 수 없습니다.')
                            kind=message['kind']; role=message.get('role')
                            if kind=='teach_save':
                                pose=stable_pose(adapter,port)
                                require(port.current(message),'취소된 위치 저장 요청')
                                store.save(role,pose,adapter.epoch,message.get('overwrite',False))
                                board_note=''
                                if all(r in store.value['points'] for r in CORNERS):
                                    try: store.board(adapter.epoch); board_note=' / 네 모서리 순서 확인 (도달/충돌 검증 아님)'
                                    except ValueError as exc: board_note=' / 작업판 확인 필요: '+str(exc)
                                port.completed.emit({'teaching':True,'message':ROLES[role]+' 초안 저장'+board_note})
                            elif kind=='teach_delete':
                                store.delete(role); port.completed.emit({'teaching':True,'message':'삭제 완료'})
                            elif kind=='teach_offset':
                                offset=store.offset(adapter.epoch)
                                port.completed.emit({'teaching':True,'message':f"오프셋 초안: ΔX {offset['dx']:.2f}, ΔY {offset['dy']:.2f}, ΔZ {offset['dz']:.2f} mm (자동 적용 안 함)"})
                            elif kind=='teach_move':
                                require(message.get('empty_tool_confirmed') is True and message.get('path_confirmed') is True,
                                        '빈 집게와 전체 경로 확인이 필요합니다.')
                                pose=stable_pose(adapter,port)
                                expected=Pose(**message['expected_start'])
                                require(all(abs(getattr(pose,k)-getattr(expected,k))<=.5 for k in 'xyzr'),
                                        '경로 확인 후 현재 위치가 바뀌었습니다. 이동을 다시 요청하세요.')
                                require(message['draft_revision']==store.value.get('updated_at'),
                                        '경로 확인 후 저장 위치가 바뀌었습니다.')
                                require(adapter.holding() is not True,'물체 감지 상태에서는 빈 집게 시험 이동을 할 수 없습니다.')
                                path=store.route(role,pose,adapter.epoch,message['bounds'])
                                require(port.current(message),'취소된 이동 요청')
                                controller.transition('RECOVERY','TEACHING_MOVE_INTENT',{'role':role,'path':[asdict(p) for p in path],
                                    'speed_mm_s':message['speed'],'operator_empty_tool_confirmed':True,'bounds':message['bounds']})
                                adapter.prepare_teaching_move(message['speed'])
                                require(port.current(message),'취소된 이동 요청')
                                active={'path':path,'pending':None,'generation':message['generation'],'speed':message['speed']}
                            elif kind=='teach_axes':
                                pose=stable_pose(adapter,port)
                                target=axes_target(message,pose,getattr(adapter,'actual_joints',None),adapter.epoch,time.monotonic())
                                require(adapter.holding() is not True,'물체 감지 중에는 빈 집게 축 이동을 할 수 없습니다.')
                                require(port.current(message),'취소된 축 이동 요청')
                                joint=isinstance(target,JointPose)
                                controller.transition('RECOVERY','TEACHING_AXES_INTENT',{
                                    'space':message['space'],'target':asdict(target),'bounds':message['bounds'],
                                    'speed':message['speed'],'empty_tool_confirmed':True,'mode':message['mode']})
                                adapter.prepare_teaching_move(message['speed'],joint=joint)
                                require(port.current(message),'취소된 축 이동 요청')
                                active={'path':[target],'pending':None,'generation':message['generation'],
                                        'action':'move_joints' if joint else 'move'}
                        except Exception as exc:
                            if message['kind'] in ('teach_move','teach_axes'):
                                controller.stop('TEACHING_MOVE_REJECTED'); active=None
                            port.completed.emit({'teaching':True,'error':str(exc)})
                    if active and not port.stop_requested.is_set() and port.current(active):
                        if active['pending'] is None:
                            target=active['path'].pop(0)
                            command=Command(uuid4().hex,'manual-teaching','TEACHING',active.get('action','move'),target)
                            adapter.submit(command); active['pending']=command
                            active['deadline']=time.monotonic()+120
                        else:
                            require(time.monotonic()<active['deadline'],'티칭 이동 시간 초과. 재연결 후 다시 티칭하세요.')
                            result=adapter.poll(active['pending'].command_id)
                            require(result.state in {'PENDING','DONE'},'이동 결과 불확실')
                            if result.state=='DONE':
                                active['pending']=None
                                if not active['path']:
                                    controller.stop('TEACHING_MOVE_COMPLETE'); active=None
                                    port.completed.emit({'teaching':True,'message':'목표 위치 도착 확인'})
                    if time.monotonic()-last_status>=.3:
                        status=adapter.status(); last_status=time.monotonic()
                        port.changed.emit({'state':'TEACHING','status':asdict(status),'busy':active is not None,
                                           'joints':asdict(adapter.actual_joints) if getattr(adapter,'actual_joints',None) is not None else None,
                                           'observed_at':stamp(),'draft':deepcopy(store.value)})
                    port.stopping.wait(.02)
    except Exception as exc:
        port.failed.emit(str(exc))
    finally:
        port.state='DISCONNECTED'; port.changed.emit({'state':'DISCONNECTED','busy':False})

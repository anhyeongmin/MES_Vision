"""Magician serial protocol V1.1.5. Original implementation, no vendor DLL required.

Wire definitions: https://download.dobot.cc/product-manual/dobot-magician/pdf/en/Dobot-Communication-Protocol-V1.1.5.pdf
Digital input address encoding cross-checked against Dobot-Arm/DobotLink Protocol2.
No command is retried after an ambiguous timeout. Physical commissioning remains required.
"""
import struct
import time
import math
from uuid import uuid4
from mes_vision.training.data import require
from .contracts import Pose,JointPose,RobotStatus,Completion


def packet(command,write=False,queued=False,parameters=b""):
    payload=bytes((command,int(write)|(int(queued)<<1)))+parameters
    require(2<=len(payload)<=244,"invalid packet length")
    return b"\xaa\xaa"+bytes((len(payload),))+payload+bytes(((-sum(payload))&255,))


class SerialProtocol:
    def __init__(self,port,*,transport=None,timeout=.4):
        if transport is None:
            import serial
            transport=serial.Serial(port,115200,timeout=.05,write_timeout=timeout)
        self.transport=transport; self.timeout=timeout; self.tainted=False
        self.transport.reset_input_buffer()
    def request(self,command,write=False,queued=False,parameters=b"",*,emergency=False):
        require(not self.tainted or emergency,"통신 결과가 불확실합니다. 장비를 재연결하세요.")
        deadline=time.monotonic()+self.timeout; buffer=bytearray()
        try:
            message=packet(command,write,queued,parameters)
            require(self.transport.write(message)==len(message),"로봇 명령 전송이 완료되지 않았습니다.")
            while time.monotonic()<deadline:
                buffer.extend(self.transport.read(256))
                while len(buffer)>=4:
                    start=buffer.find(b"\xaa\xaa")
                    if start<0: buffer[:]=buffer[-1:]; break
                    if start: del buffer[:start]
                    if len(buffer)<4: break
                    length=buffer[2]
                    require(2<=length<=244,"로봇 응답 길이 오류")
                    if len(buffer)<length+4: break
                    body=bytes(buffer[3:3+length]); checksum=buffer[3+length]; del buffer[:length+4]
                    require((sum(body)+checksum)&255==0,"로봇 응답 체크섬 오류")
                    require(body[0]==command and (body[1]&3)==(int(write)|(int(queued)<<1)),"이전 또는 다른 로봇 명령의 응답입니다.")
                    return body[2:]
            raise TimeoutError("로봇 응답 시간이 초과됐습니다. 명령을 자동 재전송하지 않습니다.")
        except Exception:
            self.tainted=True; raise
    def close(self): self.transport.close()


class Magician:
    kind="real"
    def __init__(self,settings,*,protocol=None):
        self.settings=settings; self.protocol=protocol or SerialProtocol(settings["port"])
        self.epoch=uuid4().hex; self.connected=True; self.motion_state="STOPPED"; self.pending=None
        self.pose_tolerance=settings["pose_tolerance_mm"]; self.rotation_tolerance=settings["rotation_tolerance_deg"]
        self.tool=None; self.stop_confirmed=False; self.actual_pose=None
        self.actual_joints=None; self.teaching_joint_ready=False
        try:
            self.request_stop()
            require(self.stop_confirmed,"로봇 정지를 확인할 수 없습니다.")
            self.protocol.request(245,True)
            self.read_pose()
            address=settings["input_address"]
            if address is not None: self.protocol.request(130,True,parameters=bytes((address,3)))
        except Exception:
            self.close(); raise
    @staticmethod
    def ports():
        from serial.tools import list_ports
        return [{"port":p.device,"description":p.description} for p in list_ports.comports()]
    def read_pose(self):
        data=self.protocol.request(10)
        require(len(data)==32,"Magician 위치 응답 형식이 다릅니다.")
        values=struct.unpack("<8f",data)
        require(all(math.isfinite(v) for v in values),"로봇 위치 값이 유효하지 않습니다.")
        self.actual_pose=Pose(*values[:4]); self.actual_joints=JointPose(*values[4:]); return self.actual_pose
    def holding(self):
        address=self.settings["input_address"]
        if address is None: return None
        data=self.protocol.request(133,parameters=bytes((address,)))
        require(len(data)==2 and data[0]==address and data[1] in {0,1},"집기 확인 센서 응답 오류")
        return data[1]==self.settings["holding_level"]
    def status(self):
        if not self.connected: return RobotStatus(self.epoch,False,"UNKNOWN",None,None)
        try:
            alarms=self.protocol.request(20)
            require(not any(alarms),"로봇 알람이 발생했습니다. 장비 상태를 확인하세요.")
            return RobotStatus(self.epoch,True,self.motion_state,self.holding(),self.read_pose())
        except Exception:
            self.motion_state="UNKNOWN"; raise
    def configure(self,profile):
        self.teaching_joint_ready=False
        require(profile.kind=="real" and profile.validated,"검증된 실물 로봇 설정이 필요합니다.")
        require(profile.verification_method=="digital_input", "독립 집기 확인 센서 설정이 필요합니다.")
        require(self.settings["input_address"] is not None,"집기 확인 센서 입력을 등록하세요.")
        speed=self.settings.get("speed_ratio"); acceleration=self.settings.get("acceleration_ratio")
        require(all(type(v) in {int,float} and math.isfinite(v) and 0<v<=100 for v in (speed,acceleration))
                and self.settings.get("motion_validation_reference"),"검증된 이동 속도·가속도와 기록을 등록하세요.")
        self.protocol.request(83,True,parameters=struct.pack("<2f",speed,acceleration))
        readback=self.protocol.request(83)
        require(len(readback)==8 and all(abs(a-b)<.01 for a,b in zip(struct.unpack("<2f",readback),(speed,acceleration))),"이동 속도 설정을 확인하지 못했습니다.")
        self.tool=profile.end_effector
    def ready_after_recovery(self):
        require(self.stop_confirmed and self.holding() is False,"정지·빈 집기 상태를 먼저 확인하세요.")
        self.protocol.request(245,True)
        self.epoch=uuid4().hex; self.motion_state="READY"; self.pending=None
    def prepare_teaching_move(self,speed,*,joint=False):
        """Explicit empty-tool commissioning only; never used by production recovery."""
        require(type(speed) in {int,float} and math.isfinite(speed) and 0<speed<=10,
                '티칭 이동 속도는 0 초과 10 mm/s 이하로 입력하세요.')
        require(self.pending is None and self.stop_confirmed,'먼저 정지 상태를 확인하세요.')
        # Set absolute Cartesian limits as well as ratio: do not inherit Studio settings.
        params=(80,(speed,)*4+(20.,)*4) if joint else (81,(speed,speed,20.,20.))
        self.teaching_joint_ready=False
        for code,values in (params,(83,(100.,100.))):
            self.protocol.request(code,True,parameters=struct.pack('<'+'f'*len(values),*values))
            data=self.protocol.request(code)
            require(len(data)==4*len(values) and all(abs(a-b)<.01 for a,b in zip(struct.unpack('<'+'f'*len(values),data),values)),
                    '티칭 속도 설정 읽기 확인 실패')
        self.protocol.request(245,True)
        self.teaching_joint_ready=joint
        self.tool='gripper'; self.motion_state='READY'
    def submit(self,command):
        require(self.motion_state=="READY" and self.pending is None and self.tool in {"suction","gripper"},"로봇 운전 준비 상태가 아닙니다.")
        self.stop_confirmed=False
        if command.action=="move":
            p=command.target; payload=struct.pack("<B4f",2,p.x,p.y,p.z,p.r)
            response=self.protocol.request(84,True,True,payload)
        elif command.action=='move_joints':
            require(self.teaching_joint_ready and isinstance(command.target,JointPose),'조인트 티칭 준비가 필요합니다.')
            p=command.target
            response=self.protocol.request(84,True,True,struct.pack('<B4f',4,p.j1,p.j2,p.j3,p.j4))
        elif command.action in {"engage","release"}:
            response=self.protocol.request(62 if self.tool=="suction" else 63,True,True,bytes((1,int(command.action=="engage"))))
        elif command.action in {"verify_pick","verify_place"}:
            self.pending=(command,None); return True
        else: raise ValueError("지원하지 않는 로봇 명령입니다.")
        require(len(response)==8,"명령 접수 번호를 받지 못했습니다.")
        index=struct.unpack("<Q",response)[0]
        self.pending=(command,index)
        self.protocol.request(240,True)
        return True
    def poll(self,command_id):
        require(self.pending is not None and self.pending[0].command_id==command_id,"현재 로봇 명령 ID가 다릅니다.")
        command,index=self.pending
        if index is not None:
            response=self.protocol.request(246); require(len(response)==8,"명령 완료 번호 응답 오류")
            current=struct.unpack("<Q",response)[0]
            if current<index: return Completion(command_id,"PENDING")
            require(current==index,"알 수 없는 명령이 로봇에서 실행됐습니다.")
        if command.action=="move":
            actual=self.read_pose(); target=command.target
            close=all(abs(getattr(actual,k)-getattr(target,k))<=self.pose_tolerance for k in ("x","y","z")) and abs(actual.r-target.r)<=self.rotation_tolerance
            if not close: return Completion(command_id,"PENDING",actual,detail="목표 위치 도달 확인 중")
            response=Completion(command_id,"DONE",target,detail=f"measured={actual}; tolerance_mm={self.pose_tolerance}")
        elif command.action=='move_joints':
            self.read_pose(); target=command.target
            if not all(abs(getattr(self.actual_joints,k)-getattr(target,k))<=self.rotation_tolerance for k in ('j1','j2','j3','j4')):
                return Completion(command_id,'PENDING',detail='실측 조인트 도착 확인 중')
            response=Completion(command_id,'DONE',detail=f'measured_joints={self.actual_joints}')
        elif command.action.startswith("verify_"):
            verified=self.holding() is (command.action=="verify_pick")
            if not verified: return Completion(command_id,"PENDING",verified=False)
            response=Completion(command_id,"DONE",verified=True,detail="독립 디지털 입력 확인")
        else: response=Completion(command_id,"DONE")
        self.pending=None; return response
    def request_stop(self):
        try:
            self.protocol.request(242,True,emergency=True)
            self.motion_state="STOPPED"; self.pending=None; self.stop_confirmed=True; return True
        except Exception:
            self.motion_state="UNKNOWN"; self.stop_confirmed=False; return False
    def close(self):
        if self.connected:
            if not self.stop_confirmed: self.request_stop()
            self.protocol.close(); self.connected=False

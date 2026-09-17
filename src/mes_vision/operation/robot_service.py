from dataclasses import asdict
from pathlib import Path
import queue
import threading
import time
from PySide6.QtCore import QThread,Signal
from mes_vision.robot import RobotController,build_plan,SceneStamp,load_profile
from mes_vision.robot.magician import Magician
from mes_vision.calibration import load,map_target,GeometryContext
from mes_vision.decision.io import run_from_dict
from mes_vision.training.data import require,sha256
from mes_vision.station.owner import OwnedAdapter


def geometry_context(equipment,frame):
    from .acquisition import inspection_camera
    c=inspection_camera(equipment["camera"])
    return GeometryContext(c["serial"],c["mount_revision"],c["acquisition_revision"],c["robot_base_id"],c["tool_frame_id"],
        (frame["width"],frame["height"]),tuple(frame["transformations"]))


def real_plan(result,track,product,equipment,epoch,now):
    require(track["result"] is not None and track["result"]["object_id"] in {o["object_id"] for o in result["objects"]},"현재 물체와 검사 결과가 다릅니다.")
    require(not track["missing"] and track["status"] in {"OK","NG"},"현재 유효한 OK/NG 물체가 필요합니다.")
    require(product["grasp"]["validation_reference"] and product["grasp"]["version"],"품목 집기 기준 검증 기록이 필요합니다.")
    pinned=result["config"].get("equipment_asset_digests",{})
    for key,path in (("calibration",equipment["calibration"]),("robot_profile",equipment["robot"]["profile"])):
        require(path and pinned.get(key)==sha256(Path(path)),"검사 이후 보정 또는 로봇 설정 파일이 변경됐습니다. 설정을 확인하고 모델을 다시 준비하세요.")
    calibration=load(Path(equipment["calibration"])); profile=load_profile(Path(equipment["robot"]["profile"]))
    inspection=run_from_dict(result); obj=next(o for o in inspection.objects if o.object_id==track["result"]["object_id"])
    g=product["grasp"]; b=obj.effective_box
    require(result["config"]["product_version"]==product["version"] and result["config"]["equipment_version"]==equipment["version"],"검사 후 설정이 변경됐습니다.")
    saved=result["config"]["tracking_links"][obj.object_id]
    require(saved["track_id"]==track["track_id"] and saved["revision"]==track["revision"],"검사 후 물체가 이동하거나 추적이 변경됐습니다.")
    require(0<g["u"]<1 and 0<g["v"]<1,"집기점은 물체 내부여야 합니다.")
    target=map_target(calibration,inspection,obj.object_id,(b.x1+(b.x2-b.x1)*g["u"],b.y1+(b.y2-b.y1)*g["v"]),
        geometry_context(equipment,result["frame"]),plane_z_mm=g["plane_z_mm"],r_deg=g["rotation_deg"],grasp_policy_version=g["version"])
    scene=SceneStamp(result["run_id"],result["frame"]["frame_id"],track["revision"],track["result"]["observed_at"],calibration.identity,epoch)
    return build_plan(inspection,obj.object_id,target,profile,scene,now=now),scene


class RobotThread(QThread):
    changed=Signal(object); failed=Signal(str); completed=Signal(object)
    def __init__(self,root,equipment,store,*,adapter_factory=Magician):
        super().__init__(); self.root=Path(root); self.equipment=equipment; self.store=store; self.adapter_factory=adapter_factory
        self.commands=queue.Queue(8); self.stopping=threading.Event(); self.stop_requested=threading.Event()
        self.state="DISCONNECTED"; self.epoch=None; self.command_lock=threading.Lock(); self.command_generation=0
    def request(self,kind,**kwargs):
        with self.command_lock:
            require(kind in {"pick","recover"} and not self.stopping.is_set() and not self.stop_requested.is_set(),"정지 처리 중에는 새 로봇 명령을 요청할 수 없습니다.")
            self.commands.put_nowait({**kwargs,"type":kind,"generation":self.command_generation})
    def stop_motion(self):
        with self.command_lock:
            self.command_generation+=1; self.stop_requested.set()
            while True:
                try: self.commands.get_nowait()
                except queue.Empty: break
    def valid_command(self,message):
        with self.command_lock: return not self.stopping.is_set() and not self.stop_requested.is_set() and message["generation"]==self.command_generation
    def run(self):
        adapter=None; controller=None; owned=None; scene=None; current=None; last_status=0.
        try:
            owned=OwnedAdapter(self.root,lambda:self.adapter_factory(self.equipment["robot"]))
            owned.__enter__(); adapter=owned.adapter; controller=owned.controller; self.epoch=adapter.epoch
            # Connection never arms motion. An operator must explicitly confirm recovery/readiness.
            if controller.state in {"IDLE","RECAPTURE"}: controller.transition("RECOVERY","CONNECTED_REQUIRES_CONFIRMATION")
            while not self.stopping.is_set():
                if self.stop_requested.is_set():
                    controller.stop("OPERATOR_STOP"); self.stop_requested.clear()
                    if current:
                        event={"state":controller.state,"track_id":current["track"]["track_id"],"object_id":current["track"]["result"]["object_id"]}
                        self.store.event("ROBOT_FINISHED",event,session=current["session"],track_id=event["track_id"]); self.completed.emit(event)
                    current=None
                try: message=self.commands.get_nowait()
                except queue.Empty: message=None
                if message and self.valid_command(message):
                    try:
                        if message["type"]=="recover":
                            require(self.equipment["robot"]["profile"],"로봇 운전 설정을 등록하세요.")
                            profile=load_profile(Path(self.equipment["robot"]["profile"])); adapter.configure(profile)
                            controller.recover(confirmation_reference=message["reference"]); self.epoch=adapter.epoch
                            self.store.event("ROBOT_RECOVERY",{"reference":message["reference"],"epoch":self.epoch})
                        elif message["type"]=="pick":
                            plan,scene=real_plan(message["result"],message["track"],message["product"],self.equipment,adapter.epoch,time.monotonic())
                            if not self.valid_command(message): continue
                            controller.start(plan,scene,now=time.monotonic()); current=message
                            self.store.event("ROBOT_PLAN",{"plan_id":plan.plan_id,"object_id":plan.object_id,"run_id":scene.run_id},
                                session=message["session"],track_id=message["track"]["track_id"])
                    except Exception as exc:
                        self.stop_motion()
                        # Includes failure to persist the operation event after reserving a plan.
                        # Do not reach tick() and dispatch motion after any request failure.
                        controller.stop("REQUEST_OR_RECORD_FAILED"); current=None; self.failed.emit(str(exc)); continue
                if current:
                    if self.stop_requested.is_set(): continue
                    state=controller.tick(scene,now=time.monotonic())
                    if state in {"RECAPTURE","FAULT","STOPPED","RECOVERY"}:
                        event={"state":state,"track_id":current["track"]["track_id"],"object_id":current["track"]["result"]["object_id"]}
                        self.store.event("ROBOT_FINISHED",event,session=current["session"],track_id=event["track_id"])
                        self.completed.emit(event); current=None
                if time.monotonic()-last_status>.3:
                    status=adapter.status(); last_status=time.monotonic(); self.state=controller.state
                    self.changed.emit({"state":controller.state,"status":asdict(status),"epoch":adapter.epoch,"busy":current is not None})
                self.stopping.wait(.05)
        except Exception as exc: self.failed.emit(str(exc))
        finally:
            if owned and controller:
                try: owned.__exit__(None,None,None)
                except Exception: pass
            self.state="DISCONNECTED"; self.changed.emit({"state":"DISCONNECTED","busy":False})

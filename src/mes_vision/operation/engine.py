from dataclasses import asdict
from pathlib import Path
from contextlib import nullcontext
import multiprocessing as mp
import os
import queue
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from uuid import uuid4
import numpy as np

from mes_vision.inspection import InspectionPipeline,Mode,CheckResult,CheckStatus
from mes_vision.inspection.contracts import DetectionBatch
from mes_vision.decision import load_policy,apply_policy,FrameEvidence
from mes_vision.vlm.snapshots import save_snapshot
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.training.data import require,write_json
from .catalog import OperationStore,readiness
from .tracking import Tracker
from .quality import quality,in_workspace,inside


class FrozenDetector:
    def __init__(self,batch): self.batch=batch; self.model=batch.model
    def detect(self,frame):
        require(frame.frame_id==self.batch.frame_id,"검출 프레임이 변경됐습니다.")
        return self.batch


class SelectiveInspector:
    def __init__(self,inspector,indices):
        self.inspector=inspector; self.indices=indices
        self.check_id=inspector.check_id; self.model=inspector.model
        self.product_id=getattr(inspector,"product_id",None)
    def inspect(self,crop):
        index=int(crop.object_id.rsplit("OBJ",1)[1])-1
        if index in self.indices: return self.inspector.inspect(crop)
        return CheckResult(self.check_id,crop.frame_id,crop.object_id,CheckStatus.NOT_RUN,self.model,messages=("EXISTING_TRACK_NOT_REINSPECTED",))

    def inspect_many(self,crops):
        from mes_vision.inspection.batching import validate_crops,validate_results
        crops=validate_crops(crops)
        selected=[crop for crop in crops if int(crop.object_id.rsplit('OBJ',1)[1])-1 in self.indices]
        if not selected: return tuple(self.inspect(crop) for crop in crops)
        method=getattr(self.inspector,'inspect_many',None)
        checks=method(selected) if callable(method) else tuple(self.inspector.inspect(crop) for crop in selected)
        checks=validate_results(selected,checks,self.inspector)
        by_id={check.object_id:check for check in checks}
        return tuple(by_id[crop.object_id] if crop.object_id in by_id else self.inspect(crop) for crop in crops)


def vlm_eligible(obj,policy):
    if policy=="manual": return False
    if policy=="ng_and_review" and obj.final_decision=="NG": return True
    if obj.final_decision!="REVIEW": return False
    details=obj.decision_details
    if not details.get("identity_valid"): return False
    reasons=details.get("reasons",[])
    optional={c["check_id"] for c in details.get("checks",[]) if not c["required"]}
    reasons=[r for r in reasons if not (r.get("check_id") in optional and r["code"] in {"CHECK_NOT_RUN","OPTIONAL_CHECK_MISSING"})]
    visual={"CHECK_UNCERTAIN","DEFECT_CANDIDATE_IN_REVIEW_BAND"}
    harmless={"CHECK_PASSED"}
    return any(r["code"] in visual for r in reasons) and all(r["code"] in visual|harmless for r in reasons)


class Models:
    """One owned set of product models, loaded and warmed once per configuration."""
    def __init__(self,root,product,*,registry=None):
        from mes_vision.inspection.model_registry import default_registry
        self.registry=registry or default_registry()
        self.root=Path(root); self.product=product; self.backends=[]; self.inspectors=[]; self.detector=None; self.loads=0; self.features=None; self.anomaly_engine=None
    def load(self):
        def backend(role):
            b=self.registry.create(self.product[role],role)
            self.backends.append(b); b.load()
            self.loads+=1; return b.adapter
        self.detector=backend("objects")
        if self.product["defects"]:
            self.inspectors.append(backend("defects"))
        if self.product["anomaly"] and "backend" in self.product["anomaly"]:
            self.inspectors.append(backend("anomaly"))
        elif self.product["anomaly"]:
            from mes_vision.anomaly.features import DinoFeatures
            from mes_vision.anomaly.scoring import AnomalyEngine,AnomalyInspector
            from mes_vision.training.data import read_json
            s=self.product["anomaly"]; meta=read_json(Path(s["bank"])/"bank.json")
            features=DinoFeatures(self.root,image_size=meta["feature_signature"]["preprocessing"]["size"],max_batch_size=meta['feature_signature'].get('batch_limit',1)); self.features=features; features.load()
            engine=AnomalyEngine(s["bank"],features,product_id=self.product["id"],criteria=s["criteria"]); self.anomaly_engine=engine
            engine.prepare()
            engine.score(np.zeros((128,128,3),np.uint8))
            if features.max_batch_size>1: engine.score_many([np.zeros((128,128,3),np.uint8)]*features.max_batch_size)
            self.inspectors.append(AnomalyInspector(engine))
        if self.product["geometry"]:
            from .quality import GeometryInspector
            self.inspectors.append(GeometryInspector(self.product["geometry"]))
        self.policy=load_policy(Path(self.product["policy"]))
        require(self.policy.detector==self.detector.model,"판정 기준과 물체 모델이 다릅니다.")
        for rule in self.policy.rules:
            if rule.required:
                match=next((i for i in self.inspectors if i.check_id==rule.check_id),None)
                require(match is not None and match.model==rule.model,"필수 검사 모델·판정 기준이 다릅니다: "+rule.check_id)
    def close(self):
        if self.anomaly_engine: self.anomaly_engine.close(); self.anomaly_engine=None
        for b in self.backends: b.close()
        if self.features: self.features.close(); self.features=None
        self.backends=[]; self.inspectors=[]; self.detector=None


class LiveEngine:
    """Independent of Qt. All mutations are serialized in one process."""
    def __init__(self,root,runtime,product,equipment,session,*,models=None):
        self.root=Path(root); self.store=OperationStore(runtime); self.product=product; self.equipment=equipment; self.session=session
        self.models=models or Models(root,product); self.tracker=Tracker(session=session,**equipment["tracking"])
        self.queue=AnalysisQueue(self.store.root/"vlm"); self.gpu=GpuCoordinator(self.queue.root/"gpu-coordination")
        self.enabled=False; self.generation=0; self.ready=False; self.concurrent=False
        self.last_camera_session=None; self.last_sequence=-1; self.run_count=0
        from mes_vision.training.data import sha256
        self.hardware_digests={k:sha256(Path(p)) for k,p in {"calibration":equipment["calibration"],"robot_profile":equipment["robot"]["profile"]}.items() if p}
        self.writer=ThreadPoolExecutor(max_workers=1,thread_name_prefix="inspection-evidence"); self.saving=None
    def prepare(self):
        from filelock import FileLock
        self.resident_lock=FileLock(str(self.gpu.root/"resident.lock"),timeout=0); self.resident_lock.acquire()
        with self.gpu.foreground(timeout=60): self.models.load()
        self.ready=True
        # Coexist only with an explicit memory reserve. Otherwise VLM waits while tracking owns GPU.
        import torch
        if torch.cuda.is_available():
            free,total=torch.cuda.mem_get_info(); self.concurrent=free>=12*1024**3 and total>=20*1024**3
        write_json(self.gpu.root/"resident.json",{"pid":os.getpid(),"allow_vlm":self.concurrent,"active":True})
    def set_enabled(self,enabled,generation,*,reset_all=False):
        require(type(enabled) is bool and generation>=self.generation,"검사 제어 번호 오류")
        self.generation=generation; self.enabled=enabled
        for track in self.tracker.tracks.values():
            if reset_all or track.result is None or track.missing: track.invalidate(time.monotonic(),"RECHECK")
        self.store.event("INSPECTION_STARTED" if enabled else "INSPECTION_PAUSED",{"generation":generation},session=self.session)
    def detect(self,frame,now):
        require(frame.is_live and frame.source_kind.value in {"d405","uvc"},"운영 검사는 연결된 카메라의 실시간 입력을 사용합니다.")
        if self.last_camera_session!=frame.session_id:
            self.tracker=Tracker(session=self.session+":"+uuid4().hex[:8],**self.equipment["tracking"])
            self.last_camera_session=frame.session_id; self.last_sequence=-1
        require(frame.sequence>self.last_sequence,"이전 카메라 프레임을 재사용할 수 없습니다.")
        self.last_sequence=frame.sequence
        self.observed_at=now
        with nullcontext() if self.concurrent else self.gpu.foreground(): batch=self.models.detector.detect(frame)
        workspace=self.equipment["workspace"]
        # Completely outside the workspace is not a product in the current work area.
        detections=tuple(d for d in batch.detections if inside(((d.box.x1+d.box.x2)/2,(d.box.y1+d.box.y2)/2),workspace["roi"]))
        batch=DetectionBatch(batch.frame_id,batch.image_size,detections,batch.model,
                             candidate_threshold=batch.candidate_threshold,excluded_reserved_slots=batch.excluded_reserved_slots)
        assigned=self.tracker.update(detections,frame.rgb,now)
        for event in self.tracker.events: self.store.event(event["event"],{},session=self.session,track_id=event["track_id"])
        return batch,assigned
    def inspect(self,frame,batch,assigned):
        if not self.enabled: return None
        selected={i:t for i,t in enumerate(assigned) if t is not None and t.status=="STABLE"}
        if not selected: return None
        good,measurements=quality(frame.rgb,[batch.detections[i].box for i in selected],self.product["quality"])
        workspace_good=all(in_workspace(batch.detections[i].box,self.equipment["workspace"]) for i in selected)
        if not good or not workspace_good:
            for t in selected.values():
                t.status="IMAGE_REVIEW"; t.stable_since=time.monotonic()
            return None
        generations={t.id:t.revision for t in selected.values()}
        for t in selected.values(): t.status="INSPECTING"
        inspectors=tuple(SelectiveInspector(i,set(selected)) for i in self.models.inspectors)
        pipeline=InspectionPipeline(FrozenDetector(batch),inspectors,mode=Mode.LIVE,product_id=self.product["id"],
            expected_count=self.product["expected_count"] if self.product["count_mode"]=="fixed" else None)
        with nullcontext() if self.concurrent else self.gpu.foreground(): raw=pipeline.run(frame)
        raw.config.update({"operation_session":self.session,"product_version":self.product["version"],"equipment_version":self.equipment["version"],
            "equipment_asset_digests":self.hardware_digests,
            "generation":self.generation,"quality_measurements":measurements,"tracking_links":{
                raw.objects[i].object_id:{"track_id":t.id,"revision":t.revision} for i,t in selected.items()}})
        q=self.product["quality"]
        evidence=FrameEvidence(raw.run_id,frame.frame_id,self.product["id"],"real",q["version"],CheckStatus.PASS,CheckStatus.PASS,True,q["validation_reference"])
        result=apply_policy(raw,self.models.policy,evidence)
        return {"frame":frame,"result":result,"generation":self.generation,"revisions":generations,"indices":list(selected),"observed_at":self.observed_at}
    def valid_links(self,pending):
        if not pending or not self.enabled or pending["generation"]!=self.generation: return None
        result=pending["result"]; links=[]
        for i in pending["indices"]:
            obj=result.objects[i]; link=result.config["tracking_links"][obj.object_id]; t=self.tracker.tracks.get(link["track_id"])
            if t and not t.missing and t.revision==link["revision"] and t.status in {"STABLE","INSPECTING"}:
                links.append(dict(link,object_id=obj.object_id))
        return links
    def write_snapshot(self,output,frame,result,**kwargs):
        from .storage import check_free_space
        check_free_space(self.store,frame.rgb.nbytes*3)
        return save_snapshot(output,frame,result,**kwargs)
    def begin_save(self,pending):
        if self.saving or not self.valid_links(pending): return False
        result=pending["result"]
        output=self.store.root/"inspections"/result.run_id
        future=self.writer.submit(self.write_snapshot,output,pending["frame"],result,kind="real",normal_export=self.product["normal_reference"],
            criteria={"product":self.product["name"],"instructions":self.product["vlm_criteria"]} if self.product["vlm_criteria"].strip() else None)
        self.saving=(pending,output,future); self.save_started=time.monotonic(); return True
    def finish_save(self):
        if self.saving and not self.saving[2].done() and time.monotonic()-self.save_started>30:
            raise TimeoutError("검사 원본 저장 시간이 초과됐습니다. 저장 장치와 권한을 확인하세요.")
        if not self.saving or not self.saving[2].done(): return None
        pending,output,future=self.saving; self.saving=None; future.result()
        links=self.valid_links(pending)
        if not links:
            self.store.event("UNPUBLISHED_SNAPSHOT",{"path":str(output),"reason":"TRACK_OR_INSPECTION_CHANGED"},session=self.session)
            return None
        result=pending["result"]
        self.store.register(output,self.session,links)
        accepted=[]
        for link in links:
            obj=next(o for o in result.objects if o.object_id==link["object_id"])
            record={"object_id":obj.object_id,"run_id":result.run_id,"path":str(output),"decision":obj.final_decision,
                "inspected_at":time.monotonic(),"generation":self.generation}
            record["observed_at"]=pending["observed_at"]
            if self.tracker.accept(link["track_id"],link["revision"],record): accepted.append(dict(link,**record))
            try:
                if self.queue.enabled() and vlm_eligible(obj,self.product["vlm_policy"]): self.queue.enqueue(output,obj.object_id,asdict(GenerationConfig()))
            except Exception as exc: self.store.event("VLM_REQUEST_FAILED",{"object_id":obj.object_id,"error":str(exc)},session=self.session)
        self.run_count+=1
        return {"run_id":result.run_id,"path":str(output),"links":accepted,"counts":self.store.counts(self.session)}
    def commit(self,pending):
        """Synchronous API for headless callers; production uses bounded asynchronous storage."""
        if not self.begin_save(pending): return None
        self.saving[2].result(); return self.finish_save()
    def close(self):
        self.writer.shutdown(wait=True,cancel_futures=True)
        self.models.close()
        write_json(self.gpu.root/"resident.json",{"pid":os.getpid(),"allow_vlm":True,"active":False})
        if hasattr(self,"resident_lock"): self.resident_lock.release()


def latest_put(q,value):
    try: q.put_nowait(value); return True
    except queue.Full:
        try: q.get_nowait()
        except queue.Empty: return False
        try: q.put_nowait(value); return True
        except queue.Full: return False


def engine_process(root,runtime,product,equipment,session,frames,controls,events,states,parent_pid,models_factory=None,progress=None,phase=None):
    import sys
    log_stream=None
    if sys.stdout is None or sys.stderr is None:
        logs=Path(runtime)/"logs"; logs.mkdir(parents=True,exist_ok=True)
        log_stream=(logs/("engine-"+session+".log")).open("a",encoding="utf-8",buffering=1)
        if sys.stdout is None: sys.stdout=log_stream
        if sys.stderr is None: sys.stderr=log_stream
    from mes_vision.vlm.worker import watch_parent
    watch_parent(parent_pid)
    engine=None; quitting=False; last_frame_time=time.monotonic()
    def heartbeat(value):
        if progress is not None: progress.value=time.monotonic(); phase.value=value
    def control():
        nonlocal quitting
        while True:
            try: message=controls.get_nowait()
            except queue.Empty: return
            if message["type"]=="quit": quitting=True; return
            if message["type"]=="enable": engine.set_enabled(message["value"],message["generation"],reset_all=message.get("reset_all",False))
            if message["type"]=="recheck": engine.tracker.recheck(message["track_id"],time.monotonic())
            if message["type"]=="invalidate": engine.tracker.invalidate_all(time.monotonic())
    try:
        heartbeat(1)
        engine=LiveEngine(root,runtime,product,equipment,session,models=models_factory() if models_factory else None)
        events.put({"type":"loading"},timeout=1); engine.prepare()
        events.put({"type":"ready","concurrent_vlm":engine.concurrent},timeout=1)
        while not quitting:
            heartbeat(2)
            control()
            if quitting: break
            try: frame,received_at=frames.get(timeout=.1)
            except queue.Empty:
                if time.monotonic()-last_frame_time>2: engine.tracker.invalidate_all(time.monotonic())
                continue
            if time.monotonic()-received_at>.5: continue
            last_frame_time=time.monotonic(); began=time.perf_counter()
            heartbeat(3)
            batch,assigned=engine.detect(frame,received_at)
            control()
            if quitting: break
            committed=engine.finish_save()
            if committed: events.put({"type":"inspection","generation":engine.generation,**committed},timeout=1)
            heartbeat(4)
            pending=engine.inspect(frame,batch,assigned) if engine.saving is None else None
            control()
            if pending and not quitting:
                # Re-observe a newer available frame before publishing an expensive inspection.
                try:
                    newer,new_at=frames.get_nowait()
                    if time.monotonic()-new_at<=.5: engine.detect(newer,new_at)
                    else: engine.tracker.invalidate_all(time.monotonic())
                except queue.Empty:
                    if time.monotonic()-received_at>engine.equipment["tracking"]["max_gap"]: engine.tracker.invalidate_all(time.monotonic())
                control()
                if not quitting:
                    engine.begin_save(pending)
            latest_put(states,{"type":"tracks","tracks":[t.data() for t in engine.tracker.tracks.values()],"generation":engine.generation,
                "frame_id":frame.frame_id,"updated":engine.tracker.last_time,"loop_ms":(time.perf_counter()-began)*1000})
    except Exception as exc:
        from .faults import fault_code
        try: events.put({"type":"error","code":fault_code(type(exc).__name__+": "+str(exc)),"message":str(exc),"trace":traceback.format_exc()},timeout=1)
        except queue.Full: pass
    finally:
        heartbeat(5)
        if engine: engine.close()
        if log_stream: log_stream.flush()


class EngineProcess:
    def __init__(self,root,runtime,product,equipment,session,*,models_factory=None):
        ctx=mp.get_context("spawn")
        self.frames=ctx.Queue(1); self.controls=ctx.Queue(16); self.events=ctx.Queue(16); self.states=ctx.Queue(1)
        self.progress=ctx.Value('d',time.monotonic(),lock=False); self.phase=ctx.Value('i',1,lock=False)
        self.process=ctx.Process(target=engine_process,args=(str(root),str(runtime),product,equipment,session,
            self.frames,self.controls,self.events,self.states,os.getpid(),models_factory,self.progress,self.phase),daemon=True)
        try: self.process.start()
        except Exception:
            for q in (self.frames,self.controls,self.events,self.states): q.cancel_join_thread(); q.close()
            raise
        self.sent_frame=None; self.stopping=False
    def frame(self,frame,received):
        if frame.frame_id==self.sent_frame: return
        if latest_put(self.frames,(frame,received)): self.sent_frame=frame.frame_id
    def command(self,kind,**kwargs): self.controls.put({"type":kind,**kwargs},timeout=.1)
    def watchdog_error(self,now=None):
        if self.stopping: return None
        elapsed=(time.monotonic() if now is None else now)-self.progress.value
        limit=180 if self.phase.value==1 else 30
        if elapsed<=limit: return None
        label={1:"모델 준비",2:"입력 처리",3:"물체 검출",4:"물체 검사",5:"모델 종료"}.get(self.phase.value,"검사 처리")
        return f"{label} 응답이 {limit}초 동안 없습니다. 모델을 해제하고 원인을 확인하세요."
    def poll(self):
        result=[]
        for q in (self.events,self.states):
            for _ in range(32):
                try: result.append(q.get_nowait())
                except queue.Empty: break
        return result
    def stop(self):
        if not self.stopping:
            self.stopping=True
            try: self.command("quit")
            except queue.Full: self.process.terminate()
    def dispose(self):
        if self.process.is_alive(): self.process.terminate()
        self.process.join(timeout=1)
        if self.process.is_alive(): self.process.kill(); self.process.join(timeout=1)
        require(not self.process.is_alive(),"검사 프로세스가 종료되지 않았습니다. 새 모델을 시작할 수 없습니다.")
        for q in (self.frames,self.controls,self.events,self.states): q.cancel_join_thread(); q.close()

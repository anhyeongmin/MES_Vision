"""Software contract tests with injected outputs only; no physical product validation."""
from copy import deepcopy
from dataclasses import asdict,replace
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace
import struct
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np

from mes_vision.inputs import Frame,SourceKind,D405Source,InputStatus
from mes_vision.inspection import Detection,Box,CheckResult,CheckStatus,ModelRef,Mode,InspectionPipeline
from mes_vision.inspection.contracts import DetectionBatch
from mes_vision.decision import DecisionPolicy,CheckRule
from mes_vision.anomaly.scoring import Criteria
from mes_vision.anomaly.features import fingerprint
from mes_vision.operation.catalog import OperationStore,new_product,default_equipment,readiness
from mes_vision.operation.tracking import Tracker
from mes_vision.operation.engine import LiveEngine,vlm_eligible
from mes_vision.operation.quality import in_workspace,quality,GeometryInspector
from mes_vision.robot.magician import SerialProtocol,Magician,packet
from mes_vision.robot.contracts import Pose


def detection(x=20,y=30,w=110,h=120): return Detection(Box(x,y,x+w,y+h),.99,0,"part")
def frame(sequence=0,rgb=None,session="injected-camera"):
    if rgb is None: rgb=np.full((180,480,3),125,np.uint8)
    return Frame(f"{session}:{sequence}",session,sequence,SourceKind.D405,"realsense://TEST",datetime.now(timezone.utc),rgb,is_live=True)


class InjectedModels:
    """Exercises the live contracts without loading or representing a trained model."""
    def __init__(self):
        self.loads=0; self.detections=(detection(),); self.closed=False; self.inspectors=[]; self.calls=0
        self.model=ModelRef("injected object output","1","model","d"*64,"product_objects"); self.detector=self
        self.criteria=Criteria("test-criteria","b"*64,"test-part",.2,.8,.5,True,"UNIT-TEST-ONLY","real")
        self.anomaly_score=.1; self.defect=False
        rules=[]
        for key,sha,scope in (("known_defects","a","product_defects"),("anomaly","b","product_normal_reference")):
            model=ModelRef("injected "+key,"1","model",sha*64,scope)
            owner=self
            class Inspector:
                def inspect(self,crop):
                    owner.calls+=1
                    if self.check_id=="known_defects":
                        from mes_vision.inspection import Finding
                        return CheckResult(self.check_id,crop.frame_id,crop.object_id,CheckStatus.UNCERTAIN,self.model,
                            (Finding("균열","NG03",.95),) if owner.defect else (),messages=("CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED",),candidate_threshold=.4,details={"class_codes":{"0":"NG03"}})
                    return CheckResult(self.check_id,crop.frame_id,crop.object_id,CheckStatus.PASS if owner.anomaly_score<=.2 else CheckStatus.UNCERTAIN,self.model,raw_score=owner.anomaly_score,
                        criteria_version=owner.criteria.version,details={"criteria":asdict(owner.criteria),"product_id":"test-part","kind":"real"})
            inspector=Inspector(); inspector.check_id=key; inspector.model=model; self.inspectors.append(inspector)
            if key=="known_defects": rules.append(CheckRule(key,True,"defect_candidates",model=model,criteria_version="test-defects",validated=True,
                validation_reference="UNIT-TEST-ONLY",candidate_threshold=.4,fail_thresholds={"NG03":.8},class_codes={"0":"NG03"}))
            else: rules.append(CheckRule(key,True,"anomaly_distance",model=model,criteria_version=self.criteria.version,validated=True,
                validation_reference="UNIT-TEST-ONLY",criteria_digest=fingerprint(asdict(self.criteria))))
        rules.append(CheckRule("geometry",False,optional_reason="UNIT-TEST-ONLY"))
        self.policy=DecisionPolicy("test-policy","test-part",tuple(rules),"real",True,"UNIT-TEST-ONLY",self.model,"test-quality",False)
    def detect(self,f): return DetectionBatch(f.frame_id,(f.width,f.height),self.detections,self.model,candidate_threshold=.4)
    def load(self): self.loads+=1
    def close(self): self.closed=True


class TrackingTests(unittest.TestCase):
    def setUp(self): self.t=Tracker(session="test",stable_seconds=.2,max_gap=1); self.rgb=frame().rgb
    def stable(self):
        for now in (1.,1.1,1.3): tracks=self.t.update((detection(),),self.rgb,now)
        return tracks[0]
    def test_stable_capture_once_and_id_preserved(self):
        t=self.stable(); self.assertEqual(t.status,"STABLE"); self.assertTrue(self.t.accept(t.id,t.revision,{"decision":"OK"}))
        after=self.t.update((detection(),),self.rgb,1.4)[0]; self.assertEqual(after.id,t.id); self.assertEqual(after.status,"OK")
    def test_motion_invalidates_and_rejects_late_result(self):
        t=self.stable(); revision=t.revision; self.t.accept(t.id,revision,{"decision":"OK"})
        self.t.update((detection(x=27),),self.rgb,1.4)
        self.assertIsNone(t.result); self.assertFalse(self.t.accept(t.id,revision,{"decision":"OK"}))
    def test_cumulative_slow_movement_is_not_hidden(self):
        t=self.stable(); revision=t.revision
        for i in range(1,7): self.t.update((detection(x=20+i),),self.rgb,1.3+i*.05)
        self.assertGreater(t.revision,revision)
    def test_missing_return_requires_new_inspection_and_removed_id_never_reused(self):
        t=self.stable(); self.t.accept(t.id,t.revision,{"decision":"OK"}); self.t.update((),self.rgb,1.4)
        self.assertIsNone(t.result); self.t.update((detection(),),self.rgb,1.5); self.assertNotEqual(t.status,"OK")
        self.t.update((),self.rgb,3); new=self.t.update((detection(),),self.rgb,3.1)[0]; self.assertNotEqual(t.id,new.id)
    def test_ambiguous_and_overlapping_objects_never_keep_verdict(self):
        t=self.stable(); self.t.accept(t.id,t.revision,{"decision":"OK"})
        self.t.update((detection(),detection(x=40)),self.rgb,1.4); self.assertIsNone(t.result)
        self.assertFalse(any(x.status=="OK" for x in self.t.tracks.values()))
    def test_appearance_and_frame_gap_force_recheck(self):
        t=self.stable(); revision=t.revision
        self.t.update((detection(),),np.zeros_like(self.rgb),1.4); self.assertGreater(t.revision,revision)
        revision=t.revision; self.t.update((detection(),),np.zeros_like(self.rgb),4); self.assertGreater(t.revision,revision)
    def test_reverse_time_rejected(self):
        self.stable()
        with self.assertRaises(ValueError): self.t.update((),self.rgb,1)


class OperationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.store=OperationStore(self.root)
        self.product=new_product(); self.product.update(id="test-part",workspace_id="main",version=1)
        self.product["quality"]={"version":"test-quality","validation_reference":"UNIT-TEST-ONLY","blur_min":0,"brightness_min":0,"brightness_max":255}
        self.equipment=default_equipment(); self.equipment["workspace"].update(roi=[[1,1],[479,1],[479,179],[1,179]],width_mm=100,height_mm=50,validation_reference="UNIT-TEST-ONLY")
        self.models=InjectedModels(); self.session=self.store.start_session(self.product,self.equipment)
        self.engine=LiveEngine(self.root,self.root,self.product,self.equipment,self.session,models=self.models)
        self.engine.concurrent=True; self.engine.set_enabled(True,1); self.sequence=0; self.base=time.monotonic()
    def tearDown(self): self.engine.close(); self.temp.cleanup()
    def stable(self,start=None):
        base=start if start is not None else self.base
        for offset in (0,.35,.7):
            f=frame(self.sequence); self.sequence+=1; batch,tracks=self.engine.detect(f,base+offset)
        return f,batch,tracks
    def inspect(self,start=None):
        f,b,t=self.stable(start); pending=self.engine.inspect(f,b,t); self.assertIsNotNone(pending); return pending
    def test_catalog_versions_optimistic_conflict_and_inactive(self):
        p=new_product(); first=self.store.save_product(p); second=self.store.save_product(first)
        self.assertEqual(second["version"],2)
        with self.assertRaises(ValueError): self.store.save_product(first)
        self.assertEqual(len(self.store.products()),1)
    def test_unconfigured_readiness_and_bad_files_fail(self):
        self.assertGreater(len(readiness(new_product(),default_equipment())),3)
        self.product["objects"]={"weights":str(self.root/"missing"),"sha256":"0"*64}
        self.assertTrue(readiness(self.product,self.equipment))
    def test_pause_discards_inflight_and_keeps_tracking(self):
        pending=self.inspect(); self.engine.set_enabled(False,2); self.assertIsNone(self.engine.commit(pending)); self.assertEqual(self.store.history(),[])
        f,b,t=self.stable(self.base+1); self.assertTrue(t); self.assertIsNone(self.engine.inspect(f,b,t))
    def test_pause_resume_preserves_confirmed_stationary_object_without_duplicate(self):
        self.engine.commit(self.inspect()); count=self.models.calls; identity=next(iter(self.engine.tracker.tracks))
        self.engine.set_enabled(False,2); self.engine.set_enabled(True,3)
        f,b,t=self.stable(self.base+1)
        self.assertEqual(t[0].id,identity); self.assertEqual(t[0].status,"OK"); self.assertIsNone(self.engine.inspect(f,b,t)); self.assertEqual(self.models.calls,count)
    def test_commit_snapshot_history_idempotence_and_recheck_chain(self):
        pending=self.inspect(); committed=self.engine.commit(pending); self.assertIsNotNone(committed)
        self.assertEqual(self.store.history()[0]["decision"],"OK")
        path=committed["path"]; self.store.register(path,self.session,committed["links"])
        self.assertEqual(len(self.store.history()),1)
        f,b,t=self.stable(self.base+1); self.assertIsNone(self.engine.inspect(f,b,t)); calls=self.models.calls
        self.engine.tracker.recheck(t[0].id,self.base+1.8)
        second=self.engine.commit(self.inspect(self.base+1.8)); self.assertIsNotNone(second)
        records=self.store.history(); self.assertEqual(records[0]["prior_object_id"],records[1]["object_id"])
        self.assertEqual(self.store.counts(self.session),{"OK":1,"NG":0,"REVIEW":0,"reinspections":1}); self.assertGreater(self.models.calls,calls)
    def test_selective_inspection_only_new_object(self):
        self.engine.commit(self.inspect()); calls=self.models.calls
        self.models.detections=(detection(),detection(x=180)); pending=self.inspect(self.base+1)
        self.assertEqual(self.models.calls-calls,2); committed=self.engine.commit(pending); self.assertEqual(len(committed["links"]),1)
        self.assertEqual(self.store.counts(self.session)["OK"],2)
    def test_movement_after_inference_discards_old_revision(self):
        pending=self.inspect(); self.models.detections=(detection(x=30),)
        self.engine.detect(frame(self.sequence),self.base+.8); self.assertIsNone(self.engine.commit(pending))
    def test_storage_does_not_block_tracking_or_publish_after_movement(self):
        import threading
        from mes_vision.operation import engine as module
        release=threading.Event(); original=module.save_snapshot; pending=self.inspect()
        def delayed(*args,**kwargs):
            if not release.wait(3): raise TimeoutError("test writer release missing")
            return original(*args,**kwargs)
        try:
            with patch.object(module,"save_snapshot",side_effect=delayed):
                self.assertTrue(self.engine.begin_save(pending)); self.assertFalse(self.engine.begin_save(pending))
                self.models.detections=(detection(x=30),); self.engine.detect(frame(self.sequence),self.base+.8)
                self.assertIsNone(self.engine.finish_save()); self.assertFalse(self.store.history())
                release.set(); self.engine.saving[2].result(timeout=3)
                self.assertIsNone(self.engine.finish_save()); self.assertFalse(self.store.history())
        finally: release.set()
    def test_different_camera_session_drops_old_result(self):
        pending=self.inspect(); self.engine.detect(frame(0,session="new-camera"),self.base+.8); self.assertIsNone(self.engine.commit(pending))
    def test_quality_and_roi_fail_do_not_publish_ng(self):
        self.product["quality"]["blur_min"]=999999; f,b,t=self.stable(); self.assertIsNone(self.engine.inspect(f,b,t)); self.assertEqual(t[0].status,"IMAGE_REVIEW")
        self.product["quality"]["blur_min"]=0; self.equipment["workspace"]["excluded"]=[[[10,10],[140,10],[140,160],[10,160]]]
        f,b,t=self.stable(self.base+1); self.assertIsNone(self.engine.inspect(f,b,t)); self.assertFalse(self.store.history())
    def test_snapshot_tamper_detected_and_review_does_not_change_decision(self):
        result=self.engine.commit(self.inspect()); identity=result["links"][0]["object_id"]
        self.store.review(identity,"tester","software contract only"); self.assertEqual(self.store.open_record(identity)[3]["final_decision"],"OK")
        (Path(result["path"])/"frame.png").write_bytes(b"changed")
        with self.assertRaises(ValueError): self.store.open_record(identity)
    def test_restart_marks_session_interrupted_without_robot_requests(self):
        self.assertEqual(self.store.recover_sessions(),[self.session]); self.assertEqual(self.store.recover_sessions(),[])
        with self.store.connect() as db: self.assertEqual(db.execute("SELECT count(*) FROM events WHERE type LIKE 'ROBOT_%'").fetchone()[0],0)
    def test_known_ng_has_reason_before_vlm_and_never_enqueues_when_off(self):
        self.models.defect=True; result=self.engine.commit(self.inspect()); obj=self.store.open_record(result["links"][0]["object_id"])[3]
        self.assertEqual(obj["final_decision"],"NG"); self.assertIn("NG03",obj["decision_details"]["defect_codes"]); self.assertEqual(self.engine.queue.list(),[])
    def test_visual_review_eligible_but_config_error_not(self):
        self.models.anomaly_score=.5; pending=self.inspect(); obj=pending["result"].objects[0]
        self.assertEqual(obj.final_decision,"REVIEW"); self.assertTrue(vlm_eligible(obj,"review_visual"))
        obj.decision_details["reasons"].append({"code":"CAMERA_ERROR"}); self.assertFalse(vlm_eligible(obj,"review_visual"))
        self.assertFalse(vlm_eligible(obj,"manual"))
    def test_file_live_mode_mismatch(self):
        with self.assertRaises(ValueError): InspectionPipeline(self.models,mode=Mode.LIVE).run(replace(frame(),is_live=False))
        with self.assertRaises(ValueError): InspectionPipeline(self.models,mode=Mode.MODEL_FILE).run(frame())
    def test_loaded_models_are_not_reloaded_per_frame(self):
        with patch("torch.cuda.is_available",return_value=False): self.engine.prepare()
        self.stable(); self.stable(self.base+1); self.assertEqual(self.models.loads,1)
    def test_real_robot_plan_binds_calibration_profile_track_and_freshness(self):
        from mes_vision.calibration.fixtures import make_spec
        from mes_vision.calibration import fit
        from mes_vision.robot import RobotProfile,Workspace
        from mes_vision.operation.robot_service import real_plan
        from mes_vision.training.data import write_json,sha256
        spec=replace(make_spec(),kind="real",product_id=self.product["id"])
        calibration=fit(spec).accept("UNIT-TEST-ONLY"); cp=self.root/"calibration.json"; calibration.save(cp)
        c=spec.context; self.equipment["camera"].update(serial=c.camera_id,mount_revision=c.mount_revision,acquisition_revision=c.acquisition_revision,
            robot_base_id=c.robot_base_id,tool_frame_id=c.tool_frame_id,width=480,height=180)
        self.product["grasp"].update(version="test-grasp",validation_reference="UNIT-TEST-ONLY",plane_z_mm=20.)
        profile=RobotProfile("test-profile",self.product["id"],"real",True,"UNIT-TEST-ONLY",calibration.identity,"test-grasp",
            Workspace(-300,300,-300,300,0,150,-180,180),20.,100.,{"OK":Pose(200,100,20,0),"NG":Pose(200,-100,20,0)},"suction","digital_input",2.,10.,False)
        rp=self.root/"robot.json"; write_json(rp,asdict(profile)); self.equipment["calibration"]=str(cp); self.equipment["robot"]["profile"]=str(rp)
        self.engine.hardware_digests={"calibration":sha256(cp),"robot_profile":sha256(rp)}
        pending=self.inspect(); self.engine.commit(pending); track=next(iter(self.engine.tracker.tracks.values())).data()
        result=self.store.open_record(track["result"]["object_id"])[2]
        plan,scene=real_plan(result,track,self.product,self.equipment,"epoch",self.base+.8)
        self.assertEqual(plan.profile.kind,"real"); self.assertEqual(len(plan.commands),10)
        with self.assertRaises(ValueError): real_plan(result,track,self.product,self.equipment,"epoch",self.base+9)
        wrong=deepcopy(track); wrong["revision"]+=1
        with self.assertRaises(ValueError): real_plan(result,wrong,self.product,self.equipment,"epoch",self.base+.8)
        changed=asdict(profile); changed["version"]="other"; write_json(rp,changed)
        with self.assertRaises(ValueError): real_plan(result,track,self.product,self.equipment,"epoch",self.base+.8)
    def test_spawned_engine_preload_tracking_pause_and_shutdown(self):
        from mes_vision.operation.engine import EngineProcess
        process=EngineProcess(self.root,self.root,self.product,self.equipment,self.session,models_factory=InjectedModels)
        try:
            started=time.monotonic(); ready=False; messages=[]
            while time.monotonic()-started<30 and not ready:
                messages+=process.poll(); ready=any(m["type"]=="ready" for m in messages)
                if any(m["type"]=="error" for m in messages): self.fail(str(messages))
                time.sleep(.05)
            self.assertTrue(ready,str(messages)); process.command("enable",value=True,generation=7)
            started=time.monotonic(); seq=0; found=False
            while time.monotonic()-started<10 and not found:
                process.frame(frame(seq),time.monotonic()); seq+=1; time.sleep(.12)
                messages=process.poll(); self.assertFalse(any(m["type"]=="error" for m in messages),str(messages))
                found=any(m["type"]=="inspection" for m in messages)
            self.assertTrue(found); count=len(self.store.history()); process.command("enable",value=False,generation=8)
            paused_states=[]
            for i in range(8):
                process.frame(frame(seq),time.monotonic()); seq+=1; time.sleep(.12); paused_states+=process.poll()
            self.assertTrue(any(m["type"]=="tracks" and m["generation"]==8 and m["tracks"] for m in paused_states))
            self.assertEqual(len(self.store.history()),count); process.stop(); process.process.join(timeout=5); self.assertFalse(process.process.is_alive())
        finally: process.dispose()


class SerialTransport:
    def __init__(self,response): self.response=bytearray(response); self.writes=[]; self.closed=False
    def reset_input_buffer(self): pass
    def write(self,data): self.writes.append(data); return len(data)
    def read(self,count): data=self.response[:1]; del self.response[:1]; return data
    def close(self): self.closed=True


class ProtocolTests(unittest.TestCase):
    def test_fragmented_packet_noise_and_checksum(self):
        payload=struct.pack("<8f",*range(8)); transport=SerialTransport(b"noise"+packet(10,parameters=payload))
        self.assertEqual(SerialProtocol("",transport=transport).request(10),payload)
        self.assertEqual(transport.writes,[bytes.fromhex("aaaa020a00f6")])
    def test_timeout_never_retries_ambiguous_motion(self):
        t=SerialTransport(b""); p=SerialProtocol("",transport=t,timeout=.001)
        with self.assertRaises(TimeoutError): p.request(84,True,True,b"a")
        with self.assertRaises(ValueError): p.request(84,True,True,b"a")
        self.assertEqual(len(t.writes),1)
    def test_corrupt_or_wrong_response_taints_connection(self):
        for response in (packet(11),packet(10)[:-1]+b"\x00"):
            p=SerialProtocol("",transport=SerialTransport(response))
            with self.assertRaises(ValueError): p.request(10)
            self.assertTrue(p.tainted)
    def test_di_request_carries_address(self):
        m=Magician.__new__(Magician); m.settings={"input_address":7,"holding_level":1}
        t=SerialTransport(packet(133,parameters=b"\x07\x01")); m.protocol=SerialProtocol("",transport=t)
        self.assertTrue(m.holding()); self.assertEqual(t.writes[0],packet(133,parameters=b"\x07"))
    def test_queue_index_requires_measured_pose(self):
        m=Magician.__new__(Magician); target=Pose(1,2,3,4); command=SimpleNamespace(command_id="c",target=target,action="move")
        m.pending=(command,10); m.pose_tolerance=.1; m.rotation_tolerance=.1
        m.protocol=SimpleNamespace(request=lambda *a:struct.pack("<Q",10)); m.read_pose=lambda:Pose(1,2,9,4)
        self.assertEqual(m.poll("c").state,"PENDING"); m.read_pose=lambda:target; self.assertEqual(m.poll("c").state,"DONE")
    def test_pick_and_place_need_independent_sensor(self):
        for action,level in (("verify_pick",True),("verify_place",False)):
            m=Magician.__new__(Magician); m.pending=(SimpleNamespace(command_id="c",action=action),None); m.holding=lambda:not level
            self.assertEqual(m.poll("c").state,"PENDING"); m.holding=lambda:level; self.assertTrue(m.poll("c").verified)
    def test_stop_does_not_release_tool(self):
        calls=[]; m=Magician.__new__(Magician); m.protocol=SimpleNamespace(request=lambda *a,**k:calls.append((a,k)))
        self.assertTrue(m.request_stop()); self.assertEqual([a[0] for a,k in calls],[242]); self.assertEqual(m.motion_state,"STOPPED")


class CameraContractTests(unittest.TestCase):
    def test_unconfigured_never_connects(self):
        with D405Source() as source: self.assertEqual(source.read().status,InputStatus.NOT_CONFIGURED)
    def test_configured_source_emits_live_frame_and_closes(self):
        device=SimpleNamespace(open=lambda:None,read=lambda:frame(),close=lambda:None)
        with D405Source({"serial":"test"},factory=lambda settings:device) as source:
            event=source.read(); self.assertEqual(event.status,InputStatus.FRAME); self.assertTrue(event.frame.is_live)
        self.assertEqual(source.status,InputStatus.CLOSED)
    def test_camera_disconnect_never_returns_previous_frame(self):
        device=SimpleNamespace(open=lambda:None,read=lambda:frame(),close=lambda:None)
        with D405Source({"serial":"test"},factory=lambda settings:device) as source:
            self.assertEqual(source.read().status,InputStatus.FRAME)
            def failed(): raise RuntimeError("disconnected")
            device.read=failed; event=source.read(); self.assertEqual(event.status,InputStatus.ERROR); self.assertIsNone(event.frame)


if __name__=="__main__": unittest.main()

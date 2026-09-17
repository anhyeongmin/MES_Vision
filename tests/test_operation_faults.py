"""Injected faults and real child-process termination, without physical equipment."""
import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
from pathlib import Path
from types import SimpleNamespace
import json
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from PySide6.QtWidgets import QApplication,QInputDialog
from mes_vision.operation.faults import FaultJournal,fault_code
from mes_vision.operation.camera import D405Camera,CameraProcess
from mes_vision.operation.engine import EngineProcess
from mes_vision.operation.robot_service import RobotThread
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.catalog import OperationStore,default_equipment
import test_operation

ROOT=Path(__file__).resolve().parents[1]
APP=QApplication.instance() or QApplication([])


def until(condition,timeout=12):
    end=time.monotonic()+timeout
    while time.monotonic()<end:
        APP.processEvents()
        if condition(): return
        time.sleep(.01)
    raise AssertionError("Operation did not complete before test deadline")


class HealthyCamera:
    def __init__(self,settings): self.settings=settings; self.info={"serial":"injected-camera"}; self.seq=0
    def open(self): pass
    def read(self):
        time.sleep(.02); value=test_operation.frame(self.seq); self.seq+=1; return value
    def close(self): pass


class BlockedCamera(HealthyCamera):
    def read(self): time.sleep(60)


class BlockedOpenCamera(HealthyCamera):
    def open(self): Path(self.settings["marker"]).write_text("opening"); time.sleep(60)


class CrashedCamera(HealthyCamera):
    def read(self): os._exit(42)


class MemoryFailureModels(test_operation.InjectedModels):
    def load(self): raise MemoryError("injected allocation failure")


class BlockedModels(test_operation.InjectedModels):
    def detect(self,f): time.sleep(60)


class FaultJournalTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.j=FaultJournal(self.root)
    def tearDown(self): self.temp.cleanup()
    def test_fault_survives_restart_and_deduplicates(self):
        first,new=self.j.raise_fault("CAMERA_FAILURE","disconnected"); self.assertTrue(new)
        again,new=self.j.raise_fault("CAMERA_FAILURE","still disconnected"); self.assertFalse(new); self.assertEqual(first["id"],again["id"])
        self.assertEqual(len(FaultJournal(self.root).active),1)
    def test_clear_requires_reference_and_survives_restart(self):
        self.j.raise_fault("X","failure")
        with self.assertRaises(ValueError): self.j.acknowledge("","")
        self.j.acknowledge("operator","checked hardware offline")
        restored=FaultJournal(self.root); self.assertFalse(restored.active); self.assertEqual(restored.entries[0]["operator"],"operator")
    def test_disk_failure_latches_in_memory_and_failed_ack_never_clears(self):
        with patch("mes_vision.operation.faults.os.replace",side_effect=PermissionError("injected permission")):
            self.j.raise_fault("STORAGE_FAILURE","not writable"); self.assertTrue(self.j.active); self.assertTrue(self.j.persistence_error)
            with self.assertRaises(PermissionError): self.j.acknowledge("operator","failed recovery")
        self.assertTrue(self.j.active); self.j.acknowledge("operator","disk restored"); self.assertFalse(self.j.active)
    def test_corrupted_journal_preserved_before_explicit_recovery(self):
        self.j.path.write_bytes(b"truncated json"); j=FaultJournal(self.root); self.assertTrue(j.corrupt); self.assertTrue(j.active)
        j.raise_fault("X","second fault"); self.assertEqual(j.path.read_bytes(),b"truncated json")
        j.acknowledge("operator","diagnostic copy checked")
        self.assertFalse(FaultJournal(self.root).active); self.assertEqual(next(self.root.glob("operator-faults-damaged-*.json")).read_bytes(),b"truncated json")
    def test_oversized_history_does_not_discard_active_fault(self):
        for i in range(1002):
            entry=self.j.entry(str(i),"history"); entry["cleared"]=time.time(); self.j.entries.append(entry)
        self.j.raise_fault("ACTIVE","keep this"); restored=FaultJournal(self.root)
        self.assertEqual(len(restored.entries),1000); self.assertEqual(restored.active[0]["code"],"ACTIVE")
    def test_exception_classification(self):
        self.assertEqual(fault_code("MemoryError: allocation"),"MODEL_MEMORY")
        self.assertEqual(fault_code("CUDA out of memory"),"MODEL_MEMORY")
        self.assertEqual(fault_code("database is locked"),"STORAGE_FAILURE")
    def test_fault_file_loss_recovers_from_unacknowledged_database_event(self):
        store=OperationStore(self.root); entry,_=self.j.raise_fault("STORAGE_FAILURE","write failed"); store.event("OPERATOR_FAULT",entry)
        self.j.path.unlink(); restored=FaultJournal(self.root); restored.recover_from_store(store)
        self.assertEqual(restored.active[0]["id"],entry["id"])
        restored.acknowledge("operator","checked"); store.event("OPERATOR_FAULT_ACK",{"ids":[entry["id"]]})
        again=FaultJournal(self.root); again.recover_from_store(store); self.assertFalse(again.active)
    def test_missing_database_ack_keeps_fault_active_after_restart(self):
        store=OperationStore(self.root); entry,_=self.j.raise_fault("CAMERA_FAILURE","unplugged"); store.event("OPERATOR_FAULT",entry)
        self.j.acknowledge("operator","file saved but database acknowledgement failed")
        restored=FaultJournal(self.root); restored.recover_from_store(store); self.assertTrue(restored.active)


class CameraFaultTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.camera=None
    def tearDown(self):
        if self.camera:
            self.camera.stopping.set(); until(lambda:self.camera.disposed)
        self.temp.cleanup()
    def start(self,factory,**kwargs):
        self.camera=CameraProcess({"marker":str(self.root/"marker")},factory=factory,stop_timeout=.2,**kwargs)
        self.errors=[]; self.camera.failed.connect(self.errors.append); self.camera.start(); return self.camera
    def test_real_child_frames_and_graceful_close_clear_latest(self):
        c=self.start(HealthyCamera); until(lambda:c.latest is not None); self.assertTrue(c.latest.is_live)
        c.stopping.set(); until(lambda:c.disposed); self.assertEqual(c.get_latest(),(None,0.)); self.assertFalse(self.errors)
    def test_blocked_sdk_read_is_terminated_without_hanging_ui(self):
        c=self.start(BlockedCamera,frame_timeout=.3); until(lambda:c.disposed)
        self.assertTrue(self.errors); self.assertFalse(c.process.is_alive()); self.assertIsNone(c.latest)
    def test_blocked_sdk_open_has_bounded_cancellation(self):
        c=self.start(BlockedOpenCamera,connect_timeout=10); until(lambda:(self.root/"marker").exists())
        c.started=time.monotonic()-11; until(lambda:c.disposed)
        self.assertIn("연결 시간이 초과",self.errors[0]); self.assertFalse(c.process.is_alive())
    def test_camera_process_crash_never_leaves_last_frame(self):
        c=self.start(CrashedCamera); until(lambda:c.disposed); self.assertTrue(self.errors); self.assertIsNone(c.latest)
    def device(self,numbers,timestamps):
        self.number=iter(numbers); self.timestamp=iter(timestamps)
        color=SimpleNamespace(get_frame_number=lambda:next(self.number),get_timestamp=lambda:next(self.timestamp),get_data=lambda:np.zeros((180,480,3),np.uint8))
        camera=D405Camera({"width":480,"height":180,"serial":"unit"},sdk=object())
        camera.pipeline=SimpleNamespace(try_wait_for_frames=lambda _: (True,SimpleNamespace(get_color_frame=lambda:color)))
        return camera
    def test_duplicate_hardware_frame_not_counted_as_new(self):
        camera=self.device([1,1,2],[1.,1.,2.]); self.assertIsNotNone(camera.read()); self.assertIsNone(camera.read()); self.assertEqual(camera.read().sequence,1)
    def test_counter_reset_and_invalid_timestamps_rejected(self):
        for numbers,timestamps in (([2,1],[2.,3.]),([1,2],[2.,1.]),([1,2],[2.,2.]),([1],[float('nan')])):
            camera=self.device(numbers,timestamps)
            if len(numbers)>1: camera.read()
            with self.assertRaises(RuntimeError): camera.read()


class EngineFaultTests(unittest.TestCase):
    def setUp(self): self.f=test_operation.OperationTests(); self.f.setUp(); self.process=None
    def tearDown(self):
        if self.process: self.process.dispose()
        self.f.tearDown()
    def spawn(self,factory):
        f=self.f; self.process=EngineProcess(f.root,f.root,f.product,f.equipment,f.session,models_factory=factory); self.messages=[]
        return self.process
    def events(self): self.messages+=self.process.poll(); return self.messages
    def test_memory_failure_emits_classified_error_and_no_record(self):
        p=self.spawn(MemoryFailureModels); until(lambda:any(e["type"]=="error" for e in self.events()),30)
        event=next(e for e in self.messages if e["type"]=="error"); self.assertEqual(event["code"],"MODEL_MEMORY"); self.assertFalse(self.f.store.history())
    def test_blocked_inference_watchdog_and_process_termination(self):
        p=self.spawn(BlockedModels); until(lambda:any(e["type"]=="ready" for e in self.events()),30)
        p.frame(test_operation.frame(),time.monotonic()); until(lambda:p.phase.value==3)
        self.assertIn("물체 검출",p.watchdog_error(p.progress.value+31)); p.stop(); p.dispose(); self.process=None
        self.assertFalse(p.process.is_alive()); self.assertFalse(self.f.store.history())
    def test_idle_model_heartbeat_and_prepare_deadline(self):
        p=self.spawn(test_operation.InjectedModels); until(lambda:any(e["type"]=="ready" for e in self.events()),30)
        before=p.progress.value; until(lambda:p.phase.value==2 and p.progress.value>before)
        self.assertIsNone(p.watchdog_error()); p.phase.value=1
        self.assertIsNone(p.watchdog_error(p.progress.value+179)); self.assertIsNotNone(p.watchdog_error(p.progress.value+181))
    def test_async_writer_timeout_never_publishes_result(self):
        release=threading.Event(); original=self.f.engine.write_snapshot
        def blocked(*args,**kwargs): release.wait(3); return original(*args,**kwargs)
        try:
            with patch.object(self.f.engine,"write_snapshot",blocked):
                self.f.engine.begin_save(self.f.inspect()); self.f.engine.save_started=time.monotonic()-31
                with self.assertRaises(TimeoutError): self.f.engine.finish_save()
                self.assertFalse(self.f.store.history()); self.f.engine.set_enabled(False,99)
                release.set(); self.f.engine.saving[2].result(timeout=3)
                self.assertIsNone(self.f.engine.finish_save()); self.assertFalse(self.f.store.history())
        finally: release.set()
    def test_database_register_failure_never_creates_robot_eligible_track(self):
        pending=self.f.inspect()
        with patch.object(self.f.engine.store,"register",side_effect=OSError("database unavailable")):
            with self.assertRaises(OSError): self.f.engine.commit(pending)
        self.assertFalse(self.f.store.history()); self.assertFalse(any(t.result for t in self.f.engine.tracker.tracks.values()))
    def test_optional_analysis_database_failure_preserves_primary_result(self):
        pending=self.f.inspect()
        with patch.object(self.f.engine.queue,"enabled",side_effect=OSError("VLM database inaccessible")):
            result=self.f.engine.commit(pending)
        self.assertIsNotNone(result); self.assertEqual(self.f.store.history()[0]["decision"],"OK")


class RecoveryUiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.w=DesktopWindow(ROOT,self.root,start_worker=False)
        self.w.timer.stop(); self.w.analysis.timer.stop(); self.w.show(); APP.processEvents()
    def tearDown(self):
        end=time.monotonic()+5
        while (self.w.vlm_read.busy or self.w.analysis.refresh_read.busy or self.w.readiness_read.busy) and time.monotonic()<end:
            APP.processEvents(); time.sleep(.005)
        self.w.engine=None; self.w.camera=None; self.w.robot=None; self.w.closing=True; self.w.close(); APP.processEvents(); self.temp.cleanup()
    def fake_running(self):
        self.stops=[]; w=self.w; w.running=True; w.engine_ready=True
        w.engine=SimpleNamespace(stop=lambda:self.stops.append("engine"),command=lambda *a,**k:None,stopping=False)
        w.robot=SimpleNamespace(stop_motion=lambda:self.stops.append("robot"))
        w.tracks=[{"track_id":"old","status":"OK","result":{"object_id":"old"},"revision":1,"missing":False,"box":[1,1,2,2]}]
    def test_camera_failure_invalidates_verdict_stops_robot_and_latches(self):
        self.fake_running(); generation=self.w.generation; self.w.camera_failed("unplugged")
        self.assertFalse(self.w.running); self.assertIsNone(self.w.tracks[0]["result"]); self.assertEqual(set(self.stops),{"engine","robot"})
        self.assertGreater(self.w.generation,generation); self.assertTrue(FaultJournal(self.root).active)
        with self.assertRaises(ValueError): self.w.begin()
    def test_fault_during_database_failure_still_stops_motion(self):
        self.fake_running()
        with patch.object(self.w.store,"event",side_effect=OSError("disk full")):
            self.w.trip_fault("STORAGE_FAILURE","disk full")
        self.assertIn("robot",self.stops); self.assertFalse(self.w.running); self.assertTrue(self.w.faults.active)
    def test_failed_operational_audit_write_while_running_latches_stop(self):
        self.fake_running()
        with patch.object(self.w.store,"event",side_effect=OSError("database full")): self.w.notice("start requested")
        self.assertFalse(self.w.running); self.assertIn("robot",self.stops); self.assertTrue(self.w.faults.active)
    def test_disconnect_mid_pick_requires_recovery(self):
        self.fake_running(); self.w.robot.stopping=threading.Event(); self.w.robot_pending=True
        self.w.disconnect_robot(); self.assertFalse(self.w.running); self.assertTrue(self.w.robot.stopping.is_set()); self.assertTrue(self.w.faults.active)
    def test_missing_robot_status_is_detected_by_ui_even_if_worker_stalls(self):
        self.stops=[]; self.w.robot=SimpleNamespace(stop_motion=lambda:self.stops.append("stop")); self.w.running=True
        self.w.robot_seen_at=time.monotonic()-6; self.w.safe_tick()
        self.assertFalse(self.w.running); self.assertEqual(self.stops,["stop"])
        self.assertTrue(any(e['code']=='ROBOT_STATUS_TIMEOUT' for e in self.w.faults.active))
    def test_late_ready_and_tracks_cannot_rearm_faulted_ui(self):
        self.fake_running(); old=self.w.generation; self.w.trip_fault("ENGINE_FAILURE","failure")
        self.w.handle_engine({"type":"ready","concurrent_vlm":False})
        self.w.handle_engine({"type":"tracks","generation":old,"tracks":[{"bad":"old"}]})
        self.assertFalse(self.w.engine_ready); self.assertFalse(self.w.running)
    def test_operator_ack_requires_disconnected_devices_and_no_auto_restart(self):
        self.w.trip_fault("CAMERA_FAILURE","unplugged"); self.w.camera=object()
        with self.assertRaises(ValueError): self.w.acknowledge_faults()
        self.w.camera=None
        with patch.object(QInputDialog,"getText",return_value=("operator",True)),patch.object(QInputDialog,"getMultiLineText",return_value=("cable checked",True)):
            self.w.acknowledge_faults()
        self.assertFalse(self.w.faults.active); self.assertFalse(self.w.running); self.assertIsNone(self.w.engine)
    def test_optional_vlm_worker_failure_leaves_primary_inspection_running(self):
        self.w.running=True; self.w.queue.set_enabled(True); self.w.vlm_worker_failed("injected model memory failure")
        self.assertTrue(self.w.running); self.assertFalse(self.w.queue.enabled()); self.assertFalse(self.w.faults.active)
    def test_optional_vlm_database_read_failure_does_not_stop_primary(self):
        self.w.running=True
        with patch.object(self.w.queue,"enabled",side_effect=OSError("VLM database inaccessible")):
            self.w.safe_tick(); self.w.safe_tick()
            end=time.monotonic()+5
            while self.w.vlm_read.busy and time.monotonic()<end: APP.processEvents(); time.sleep(.005)
        self.assertTrue(self.w.running); self.assertFalse(self.w.faults.active); self.assertIn("기록 오류",self.w.vlm.text())
    def test_control_queue_failure_is_latched(self):
        self.fake_running()
        def full(*a,**k): raise queue.Full()
        self.w.engine.command=full; self.w.control_enabled(True)
        self.assertTrue(self.w.faults.active); self.assertFalse(self.w.running)
    def test_stop_failure_cannot_prevent_generation_invalidation(self):
        self.fake_running()
        def failed(): raise RuntimeError("broken IPC")
        self.w.engine.stop=failed; self.w.robot.stop_motion=failed; old=self.w.generation
        self.w.trip_fault("ENGINE_FAILURE","broken IPC")
        self.assertGreater(self.w.generation,old); self.assertFalse(self.w.running); self.assertTrue(self.w.faults.active)
    def test_cancelled_robot_queue_clears_ui_pending_without_restarting(self):
        self.w.robot_pending=True; self.w.pick_block={"box":[]}; self.w.running=False
        self.w.robot_changed({"state":"STOPPED","busy":False})
        self.assertFalse(self.w.robot_pending); self.assertIsNone(self.w.pick_block); self.assertFalse(self.w.running)
    def test_storage_error_during_ack_cannot_clear_interlock(self):
        self.w.trip_fault("CAMERA_FAILURE","unplugged"); original=self.w.store.event
        def failed(name,*a,**k):
            if name=="OPERATOR_FAULT_ACK": raise OSError("database write failed")
            return original(name,*a,**k)
        with patch.object(self.w.store,"event",side_effect=failed),patch.object(QInputDialog,"getText",return_value=("operator",True)),patch.object(QInputDialog,"getMultiLineText",return_value=("checked",True)):
            with self.assertRaises(OSError): self.w.acknowledge_faults()
        self.assertTrue(self.w.faults.active); self.assertFalse(self.w.running)


class RobotRequestFaultTests(unittest.TestCase):
    def setUp(self): self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name); self.store=OperationStore(self.root)
    def tearDown(self): self.temp.cleanup()
    def test_stop_discards_queued_recover_and_pick_and_invalidates_dequeued(self):
        robot=RobotThread(self.root/"robot",default_equipment(),self.store)
        robot.request("pick"); dequeued=robot.commands.get_nowait(); robot.request("recover",reference="old")
        robot.stop_motion(); self.assertTrue(robot.commands.empty()); self.assertFalse(robot.valid_command(dequeued))
        with self.assertRaises(ValueError): robot.request("pick")
        robot.stop_requested.clear(); robot.request("recover",reference="new"); self.assertTrue(robot.valid_command(robot.commands.get_nowait()))
    def test_record_failure_after_start_does_not_dispatch_tick(self):
        robot=RobotThread(self.root/"robot",default_equipment(),self.store); calls=[]
        adapter=SimpleNamespace(epoch="epoch",close=lambda:None)
        class Controller:
            state="RECOVERY"
            def __init__(self,*a,**k): pass
            def start(self,*a,**k): calls.append("reserved")
            def stop(self,*a): calls.append("stopped"); robot.stopping.set()
            def tick(self,*a,**k): calls.append("UNSAFE_DISPATCH")
            def close(self): pass
        robot.adapter_factory=lambda _:adapter
        robot.request("pick",result={},track={"track_id":"T"},product={},session="S")
        with patch("mes_vision.station.owner.RobotController",Controller),patch("mes_vision.operation.robot_service.real_plan",return_value=(SimpleNamespace(plan_id="P",object_id="O"),SimpleNamespace(run_id="R"))),patch.object(self.store,"event",side_effect=OSError("database full")):
            robot.run()
        self.assertEqual(calls,["reserved","stopped"])


if __name__=="__main__": unittest.main()

"""Persistence tests use injected inspection outputs; no hardware or accuracy claims."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import os
import sqlite3
import tempfile
import time
import unittest
from unittest.mock import patch
from filelock import FileLock,Timeout
import test_operation
from mes_vision.operation.catalog import OperationStore,new_product
from mes_vision.operation.storage import Search,StorageService,inventory,plain_path,copy_tree_verified
from mes_vision.operation.maintenance import Maintenance,connection
from mes_vision.training.data import read_json,write_json,sha256
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.robot.journal import RobotJournal


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_operation.OperationTests(); self.fixture.setUp(); self.store=self.fixture.store
        self.external=tempfile.TemporaryDirectory(); self.out=Path(self.external.name); self.service=Maintenance(self.store)
        self.result=None

    def tearDown(self): self.fixture.tearDown(); self.external.cleanup()

    def capture(self,*,defect=False,anomaly=.1,two=False):
        f=self.fixture; f.models.defect=defect; f.models.anomaly_score=anomaly
        if two: f.models.detections=(test_operation.detection(),test_operation.detection(x=210))
        self.result=f.engine.commit(f.inspect()); self.assertIsNotNone(self.result)
        f.engine.close(); self.store.end_session(f.session)
        return self.store.history()

    def old(self):
        with self.store.connect() as db: db.execute("UPDATE inspections SET created=?",(time.time()-900*86400,))
        self.service.save_policy(dict(self.service.policy(),archive_directory=str(self.out/"archive")))

    def enqueue(self):
        row=self.store.history()[0]; return self.service.queue.enqueue(row["path"],row["object_id"],asdict(GenerationConfig()))

    def test_search_filters_period_summary_and_pagination(self):
        rows=self.capture(defect=True,two=True); job=self.enqueue(); row=rows[0]
        self.store.review(row["object_id"],"operator","확인 완료")
        self.store.event("ROBOT_FINISHED",{"object_id":row["object_id"],"state":"RECAPTURE"},track_id=row["track_id"])
        result=self.service.search(Search(decision="NG",defect="NG03"),page_size=1)
        self.assertEqual(result["summary"]["inspections"],2); self.assertEqual(len(result["rows"]),1)
        second=self.service.search(Search(),page=1,page_size=1)["rows"][0]
        self.assertNotEqual(result["rows"][0]["object_id"],second["object_id"])
        self.assertEqual(self.service.search(Search(review="reviewed",robot="RECAPTURE",vlm="SKIPPED_DISABLED"))["summary"]["inspections"],1)
        self.assertEqual(self.service.search(Search(review="unreviewed",decision="ATTENTION"))["summary"]["inspections"],1)
        self.assertEqual(self.service.search(Search(product="test-part"))["summary"]["inspections"],2)
        self.assertEqual(self.service.search(Search(text="' OR 1=1 --"))["rows"],[])
        self.assertEqual(self.service.search(Search(until=1))["rows"],[])

    def test_reinspection_summary_counts_prior_outside_period(self):
        f=self.fixture; f.engine.commit(f.inspect()); track=next(iter(f.engine.tracker.tracks))
        f.stable(f.base+1); f.engine.tracker.recheck(track,f.base+1.8); f.engine.commit(f.inspect(f.base+1.8))
        rows=self.store.history(); since=(rows[0]["created"]+rows[1]["created"])/2
        result=self.service.search(Search(since=since))["summary"]
        self.assertEqual(result["inspections"],1); self.assertEqual(result["reinspections"],1)

    def test_index_upgrade_and_tamper_report(self):
        self.capture(defect=True)
        with self.store.connect() as db: db.execute("DELETE FROM inspection_index")
        self.assertEqual(self.service.search()["index_missing"],1)
        self.assertEqual(self.service.rebuild_index()["indexed"],1)
        self.assertEqual(len(self.service.search(Search(defect="NG03"))["rows"]),1)
        with self.store.connect() as db: db.execute("DELETE FROM inspection_index")
        (Path(self.result["path"])/"frame.png").write_bytes(b"tampered")
        self.assertEqual(len(self.service.rebuild_index()["errors"]),1)

    def test_export_all_not_visible_page_and_shared_evidence_once(self):
        self.capture(two=True); self.service.search(page_size=1)
        with self.store.connect() as db: db.execute("UPDATE inspection_index SET product_name=?",("=1+1",))
        destination=self.out/"report"; result=self.service.export(destination,Search())
        self.assertEqual(result["inspections"],2); self.assertEqual(len(list((destination/"evidence").iterdir())),1)
        report=read_json(destination/"report.json"); self.assertEqual(len(report["records"]),2)
        with (destination/"inspections.csv").open(encoding="utf-8-sig",newline="") as stream: rows=list(csv.DictReader(stream))
        self.assertEqual(rows[0]["product_name"],"'=1+1")
        files=inventory(destination); manifest=read_json(destination/"export-manifest.json"); files.pop("export-manifest.json")
        self.assertEqual(files,manifest["files"])
        with self.assertRaises(ValueError): self.service.export(destination,Search())

    def test_export_without_images_keeps_reasons_and_detects_corrupt_original(self):
        self.capture(defect=True); target=self.out/"report"; self.service.export(target,Search(),images=False)
        self.assertFalse((target/"evidence").exists()); self.assertEqual(read_json(target/"report.json")["records"][0]["inspection"]["final_decision"],"NG")
        (Path(self.result["path"])/"frame.png").write_bytes(b"bad")
        with self.assertRaises(ValueError): self.service.export(self.out/"bad-report",Search(),images=False)
        self.assertFalse((self.out/"bad-report").exists())

    def test_retention_protects_unknown_review_and_hold(self):
        rows=self.capture(anomaly=.95); self.old(); result=self.service.preview_archive()
        self.assertEqual(result["candidates"],[]); self.assertIn("미검토",result["protected"][0]["reason"])
        self.store.review(rows[0]["object_id"],"operator","미등록 형상 확인")
        self.assertEqual(len(self.service.preview_archive()["candidates"]),1)
        self.service.hold(rows[0]["run_id"],"고객 검토 자료")
        self.assertFalse(self.service.preview_archive()["candidates"])
        self.assertEqual(len(self.service.search(Search(archived="held"))["rows"]),1)
        self.service.hold(rows[0]["run_id"],None); self.assertEqual(len(self.service.preview_archive()["candidates"]),1)

    def test_shared_frame_retained_until_every_record_old(self):
        rows=self.capture(two=True); self.old()
        with self.store.connect() as db: db.execute("UPDATE inspections SET created=? WHERE object_id=?",(time.time(),rows[0]["object_id"]))
        self.assertFalse(self.service.preview_archive()["candidates"])

    def test_archive_relocates_history_and_vlm_without_changing_evidence(self):
        rows=self.capture(); job=self.enqueue(); self.old(); original=Path(rows[0]["path"]); before=inventory(original)
        result=self.service.archive(self.service.preview_archive()); self.assertEqual(result["archived"],1)
        self.assertFalse(original.exists()); row,_,_,obj=self.store.open_record(rows[0]["object_id"])
        self.assertEqual(inventory(Path(row["path"])),before); self.assertEqual(obj["final_decision"],"OK")
        self.assertEqual(self.service.queue.get(job)["snapshot_path"],row["path"])
        self.assertEqual(len(self.service.search(Search(archived="archived"))["rows"]),1)

    def test_archive_recovery_after_link_interrupt(self):
        rows=self.capture(); self.old(); original=Path(rows[0]["path"]); preview=self.service.preview_archive()
        real_intent=self.service._intent
        def interrupt(identity,state,data):
            real_intent(identity,state,data)
            if state=="LINKED": raise OSError("injected power interruption")
        with patch.object(self.service,"_intent",side_effect=interrupt):
            with self.assertRaises(OSError): self.service.archive(preview)
        self.assertTrue(original.exists()); self.store.open_record(rows[0]["object_id"])
        with self.assertRaises(ValueError): self.service.backup(self.out/"blocked")
        self.assertEqual(self.service.repair_archive()[0]["state"],"COMPLETED"); self.assertFalse(original.exists())
        self.assertEqual(self.service.repair_archive(),[])

    def test_archive_partial_copy_recovers_and_tamper_refuses_delete(self):
        rows=self.capture(); self.old(); preview=self.service.preview_archive(); original=Path(rows[0]["path"])
        from mes_vision.operation import maintenance
        def partial(source,target):
            target.mkdir(); first=next(source.iterdir()); (target/first.name).write_bytes(first.read_bytes()); raise OSError("interrupted copy")
        with patch.object(maintenance,"copy_tree_verified",side_effect=partial):
            with self.assertRaises(OSError): self.service.archive(preview)
        with self.store.connect() as db: data=json.loads(db.execute("SELECT data FROM storage_operations").fetchone()[0])
        bad=Path(data["destination"])/"unexpected"; bad.write_text("tampered")
        self.assertEqual(self.service.repair_archive()[0]["state"],"ATTENTION"); self.assertTrue(original.exists())
        bad.unlink(); self.assertEqual(self.service.repair_archive()[0]["state"],"COMPLETED")

    def test_stale_preview_changed_policy_and_active_session_refused(self):
        self.capture(); self.old(); preview=self.service.preview_archive(); preview["candidates"][0]["bytes"]+=1
        with self.assertRaises(ValueError): self.service.archive(preview)
        preview=self.service.preview_archive(); self.service.save_policy(dict(self.service.policy(),days_ok=31))
        with self.assertRaises(ValueError): self.service.archive(preview)
        session=self.store.start_session(self.fixture.product,self.fixture.equipment)
        with self.assertRaises(ValueError): self.service.backup(self.out/"active")
        self.store.end_session(session)
        with FileLock(str(self.store.root/"vlm/worker.lock")):
            with self.assertRaises(Timeout): self.service.archive(self.service.preview_archive())

    def test_orphan_age_and_source_path_validation(self):
        self.capture(); self.old(); orphan=self.store.root/"inspections/orphan"; orphan.mkdir(); (orphan/"partial").write_text("incomplete")
        preview=self.service.preview_archive(); self.assertEqual(len(preview["candidates"]),1)
        old=time.time()-8*86400
        for p in (orphan/"partial",orphan): os.utime(p,(old,old))
        self.assertEqual(len(self.service.preview_archive()["candidates"]),2)
        data=dict(self.service.preview_archive()["candidates"][0],source=str(self.out),destination=str(self.out/"a/b"),archive_root=str(self.out/"a"))
        with self.assertRaises(ValueError): self.service._finish_archive("bad",data)

    def test_backup_restore_assets_archived_frames_wal_vlm_and_robot_safe(self):
        rows=self.capture(defect=True); job=self.enqueue(); self.old(); self.service.archive(self.service.preview_archive())
        asset=self.out/"weights.pth"; asset.write_bytes(b"registered asset content")
        p=new_product("복원 대상"); p["objects"]={"weights":str(asset),"sha256":sha256(asset)}; self.store.save_product(p)
        journal=RobotJournal(self.store.root/"robot")
        with journal.db() as db: db.execute("INSERT INTO plans VALUES('plan','digest','frame','{}','PREPARE_PICK')")
        journal.record("FAULT","TEST_EVENT")
        with self.service.queue.connect() as db: db.execute("UPDATE jobs SET state='COMPLETED',result=? WHERE id=?",('{"note":"추가 관찰"}',job))
        before=self.store.open_record(rows[0]["object_id"])[0]
        self.store.event("WAL_TEST",{"value":"보존 확인"})
        backup=self.out/"backup"; result=self.service.backup(backup); self.assertGreater(result["files"],3)
        manifest=self.service.validate_backup(backup); self.assertGreater(len(manifest["mappings"]),1)
        restored=self.service.restore(backup,self.out/"restored"); copied=OperationStore(restored["runtime"])
        after=copied.open_record(rows[0]["object_id"])[0]; self.assertEqual(before["digest"],after["digest"])
        self.assertTrue(Path(after["path"]).is_relative_to(self.out/"restored")); self.assertEqual(self.store.open_record(rows[0]["object_id"])[0]["path"],before["path"])
        asset_copy=Path(copied.products()[0]["objects"]["weights"]); self.assertEqual(asset_copy.read_bytes(),asset.read_bytes())
        q=AnalysisQueue(copied.root/"vlm"); self.assertFalse(q.enabled()); self.assertEqual(q.get(job)["state"],"COMPLETED")
        self.assertEqual(q.get(job)["result"]["note"],"추가 관찰"); self.assertEqual(RobotJournal(copied.root/"robot").state(),"RECOVERY")
        with copied.connect() as db:
            self.assertEqual(db.execute("SELECT COUNT(*) FROM events WHERE type='WAL_TEST'").fetchone()[0],1)
            self.assertEqual(db.execute("SELECT state FROM storage_operations").fetchone()[0],"RESTORED_HISTORY")
        self.assertTrue((self.out/"restored/restore-source.json").exists())

    def test_backup_missing_assets_and_tampered_bundle_refused(self):
        self.capture(); p=new_product(); p["policy"]=str(self.out/"missing.json"); self.store.save_product(p)
        with self.assertRaises(ValueError): self.service.backup(self.out/"missing-backup")
        p=self.store.products()[0]; p["policy"]=None; self.store.save_product(p)
        # All historical versions are included; repair only the deliberately broken test setup.
        with self.store.connect() as db: db.execute("DELETE FROM products WHERE version=1")
        self.service.backup(self.out/"backup"); file=self.out/"backup/runtime/operation.sqlite3"; file.write_bytes(b"tampered")
        with self.assertRaises(ValueError): self.service.restore(self.out/"backup",self.out/"restored")
        self.assertFalse((self.out/"restored").exists())

    def test_backup_manifest_traversal_and_mapping_alias_rejected(self):
        self.capture(); self.service.backup(self.out/"backup"); path=self.out/"backup/backup.json"; original=read_json(path)
        broken=json.loads(json.dumps(original)); broken["files"]["../outside"]={"sha256":"x","bytes":1}; write_json(path,broken)
        with self.assertRaises(ValueError): self.service.validate_backup(path.parent)
        broken=json.loads(json.dumps(original)); broken["mappings"][0]["target"]="assets/../runtime"; write_json(path,broken)
        with self.assertRaises(ValueError): self.service.validate_backup(path.parent)

    def test_storage_configuration_and_existing_destination_guards(self):
        self.capture(); self.assertGreater(self.service.usage()["runtime_bytes"],0)
        with self.assertRaises(ValueError): self.service.save_policy(dict(self.service.policy(),archive_directory=str(self.store.root/"inside")))
        with self.assertRaises(ValueError): self.service.save_policy(dict(self.service.policy(),days_ng=-1))
        with self.assertRaises(ValueError): self.service.backup(self.out)
        with self.assertRaises(ValueError): self.service.export(self.store.root/"inside",Search())

    def test_low_disk_stops_publication_before_creating_evidence(self):
        from types import SimpleNamespace
        pending=self.fixture.inspect()
        with patch("mes_vision.operation.storage.shutil.disk_usage",return_value=SimpleNamespace(free=1024)):
            with self.assertRaisesRegex(ValueError,"공간이 부족"):
                self.fixture.engine.commit(pending)
        self.assertFalse(self.store.history()); self.assertFalse((self.store.root/"inspections").exists())

    def test_backup_verification_is_read_only_and_repeatable(self):
        self.capture(); self.service.backup(self.out/"backup"); before=inventory(self.out/"backup")
        for _ in range(3): self.service.validate_backup(self.out/"backup")
        self.assertEqual(before,inventory(self.out/"backup"))

    def test_archived_unregistered_files_are_included_in_backup(self):
        self.capture(); self.old(); orphan=self.store.root/"inspections/orphan"; orphan.mkdir(); (orphan/"partial").write_bytes(b"recoverable raw output")
        old=time.time()-8*86400
        for p in (orphan/"partial",orphan): os.utime(p,(old,old))
        self.service.archive(self.service.preview_archive()); self.service.backup(self.out/"backup")
        self.assertTrue(any(p.is_file() and p.name=="partial" for p in (self.out/"backup").rglob("partial")))

    def test_restored_deferred_analysis_never_restarts_automatically(self):
        self.capture(); job=self.enqueue()
        with self.service.queue.connect() as db: db.execute("UPDATE jobs SET state='DEFERRED' WHERE id=?",(job,))
        self.service.backup(self.out/"backup"); result=self.service.restore(self.out/"backup",self.out/"restored")
        q=AnalysisQueue(Path(result["runtime"])/"vlm"); self.assertFalse(q.enabled()); self.assertEqual(q.get(job)["state"],"CANCELLED")
        self.assertEqual(self.service.queue.get(job)["state"],"DEFERRED")


if __name__=="__main__": unittest.main()

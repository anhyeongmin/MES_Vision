"""Verified archive moves and portable backups, with durable recovery intents."""
from pathlib import Path,PurePosixPath
from copy import deepcopy
from contextlib import contextmanager
from datetime import datetime,timezone
import json
import os
import shutil
import sqlite3
import time
from uuid import uuid4
from mes_vision.training.data import require,read_json,write_json,sha256
from mes_vision.anomaly.features import fingerprint
from mes_vision.vlm.snapshots import load_snapshot
from .storage import StorageService,plain_path,within,file_tree,inventory,copy_tree_verified
from .catalog import OperationStore


@contextmanager
def connection(*args,**kwargs):
    db=sqlite3.connect(*args,**kwargs)
    try:
        with db: yield db
    finally: db.close()


def snapshot_database(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    with connection(source) as incoming,connection(destination) as outgoing:
        incoming.backup(outgoing)
        outgoing.execute("PRAGMA journal_mode=DELETE")
        require(outgoing.execute("PRAGMA integrity_check").fetchone()[0]=="ok","백업 데이터베이스 검증 실패")


def asset_paths(product=None,equipment=None):
    paths=[]
    if product:
        for key in ("objects","defects"):
            if product.get(key): paths.append(product[key]["weights"])
        for key in ("policy","geometry","normal_reference"):
            if product.get(key): paths.append(product[key])
        if product.get("anomaly"): paths.extend(v for k,v in product["anomaly"].items() if k in {"bank","criteria"} and v)
    if equipment:
        if equipment.get("calibration"): paths.append(equipment["calibration"])
        if equipment.get("robot",{}).get("profile"): paths.append(equipment["robot"]["profile"])
    return paths


class Maintenance(StorageService):
    def preview_archive(self,*,now=None):
        now=time.time() if now is None else now; policy=self.policy(); candidates=[]; protected=[]; errors=[]
        with self.store.connect() as db:
            captures=[dict(r) for r in db.execute("SELECT c.*,COALESCE(s.archived,0) archived,s.hold_note FROM captures c LEFT JOIN capture_storage s USING(run_id)")]
            for c in captures:
                if c["archived"]: continue
                rows=[dict(r) for r in db.execute("SELECT i.*,x.codes FROM inspections i LEFT JOIN inspection_index x USING(object_id) WHERE run_id=?",(c["run_id"],))]
                reason=None
                if c["hold_note"]: reason="별도 보존 지정"
                elif any(r["codes"] is None for r in rows): reason="이전 기록 색인 필요"
                elif policy["protect_unreviewed"] and any(not r["review"] and (r["decision"]=="REVIEW" or "NG_UNKNOWN" in json.loads(r["codes"])) for r in rows): reason="미검토 보류·미등록 이상"
                elif any(now-r["created"]<policy[{"OK":"days_ok","NG":"days_ng","REVIEW":"days_review"}[r["decision"]]]*86400 for r in rows): reason="보존 기간 이내"
                if reason: protected.append({"run_id":c["run_id"],"reason":reason}); continue
                try:
                    source=within(c["path"],self.root/"inspections"); require(source.parent==self.root/"inspections","검사 원본 경로 구조 오류")
                    load_snapshot(source,expected_digest=c["digest"]); files=inventory(source)
                    candidates.append({"run_id":c["run_id"],"source":str(source),"digest":c["digest"],"files":files,"bytes":sum(f["bytes"] for f in files.values()),"kind":"inspection"})
                except Exception as exc: errors.append({"path":c["path"],"error":str(exc)})
        registered={Path(os.path.abspath(c["path"])) for c in captures}
        directory=self.root/"inspections"
        if directory.exists():
            for source in directory.iterdir():
                try:
                    source=within(source,directory)
                    if not source.is_dir() or source in registered: continue
                    files=inventory(source)
                    newest=max([source.stat().st_mtime,*[p.stat().st_mtime for p in file_tree(source)]])
                    if now-newest<policy["orphan_days"]*86400: continue
                    candidates.append({"run_id":None,"source":str(source),"digest":None,"files":files,"bytes":sum(f["bytes"] for f in files.values()),"kind":"unregistered"})
                except Exception as exc: errors.append({"path":str(source),"error":str(exc)})
        from mes_vision.station.retention import preview
        extra,kept,issues=preview(self,now,policy); candidates+=extra; protected+=kept; errors+=issues
        report={"schema_version":1,"runtime":str(self.root),"created":now,"policy":policy,"candidates":candidates,"protected":protected,"errors":errors}
        report["digest"]=fingerprint(report); return report

    def _intent(self,identity,state,data):
        with self.store.connect() as db:
            db.execute("INSERT INTO storage_operations VALUES(?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,data=excluded.data,updated=excluded.updated",(identity,state,json.dumps(data,ensure_ascii=False),time.time()))

    def archive(self,preview):
        expected=deepcopy(preview); digest=expected.pop("digest",None); require(fingerprint(expected)==digest,"정리 미리보기 내용이 변경됐습니다.")
        require(preview["runtime"]==str(self.root) and 0<=time.time()-preview["created"]<=900,"다시 정리 미리보기를 실행하세요.")
        with self.exclusive(idle=True):
            with self.store.connect() as db: require(not db.execute("SELECT 1 FROM storage_operations WHERE state NOT IN ('COMPLETED','RESTORED_HISTORY') LIMIT 1").fetchone(),"중단된 보관 작업을 먼저 복구하세요.")
            current=self.preview_archive(); require(current["policy"]==preview["policy"],"보존 설정이 바뀌었습니다.")
            allowed={c["source"]:c for c in current["candidates"]}
            root=current["policy"]["archive_directory"]; require(root,"보관 폴더를 먼저 지정하세요."); root=plain_path(root)
            root.mkdir(parents=True,exist_ok=True)
            require(shutil.disk_usage(root).free>sum(c["bytes"] for c in preview["candidates"])+1024**2,"보관 위치의 남은 용량이 부족합니다.")
            completed=[]
            for item in preview["candidates"]:
                require(allowed.get(item["source"])==item,"정리 대상 상태가 바뀌었습니다. 미리보기를 다시 실행하세요.")
                identity=uuid4().hex; destination=within(root/identity,root)
                data={**item,"destination":str(destination),"archive_root":str(root),"preview_digest":digest}
                self._intent(identity,"COPYING",data); self._finish_archive(identity,data); completed.append(identity)
            return {"archived":len(completed),"bytes":sum(c["bytes"] for c in preview["candidates"]),"operations":completed}

    def _finish_archive(self,identity,data):
        if data.get('kind')=='station':
            from mes_vision.station.retention import finish
            return finish(self,identity,data)
        source=within(data["source"],self.root/"inspections"); require(source.parent==self.root/"inspections","허용된 검사 폴더가 아닙니다.")
        archive_root=plain_path(data["archive_root"]); destination=within(data["destination"],archive_root)
        require(destination.parent==archive_root and not archive_root.is_relative_to(self.root) and not self.root.is_relative_to(archive_root),"보관 경로가 잘못됐습니다.")
        if not destination.exists():
            require(inventory(source)==data["files"],"복사 전 원본 파일이 변경됐습니다.")
            copy_tree_verified(source,destination)
        elif inventory(destination)!=data["files"]:
            partial=inventory(destination)
            require(all(name in data["files"] and data["files"][name]==value for name,value in partial.items()),"보관 중간 파일이 변경됐습니다.")
            require(inventory(source)==data["files"],"중단된 복사를 재개할 원본을 확인할 수 없습니다.")
            for name in data["files"]:
                if name not in partial:
                    target=within(destination/name,archive_root); target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(within(source/name,self.root/"inspections"),target)
        require(inventory(destination)==data["files"],"보관 원본의 무결성을 확인할 수 없습니다.")
        if data["digest"]: load_snapshot(destination,expected_digest=data["digest"])
        self._intent(identity,"COPIED",data)
        if data["run_id"]:
            with self.store.connect() as db:
                row=db.execute("SELECT path,digest FROM captures WHERE run_id=?",(data["run_id"],)).fetchone()
                require(row and row["digest"]==data["digest"] and plain_path(row["path"]) in {source,destination},"기록과 보관 작업의 연결이 다릅니다.")
                db.execute("UPDATE captures SET path=? WHERE run_id=?",(str(destination),data["run_id"]))
                db.execute("INSERT INTO capture_storage(run_id,archived) VALUES(?,1) ON CONFLICT(run_id) DO UPDATE SET archived=1",(data["run_id"],))
            with self.queue.connect() as db:
                db.execute("UPDATE jobs SET snapshot_path=? WHERE snapshot_path=? AND snapshot_digest=?",(str(destination),str(source),data["digest"]))
        self._intent(identity,"LINKED",data)
        if source.exists():
            # All destructive paths are resolved under the original managed inspections root;
            # remove only inventoried files after a complete independent archive was verified.
            remaining=inventory(source)
            require(all(name in data["files"] and data["files"][name]==value for name,value in remaining.items()),"정리 중 원본 파일이 변경됐습니다.")
            for name in remaining: within(source/name,self.root/"inspections").unlink()
            for directory,folders,_ in os.walk(source,topdown=False,followlinks=False):
                for folder in folders: within(Path(directory)/folder,self.root/"inspections").rmdir()
            within(source,self.root/"inspections").rmdir()
        self._intent(identity,"COMPLETED",data)
        self.store.event("SNAPSHOT_ARCHIVED",{"operation_id":identity,"run_id":data["run_id"],"path":str(destination),"kind":data["kind"]})

    def repair_archive(self):
        results=[]
        with self.exclusive(idle=True),self.store.connect() as db:
            rows=[dict(r) for r in db.execute("SELECT * FROM storage_operations WHERE state NOT IN ('COMPLETED','RESTORED_HISTORY')")]
        # Each recovery rechecks quiescence and OS-held leases. Never repeats robot commands.
        for row in rows:
            try:
                with self.exclusive(idle=True): self._finish_archive(row["id"],json.loads(row["data"]))
                results.append({"id":row["id"],"state":"COMPLETED"})
            except Exception as exc: results.append({"id":row["id"],"state":"ATTENTION","error":str(exc)})
        return results

    def backup(self,destination):
        destination=plain_path(destination); require(not destination.exists() and not destination.is_relative_to(self.root),"백업은 운영 폴더 밖의 새 폴더를 선택하세요.")
        staging=destination.with_name(destination.name+".partial-"+uuid4().hex); staging.mkdir(parents=True)
        with self.exclusive(idle=True):
            with self.store.connect() as db: require(not db.execute("SELECT 1 FROM storage_operations WHERE state NOT IN ('COMPLETED','RESTORED_HISTORY') LIMIT 1").fetchone(),"중단된 보관 작업을 먼저 복구하세요.")
            mappings=[{"source":str(self.root),"target":"runtime","directory":True}]
            for source in file_tree(self.root,skip_runtime_transients=True):
                relative=source.relative_to(self.root)
                if source.name.endswith((".lock","-wal","-shm")) or "gpu-coordination" in relative.parts: continue
                target=within(staging/"runtime"/relative,staging); target.parent.mkdir(parents=True,exist_ok=True)
                if source.suffix==".sqlite3": snapshot_database(source,target)
                else:
                    before=sha256(source); shutil.copyfile(source,target); require(before==sha256(source)==sha256(target),"백업 중 파일이 변경됐습니다.")
            from mes_vision.station.portability import external_assets
            external=external_assets(self.root)
            with self.store.connect() as db:
                for row in db.execute("SELECT data FROM products"): external+=asset_paths(product=json.loads(row[0]))
                for row in db.execute("SELECT data FROM equipment"): external+=asset_paths(equipment=json.loads(row[0]))
                for row in db.execute("SELECT product,equipment FROM sessions"): external+=asset_paths(product=json.loads(row[0]),equipment=json.loads(row[1]))
                for row in db.execute("SELECT path,digest FROM captures"):
                    load_snapshot(row["path"],expected_digest=row["digest"]); external.append(row["path"])
                for row in db.execute("SELECT data FROM storage_operations WHERE state IN ('COMPLETED','RESTORED_HISTORY')"):
                    data=json.loads(row[0])
                    if data.get("kind")=="unregistered":
                        path=plain_path(data["destination"]); require(inventory(path)==data["files"],"보관된 미등록 저장물이 누락되거나 변경됐습니다."); external.append(path)
            with self.queue.connect() as db:
                for row in db.execute("SELECT DISTINCT snapshot_path,snapshot_digest FROM jobs"):
                    load_snapshot(row[0],expected_digest=row[1]); external.append(row[0])
            sources=sorted({plain_path(p) for p in external},key=lambda p:len(p.parts))
            for source in sources:
                if any(source==Path(m["source"]) or m["directory"] and source.is_relative_to(Path(m["source"])) for m in mappings): continue
                require(source.exists(),"등록된 외부 파일이 없습니다: "+str(source)); relative="assets/"+uuid4().hex
                target=within(staging/relative,staging); target.parent.mkdir(parents=True,exist_ok=True)
                if source.is_dir(): copy_tree_verified(source,target)
                else:
                    before=sha256(source); shutil.copyfile(source,target); require(before==sha256(source)==sha256(target),"외부 파일 백업 중 변경 감지")
                mappings.append({"source":str(source),"target":relative,"directory":source.is_dir()})
            manifest={"schema_version":1,"kind":"mes-operation-backup","created":time.time(),"mappings":mappings,"files":inventory(staging),
                "scope":"운영 기록과 등록된 모델·기준·참조 파일. 프로그램 설치본과 공통 기본 모델은 별도 설치 필요."}
            write_json(staging/"backup.json",manifest); self.validate_backup(staging)
            require(not destination.exists(),"백업 대상 폴더가 이미 만들어졌습니다."); staging.rename(destination)
        self.store.event("BACKUP_CREATED",{"path":str(destination),"files":len(manifest["files"]),"manifest_digest":fingerprint(manifest)})
        return {"path":str(destination),"files":len(manifest["files"]),"bytes":sum(f["bytes"] for f in manifest["files"].values())}

    @staticmethod
    def validate_backup(directory):
        directory=plain_path(directory); manifest=read_json(directory/"backup.json")
        require(manifest.get("schema_version")==1 and manifest.get("kind")=="mes-operation-backup","지원하는 운영 백업 폴더가 아닙니다.")
        require(isinstance(manifest["files"],dict) and len(manifest["files"])<=100000,"백업 파일 목록 오류")
        for name in manifest["files"]:
            p=PurePosixPath(name)
            require(name and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name and str(p)==name,"백업 경로가 올바르지 않습니다.")
            within(directory/name,directory)
        actual=inventory(directory); actual.pop("backup.json",None); require(actual==manifest["files"],"백업 파일이 누락되거나 변경됐습니다.")
        require("runtime/operation.sqlite3" in actual,"운영 데이터베이스가 없습니다.")
        mappings=manifest["mappings"]
        require(isinstance(mappings,list) and sum(m.get("target")=="runtime" and m.get("directory") is True for m in mappings)==1,"운영 경로 매핑 오류")
        require(len({m["source"] for m in mappings})==len(mappings) and len({m["target"] for m in mappings})==len(mappings),"중복된 백업 매핑")
        for mapping in mappings:
            require(type(mapping["directory"]) is bool and Path(mapping["source"]).is_absolute(),"백업 경로 매핑 오류")
            p=PurePosixPath(mapping["target"])
            require(str(p)==mapping["target"] and not p.is_absolute() and ".." not in p.parts and "\\" not in str(p) and ":" not in str(p) and (str(p)=="runtime" or len(p.parts)==2 and p.parts[0]=="assets"),"백업 자산 경로 오류")
            target=within(directory/mapping["target"],directory); require(target.exists() and target.is_dir()==mapping["directory"],"백업 자산 매핑 오류")
        for name in actual:
            if name.endswith(".sqlite3"):
                with connection(f"{(directory/name).as_uri()}?mode=ro&immutable=1",uri=True) as db: require(db.execute("PRAGMA integrity_check").fetchone()[0]=="ok","백업 데이터베이스 손상")
        return manifest

    def restore(self,backup,destination):
        backup=plain_path(backup); destination=plain_path(destination)
        require(not destination.exists() and not destination.is_relative_to(self.root) and not destination.is_relative_to(backup),"복원은 기존 운영·백업 폴더 밖의 새 폴더에만 가능합니다.")
        manifest=self.validate_backup(backup); staging=destination.with_name(destination.name+".partial-"+uuid4().hex)
        copy_tree_verified(backup,staging)
        # The destination contains runtime plus restored assets, so old immutable evidence is unchanged.
        def relocate(value):
            if isinstance(value,str):
                for mapping in sorted(manifest["mappings"],key=lambda m:len(m["source"]),reverse=True):
                    old=Path(mapping["source"]); candidate=Path(value)
                    if candidate==old: return str(destination/mapping["target"])
                    if mapping["directory"] and candidate.is_absolute() and candidate.is_relative_to(old): return str(destination/mapping["target"]/candidate.relative_to(old))
                return value
            if isinstance(value,list): return [relocate(v) for v in value]
            if isinstance(value,dict): return {k:relocate(v) for k,v in value.items()}
            return value
        runtime=staging/"runtime"
        with connection(runtime/"operation.sqlite3") as db:
            for table,column,key in (("products","data","rowid"),("equipment","data","rowid"),("sessions","product","id"),("sessions","equipment","id"),("events","data","id"),("storage_operations","data","id")):
                for identity,value in db.execute(f"SELECT {key},{column} FROM {table}").fetchall():
                    db.execute(f"UPDATE {table} SET {column}=? WHERE {key}=?",(json.dumps(relocate(json.loads(value)),ensure_ascii=False),identity))
            for run_id,path in db.execute("SELECT run_id,path FROM captures").fetchall(): db.execute("UPDATE captures SET path=? WHERE run_id=?",(relocate(path),run_id))
            for path,digest in db.execute("SELECT path,digest FROM captures"):
                target=within(path,destination); staged=staging/target.relative_to(destination); load_snapshot(staged,expected_digest=digest)
            for value, in db.execute("SELECT data FROM products"):
                for path in asset_paths(product=json.loads(value)): within(path,destination)
            for value, in db.execute("SELECT data FROM equipment"):
                for path in asset_paths(equipment=json.loads(value)): within(path,destination)
            db.execute("UPDATE sessions SET state='INTERRUPTED',ended=COALESCE(ended,?) WHERE state='ACTIVE'",(time.time(),))
            # Restored history is never queued for physical execution or automatic cleanup.
            db.execute("DELETE FROM preferences WHERE key='storage_policy'")
            db.execute("UPDATE storage_operations SET state='RESTORED_HISTORY'")
        queue_path=runtime/"vlm/queue.sqlite3"
        if queue_path.exists():
            with connection(queue_path) as db:
                for identity,path in db.execute("SELECT id,snapshot_path FROM jobs").fetchall(): db.execute("UPDATE jobs SET snapshot_path=? WHERE id=?",(relocate(path),identity))
                for path,digest in db.execute("SELECT snapshot_path,snapshot_digest FROM jobs"):
                    target=within(path,destination); load_snapshot(staging/target.relative_to(destination),expected_digest=digest)
                db.execute("UPDATE settings SET value='false' WHERE key='enabled'")
                db.execute("DELETE FROM settings WHERE key='runtime'")
                db.execute("UPDATE jobs SET state='CANCELLED',owner=NULL,token=NULL,cancel_requested=0,error='RESTORED_REQUIRES_MANUAL_REQUEST' WHERE state IN ('PENDING','RUNNING','DEFERRED')")
        robot_path=runtime/"robot/robot.sqlite3"
        if robot_path.exists():
            with connection(robot_path) as db:
                db.execute("UPDATE settings SET value='RECOVERY' WHERE key='state'"); db.execute("UPDATE plans SET state='RESTORED_HISTORY'")
        from mes_vision.station.portability import restore_station
        restore_station(runtime,relocate)
        (staging/"backup.json").rename(staging/"restore-source.json")
        # JSON settings embedded inside registered files keep original content/digests. Only catalog path references relocate.
        require(not destination.exists(),"복원 위치가 이미 만들어졌습니다."); staging.rename(destination)
        restored=OperationStore(destination/"runtime")
        with restored.connect() as db: captures=[dict(r) for r in db.execute("SELECT * FROM captures")]
        for row in captures: load_snapshot(row["path"],expected_digest=row["digest"])
        restored.event("BACKUP_RESTORED",{"source":str(backup),"source_manifest_digest":fingerprint(manifest),"robot_resume":False,"vlm_enabled":False})
        return {"path":str(destination),"runtime":str(destination/"runtime"),"captures":len(captures)}

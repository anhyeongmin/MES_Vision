"""Search and portable evidence exports. No implicit deletion or verdict rewriting."""
from contextlib import contextmanager
from dataclasses import dataclass,asdict
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
import csv
import json
import math
import os
import shutil
import sqlite3
import stat
import time
from uuid import uuid4
from filelock import FileLock
from mes_vision.training.data import require,read_json,write_json,sha256
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.snapshots import load_snapshot
from .catalog import OperationStore


def plain_path(path):
    """Reject symbolic links and Windows junction/reparse points, including ancestors."""
    path=Path(os.path.abspath(path))
    for part in (path,*path.parents):
        if part.exists() or part.is_symlink():
            info=part.lstat()
            require(not stat.S_ISLNK(info.st_mode) and not getattr(info,"st_file_attributes",0)&0x400,"연결된 경로(심볼릭 링크/접합점)는 저장 작업에 사용할 수 없습니다.")
    return path.resolve()


def within(path,root):
    path=plain_path(path); root=plain_path(root)
    require(path!=root and path.is_relative_to(root),"저장 작업 경로가 지정 폴더 밖입니다.")
    return path


def file_tree(root,*,skip_runtime_transients=False):
    root=plain_path(root); require(root.is_dir(),"폴더를 찾을 수 없습니다: "+str(root))
    for directory,folders,files in os.walk(root,followlinks=False):
        for name in folders: within(Path(directory)/name,root)
        for name in sorted(files):
            if skip_runtime_transients and name.endswith(('.lock','-wal','-shm')): continue
            path=within(Path(directory)/name,root); require(path.is_file(),"일반 파일이 아닌 항목이 있습니다."); yield path


def inventory(root): return {p.relative_to(root).as_posix():{"sha256":sha256(p),"bytes":p.stat().st_size} for p in file_tree(root)}


def copy_tree_verified(source,destination):
    source=plain_path(source); destination=plain_path(destination)
    require(not destination.exists(),"대상 폴더가 이미 존재합니다.")
    before=inventory(source); destination.mkdir(parents=True)
    for name in before:
        target=within(destination/name,destination); target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(within(source/name,source),target)
    require(before==inventory(source)==inventory(destination),"복사 중 원본 변경 또는 저장 오류가 발생했습니다.")
    return before


def csv_value(value):
    if value is None: return ""
    if isinstance(value,str) and value.lstrip().startswith(("=","+","-","@")): return "'"+value
    return value


def check_free_space(store,required_bytes=0):
    with store.connect() as db: row=db.execute("SELECT value FROM preferences WHERE key='storage_policy'").fetchone()
    minimum=json.loads(row[0])["min_free_gib"] if row else 5.
    require(shutil.disk_usage(store.root).free>minimum*1024**3+required_bytes,"검사 원본을 저장할 공간이 부족합니다. 저장 관리에서 용량을 확인하세요.")


@dataclass(frozen=True)
class Search:
    text: str=""
    product: str=""
    decision: str=""
    defect: str=""
    review: str=""
    vlm: str=""
    robot: str=""
    archived: str=""
    since: float=0.
    until: float=32503680000.
    as_of: float=32503680000.

    def __post_init__(self):
        require(self.decision in {"","OK","NG","REVIEW","ATTENTION"} and self.defect in {"","NG01","NG02","NG03","NG04","NG05","NG06","NG_UNKNOWN"},"판정·불량 검색 조건 오류")
        require(self.review in {"","unreviewed","reviewed"} and self.archived in {"","active","archived","held"},"검토·보존 검색 조건 오류")
        require(self.vlm in {"","NONE","PENDING","RUNNING","COMPLETED","FAILED","CANCELLED","DEFERRED","SKIPPED_DISABLED"},"VLM 검색 조건 오류")
        require(self.robot in {"","NONE","REQUESTED","RECAPTURE","FAULT","STOPPED","RECOVERY"},"로봇 검색 조건 오류")
        require(all(type(v) in {int,float} and math.isfinite(v) for v in (self.since,self.until,self.as_of)) and 0<=self.since<self.until,"검색 기간 오류")
        require(all(isinstance(v,str) and len(v)<=500 for v in (self.text,self.product)),"검색어가 너무 깁니다.")


class StorageService:
    def __init__(self,store):
        self.store=store; self.root=store.root; self.queue=AnalysisQueue(self.root/"vlm")

    @contextmanager
    def exclusive(self,*,idle=False):
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(FileLock(str(self.root/"storage.lock"),timeout=0))
            if idle:
                for path in (self.root/"vlm/worker.lock",self.root/"vlm/gpu-coordination/resident.lock",self.root/"robot/controller.lock"):
                    path.parent.mkdir(parents=True,exist_ok=True); stack.enter_context(FileLock(str(path),timeout=0))
                require(not self.queue.enabled(),"VLM을 OFF로 설정하세요.")
                with self.queue.connect() as db: require(db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('RUNNING','PENDING')").fetchone()[0]==0,"분석 작업이 종료될 때까지 기다리세요.")
                with self.store.connect() as db: require(db.execute("SELECT COUNT(*) FROM sessions WHERE state='ACTIVE'").fetchone()[0]==0,"검사 모델을 해제하고 운전 회차를 종료하세요.")
                if (self.root/'station.sqlite3').exists():
                    from mes_vision.station.sequence import ScanJournal
                    with ScanJournal(self.root).connect() as db:
                        require(not db.execute("SELECT 1 FROM cycles WHERE state NOT IN ('COMPLETED','CANCELLED','FAILED','INTERRUPTED') LIMIT 1").fetchone(),'전체·상세 검사 회차가 종료되어야 합니다.')
            yield

    def rebuild_index(self):
        count=0; errors=[]
        with self.exclusive(),self.store.connect() as db:
            rows=db.execute("SELECT DISTINCT c.* FROM captures c JOIN inspections i USING(run_id) LEFT JOIN inspection_index x USING(object_id) WHERE x.object_id IS NULL").fetchall()
            for row in rows:
                try:
                    _,result,_=load_snapshot(row["path"],expected_digest=row["digest"])
                    ids={r[0] for r in db.execute("SELECT object_id FROM inspections WHERE run_id=?",(row["run_id"],))}
                    for obj in result["objects"]:
                        if obj["object_id"] in ids: self.store.index_object(db,result,obj,row["session"]); count+=1
                except Exception as exc: errors.append({"run_id":row["run_id"],"error":str(exc)})
        from mes_vision.station.history import sync
        sync(self.store,verify=True)
        with self.store.connect() as db:
            count+=db.execute('SELECT COUNT(*) FROM station_records').fetchone()[0]
            errors += [{'run_id':'station:'+r[0],'error':r[1]} for r in db.execute('SELECT id,error FROM station_index_cycles WHERE error IS NOT NULL')]
        return {"indexed":count,"errors":errors}

    def _query(self,db,filters):
        from mes_vision.station.history import sync
        sync(self.store)
        db.execute("ATTACH DATABASE ? AS analysis",(str(self.queue.database),))
        sql='''WITH records AS (
          SELECT i.*,c.path,c.digest,c.session,
           COALESCE(x.product_id,json_extract(s.product,'$.id'),'') product_id,
           COALESCE(x.product_name,json_extract(s.product,'$.name'),'') product_name,
           COALESCE(x.product_version,json_extract(s.product,'$.version')) product_version,x.equipment_version,COALESCE(x.codes,'[]') codes,
           COALESCE(st.archived,0) archived,st.hold_note,
           COALESCE((SELECT j.state FROM analysis.jobs j WHERE j.object_id=i.object_id ORDER BY j.created DESC,j.id DESC LIMIT 1),'NONE') vlm_state,
           COALESCE((SELECT CASE WHEN e.type='ROBOT_FINISHED' THEN json_extract(e.data,'$.state') ELSE 'REQUESTED' END
             FROM events e WHERE json_extract(e.data,'$.object_id')=i.object_id AND e.type IN ('ROBOT_PLAN','ROBOT_FINISHED') ORDER BY e.id DESC LIMIT 1),'NONE') robot_state,
           'legacy' record_kind,i.object_id source_object_id,NULL station_data
          FROM inspections i JOIN captures c USING(run_id) LEFT JOIN sessions s ON s.id=c.session
          LEFT JOIN inspection_index x USING(object_id) LEFT JOIN capture_storage st USING(run_id)
          UNION ALL
          SELECT r.object_id,r.run_id,r.track_id,r.revision,r.decision,r.prior_object_id,r.created,r.review,r.path,r.digest,r.session,
            r.product_id,r.product_name,r.product_version,r.equipment_version,r.codes,COALESCE(st.archived,0),st.hold_note,
            COALESCE((SELECT j.state FROM analysis.jobs j WHERE j.object_id=r.source_object_id ORDER BY j.created DESC,j.id DESC LIMIT 1),'NONE'),
            r.robot_state,'station',r.source_object_id,r.data
          FROM station_records r LEFT JOIN capture_storage st USING(run_id)
        ) SELECT * FROM records WHERE created>=? AND created<? AND created<=?'''
        args=[filters.since,filters.until,filters.as_of]
        if filters.text:
            sql+=" AND (instr(track_id,?)>0 OR instr(run_id,?)>0 OR instr(object_id,?)>0 OR instr(session,?)>0 OR instr(product_name,?)>0)"; args += [filters.text]*5
        if filters.product: sql+=" AND (instr(product_id,?)>0 OR instr(product_name,?)>0)"; args += [filters.product]*2
        for column,value in (("decision",filters.decision),("vlm_state",filters.vlm),("robot_state",filters.robot)):
            if column=="decision" and value=="ATTENTION": sql+=" AND decision IN ('NG','REVIEW')"
            elif value: sql+=f" AND {column}=?"; args.append(value)
        if filters.defect: sql+=" AND EXISTS(SELECT 1 FROM json_each(codes) WHERE value=?)"; args.append(filters.defect)
        if filters.review: sql+=" AND review IS "+("NULL" if filters.review=="unreviewed" else "NOT NULL")
        if filters.archived in {"active","archived"}: sql+=" AND archived=?"; args.append(int(filters.archived=="archived"))
        if filters.archived=="held": sql+=" AND hold_note IS NOT NULL"
        return sql,args

    def search(self,filters=None,*,page=0,page_size=100):
        filters=filters or Search(as_of=time.time()); require(type(page) is int and page>=0 and type(page_size) is int and 1<=page_size<=500,"조회 페이지 오류")
        with self.store.connect() as db:
            sql,args=self._query(db,filters); db.execute("BEGIN")
            summary=dict(db.execute("SELECT COUNT(*) inspections,COUNT(DISTINCT track_id) objects,COALESCE(SUM(prior_object_id IS NOT NULL),0) reinspections,COALESCE(SUM(decision='OK'),0) ok,COALESCE(SUM(decision='NG'),0) ng,COALESCE(SUM(decision='REVIEW'),0) review FROM ("+sql+")",args).fetchone())
            rows=[dict(r) for r in db.execute(sql+" ORDER BY created DESC,object_id DESC LIMIT ? OFFSET ?",[*args,page_size,page*page_size])]
            missing=db.execute("SELECT COUNT(*) FROM inspections i LEFT JOIN inspection_index x USING(object_id) WHERE x.object_id IS NULL").fetchone()[0]
            missing+=db.execute('SELECT COUNT(*) FROM station_index_cycles WHERE error IS NOT NULL').fetchone()[0]
        return {"rows":rows,"summary":summary,"page":page,"page_size":page_size,"index_missing":missing,"filters":asdict(filters)}

    def hold(self,run_id,note):
        require(note is None or isinstance(note,str) and note.strip(),"보존 사유를 입력하세요.")
        with self.exclusive(),self.store.connect() as db:
            require(db.execute("SELECT 1 FROM captures WHERE run_id=? UNION ALL SELECT 1 FROM station_records WHERE run_id=?",(run_id,run_id)).fetchone(),"검사 기록이 없습니다.")
            db.execute("INSERT INTO capture_storage(run_id,hold_note) VALUES(?,?) ON CONFLICT(run_id) DO UPDATE SET hold_note=excluded.hold_note",(run_id,note))
        self.store.event("STORAGE_HOLD",{"run_id":run_id,"note":note})

    def export(self,destination,filters,*,images=True):
        destination=plain_path(destination); require(not destination.exists() and not destination.is_relative_to(self.root),"내보내기는 운영 저장 폴더 밖의 새 폴더를 선택하세요.")
        staging=destination.with_name(destination.name+".partial-"+uuid4().hex); staging.mkdir(parents=True)
        with self.exclusive(),self.store.connect() as db:
            sql,args=self._query(db,filters); db.execute("BEGIN")
            rows=[dict(r) for r in db.execute(sql+" ORDER BY created,object_id",args)]
            details=[]; snapshots={}; states=[]; station_cache={}
            for row in rows:
                if row['record_kind']=='station':
                    from mes_vision.station.history import export_record
                    details.append(export_record(self,row,staging,images=images,cache=station_cache,db=db)); states.append(row['decision']); continue
                run_id=row["run_id"]
                if run_id not in snapshots:
                    _,result,digest=load_snapshot(row["path"],expected_digest=row["digest"])
                    relative="evidence/"+uuid4().hex
                    if images:
                        copy_tree_verified(row["path"],staging/relative)
                        load_snapshot(staging/relative,expected_digest=digest)
                    snapshots[run_id]=(relative if images else None,result)
                relative,result=snapshots[run_id]
                obj=next(o for o in result["objects"] if o["object_id"]==row["object_id"])
                jobs=[dict(r) for r in db.execute("SELECT id,state,payload,result,error FROM analysis.jobs WHERE object_id=? ORDER BY created",(row["object_id"],))]
                events=[dict(r) for r in db.execute("SELECT created,type,data FROM events WHERE json_extract(data,'$.object_id')=? ORDER BY id",(row["object_id"],))]
                details.append({"record":row,"evidence":relative,"inspection":obj,"configuration":result["config"],"decision_policy":result["decision_details"],"vlm":jobs,"events":events})
                states.append(row["decision"])
            columns=["created","product_name","product_id","product_version","track_id","revision","object_id","run_id","decision","codes","vlm_state","robot_state","review","archived","hold_note","record_kind","session"]
            with (staging/"inspections.csv").open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.writer(stream); writer.writerow(columns)
                for row in rows:
                    row=dict(row,created=datetime.fromtimestamp(row["created"],timezone.utc).isoformat()); writer.writerow([csv_value(row.get(k)) for k in columns])
            summary={"schema_version":1,"created_utc":datetime.now(timezone.utc).isoformat(),"filters":asdict(filters),"inspections":len(rows),
                "objects":len({r['track_id'] for r in rows}),"counts":{k:states.count(k) for k in ("OK","NG","REVIEW")},"includes_images":images,
                "counts_describe_inspections_not_final_unique_objects":True,"records":details}
            write_json(staging/"report.json",summary)
            write_json(staging/"export-manifest.json",{"schema_version":1,"kind":"inspection-export","files":inventory(staging)})
        require(not destination.exists(),"대상이 이미 만들어졌습니다."); staging.rename(destination)
        self.store.event("INSPECTIONS_EXPORTED",{"path":str(destination),"count":len(rows),"filters":asdict(filters)})
        return {"path":str(destination),"inspections":len(rows),"objects":summary["objects"]}

    def policy(self):
        with self.store.connect() as db: row=db.execute("SELECT value FROM preferences WHERE key='storage_policy'").fetchone()
        return json.loads(row[0]) if row else {"days_ok":30,"days_ng":180,"days_review":365,"protect_unreviewed":True,"orphan_days":7,"max_active_gib":100.,"min_free_gib":5.,"archive_directory":None}

    def save_policy(self,value):
        require(set(value)==set(self.policy()),"보존 설정 형식 오류")
        require(all(type(value[k]) is int and 1<=value[k]<=36500 for k in ("days_ok","days_ng","days_review","orphan_days")),"보존 기간을 1~36500일로 입력하세요.")
        require(type(value["protect_unreviewed"]) is bool,"미검토 보호 설정 오류")
        require(all(type(value[k]) in {int,float} and math.isfinite(value[k]) and value[k]>0 for k in ("max_active_gib","min_free_gib")),"용량 기준은 양수여야 합니다.")
        if value["archive_directory"]:
            archive=plain_path(value["archive_directory"]); require(not archive.is_relative_to(self.root) and not self.root.is_relative_to(archive),"보관 폴더는 운영 폴더와 분리된 위치를 선택하세요.")
            value=dict(value,archive_directory=str(archive))
        with self.exclusive(),self.store.connect() as db:
            db.execute("INSERT INTO preferences VALUES('storage_policy',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",(json.dumps(value,ensure_ascii=False),))
        self.store.event("STORAGE_POLICY_SAVED",value)

    def usage(self):
        files=list(file_tree(self.root)); sizes={str(p):p.stat().st_size for p in files}; disk=shutil.disk_usage(self.root); policy=self.policy()
        active=sum(size for path,size in sizes.items() if any(Path(path).is_relative_to(self.root/name) for name in ('inspections','station-captures','station-evidence')))
        return {"runtime_bytes":sum(sizes.values()),"active_bytes":active,"free_bytes":disk.free,"total_bytes":disk.total,"files":len(files),
            "active_limit_exceeded":active>policy["max_active_gib"]*1024**3,"free_space_low":disk.free<policy["min_free_gib"]*1024**3}

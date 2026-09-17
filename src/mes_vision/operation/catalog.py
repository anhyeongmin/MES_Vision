from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import json
import math
import sqlite3
import time
from uuid import uuid4
from mes_vision.training.data import require, read_json, sha256
from mes_vision.anomaly.features import fingerprint
from mes_vision.vlm.snapshots import load_snapshot


def new_product(name="새 품목"):
    return {"id":uuid4().hex,"name":name,"version":0,"active":True,"objects":None,"defects":None,"anomaly":None,
            "policy":None,"geometry":None,"normal_reference":None,"vlm_criteria":"", "count_mode":"free","expected_count":None,
            "quality":{"version":"","validation_reference":"","blur_min":None,"brightness_min":None,"brightness_max":None},
            "grasp":{"version":"","validation_reference":"","u":.5,"v":.5,"rotation_deg":0.,"plane_z_mm":None},
            "workspace_id":None,"vlm_policy":"review_visual"}


def default_equipment():
    return {"id":"main","version":0,"camera":{"driver":"uvc","pixel_format":"MJPG","serial":"","width":1280,"height":720,"fps":30,"auto_exposure":True,"exposure":None,
            "mount_revision":"","acquisition_revision":"","robot_base_id":"","tool_frame_id":""},
            "workspace":{"width_mm":None,"height_mm":None,"roi":[],"excluded":[],"validation_reference":""},
            "tracking":{"stable_seconds":.6,"max_gap":1.,"motion_px":4.,"appearance_delta":.1},
            "calibration":None,"robot":{"port":"","baudrate":115200,"profile":None,"input_address":None,
            "holding_level":1,"pose_tolerance_mm":1.,"rotation_tolerance_deg":1.,"speed_ratio":None,"acceleration_ratio":None,"motion_validation_reference":""},"auto_sort":False}


def valid_product(p):
    require(set(p)==set(new_product()),"품목 설정 형식 오류")
    require(isinstance(p["id"],str) and p["id"] and isinstance(p["name"],str) and p["name"].strip(),"품목 이름이 필요합니다.")
    require(p["count_mode"] in {"free","fixed"},"수량 조건 오류")
    require(p["count_mode"]=="free" or type(p["expected_count"]) is int and 1<=p["expected_count"]<=100,"고정 수량을 입력하세요.")
    require(p["vlm_policy"] in {"review_visual","ng_and_review","manual"},"VLM 전달 조건 오류")
    require(type(p["active"]) is bool,"품목 사용 상태 오류")
    json.dumps(p,allow_nan=False)


def model_asset(path, *, role, threshold=.4, class_codes=None):
    import torch
    path=Path(path).resolve()
    data=torch.load(path,map_location="cpu",weights_only=True,mmap=True)
    names=data.get("args",{}).get("class_names")
    require(isinstance(names,(list,tuple)) and len(names)>0,"학습 결과에 클래스 이름이 없습니다. 제품 전용 학습 결과를 선택하세요.")
    require(data["model"]["class_embed.weight"].shape[0]==len(names)+1,"학습 모델의 클래스 수가 다릅니다.")
    result={"weights":str(path),"sha256":sha256(path),"class_names":list(names),"threshold":threshold}
    if role=="defects":
        from mes_vision.inspection.contracts import DEFECT_CODES
        codes=class_codes or {str(i):n for i,n in enumerate(names)}
        require(set(codes)=={str(i) for i in range(len(names))} and all(v in DEFECT_CODES-{"NG_UNKNOWN"} for v in codes.values()),"모든 불량 클래스에 NG01~NG06 코드를 연결하세요.")
        result["class_codes"]=codes
    return result


def readiness(product,equipment,*,verify_files=True):
    problems=[]
    if not product: return ["사용할 품목을 선택하세요."]
    try: valid_product(product)
    except Exception as exc: return [str(exc)]
    if not product["active"]: problems.append("비활성 품목입니다.")
    if product["workspace_id"]!=equipment["id"]: problems.append("품목에 작업 환경을 연결하세요.")
    policy=None
    if not product["objects"]: problems.append("물체 검출 모델을 등록하세요.")
    if not product["policy"]: problems.append("검증된 판정 기준을 등록하세요.")
    else:
        try:
            from mes_vision.decision import load_policy
            policy=load_policy(Path(product["policy"]))
            require(policy.kind=="real" and policy.validated and policy.product_id==product["id"],"품목과 연결된 실물 검증 판정 기준이 필요합니다.")
            require(policy.require_expected_count==(product["count_mode"]=="fixed"),"판정 기준과 수량 조건이 다릅니다.")
            q=product["quality"]
            require(q["version"]==policy.frame_criteria_version and q["validation_reference"],"촬영 품질 기준의 검증 기록이 필요합니다.")
            require(all(type(q[k]) in {int,float} and math.isfinite(q[k]) for k in ("blur_min","brightness_min","brightness_max"))
                    and q["blur_min"]>=0 and 0<=q["brightness_min"]<q["brightness_max"]<=255,"촬영 품질 범위를 입력하세요.")
            for rule in policy.rules:
                asset={"known_defects":"defects","anomaly":"anomaly","geometry":"geometry"}.get(rule.check_id)
                if rule.required and asset and not product.get(asset): problems.append(f"필수 검사 {rule.check_id}의 모델 또는 기준이 없습니다.")
        except Exception as exc: problems.append(str(exc))
    w=equipment["workspace"]
    if len(w["roi"])<3 or not w["validation_reference"]: problems.append("검사 영역을 지정하고 검증 기록을 입력하세요.")
    if not all(type(w[k]) in {int,float} and math.isfinite(w[k]) and w[k]>0 for k in ("width_mm","height_mm")):
        problems.append("작업 공간 크기를 입력하세요.")
    camera=equipment["camera"]
    if not all(camera[k] for k in ("serial","mount_revision","acquisition_revision")): problems.append("카메라와 촬영·설치 조건을 등록하세요.")
    if verify_files:
        for role in ("objects","defects"):
            spec=product.get(role)
            if spec:
                try: require(sha256(Path(spec["weights"]))==spec["sha256"],role+" 모델 파일이 변경됐습니다.")
                except Exception as exc: problems.append(str(exc))
        if product.get("normal_reference"):
            try:
                from mes_vision.anomaly.bank import read_normal_export
                _,reference,_=read_normal_export(product["normal_reference"],allow_synthetic=False)
                require(reference["product_id"]==product["id"],"정상 참조 품목이 다릅니다.")
            except Exception as exc: problems.append(str(exc))
    return problems


class OperationStore:
    def __init__(self,root):
        self.root=Path(root).resolve(); self.root.mkdir(parents=True,exist_ok=True)
        self.database=self.root/"operation.sqlite3"
        with self.connect() as db:
            db.executescript('''PRAGMA journal_mode=WAL;
              CREATE TABLE IF NOT EXISTS products(id TEXT,version INTEGER,data TEXT,created REAL,PRIMARY KEY(id,version));
              CREATE TABLE IF NOT EXISTS equipment(version INTEGER PRIMARY KEY,data TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS preferences(key TEXT PRIMARY KEY,value TEXT);
              CREATE TABLE IF NOT EXISTS sessions(id TEXT PRIMARY KEY,started REAL,ended REAL,product TEXT,equipment TEXT,state TEXT);
              CREATE TABLE IF NOT EXISTS captures(run_id TEXT PRIMARY KEY,path TEXT,digest TEXT,session TEXT,created REAL);
              CREATE TABLE IF NOT EXISTS inspections(object_id TEXT PRIMARY KEY,run_id TEXT,track_id TEXT,revision INTEGER,decision TEXT,
                  prior_object_id TEXT,created REAL,review TEXT,UNIQUE(track_id,revision));
              CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY,created REAL,session TEXT,track_id TEXT,type TEXT,data TEXT);
              CREATE TABLE IF NOT EXISTS inspection_index(object_id TEXT PRIMARY KEY,product_id TEXT,product_name TEXT,product_version INTEGER,equipment_version INTEGER,codes TEXT);
              CREATE TABLE IF NOT EXISTS capture_storage(run_id TEXT PRIMARY KEY,archived INTEGER NOT NULL DEFAULT 0,hold_note TEXT);
              CREATE TABLE IF NOT EXISTS storage_operations(id TEXT PRIMARY KEY,state TEXT,data TEXT,updated REAL);
              CREATE INDEX IF NOT EXISTS inspection_time ON inspections(created,object_id);
              CREATE INDEX IF NOT EXISTS inspection_track ON inspections(track_id,created);
              CREATE INDEX IF NOT EXISTS capture_session ON captures(session);
              CREATE INDEX IF NOT EXISTS event_object ON events(json_extract(data,'$.object_id'),id);
            ''')
            from mes_vision.station.history import schema
            schema(db)

    @contextmanager
    def connect(self):
        db=sqlite3.connect(self.database,timeout=5); db.row_factory=sqlite3.Row
        try: yield db; db.commit()
        except BaseException: db.rollback(); raise
        finally: db.close()

    def products(self):
        with self.connect() as db:
            return [json.loads(r[0]) for r in db.execute("SELECT p.data FROM products p JOIN (SELECT id,MAX(version) v FROM products GROUP BY id) n ON p.id=n.id AND p.version=n.v ORDER BY p.created DESC")]

    def save_product(self,p):
        valid_product(p); p=deepcopy(p)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            version=db.execute("SELECT COALESCE(MAX(version),0) FROM products WHERE id=?",(p["id"],)).fetchone()[0]
            require(version==p["version"],"다른 창에서 품목을 수정했습니다. 다시 불러오세요.")
            p["version"]=version+1
            db.execute("INSERT INTO products VALUES(?,?,?,?)",(p["id"],p["version"],json.dumps(p,ensure_ascii=False,allow_nan=False),time.time()))
        return p

    def equipment(self):
        with self.connect() as db: row=db.execute("SELECT data FROM equipment ORDER BY version DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else default_equipment()

    def save_equipment(self,e):
        e=deepcopy(e)
        from .quality import validate_equipment
        validate_equipment(e)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE"); version=db.execute("SELECT COALESCE(MAX(version),0) FROM equipment").fetchone()[0]
            require(version==e["version"],"장비 설정이 다른 곳에서 변경됐습니다.")
            e["version"]=version+1
            db.execute("INSERT INTO equipment VALUES(?,?,?)",(e["version"],json.dumps(e,ensure_ascii=False,allow_nan=False),time.time()))
        return e

    def event(self,kind,data,*,session=None,track_id=None):
        with self.connect() as db: db.execute("INSERT INTO events(created,session,track_id,type,data) VALUES(?,?,?,?,?)",
            (time.time(),session,track_id,kind,json.dumps(data,ensure_ascii=False,allow_nan=False)))

    def start_session(self,product,equipment):
        identity=uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO sessions VALUES(?,?,?,?,?,?)",(identity,time.time(),None,json.dumps(product),json.dumps(equipment),"ACTIVE"))
        return identity

    def end_session(self,identity,state="STOPPED"):
        with self.connect() as db: db.execute("UPDATE sessions SET ended=?,state=? WHERE id=?",(time.time(),state,identity))

    def recover_sessions(self):
        with self.connect() as db:
            ids=[r[0] for r in db.execute("SELECT id FROM sessions WHERE state='ACTIVE'")]
            db.execute("UPDATE sessions SET state='INTERRUPTED',ended=? WHERE state='ACTIVE'",(time.time(),))
        return ids

    def register(self,path,session,links):
        manifest,result,digest=load_snapshot(path)
        require(manifest["kind"]=="real","운영 이력에는 실제 검사 경로만 등록합니다.")
        require(result["mode"]=="live" and result["frame"]["is_live"] and result["config"].get("operation_session")==session,"현재 운전 회차의 실시간 검사만 등록할 수 있습니다.")
        records={o["object_id"]:o for o in result["objects"]}
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            existing=db.execute("SELECT digest FROM captures WHERE run_id=?",(result["run_id"],)).fetchone()
            require(existing is None or existing[0]==digest,"동일 검사 원본이 변경됐습니다.")
            db.execute("INSERT OR IGNORE INTO captures VALUES(?,?,?,?,?)",(result["run_id"],str(Path(path).resolve()),digest,session,time.time()))
            for link in links:
                o=records[link["object_id"]]
                require(result["config"].get("tracking_links",{}).get(o["object_id"])=={"track_id":link["track_id"],"revision":link["revision"]},"원본과 추적 연결이 다릅니다.")
                prior=db.execute("SELECT object_id FROM inspections WHERE track_id=? ORDER BY created DESC LIMIT 1",(link["track_id"],)).fetchone()
                prior_link=db.execute("SELECT object_id,run_id FROM inspections WHERE track_id=? AND revision=?",(link["track_id"],link["revision"])).fetchone()
                if prior_link:
                    require(prior_link[0]==o["object_id"] and prior_link[1]==result["run_id"],"동일 추적 버전에 다른 검사 결과를 덮어쓸 수 없습니다.")
                    continue
                db.execute("INSERT INTO inspections VALUES(?,?,?,?,?,?,?,?)",(o["object_id"],result["run_id"],link["track_id"],link["revision"],o["final_decision"],prior[0] if prior else None,time.time(),None))
                self.index_object(db,result,o,session)
        return result

    @staticmethod
    def index_object(db,result,obj,session):
        row=db.execute("SELECT product FROM sessions WHERE id=?",(session,)).fetchone()
        product=json.loads(row[0]) if row else {}
        config=result["config"]
        db.execute("INSERT OR REPLACE INTO inspection_index VALUES(?,?,?,?,?,?)",(obj["object_id"],config.get("product_id"),product.get("name",config.get("product_id")),
            config.get("product_version"),config.get("equipment_version"),json.dumps(obj.get("decision_details",{}).get("defect_codes",[]))))

    def history(self,query="",decision="",limit=500):
        with self.connect() as db:
            return [dict(r) for r in db.execute('''SELECT i.*,c.path,c.digest,c.session FROM inspections i JOIN captures c USING(run_id)
                WHERE (?='' OR decision=?) AND (instr(i.track_id,?)>0 OR instr(i.run_id,?)>0)
                ORDER BY i.created DESC LIMIT ?''',(decision,decision,query,query,limit))]

    def open_record(self,object_id):
        with self.connect() as db:
            row=db.execute("SELECT i.*,c.path,c.digest,c.session FROM inspections i JOIN captures c USING(run_id) WHERE object_id=?",(object_id,)).fetchone()
        if row is None:
            from mes_vision.station.history import open_record
            return open_record(self,object_id)
        manifest,result,_=load_snapshot(row["path"],expected_digest=row["digest"])
        return dict(row),manifest,result,next(o for o in result["objects"] if o["object_id"]==object_id)

    def review(self,object_id,operator,note):
        from mes_vision.station.history import get_row,review
        if get_row(self,object_id) is not None: return review(self,object_id,operator,note)
        require(operator.strip() and note.strip(),"확인자와 검토 내용을 입력하세요.")
        self.open_record(object_id)
        self.event("OPERATOR_REVIEW",{"object_id":object_id,"operator":operator,"note":note})
        with self.connect() as db: db.execute("UPDATE inspections SET review=? WHERE object_id=?",(json.dumps({"operator":operator,"note":note,"at":time.time()},ensure_ascii=False),object_id))

    def counts(self,session):
        with self.connect() as db:
            rows=db.execute('''SELECT i.decision,COUNT(*) n FROM inspections i JOIN captures c USING(run_id)
                JOIN (SELECT track_id,MAX(created) latest FROM inspections GROUP BY track_id) t
                ON i.track_id=t.track_id AND i.created=t.latest WHERE c.session=? GROUP BY i.decision''',(session,)).fetchall()
            total=db.execute("SELECT COUNT(*) FROM inspections i JOIN captures c USING(run_id) WHERE c.session=?",(session,)).fetchone()[0]
        counts={k:0 for k in ("OK","NG","REVIEW")}; counts.update({r[0]:r[1] for r in rows})
        counts["reinspections"]=total-sum(counts.values()); return counts

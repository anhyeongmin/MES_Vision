from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from mes_vision.anomaly.features import fingerprint
from mes_vision.training.data import require
from .snapshots import load_snapshot, object_record, basic_reasons

PROMPT_VERSION = "post-inspection-observation-v3-all-classes"
STATES = ("PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED", "DEFERRED", "SKIPPED_DISABLED")


def encoded(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class AnalysisQueue:
    def __init__(self, root, *, capacity=None):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "queue.sqlite3"
        require(capacity is None or type(capacity) is int and 1 <= capacity <= 1000, "capacity must be 1..1000")
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS jobs(
                    id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, snapshot_path TEXT NOT NULL,
                    snapshot_digest TEXT NOT NULL, object_id TEXT NOT NULL, kind TEXT NOT NULL,
                    state TEXT NOT NULL, priority INTEGER NOT NULL, created REAL NOT NULL, updated REAL NOT NULL,
                    payload TEXT NOT NULL, owner TEXT, token TEXT, attempt INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0, error TEXT, result TEXT);
                CREATE TABLE IF NOT EXISTS events(
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT, created REAL NOT NULL, event TEXT NOT NULL, detail TEXT);
                CREATE INDEX IF NOT EXISTS jobs_work ON jobs(state,priority,created);
                CREATE INDEX IF NOT EXISTS jobs_object_history ON jobs(object_id,created,id);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES('enabled','false')")
            db.execute("INSERT OR IGNORE INTO settings VALUES('capacity',?)", (str(capacity or 100),))
            db.execute("INSERT OR IGNORE INTO settings VALUES('schema','1')")
            require(db.execute("SELECT value FROM settings WHERE key='schema'").fetchone()[0] == "1", "unsupported queue schema")
            actual = int(db.execute("SELECT value FROM settings WHERE key='capacity'").fetchone()[0])
            require(capacity is None or actual == capacity, "existing queue has a different capacity")

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            db.execute("PRAGMA busy_timeout=5000")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally: db.close()

    @staticmethod
    def event(db, job_id, name, detail=None):
        db.execute("INSERT INTO events(job_id,created,event,detail) VALUES(?,?,?,?)", (job_id, time.time(), name, encoded(detail)))

    def enabled(self):
        with self.connect() as db:
            return db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] == "true"

    def runtime(self, value=None):
        with self.connect() as db:
            if value is not None:
                db.execute("INSERT INTO settings(key,value) VALUES('runtime',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (encoded(value),))
            row = db.execute("SELECT value FROM settings WHERE key='runtime'").fetchone()
            return json.loads(row[0]) if row else None

    def set_enabled(self, value):
        require(type(value) is bool, "enabled must be a boolean")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE settings SET value=? WHERE key='enabled'", ("true" if value else "false",))
            if not value:
                for row in db.execute("SELECT id FROM jobs WHERE state='PENDING'").fetchall():
                    self.event(db, row[0], "CANCELLED", "VLM_DISABLED")
                db.execute("UPDATE jobs SET state='CANCELLED',error='VLM_DISABLED',updated=? WHERE state='PENDING'", (time.time(),))
                db.execute("UPDATE jobs SET cancel_requested=1,updated=? WHERE state='RUNNING'", (time.time(),))
            self.event(db, None, "ENABLED" if value else "DISABLED")

    def enqueue(self, snapshot, object_id, generation):
        from .backend import GenerationConfig
        GenerationConfig(**generation)
        manifest, inspection, digest = load_snapshot(snapshot)
        record = object_record(manifest, inspection, object_id)
        priority = 0 if record["final_decision"] == "NG" or any(c["status"] == "FAIL" for c in record["checks"]) else 1 if record["final_decision"] in {None, "REVIEW"} else 2
        payload = {"prompt_version": PROMPT_VERSION, "generation": generation, "product_id": manifest["product_id"],
                   "run_id": manifest["run_id"], "frame_id": manifest["frame_id"], "base_decision": record["final_decision"],
                   "basic_reasons": basic_reasons(record)}
        request_key = fingerprint({"snapshot": digest, "object": object_id, "payload": payload})
        now, identity = time.time(), uuid4().hex
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT id FROM jobs WHERE request_key=?", (request_key,)).fetchone()
            if old: return old[0]
            enabled = db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] == "true"
            capacity = int(db.execute("SELECT value FROM settings WHERE key='capacity'").fetchone()[0])
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')").fetchone()[0]
            state = "SKIPPED_DISABLED" if not enabled else "DEFERRED" if active >= capacity else "PENDING"
            db.execute("INSERT INTO jobs(id,request_key,snapshot_path,snapshot_digest,object_id,kind,state,priority,created,updated,payload) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                       (identity, request_key, str(Path(snapshot).resolve()), digest, object_id, manifest["kind"], state, priority, now, now, encoded(payload)))
            self.event(db, identity, state)
        return identity

    def claim(self, owner):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] != "true": return None
            row = db.execute("SELECT id FROM jobs WHERE state='PENDING' ORDER BY priority,created,id LIMIT 1").fetchone()
            if row is None: return None
            token = uuid4().hex
            db.execute("UPDATE jobs SET state='RUNNING',owner=?,token=?,attempt=attempt+1,cancel_requested=0,error=NULL,updated=? WHERE id=?",
                       (owner, token, time.time(), row[0]))
            self.event(db, row[0], "RUNNING", {"token": token})
        return self.get(row[0])

    @staticmethod
    def decode(row):
        if row is None: return None
        result = dict(row)
        result["payload"] = json.loads(result["payload"])
        result["result"] = json.loads(result["result"]) if result["result"] is not None else None
        return result

    def get(self, identity):
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (identity,)).fetchone()
            require(row is not None, "unknown analysis request")
            return self.decode(row)

    def list(self, limit=200):
        require(type(limit) is int and 1 <= limit <= 1000, "invalid page limit")
        with self.connect() as db:
            return [self.decode(r) for r in db.execute("SELECT * FROM jobs ORDER BY created DESC,id LIMIT ?", (limit,)).fetchall()]

    def summary(self):
        with self.connect() as db:
            counts = {r[0]: r[1] for r in db.execute("SELECT state,COUNT(*) FROM jobs GROUP BY state")}
            return {"enabled": self.enabled(), "counts": {s: counts.get(s, 0) for s in STATES},
                    "capacity": int(db.execute("SELECT value FROM settings WHERE key='capacity'").fetchone()[0])}

    def finish(self, identity, token, state, *, result=None, error=None):
        require(state in {"COMPLETED", "FAILED", "CANCELLED", "PENDING"}, "invalid terminal/requeue state")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE id=? AND state='RUNNING' AND token=?", (identity, token)).fetchone()
            if row is None: return False
            enabled = db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] == "true"
            if row["cancel_requested"] or not enabled:
                state, result, error = "CANCELLED", None, "VLM_DISABLED" if not enabled else "USER_CANCELLED"
            db.execute("UPDATE jobs SET state=?,result=?,error=?,owner=NULL,token=NULL,updated=? WHERE id=?",
                       (state, encoded(result) if result is not None else None, error, time.time(), identity))
            self.event(db, identity, state, {"error": error, "attempt": row["attempt"]})
            return True

    def recover(self):
        """Call only while holding the single-worker lock; old completion tokens are invalidated."""
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            enabled = db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] == "true"
            for row in db.execute("SELECT * FROM jobs WHERE state='RUNNING'").fetchall():
                state = "CANCELLED" if not enabled or row["cancel_requested"] else "PENDING"
                db.execute("UPDATE jobs SET state=?,token=NULL,owner=NULL,updated=?,error='WORKER_INTERRUPTED' WHERE id=?", (state, time.time(), row["id"]))
                self.event(db, row["id"], "RECOVERED", {"state": state})

    def cancel(self, identity):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT state FROM jobs WHERE id=?", (identity,)).fetchone()
            require(row is not None, "unknown request")
            if row[0] == "PENDING":
                db.execute("UPDATE jobs SET state='CANCELLED',error='USER_CANCELLED',updated=? WHERE id=?", (time.time(), identity))
            elif row[0] == "RUNNING":
                db.execute("UPDATE jobs SET cancel_requested=1,updated=? WHERE id=?", (time.time(), identity))
            self.event(db, identity, "CANCEL_REQUESTED")

    def retry(self, identity):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            require(db.execute("SELECT value FROM settings WHERE key='enabled'").fetchone()[0] == "true", "VLM is OFF")
            row = db.execute("SELECT state FROM jobs WHERE id=?", (identity,)).fetchone()
            require(row is not None and row[0] in {"FAILED", "CANCELLED", "DEFERRED", "SKIPPED_DISABLED"}, "request cannot be retried")
            active = db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('PENDING','RUNNING')").fetchone()[0]
            capacity = int(db.execute("SELECT value FROM settings WHERE key='capacity'").fetchone()[0])
            require(active < capacity, "analysis queue is full")
            db.execute("UPDATE jobs SET state='PENDING',cancel_requested=0,error=NULL,result=NULL,updated=? WHERE id=?", (time.time(), identity))
            self.event(db, identity, "RETRY_REQUESTED")

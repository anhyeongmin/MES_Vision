from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time

from mes_vision.training.data import require


class RobotJournal:
    def __init__(self, root):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "robot.sqlite3"
        with self.db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY,value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS plans(id TEXT PRIMARY KEY, digest TEXT NOT NULL, frame_id TEXT NOT NULL UNIQUE,
                    payload TEXT NOT NULL,state TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS events(seq INTEGER PRIMARY KEY AUTOINCREMENT,created REAL NOT NULL,
                    plan_id TEXT,state TEXT NOT NULL,event TEXT NOT NULL,payload TEXT NOT NULL);
            """)
            db.execute("INSERT OR IGNORE INTO settings VALUES('state','IDLE')")
            db.execute("INSERT OR IGNORE INTO settings VALUES('schema','1')")
            require(db.execute("SELECT value FROM settings WHERE key='schema'").fetchone()[0] == "1", "unsupported robot journal schema")

    @contextmanager
    def db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback(); raise
        finally: db.close()

    def state(self):
        with self.db() as db: return db.execute("SELECT value FROM settings WHERE key='state'").fetchone()[0]

    @staticmethod
    def write(db, state, event, plan_id, payload):
        db.execute("UPDATE settings SET value=? WHERE key='state'", (state,))
        if plan_id: db.execute("UPDATE plans SET state=? WHERE id=?", (state, plan_id))
        db.execute("INSERT INTO events(created,plan_id,state,event,payload) VALUES(?,?,?,?,?)",
                   (time.time(), plan_id, state, event, json.dumps(payload, ensure_ascii=False, allow_nan=False)))

    def record(self, state, event, *, plan_id=None, payload=None):
        with self.db() as db: self.write(db, state, event, plan_id, payload)

    def reserve(self, plan):
        from dataclasses import asdict
        with self.db() as db:
            db.execute("BEGIN IMMEDIATE")
            require(db.execute("SELECT 1 FROM plans WHERE id=? OR frame_id=?", (plan.plan_id, plan.scene.frame_id)).fetchone() is None,
                    "PLAN_OR_FRAME_ALREADY_CONSUMED; recapture before another pick")
            db.execute("INSERT INTO plans VALUES(?,?,?,?,?)", (plan.plan_id, plan.digest, plan.scene.frame_id,
                       json.dumps(asdict(plan), ensure_ascii=False, allow_nan=False), "PREPARE_PICK"))
            self.write(db, "PREPARE_PICK", "PLAN_RESERVED", plan.plan_id, {"digest": plan.digest, "synthetic": plan.profile.kind == "synthetic"})

    def events(self):
        with self.db() as db:
            return [dict(r, payload=json.loads(r["payload"])) for r in db.execute("SELECT * FROM events ORDER BY seq")]

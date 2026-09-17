from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from uuid import uuid4

from mes_vision.training.data import read_json, write_json, require
from mes_vision.vlm.snapshots import save_snapshot, load_snapshot


DEFAULTS = {"product_id": "unconfigured-product", "workspace_width_mm": 0., "workspace_height_mm": 0.,
            "expected_count": 0, "threshold": .4, "recipe": None}


def validate_settings(settings):
    import math
    require(isinstance(settings, dict) and set(settings) == set(DEFAULTS), "설정 항목이 올바르지 않습니다.")
    require(isinstance(settings["product_id"], str) and bool(settings["product_id"].strip()), "품목 ID를 입력하세요.")
    for name in ("workspace_width_mm", "workspace_height_mm", "threshold"):
        value = settings[name]
        require(type(value) in {int, float} and math.isfinite(value), "설정값은 유한한 숫자여야 합니다.")
    require(0 <= settings["threshold"] <= 1, "검출 임계값 범위: 0~1")
    require(all(0 <= settings[n] <= 10000 for n in ("workspace_width_mm", "workspace_height_mm")), "작업 영역 범위: 0~10000 mm")
    require(type(settings["expected_count"]) is int and 0 <= settings["expected_count"] <= 100, "물체 수 범위: 0~100")
    require(settings["recipe"] is None or isinstance(settings["recipe"], dict), "모델 설정이 올바르지 않습니다.")
    if settings["recipe"] is not None:
        require(settings["recipe"]["product_id"] == settings["product_id"], "모델 설정의 품목 ID가 다릅니다.")
    return dict(settings)


def load_recipe(path):
    """Resolve explicit local assets once on import. Never execute config-supplied code."""
    path = Path(path).resolve()
    data = read_json(path)
    require(set(data) <= {"schema_version", "product_id", "objects", "defects", "anomaly", "policy"}
            and data.get("schema_version") == 1, "지원하지 않는 모델 설정 형식입니다.")
    require(isinstance(data.get("product_id"), str) and data["product_id"].strip(), "품목 ID가 필요합니다.")
    require(isinstance(data.get("objects"), dict), "물체 검출 모델이 필요합니다.")
    def local(value):
        require(isinstance(value, str) and value.strip(), "모델 경로가 필요합니다.")
        resolved = (path.parent / value).resolve()
        require(resolved.exists(), "파일 또는 폴더가 없습니다: " + str(resolved))
        return str(resolved)
    for role in ("objects", "defects"):
        spec = data.get(role)
        if spec is None: continue
        require(set(spec) == ({"weights", "sha256", "class_names", "threshold"} | ({"class_codes"} if role == "defects" else set())), "모델 항목을 확인하세요.")
        spec["weights"] = local(spec["weights"])
        require(isinstance(spec["class_names"], list), "class_names는 순서가 고정된 목록이어야 합니다.")
        require(isinstance(spec["sha256"], str) and len(spec["sha256"]) == 64
                and all(c in "0123456789abcdef" for c in spec["sha256"]), "SHA256 형식 오류")
        from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDefectInspector
        backend = RFDETRBackend(spec["weights"], spec["sha256"], threshold=spec["threshold"],
            training_scope="product_" + role, class_names=tuple(spec["class_names"]))
        if role == "defects": RFDETRDefectInspector(backend, class_codes={int(k): v for k, v in spec["class_codes"].items()})
    if data.get("anomaly") is not None:
        spec = data["anomaly"]
        require(set(spec) == {"bank", "criteria"}, "이상 탐지 설정 항목 오류")
        spec["bank"] = local(spec["bank"])
        if spec["criteria"]: spec["criteria"] = local(spec["criteria"])
    if data.get("policy"):
        from mes_vision.decision import load_policy
        data["policy"] = local(data["policy"])
        policy = load_policy(data["policy"])
        require(policy.kind == "real" and policy.product_id == data["product_id"], "실물 품목의 판정 규칙이 필요합니다.")
    return data


class AppStore:
    """Minimum durable UI index. Snapshot integrity remains checked on every open."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / "desktop.sqlite3"
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS runs(run_id TEXT PRIMARY KEY, created REAL NOT NULL,
                    path TEXT NOT NULL UNIQUE, digest TEXT NOT NULL, product TEXT NOT NULL,
                    kind TEXT NOT NULL, verdict TEXT NOT NULL, objects INTEGER NOT NULL);
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.database, timeout=3)
        db.row_factory = sqlite3.Row
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally: db.close()

    def settings(self):
        with self.connect() as db:
            row = db.execute("SELECT value FROM settings WHERE key='profile'").fetchone()
        return validate_settings(json.loads(row[0]) if row else dict(DEFAULTS))

    def save_settings(self, value):
        data = validate_settings(value)
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO settings VALUES('profile',?)", (json.dumps(data, ensure_ascii=False, allow_nan=False),))

    def register(self, path):
        manifest, result, digest = load_snapshot(path)
        with self.connect() as db:
            existing = db.execute("SELECT digest,path FROM runs WHERE run_id=?", (manifest["run_id"],)).fetchone()
            require(existing is None or (existing["digest"] == digest and existing["path"] == str(Path(path).resolve())), "검사 ID가 중복되거나 저장 결과가 변경됐습니다.")
            db.execute("INSERT OR IGNORE INTO runs VALUES(?,?,?,?,?,?,?,?)", (manifest["run_id"], time.time(),
                str(Path(path).resolve()), digest, manifest["product_id"], manifest["kind"], result["final_decision"] or "REVIEW", len(result["objects"])))
        return manifest, result

    def list(self, query=""):
        with self.connect() as db:
            return [dict(r) for r in db.execute("SELECT * FROM runs WHERE instr(product,?)>0 OR instr(run_id,?)>0 ORDER BY created DESC LIMIT 500", (query, query))]

    def open(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        require(row is not None, "검사 이력을 찾을 수 없습니다.")
        manifest, result, _ = load_snapshot(row["path"], expected_digest=row["digest"])
        return Path(row["path"]), manifest, result


def run_inspection(request, project_root):
    """Executed only in an owned child process; killing it cannot publish a UI result."""
    root = Path(project_root).resolve()
    settings = validate_settings(request["settings"])
    mode = request["mode"]
    require(mode in {"synthetic", "file"}, "실제 장비 운전은 아직 준비되지 않았습니다.")
    started = time.perf_counter()
    if mode == "synthetic":
        from mes_vision.decision.fixtures import make_case
        from mes_vision.decision import apply_policy
        case = make_case(mixed=True)
        identity = "desktop-synthetic-" + uuid4().hex
        frame = replace(case["frame"], frame_id=identity + ":0", session_id=identity, read_at_utc=datetime.now(timezone.utc))
        raw = case["pipeline"].run(frame)
        raw.config["desktop_settings"] = settings
        evidence = replace(case["evidence"], run_id=raw.run_id, frame_id=frame.frame_id)
        result = apply_policy(raw, case["policy"], evidence)
    else:
        from mes_vision.inputs import ImageSource
        from mes_vision.inspection import InspectionPipeline, Mode
        from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDetector, RFDETRDefectInspector
        from mes_vision.decision import load_policy, apply_policy
        from mes_vision.vlm.gpu import GpuCoordinator
        with ImageSource(Path(request["image"])) as source:
            event = source.read()
            require(event.frame is not None, "이미지를 읽을 수 없습니다.")
            frame = event.frame
        recipe = settings["recipe"]
        backends, inspectors = [], []
        # Include loading in the foreground lease; a second model must not allocate over VLM.
        with GpuCoordinator(Path(request["queue_root"]) / "gpu-coordination").foreground() as wait_ms:
            try:
                def trained(role):
                    spec = recipe[role]
                    backend = RFDETRBackend(spec["weights"], spec["sha256"], threshold=spec["threshold"],
                        training_scope="product_" + role, class_names=tuple(spec["class_names"]))
                    backends.append(backend); backend.load()
                    return backend
                if recipe is None:
                    manifest = read_json(root / "models/manifest.json")
                    detector_backend = RFDETRBackend(root / "models/rf-detr-small.pth", manifest["sha256"],
                        threshold=settings["threshold"], training_scope="coco_general")
                    backends.append(detector_backend); detector_backend.load()
                else:
                    detector_backend = trained("objects")
                    if recipe.get("defects"):
                        inspectors.append(RFDETRDefectInspector(trained("defects"),
                            class_codes={int(k): v for k, v in recipe["defects"]["class_codes"].items()}))
                    if recipe.get("anomaly"):
                        from mes_vision.anomaly.features import DinoFeatures
                        from mes_vision.anomaly.scoring import AnomalyEngine, AnomalyInspector
                        spec = recipe["anomaly"]
                        bank_meta=read_json(Path(spec['bank'])/'bank.json')
                        extractor=DinoFeatures(root,image_size=bank_meta['feature_signature']['preprocessing']['size'],
                            max_batch_size=bank_meta['feature_signature'].get('batch_limit',1))
                        backends.append(extractor)
                        anomaly_engine=AnomalyEngine(spec["bank"], extractor,
                            product_id=settings["product_id"], criteria=spec["criteria"])
                        backends.append(anomaly_engine); anomaly_engine.prepare()
                        inspectors.append(AnomalyInspector(anomaly_engine))
                policy = load_policy(recipe["policy"] if recipe and recipe.get("policy") else root / "configs/decision/unconfigured.json")
                result = InspectionPipeline(RFDETRDetector(detector_backend), inspectors, mode=Mode.MODEL_FILE,
                    product_id=settings["product_id"], expected_count=settings["expected_count"] or None).run(frame)
                result.config["gpu_wait_ms"] = wait_ms
                result.config["desktop_settings"] = settings
                result = apply_policy(result, policy)
            finally:
                for backend in reversed(backends): backend.close()
    output = Path(request["output"])
    save_snapshot(output, frame, result, kind="synthetic" if mode == "synthetic" else "real")
    write_json(output / "desktop-timing.json", {"total_ms": (time.perf_counter()-started)*1000,
        "inference_pipeline_ms": result.elapsed_ms, "includes_model_load": mode == "file", "production_ready": False})
    return output

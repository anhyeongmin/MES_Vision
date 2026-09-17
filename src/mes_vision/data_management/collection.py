from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from uuid import uuid4

from filelock import FileLock
from PIL import Image

from mes_vision.inputs import ImageSource, InputStatus
from mes_vision.inspection import Box
from mes_vision.training.data import read_json, require, sha256, write_json

SPLITS = ("unassigned", "train", "valid", "test", "challenge")
CONDITIONS = ("UNREVIEWED", "NORMAL", "KNOWN_NG", "UNKNOWN_NG", "UNCERTAIN")
CODES = {"NG01": "누락", "NG02": "돌출·버", "NG03": "균열", "NG04": "변형", "NG05": "구멍 불량", "NG06": "표면 결함"}


def utc():
    return datetime.now(timezone.utc).isoformat()


def new_object(specimen_id, bbox):
    return {"id": uuid4().hex, "specimen_id": specimen_id, "bbox": list(bbox),
            "condition": "UNREVIEWED", "note": "", "defects": []}


def validate_record(record, *, complete=False):
    width, height = record["width"], record["height"]
    require(type(width) is int and type(height) is int and width > 0 and height > 0, "이미지 크기가 잘못되었습니다")
    require(record["split"] in SPLITS, "분할이 잘못되었습니다")
    require(isinstance(record["capture_session_id"], str) and record["capture_session_id"].strip(), "촬영 회차 ID가 필요합니다")
    require(type(record["excluded"]) is bool and type(record["empty_confirmed"]) is bool, "확인 상태가 잘못되었습니다")
    if record["excluded"]:
        require(record["exclude_reason"].strip(), "제외 사유가 필요합니다")
    require(not (record["empty_confirmed"] and record["objects"]), "물체가 있으면 빈 작업대로 표시할 수 없습니다")
    ids, specimens, defect_ids = set(), set(), set()
    for obj in record["objects"]:
        require(obj["id"] not in ids and obj["specimen_id"].strip() and obj["specimen_id"] not in specimens, "물체 ID 또는 실물 ID가 중복되거나 비어 있습니다")
        ids.add(obj["id"])
        specimens.add(obj["specimen_id"])
        box = Box(*obj["bbox"])
        require(box.clip(width, height) == box, "물체 상자가 원본 이미지 밖에 있습니다")
        require(obj["condition"] in CONDITIONS, "물체 상태가 잘못되었습니다")
        for defect in obj["defects"]:
            require(defect["id"] not in defect_ids and defect["code"] in CODES, "불량 ID 또는 코드가 잘못되었습니다")
            defect_ids.add(defect["id"])
            if defect["bbox"] is not None:
                region = Box(*defect["bbox"])
                require(region.x1 >= box.x1 and region.y1 >= box.y1 and region.x2 <= box.x2 and region.y2 <= box.y2, "불량 영역은 선택한 물체 상자 안에 있어야 합니다")
            else:
                require(defect["note"].strip(), "영역 없는 불량에는 관찰 설명이 필요합니다")
        require(obj["condition"] != "NORMAL" or not obj["defects"], "정상 물체에 불량 라벨이 있습니다")
        require(obj["condition"] != "KNOWN_NG" or obj["defects"], "알려진 불량은 불량 라벨이 필요합니다")
        if obj["condition"] in {"UNKNOWN_NG", "UNCERTAIN"}:
            require(obj["note"].strip(), "미등록 불량·판독 불확실에는 관찰 설명이 필요합니다")
        if complete:
            require(obj["condition"] != "UNREVIEWED", "모든 물체의 상태를 확인하세요")
    if complete and not record["excluded"]:
        require(record["objects"] or record["empty_confirmed"], "물체가 없으면 빈 작업대임을 명시적으로 확인하세요")


def validate_groups(records):
    assignments = {"session": {}, "specimen": {}, "pixels": {}}
    for record in records:
        if record["excluded"] or record["split"] == "unassigned":
            continue
        for kind, values in (("session", [record["capture_session_id"]]),
                             ("specimen", [o["specimen_id"] for o in record["objects"]]),
                             ("pixels", [record["pixel_sha256"]])):
            for value in values:
                previous = assignments[kind].setdefault(value, record["split"])
                require(previous == record["split"], f"{kind} '{value}'가 {previous}/{record['split']}에 섞입니다. 같은 실물·회차·사진은 같은 분할로 묶으세요")


class Collection:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.data = read_json(self.root / "collection.json")
        require(self.data["schema_version"] == 1, "지원하지 않는 수집 프로젝트 버전입니다")

    @classmethod
    def create(cls, root, product_id, *, kind="real"):
        require(isinstance(product_id, str) and product_id.strip(), "품목 ID가 필요합니다")
        require(kind in {"real", "synthetic"}, "real 또는 synthetic 모드가 필요합니다")
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=False)
        for folder in ("images", "originals", "history"):
            (root / folder).mkdir()
        write_json(root / "collection.json", {"schema_version": 1, "id": uuid4().hex, "revision": 0,
                   "product_id": product_id.strip(), "kind": kind, "created_at_utc": utc(), "records": []})
        return cls(root)

    def reload(self):
        self.data = read_json(self.root / "collection.json")

    def image_path(self, record):
        path = (self.root / record["image"]).resolve()
        require(path.is_relative_to(self.root / "images"), "이미지 경로가 수집 프로젝트 밖에 있습니다")
        return path

    def record(self, identity):
        return deepcopy(next(r for r in self.data["records"] if r["id"] == identity))

    def _commit(self, updated):
        for record in updated["records"]:
            validate_record(record, complete=record["reviewed"])
        validate_groups(updated["records"])
        with FileLock(str(self.root / ".collection.lock"), timeout=3):
            current = read_json(self.root / "collection.json")
            require(current["revision"] == self.data["revision"], "다른 창에서 수정했습니다. 저장하지 않은 내용을 보관하고 프로젝트를 다시 여세요")
            write_json(self.root / "history" / f"revision-{current['revision']:06d}.json", current)
            updated["revision"] = current["revision"] + 1
            updated["updated_at_utc"] = utc()
            write_json(self.root / "collection.json", updated)
        self.data = updated

    def import_image(self, path, session_id):
        require(isinstance(session_id, str) and session_id.strip(), "촬영 회차 ID가 필요합니다")
        session_id = session_id.strip()
        original = Path(path).resolve()
        identity = uuid4().hex
        owned = self.root / "originals" / (identity + original.suffix.lower())
        shutil.copyfile(original, owned)
        with ImageSource(owned) as source:
            event = source.read()
            require(event.status == InputStatus.FRAME, f"사진을 읽을 수 없습니다: {event.message}")
            frame = event.frame
        image_path = self.root / "images" / (identity + ".png")
        Image.fromarray(frame.rgb).save(image_path)
        split = next((r["split"] for r in self.data["records"] if r["capture_session_id"] == session_id and not r["excluded"]), "unassigned")
        record = {"id": identity, "source_name": original.name, "source_uri": original.as_uri(),
                  "original": owned.relative_to(self.root).as_posix(), "original_sha256": sha256(owned),
                  "image": image_path.relative_to(self.root).as_posix(), "image_sha256": sha256(image_path),
                  "pixel_sha256": hashlib.sha256(str((frame.width, frame.height)).encode() + frame.rgb.tobytes()).hexdigest(),
                  "width": frame.width, "height": frame.height, "transformations": list(frame.transformations),
                  "capture_session_id": session_id.strip(), "split": split, "objects": [],
                  "empty_confirmed": False, "excluded": False, "exclude_reason": "", "reviewed": False,
                  "reviewer": None, "reviewed_at_utc": None, "imported_at_utc": utc()}
        updated = deepcopy(self.data)
        updated["records"].append(record)
        self._commit(updated)
        return identity

    def save_record(self, record):
        previous = self.record(record["id"])
        # Only annotation fields can be edited; copied image identity cannot be replaced.
        allowed = {"objects", "empty_confirmed", "excluded", "exclude_reason"}
        saved = deepcopy(previous)
        saved.update({key: deepcopy(record[key]) for key in allowed})
        saved.update(reviewed=False, reviewer=None, reviewed_at_utc=None)
        updated = deepcopy(self.data)
        updated["records"] = [saved if r["id"] == saved["id"] else r for r in updated["records"]]
        self._commit(updated)

    def review(self, identity, reviewer):
        require(isinstance(reviewer, str) and reviewer.strip(), "검토자 이름이 필요합니다")
        updated = deepcopy(self.data)
        record = next(r for r in updated["records"] if r["id"] == identity)
        validate_record(record, complete=True)
        record.update(reviewed=True, reviewer=reviewer.strip(), reviewed_at_utc=utc())
        self._commit(updated)

    def assign_session(self, session_id, split):
        require(split in SPLITS, "분할이 잘못되었습니다")
        updated = deepcopy(self.data)
        found = False
        for record in updated["records"]:
            if record["capture_session_id"] == session_id:
                record["split"] = split
                found = True
        require(found, "촬영 회차를 찾을 수 없습니다")
        self._commit(updated)

    def inventory(self):
        return {"product_id": self.data["product_id"], "kind": self.data["kind"], "revision": self.data["revision"],
                "captures": len(self.data["records"]), "reviewed": sum(r["reviewed"] for r in self.data["records"]),
                "objects": sum(len(r["objects"]) for r in self.data["records"]),
                "splits": {s: sum(r["split"] == s for r in self.data["records"]) for s in SPLITS},
                "conditions": {c: sum(o["condition"] == c for r in self.data["records"] for o in r["objects"]) for c in CONDITIONS}}

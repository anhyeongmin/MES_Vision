from __future__ import annotations

from copy import deepcopy
import math
from pathlib import Path
import shutil
from uuid import uuid4

from PIL import Image

from mes_vision.inspection import Box
from mes_vision.training.data import require, sha256, validate_dataset, write_json
from .collection import Collection, CODES, validate_record, validate_groups, utc


def crop_bounds(obj):
    x1, y1, x2, y2 = obj["bbox"]
    return (math.floor(x1), math.floor(y1), math.ceil(x2), math.ceil(y2))


def clean_crop(record, obj):
    bounds = Box(*crop_bounds(obj))
    require(not any(other["id"] != obj["id"] and bounds.overlaps(Box(*other["bbox"])) for other in record["objects"]),
            f"다른 물체가 잘라낼 영역에 들어갑니다: {record['source_name']} / {obj['specimen_id']}")
    return crop_bounds(obj)


def export_collection(collection: Collection, output, role, *, defect_codes=()):
    require(role in {"object_detector", "known_defect_detector", "normal_bank", "challenge"}, "내보내기 역할이 잘못되었습니다")
    if role == "known_defect_detector":
        require(defect_codes and len(set(defect_codes)) == len(defect_codes) and all(c in CODES for c in defect_codes), "학습할 불량 코드를 중복 없이 명시하세요")
    snapshot = deepcopy(collection.data)
    validate_groups(snapshot["records"])
    records, excluded = [], []
    for record in snapshot["records"]:
        validate_record(record, complete=record["reviewed"])
        if record["excluded"]:
            excluded.append({"capture_id": record["id"], "reason": record["exclude_reason"]})
            continue
        require(record["reviewed"], f"검토되지 않은 사진: {record['source_name']}")
        require(record["split"] != "unassigned", f"분할을 지정하세요: {record['source_name']}")
        if any(o["condition"] in {"UNKNOWN_NG", "UNCERTAIN"} for o in record["objects"]):
            require(record["split"] == "challenge", "미등록 불량·불확실 사진은 촬영 회차와 관련 실물을 challenge로 분리하세요")
        require(sha256(collection.image_path(record)) == record["image_sha256"], "수집 이미지가 변경되었습니다. 라벨 좌표를 다시 확인하세요")
        records.append(record)
    output = Path(output).resolve()
    require(not output.exists(), "기존 출력 폴더를 덮어쓸 수 없습니다")
    require(not output.is_relative_to(collection.root), "내보내기 폴더는 수집 프로젝트 밖에 지정하세요")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ("." + output.name + "-building-" + uuid4().hex[:8])
    require(staging.resolve().parent == output.parent, "내보내기 임시 경로가 잘못되었습니다")
    staging.mkdir()
    report = {"status": "BUILDING", "role": role, "created_at_utc": utc(), "collection_id": snapshot["id"],
              "collection_revision": snapshot["revision"], "kind": snapshot["kind"], "product_id": snapshot["product_id"],
              "excluded": excluded, "items": [], "production_ready": False}
    selected_splits = {"train", "valid", "test"} if role in {"object_detector", "known_defect_detector"} else {"train"} if role == "normal_bank" else {"challenge"}
    report["selection"] = {"splits": sorted(selected_splits), "condition": "NORMAL" if role == "normal_bank" else "all reviewed in selected splits"}
    for record in records:
        if record["split"] not in selected_splits:
            report["excluded"].append({"capture_id": record["id"], "reason": "SPLIT_NOT_SELECTED_FOR_ROLE", "split": record["split"]})
    write_json(staging / "collection-snapshot.json", snapshot)
    try:
        if role in {"object_detector", "known_defect_detector"}:
            names = [snapshot["product_id"]] if role == "object_detector" else list(defect_codes)
            categories = [{"id": i+1, "name": name, "supercategory": "product"} for i, name in enumerate(names)]
            ids = {name: i+1 for i, name in enumerate(names)}
            for split in ("train", "valid", "test"):
                folder = staging / split
                folder.mkdir()
                images, annotations = [], []
                for record in records:
                    if record["split"] != split:
                        continue
                    with Image.open(collection.image_path(record)) as image:
                        candidates = [None] if role == "object_detector" else record["objects"]
                        for obj in candidates:
                            if obj is not None and obj["condition"] == "KNOWN_NG" and any(d["bbox"] is None or d["code"] not in ids for d in obj["defects"]):
                                report["excluded"].append({"capture_id": record["id"], "object_id": obj["id"], "reason": "NON_LOCALIZABLE_OR_UNSELECTED_DEFECT; not converted to normal"})
                                continue
                            image_id = len(images) + 1
                            name = f"{record['id']}{'-'+obj['id'] if obj else ''}.png"
                            if obj is None:
                                image.save(folder / name)
                                width, height = image.size
                                boxes = [(o["bbox"], 1) for o in record["objects"]]
                                specimens = [o["specimen_id"] for o in record["objects"]]
                                bounds = [0, 0, width, height]
                            else:
                                bounds = clean_crop(record, obj)
                                cropped = image.crop(bounds)
                                cropped.save(folder / name)
                                width, height = cropped.size
                                boxes = [([d["bbox"][0]-bounds[0], d["bbox"][1]-bounds[1], d["bbox"][2]-bounds[0], d["bbox"][3]-bounds[1]], ids[d["code"]]) for d in obj["defects"]]
                                specimens = [obj["specimen_id"]]
                            images.append({"id": image_id, "file_name": name, "width": width, "height": height,
                                           "specimen_ids": specimens, "capture_session_id": record["capture_session_id"]})
                            for box, category in boxes:
                                x1, y1, x2, y2 = box
                                annotations.append({"id": len(annotations)+1, "image_id": image_id, "category_id": category,
                                                    "bbox": [x1, y1, x2-x1, y2-y1], "area": (x2-x1)*(y2-y1), "iscrowd": 0})
                            report["items"].append({"file": f"{split}/{name}", "capture_id": record["id"], "object_id": obj["id"] if obj else None,
                                                    "crop_bounds_original": bounds, "image_sha256": sha256(folder / name)})
                write_json(folder / "_annotations.coco.json", {"info": {"description": "Reviewed MES Vision export"}, "licenses": [],
                           "images": images, "annotations": annotations, "categories": categories})
            write_json(staging / "dataset.json", {"schema_version": 1, "role": role, "kind": snapshot["kind"], "product_id": snapshot["product_id"],
                       "collection_id": snapshot["id"], "collection_revision": snapshot["revision"]})
            validation = validate_dataset(staging)
            report["dataset_fingerprint"] = validation["fingerprint"]
            report["splits"] = validation["splits"]
        else:
            folder = staging / "images"
            folder.mkdir()
            for record in records:
                if role == "normal_bank" and record["split"] != "train":
                    continue
                if role == "challenge" and record["split"] != "challenge":
                    continue
                with Image.open(collection.image_path(record)) as image:
                    for obj in record["objects"]:
                        if role == "normal_bank" and obj["condition"] != "NORMAL":
                            report["excluded"].append({"capture_id": record["id"], "object_id": obj["id"], "reason": "NOT_REVIEWED_NORMAL"})
                            continue
                        bounds = clean_crop(record, obj)
                        name = f"{record['id']}-{obj['id']}.png"
                        image.crop(bounds).save(folder / name)
                        report["items"].append({"file": f"images/{name}", "capture_id": record["id"], "object": obj,
                                                "capture_session_id": record["capture_session_id"], "split": record["split"],
                                                "crop_bounds_original": bounds, "image_sha256": sha256(folder / name)})
            require(report["items"], "내보낼 대상이 없습니다. 정상 등록은 train의 검토된 NORMAL, challenge는 분리된 사진만 사용합니다")
            report["purpose"] = "normal reference preparation; feature bank not built" if role == "normal_bank" else "held-out unknown/uncertain challenge; not supervised training"
        report["status"] = "COMPLETE"
        write_json(staging / "export.json", report)
        require(not output.exists(), "출력 폴더가 작업 중 생성되었습니다")
        staging.rename(output)
    except Exception as exc:
        report.update(status="FAILED", error=str(exc))
        write_json(staging / "export.json", report)
        raise ValueError(f"내보내기 실패: {exc}. 점검 파일: {staging}") from exc
    return report

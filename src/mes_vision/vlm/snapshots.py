from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import numpy as np
from PIL import Image

from mes_vision.anomaly.bank import read_normal_export, safe_image
from mes_vision.anomaly.features import fingerprint
from mes_vision.inspection import Box, Mode
from mes_vision.training.data import read_json, require, sha256, write_json


def save_snapshot(output, frame, result, *, kind, normal_export=None, reference_index=0, criteria=None):
    """Persist evidence before enqueue; the supplied result is never changed."""
    require(kind in {"real", "synthetic"}, "snapshot kind required")
    require(result.mode != Mode.SIMULATION or kind == "synthetic", "simulation must remain synthetic")
    require(result.frame == frame.metadata(), "inspection/frame identity mismatch")
    product = result.config.get("product_id")
    require(isinstance(product, str) and product.strip(), "snapshot product_id required")
    require(criteria is None or isinstance(criteria, dict), "inspection criteria must be an object or null")
    output = Path(output).resolve()
    require(not output.exists(), "snapshot output exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ("." + output.name + "-building-" + uuid4().hex[:8])
    staging.mkdir()
    Image.fromarray(frame.rgb).save(staging / "frame.png")
    write_json(staging / "inspection.json", result.to_dict())
    objects = []
    for index, obj in enumerate(result.objects):
        box = Box(**obj.crop["bounds_original_xyxy"])
        bounds = [box.x1, box.y1, box.x2, box.y2]
        require(all(float(x).is_integer() for x in bounds) and box.clip(frame.width, frame.height) == box, "invalid saved crop bounds")
        x1, y1, x2, y2 = map(int, bounds)
        name = f"object-{index:04d}.png"
        Image.fromarray(frame.rgb[y1:y2, x1:x2].copy()).save(staging / name)
        objects.append({"object_id": obj.object_id, "file": name, "bounds_original": bounds, "image_size": [x2-x1, y2-y1]})
    require(len({o["object_id"] for o in objects}) == len(objects), "duplicate object IDs")
    reference = None
    if normal_export is not None:
        root, report, _ = read_normal_export(normal_export, allow_synthetic=kind == "synthetic")
        require(report["product_id"] == product and report["kind"] == kind, "reference product/kind mismatch")
        require(type(reference_index) is int and 0 <= reference_index < len(report["items"]), "invalid reference index")
        item = report["items"][reference_index]
        with Image.open(safe_image(root, item["file"])) as image:
            image.save(staging / "reference.png")
        reference = {"file": "reference.png", "collection_id": report["collection_id"],
                     "collection_revision": report["collection_revision"], "source": item}
    manifest = {"schema_version": 1, "snapshot_id": uuid4().hex, "run_id": result.run_id, "frame_id": frame.frame_id,
                "product_id": product, "kind": kind, "objects": objects, "reference": reference, "criteria": criteria,
                "files": {p.name: sha256(p) for p in staging.iterdir() if p.is_file()}}
    write_json(staging / "snapshot.json", manifest)
    load_snapshot(staging)
    require(not output.exists(), "snapshot output created concurrently")
    staging.rename(output)
    return manifest


def load_snapshot(directory, *, expected_digest=None):
    root = Path(directory).resolve()
    manifest = read_json(root / "snapshot.json")
    digest = fingerprint(manifest)
    require(expected_digest is None or digest == expected_digest, "snapshot manifest changed after enqueue")
    require(manifest["schema_version"] == 1 and manifest["kind"] in {"real", "synthetic"}, "invalid snapshot schema/kind")
    expected = {"frame.png", "inspection.json"} | {o["file"] for o in manifest["objects"]}
    if manifest["reference"] is not None: expected.add(manifest["reference"]["file"])
    require(set(manifest["files"]) == expected, "snapshot file list mismatch")
    for name, digest_value in manifest["files"].items():
        require(Path(name).name == name and "/" not in name and "\\" not in name and ":" not in name, "invalid snapshot file path")
        path = (root / name).resolve()
        require(path.parent == root and sha256(path) == digest_value, "snapshot file changed or escaped directory")
    inspection = read_json(root / "inspection.json")
    require(inspection["run_id"] == manifest["run_id"] and inspection["frame"]["frame_id"] == manifest["frame_id"]
            and inspection["config"]["product_id"] == manifest["product_id"], "saved inspection identity mismatch")
    records = {o["object_id"]: o for o in inspection["objects"]}
    require(len(records) == len(inspection["objects"]) == len(manifest["objects"]), "snapshot object count mismatch")
    require(len({o["object_id"] for o in manifest["objects"]}) == len(records), "duplicate snapshot object ID")
    for item in manifest["objects"]:
        record = records[item["object_id"]]
        box = Box(**record["crop"]["bounds_original_xyxy"])
        require(item["bounds_original"] == [box.x1, box.y1, box.x2, box.y2], "saved object bounds mismatch")
        with Image.open(root / item["file"]) as image:
            require(image.mode == "RGB" and list(image.size) == item["image_size"], "saved crop image changed")
        for check in record["checks"]:
            require(check["frame_id"] == manifest["frame_id"] and check["object_id"] == item["object_id"], "check belongs to another frame/object")
    return manifest, inspection, digest


def object_record(manifest, inspection, object_id):
    require(object_id in {o["object_id"] for o in manifest["objects"]}, "object not in saved snapshot")
    return next(o for o in inspection["objects"] if o["object_id"] == object_id)


def basic_reasons(record):
    labels = {"known_defects": "알려진 불량 검사", "anomaly": "이상 탐지", "geometry": "형상 검사"}
    reasons = []
    for check in record["checks"]:
        if check["check_id"] == "vlm": continue
        findings = [{"label": f["label"], "defect_code": f["defect_code"], "original_box": f["original_box"], "score": f.get("score")}
                    for f in check["findings"]]
        reasons.append({"check_id": check["check_id"], "name": labels.get(check["check_id"], check["check_id"]),
                        "status": check["status"], "findings": findings, "raw_score": check["raw_score"],
                        "criteria_version": check["criteria_version"], "messages": check["messages"]})
    return reasons

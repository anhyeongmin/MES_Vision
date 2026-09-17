"""Matched v3 synthetic comparison. All thresholds are chosen on validation only."""
from pathlib import Path
import argparse
import gc
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ash_training import collect, collect_predicted_crops, draw_review
from mes_vision.synthetic.detection_review import summarize, select_threshold
from mes_vision.training.data import read_json, write_json, sha256, require, validate_dataset
from mes_vision.training.jobs import verified_artifact

DATA = ROOT / "datasets/ash-synthetic-reinforced-v3"
BATCH = ROOT / "artifacts/training/ash-reinforced-v3"
OLD = ROOT / "artifacts/training/ash-synthetic-v2-first"
LABELS = [f"NG{i:02d}" for i in range(1, 7)]
FINGERPRINT = "481913106444a6953cf9505ccc392f607b73a982de43feb9e0f256a938014033"


def xyxy(box):
    x, y, w, h = box
    return [x, y, x+w, y+h]


def translated_targets(coco, info):
    names = {c["id"]: c["name"] for c in coco["categories"]}
    left, top, _, _ = info["source_crop_xyxy"]
    rows = []
    for ann in coco["annotations"]:
        if ann["image_id"] == info["id"]:
            x, y, r, b = xyxy(ann["bbox"])
            rows.append({"label": names[ann["category_id"]], "box": [x+left, y+top, r+left, b+top]})
    return rows


def assert_matched(left, right):
    require(len(left) == len(right), "comparison image count differs")
    require(len({r["file_name"] for r in left}) == len(left), "duplicate comparison image")
    for a, b in zip(left, right, strict=True):
        for key in ("file_name", "image_sha256", "targets"):
            require(a[key] == b[key], f"comparison {key} differs: {a['file_name']}")


def prepare(output):
    from PIL import Image
    require(validate_dataset(DATA / "package/defect-crops")["fingerprint"] == FINGERPRINT, "dataset changed")
    output.mkdir(parents=True, exist_ok=False)
    files = []
    for split in ("valid", "test"):
        directory = output / "full-frames" / split
        directory.mkdir(parents=True)
        coco = read_json(DATA / "package/defect-crops" / split / "_annotations.coco.json")
        full = {"images": [], "annotations": [], "categories": coco["categories"]}
        names = {c["name"]: c["id"] for c in coco["categories"]}
        seen = set()
        for info in coco["images"]:
            source_id = info["source_render_id"]
            require(source_id not in seen and info["source_origin"] == "v3", "not a unique new image")
            seen.add(source_id)
            source = DATA / "raw" / (source_id + ".png")
            quality = read_json(DATA / "quality" / (source_id + ".json"))
            digest = sha256(source)
            require(digest == info["source_image_sha256"] == quality["image_sha256"], "source hash differs")
            require(quality["accepted"] and len(quality["objects"]) == 1, "invalid source quality")
            original_targets = [{"label": d["code"], "box": xyxy(d["bbox"])} for d in quality["defects"]]
            require(translated_targets(coco, info) == original_targets, "crop targets differ from raw targets")
            with Image.open(source) as image:
                width, height = image.size
            target = directory / source.name
            shutil.copy2(source, target)
            require(sha256(target) == digest, "copy differs")
            full["images"].append({"id": info["id"], "file_name": source.name,
                "width": width, "height": height, "source_sha256": digest,
                "object_box": xyxy(quality["objects"][0]["bbox"]), "crop_file_name": info["file_name"]})
            for defect in quality["defects"]:
                full["annotations"].append({"id": len(full["annotations"])+1, "image_id": info["id"],
                    "category_id": names[defect["code"]], "bbox": defect["bbox"]})
            files.append({"path": target.relative_to(output).as_posix(), "sha256": digest})
        path = directory / "_annotations.coco.json"
        write_json(path, full)
        files.append({"path": path.relative_to(output).as_posix(), "sha256": sha256(path)})
    write_json(output / "inputs.json", {"dataset_fingerprint": FINGERPRINT, "files": files,
        "protocol_sha256": sha256(BATCH / "protocol.json"), "scope": "validation/test views only; no training data"})
    print("PREPARED matched full-frame validation/test views", flush=True)


def backend_for(run, scope):
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    manifest, model = read_json(run / "run.json"), read_json(run / "model.json")
    require(manifest["status"] in ("COMPLETED", "STOPPED"), "model training still active or failed")
    weight = verified_artifact(run, manifest["checkpoints"]["inference"])
    labels = [c["name"] for c in model["categories"]]
    require(labels == (LABELS if scope == "product_defects" else ["ASH"]), "class mapping differs")
    backend = RFDETRBackend(weight, sha256(weight), threshold=.05, training_scope=scope,
                           class_names=tuple(labels), inference_profile="standard", max_batch_size=1)
    return backend, model


def close_backend(backend):
    import torch
    backend.close()
    gc.collect()
    torch.cuda.empty_cache()


def objects(backend, full, split):
    import numpy as np
    from PIL import Image
    rows = []
    for info in read_json(full / split / "_annotations.coco.json")["images"]:
        path = full / split / info["file_name"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        predictions = backend.predict_rgb(rgb)
        rows.append({"file_name": path.name, "image_sha256": sha256(path),
            "targets": [{"label": "ASH", "box": info["object_box"]}],
            "predictions": [{"label": p.label, "score": p.score,
                "box": [p.box.x1, p.box.y1, p.box.x2, p.box.y2]} for p in predictions]})
    return rows


def counts(rows, threshold):
    report = summarize(rows, threshold, LABELS)
    if rows and "crop_status" in rows[0]:
        report["unconfirmed_crop_images"] = sum(r["crop_status"] != "CONFIRMED" for r in rows)
    return report


def evaluate(output):
    require(not (output / "thresholds.json").exists(), "evaluation already started; use a new output")
    inputs = read_json(output / "inputs.json")
    require(sha256(BATCH / "protocol.json") == inputs["protocol_sha256"], "protocol changed")
    for item in inputs["files"]:
        require(sha256(output / item["path"]) == item["sha256"], "evaluation input changed")
    require(validate_dataset(DATA / "package/defect-crops")["fingerprint"] == FINGERPRINT, "dataset changed")
    protocol = read_json(BATCH / "protocol.json")
    require(protocol["comparison"]["object_threshold"] == .2, "object threshold changed")
    stress_files = []
    for row in read_json(DATA / "package/crop-ledger.json"):
        if row["path"].startswith("crop-stress/"):
            require(sha256(DATA / "package" / row["path"]) == row["sha256"], "stress image changed")
            stress_files.append({"path": row["path"], "sha256": row["sha256"]})
    require(len(stress_files) == 756, "stress count changed")
    for split in ("valid", "test"):
        path = DATA / "package/crop-stress" / split / "_annotations.coco.json"
        stress_files.append({"path": path.relative_to(DATA / "package").as_posix(), "sha256": sha256(path)})
    write_json(output / "stress-inputs.json", stress_files)
    write_json(output / "evaluation-source-hashes.json", {str(p.relative_to(ROOT)): sha256(p) for p in (
        Path(__file__), ROOT / "scripts/analyze_ash_training.py",
        ROOT / "src/mes_vision/synthetic/detection_review.py",
        ROOT / "src/mes_vision/inspection/rfdetr_adapter.py")})
    full = output / "full-frames"
    backend, object_model = backend_for(OLD / "detail-objects", "product_objects")
    object_rows = {}
    try:
        backend.load()
        for split in ("valid", "test"):
            object_rows[split] = objects(backend, full, split)
            write_json(output / f"objects-{split}.json", object_rows[split])
            write_json(output / f"objects-{split}-counts.json", summarize(object_rows[split], .2, ["ASH"]))
            print(f"OBJECTS {split} completed", flush=True)
    finally:
        close_backend(backend)
    models = {"baseline": OLD / "detail-defects", "reinforced": BATCH / "detail-defects"}
    thresholds, provenance = {}, {}
    # Seal thresholds for BOTH models before either defect model sees the test split.
    for name, run in models.items():
        destination = output / name
        destination.mkdir(exist_ok=False)
        backend, model = backend_for(run, "product_defects")
        provenance[name] = {"run": str(run), "weights": model["weights"], "trained_dataset": model["dataset_fingerprint"]}
        thresholds[name] = {}
        try:
            backend.load()
            routes = {
                "ground_truth_crop": lambda: collect(backend, DATA / "package/defect-crops", "valid", model["categories"]),
                "predicted_crop_chain": lambda: collect_predicted_crops(backend, full, "valid", object_rows["valid"], .2, model["categories"]),
            }
            for route, callback in routes.items():
                rows = callback()
                threshold, sweep = select_threshold(rows, LABELS)
                thresholds[name][route] = threshold
                write_json(destination / f"{route}-valid-predictions.json", rows)
                write_json(destination / f"{route}-threshold.json", {"split": "valid", "selected": threshold, "sweep": sweep})
                print(f"VALID {name} {route}: threshold {threshold}", flush=True)
            stress = collect(backend, DATA / "package/crop-stress", "valid", model["categories"])
            write_json(destination / "crop_stress-valid-predictions.json", stress)
            write_json(destination / "crop_stress-valid-counts.json", counts(stress, thresholds[name]["ground_truth_crop"]))
        finally:
            close_backend(backend)
    write_json(output / "thresholds.json", {"selected_from": "valid only", "models": thresholds, "provenance": provenance})
    summary = {"status": "RUNNING", "synthetic": True, "production_ready": False,
        "scope": "known CAD, new rendering conditions; detection/crop diagnostic, not an inspection PASS policy",
        "crop_stress_independent_specimens": False, "models": {}, "object_weights": object_model["weights"],
        "dataset_fingerprint": FINGERPRINT, "threshold_selection": "validation micro-F1 at IoU 0.5",
        "provenance": provenance}
    for name, run in models.items():
        destination = output / name
        backend, model = backend_for(run, "product_defects")
        summary["models"][name] = {}
        try:
            backend.load()
            routes = {
                "ground_truth_crop": (DATA / "package/defect-crops", lambda: collect(backend, DATA / "package/defect-crops", "test", model["categories"])),
                "predicted_crop_chain": (full, lambda: collect_predicted_crops(backend, full, "test", object_rows["test"], .2, model["categories"])),
                "crop_stress": (DATA / "package/crop-stress", lambda: collect(backend, DATA / "package/crop-stress", "test", model["categories"])),
            }
            for route, (image_root, callback) in routes.items():
                rows = callback()
                write_json(destination / f"{route}-test-predictions.json", rows)
                threshold = thresholds[name]["ground_truth_crop" if route == "crop_stress" else route]
                report = counts(rows, threshold)
                write_json(destination / f"{route}-test-counts.json", report)
                summary["models"][name][route] = {k: v for k, v in report.items() if k != "errors"}
                fixed = protocol["comparison"]["fixed_comparison_thresholds"]["ground_truth_crop" if route == "crop_stress" else route]
                write_json(destination / f"{route}-fixed-threshold-counts.json", counts(rows, fixed))
                review = destination / route
                review.mkdir()
                draw_review(rows, image_root, review, threshold, report)
                if name == "reinforced":
                    assert_matched(read_json(output / "baseline" / f"{route}-test-predictions.json"), rows)
                write_json(output / "summary.json", summary)
                print(f"TEST {name} {route}: {report['total']}", flush=True)
        finally:
            close_backend(backend)
    # Verify both the primary and stress validation views were identical as well.
    for route in ("ground_truth_crop", "predicted_crop_chain", "crop_stress"):
        assert_matched(read_json(output / "baseline" / f"{route}-valid-predictions.json"),
                       read_json(output / "reinforced" / f"{route}-valid-predictions.json"))
    summary["status"] = "COMPLETED"
    summary["matched_inputs_verified"] = True
    for row in stress_files:
        require(sha256(DATA / "package" / row["path"]) == row["sha256"], "stress input changed during evaluation")
    write_json(output / "summary.json", summary)
    print("MATCHED COMPARISON COMPLETED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate"))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    (prepare if args.command == "prepare" else evaluate)(args.output.resolve())

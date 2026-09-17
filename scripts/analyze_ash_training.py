"""Review held-out synthetic detections through the app's real inference adapter."""
from pathlib import Path
import argparse
import gc
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key, folder in {"HF_HOME": "huggingface", "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ[key] = str(ROOT / ".cache" / folder)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"

from mes_vision.synthetic.detection_review import summarize, select_threshold
from mes_vision.training.data import read_json, write_json, sha256, require, validate_dataset
from mes_vision.training.jobs import verified_artifact


def collect(backend, data_root, split, categories):
    import numpy as np
    from PIL import Image
    coco = read_json(data_root / split / "_annotations.coco.json")
    names = {c["category_id"]: c["name"] for c in categories}
    annotations = {}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        annotations.setdefault(a["image_id"], []).append({"label": names[a["category_id"]], "box": [x, y, x+w, y+h]})
    rows = []
    for info in coco["images"]:
        path = data_root / split / info["file_name"]
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"))
        before = time.perf_counter()
        predictions = backend.predict_rgb(rgb)
        elapsed = time.perf_counter() - before
        rows.append({"file_name": info["file_name"], "image_sha256": sha256(path),
                     "targets": annotations.get(info["id"], []),
                     "predictions": [{"label": p.label, "score": p.score,
                                      "box": [p.box.x1, p.box.y1, p.box.x2, p.box.y2]} for p in predictions],
                     "seconds": elapsed, "excluded_reserved_slots": backend.excluded_reserved_slots})
    return rows


def draw_review(rows, data_root, output, threshold, report):
    from PIL import Image, ImageDraw, ImageFont
    errors = {e["file_name"]: len(e["fp"]) + len(e["fn"]) for e in report["errors"]}
    ordered = sorted(rows, key=lambda r: (-errors.get(r["file_name"], 0), r["file_name"]))
    selected, seen = [], set()
    for row in ordered:
        group = tuple(sorted({t["label"] for t in row["targets"]}))
        if group not in seen:
            selected.append(row)
            seen.add(group)
    for row in ordered:
        if len(selected) >= 12:
            break
        if row not in selected:
            selected.append(row)
    sheet = Image.new("RGB", (1200, ((len(selected)+2)//3)*265+40), (22, 25, 30))
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 14)
    ImageDraw.Draw(sheet).text((12, 10), f"SYNTHETIC TEST | Green: GT | Red: prediction > {threshold:.2f} | Errors first", fill="white", font=font)
    for index, row in enumerate(selected):
        with Image.open(data_root / "test" / row["file_name"]) as opened:
            image = opened.convert("RGB")
        draw = ImageDraw.Draw(image)
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(14, round(image.width/394*12)))
        for target in row["targets"]:
            draw.rectangle(target["box"], outline=(50, 255, 90), width=3)
        for p in row["predictions"]:
            if p["score"] > threshold:
                draw.rectangle(p["box"], outline=(255, 70, 70), width=3)
                draw.text((max(0, p["box"][0]), max(0, p["box"][1]-label_font.size-3)), f'{p["label"]} {p["score"]:.2f}', fill=(255, 100, 100), font=label_font)
        image.thumbnail((394, 222))
        x, y = (index % 3)*400, (index//3)*265+40
        sheet.paste(image, (x, y))
        ImageDraw.Draw(sheet).text((x+5, y+224), f'{row["file_name"]} | errors {errors.get(row["file_name"], 0)}', fill="white", font=font)
    sheet.save(output / "test-review.png")
    write_json(output / "review-selection.json", [r["file_name"] for r in selected])


def collect_predicted_crops(backend, full_root, split, object_rows, object_threshold, categories):
    from mes_vision.inputs import ImageSource
    from mes_vision.inspection import Box
    from mes_vision.inspection.geometry import extract_crop
    names = {c["category_id"]: c["name"] for c in categories}
    coco = read_json(full_root / split / "_annotations.coco.json")
    targets = {}
    for ann in coco["annotations"]:
        x, y, w, h = ann["bbox"]
        targets.setdefault(ann["image_id"], []).append({"label": names[ann["category_id"]], "box": [x,y,x+w,y+h]})
    objects = {row["file_name"]: row for row in object_rows}
    rows = []
    for info in coco["images"]:
        path = full_root / split / info["file_name"]
        row = {"file_name": info["file_name"], "image_sha256": sha256(path),
               "targets": targets.get(info["id"], []), "predictions": [], "crop_status": "UNCONFIRMED"}
        require(objects[info["file_name"]]["image_sha256"] == row["image_sha256"], "object prediction belongs to another image")
        detections = [p for p in objects[info["file_name"]]["predictions"] if p["score"] > object_threshold]
        if len(detections) == 1 and detections[0]["label"] == "ASH":
            b = Box(*detections[0]["box"])
            associated = abs((b.x1+b.x2)/2/info["width"]-.5) <= .25 and abs((b.y1+b.y2)/2/info["height"]-.5) <= .25
            inside = b.clip(info["width"], info["height"]) == b
            if associated and inside:
                with ImageSource(path) as source:
                    frame = source.read().frame
                crop = extract_crop(frame, info["file_name"], b, margin_px=0)
                row["crop_status"] = "CONFIRMED"
                row["crop_xyxy"] = [crop.bounds.x1, crop.bounds.y1, crop.bounds.x2, crop.bounds.y2]
                for p in backend.predict_rgb(crop.rgb):
                    if p.box.clip(crop.width, crop.height) != p.box:
                        raise ValueError("defect outside production crop")
                    original = crop.to_original(p.box)
                    row["predictions"].append({"label": p.label, "score": p.score,
                        "box": [original.x1, original.y1, original.x2, original.y2]})
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    batch, output = args.batch.resolve(), args.output.resolve()
    require(read_json(batch / "batch.json")["status"] == "COMPLETED", "training batch not completed")
    output.mkdir(parents=True, exist_ok=False)
    from mes_vision.inspection.rfdetr_adapter import RFDETRBackend
    import torch
    summary = {"synthetic": True, "production_ready": False,
               "scope": "known CAD, held-out rendering conditions only; no physical accuracy claim",
               "threshold_selection": "validation micro-F1 at IoU 0.5; diagnostic only, not a release decision policy",
               "input": "full photographs for object models; ground-truth object crops for defect component; predicted-crop chain reported separately",
               "inference_profile": "standard", "models": {}}
    for task in ("overview-objects", "detail-objects", "detail-defects"):
        run = batch / task
        manifest, model = read_json(run / "run.json"), read_json(run / "model.json")
        data_root = Path(manifest["config"]["dataset_dir"])
        require(validate_dataset(data_root)["fingerprint"] == manifest["dataset_fingerprint"], "dataset changed")
        weights = verified_artifact(run, manifest["checkpoints"]["inference"])
        labels = [c["name"] for c in model["categories"]]
        backend = RFDETRBackend(weights, manifest["checkpoints"]["inference"]["sha256"], threshold=.05,
                               training_scope="product_defects" if task == "detail-defects" else "product_objects",
                               class_names=tuple(labels), inference_profile="standard", max_batch_size=1)
        destination = output / task
        destination.mkdir()
        try:
            backend.load()
            validation = collect(backend, data_root, "valid", model["categories"])
            threshold, sweep = select_threshold(validation, labels)
            write_json(destination / "validation-predictions.json", validation)
            write_json(destination / "threshold-selection.json", {"split": "valid", "selected": threshold, "sweep": sweep})
            test = collect(backend, data_root, "test", model["categories"])
            write_json(destination / "test-predictions.json", test)
            report = summarize(test, threshold, labels)
            write_json(destination / "test-counts.json", report)
            draw_review(test, data_root, destination, threshold, report)
            summary["models"][task] = {"weights_sha256": manifest["checkpoints"]["inference"]["sha256"],
                                        "dataset_fingerprint": manifest["dataset_fingerprint"],
                                        **{k: v for k, v in report.items() if k != "errors"},
                                        "error_images": len(report["errors"])}
            write_json(output / "summary.json", summary)
            print(f"REVIEWED {task}: {report['total']}", flush=True)
            if task == "detail-defects":
                full_root = ROOT / "datasets/ash-synthetic-v2/training/detail-defects"
                object_output = output / "detail-objects"
                object_threshold = read_json(object_output / "threshold-selection.json")["selected"]
                chain_dir = output / "detail-predicted-crop-chain"
                chain_dir.mkdir()
                chain_valid = collect_predicted_crops(backend, full_root, "valid",
                    read_json(object_output / "validation-predictions.json"), object_threshold, model["categories"])
                chain_threshold, chain_sweep = select_threshold(chain_valid, labels)
                write_json(chain_dir / "threshold-selection.json", {"split": "valid", "selected": chain_threshold,
                           "object_threshold": object_threshold, "sweep": chain_sweep})
                write_json(chain_dir / "validation-predictions.json", chain_valid)
                chain_test = collect_predicted_crops(backend, full_root, "test",
                    read_json(object_output / "test-predictions.json"), object_threshold, model["categories"])
                write_json(chain_dir / "test-predictions.json", chain_test)
                chain_report = summarize(chain_test, chain_threshold, labels)
                chain_report["unconfirmed_crop_images"] = sum(r["crop_status"] != "CONFIRMED" for r in chain_test)
                chain_report["scope"] = "detection/crop chain only; physical quality/workspace/anomaly/decision policy not evaluated"
                write_json(chain_dir / "test-counts.json", chain_report)
                draw_review(chain_test, full_root, chain_dir, chain_threshold, chain_report)
                summary["models"]["detail-predicted-crop-chain"] = {k: v for k, v in chain_report.items() if k != "errors"}
                write_json(output / "summary.json", summary)
                print(f"REVIEWED predicted-crop chain: {chain_report['total']}", flush=True)
        finally:
            backend.close()
            gc.collect()
            torch.cuda.empty_cache()
    summary["status"] = "COMPLETED"
    write_json(output / "summary.json", summary)


if __name__ == "__main__":
    main()

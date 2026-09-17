"""Derive defect training crops using the production crop geometry, keeping splits."""
from pathlib import Path
import argparse
from copy import deepcopy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.training.data import read_json, write_json, validate_dataset, require, sha256
from mes_vision.inputs import ImageSource
from mes_vision.inspection import Box
from mes_vision.inspection.geometry import extract_crop


def prepare(source, output):
    from PIL import Image
    source, output = source.resolve(), output.resolve()
    objects, defects = source / "detail-objects", source / "detail-defects"
    object_data, defect_data = validate_dataset(objects), validate_dataset(defects)
    require(not output.exists(), "crop dataset already exists")
    pending = output.with_name(output.name + ".pending")
    pending.mkdir(parents=True, exist_ok=False)
    meta = deepcopy(defect_data["metadata"])
    meta["input_preprocessing"] = {"type": "object_crop", "margin_px": 0, "box_source": "ground_truth",
                                    "implementation": "mes_vision.inspection.geometry.extract_crop",
                                    "requires_predicted_crop_evaluation": True}
    meta["parent_datasets"] = {"objects": object_data["fingerprint"], "defects": defect_data["fingerprint"]}
    write_json(pending / "dataset.json", meta)
    transforms = []
    for split in ("train", "valid", "test"):
        folder = pending / split
        folder.mkdir()
        obj = read_json(objects / split / "_annotations.coco.json")
        defect = read_json(defects / split / "_annotations.coco.json")
        object_by_name = {im["file_name"]: im for im in obj["images"]}
        object_boxes = {}
        for ann in obj["annotations"]:
            require(ann["image_id"] not in object_boxes, "detail must contain one object")
            object_boxes[ann["image_id"]] = ann["bbox"]
        new_coco = deepcopy(defect)
        by_image = {}
        for ann in new_coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        for info in new_coco["images"]:
            source_path = defects / split / info["file_name"]
            object_info = object_by_name[info["file_name"]]
            require(sha256(source_path) == sha256(objects / split / info["file_name"]), "paired object/defect RGB differs")
            x, y, w, h = object_boxes[object_info["id"]]
            with ImageSource(source_path) as image_source:
                frame = image_source.read().frame
            crop = extract_crop(frame, info["file_name"], Box(x, y, x+w, y+h), margin_px=0)
            left, top = crop.bounds.x1, crop.bounds.y1
            height, width = crop.rgb.shape[:2]
            for ann in by_image.get(info["id"], []):
                ax, ay, aw, ah = ann["bbox"]
                require(ax >= left and ay >= top and ax+aw <= left+width and ay+ah <= top+height,
                        "defect ground truth would be truncated by crop")
                ann["bbox"] = [ax-left, ay-top, aw, ah]
            info.update(width=width, height=height, source_image_sha256=sha256(source_path),
                        source_crop_xyxy=[left, top, left+width, top+height])
            Image.fromarray(crop.rgb).save(folder / info["file_name"])
            transforms.append({"split": split, "file_name": info["file_name"], "crop_xyxy": info["source_crop_xyxy"]})
        write_json(folder / "_annotations.coco.json", new_coco)
    result = validate_dataset(pending)
    require(validate_dataset(objects)["fingerprint"] == object_data["fingerprint"] and
            validate_dataset(defects)["fingerprint"] == defect_data["fingerprint"], "source data changed")
    write_json(pending / "crop-transforms.json", transforms)
    write_json(pending / "preparation.json", {"status": "VALIDATED", "fingerprint": result["fingerprint"],
               "splits": result["splits"], "source_sha256": sha256(Path(__file__)),
               "crop_geometry_sha256": sha256(ROOT / "src/mes_vision/inspection/geometry.py")})
    pending.rename(output)
    print(f"Prepared {len(transforms)} crops: {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.source, args.output)

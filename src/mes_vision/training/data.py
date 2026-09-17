from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from PIL import Image

ROLES = {"object_detector", "known_defect_detector"}
SPLITS = ("train", "valid", "test")


def read_json(path: Path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf-8-sig"), object_pairs_hook=unique,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number: {value}")))


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int


def validate_dataset(root: str | Path) -> dict:
    """Never fixes labels or splits silently. Fingerprints include annotations and image bytes."""
    root = Path(root).resolve()
    meta = read_json(root / "dataset.json")
    require(meta.get("schema_version") == 1, "dataset schema_version must be 1")
    require(meta.get("role") in ROLES, "invalid dataset role")
    require(meta.get("kind") in {"real", "synthetic"}, "dataset kind must be real or synthetic")
    require(isinstance(meta.get("product_id"), str) and meta["product_id"].strip(), "product_id required")
    files = [{"path": "dataset.json", "sha256": sha256(root / "dataset.json")}]
    seen_pixels, seen_specimens, seen_sessions = {}, {}, {}
    reference_categories = None
    split_reports = {}
    for split in SPLITS:
        directory = root / split
        annotation_path = directory / "_annotations.coco.json"
        coco = read_json(annotation_path)
        for key in ("images", "annotations", "categories"):
            require(isinstance(coco.get(key), list), f"{split}: COCO {key} list required")
        cats = coco["categories"]
        require(cats and all(integer(c.get("id")) and c["id"] >= 0 and isinstance(c.get("name"), str) and c["name"].strip() for c in cats), f"{split}: invalid categories")
        require(len({c["id"] for c in cats}) == len(cats) and len({c["name"] for c in cats}) == len(cats), f"{split}: duplicate category")
        categories = [(c["id"], c["name"]) for c in sorted(cats, key=lambda c: c["id"])]
        if meta["role"] == "known_defect_detector":
            require(all(name in {f"NG{i:02d}" for i in range(1, 7)} for _, name in categories), "defect categories must be NG01..NG06; normal crops use zero annotations")
        if reference_categories is None:
            reference_categories = categories
        require(categories == reference_categories, f"{split}: category mapping differs between splits")
        ids = {row[0] for row in categories}
        images, counts, used_paths = {}, {}, set()
        require(coco["images"], f"{split}: no images")
        for info in coco["images"]:
            image_id = info.get("id")
            require(integer(image_id) and image_id not in images, f"{split}: duplicate/invalid image id")
            name = info.get("file_name")
            require(isinstance(name, str) and name, f"{split}: image file_name required")
            relative = Path(name)
            require(not relative.is_absolute() and ".." not in relative.parts and ":" not in name and "\\" not in name, f"{split}: unsafe file_name")
            path = (directory / relative).resolve()
            require(path.is_relative_to(directory.resolve()) and path not in used_paths, f"{split}: escaped or duplicate image path")
            used_paths.add(path)
            require(path.suffix.lower() in {".png", ".jpg", ".jpeg"}, "training supports RGB PNG/JPEG only")
            width, height = info.get("width"), info.get("height")
            require(integer(width) and integer(height) and width > 0 and height > 0, f"{split}: invalid image dimensions")
            with Image.open(path) as image:
                require(image.size == (width, height), f"{split}: image size mismatch: {name}")
                require(image.mode == "RGB" and getattr(image, "n_frames", 1) == 1, f"{split}: opaque single-frame RGB required")
                require("transparency" not in image.info, f"{split}: transparent RGB PNG unsupported")
                require(image.getexif().get(274, 1) == 1, f"{split}: normalize EXIF orientation and annotation coordinates before training")
                if image.format == "PNG":
                    with path.open("rb") as encoded:
                        require(encoded.read(25)[24] == 8, f"{split}: 8-bit PNG required")
                image.load()
                pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
            require(pixels not in seen_pixels or seen_pixels[pixels] == split, "same image pixels appear across train/valid/test")
            seen_pixels[pixels] = split
            specimens, session = info.get("specimen_ids"), info.get("capture_session_id")
            require(isinstance(specimens, list) and all(isinstance(x, str) and x.strip() for x in specimens), f"{split}: specimen_ids list required")
            require(len(set(specimens)) == len(specimens), f"{split}: duplicate specimen id in image")
            require(isinstance(session, str) and session.strip(), f"{split}: capture_session_id required")
            for value in specimens:
                require(value not in seen_specimens or seen_specimens[value] == split, "same physical specimen appears across splits")
                seen_specimens[value] = split
            require(session not in seen_sessions or seen_sessions[session] == split, "same capture session appears across splits")
            seen_sessions[session] = split
            images[image_id], counts[image_id] = info, 0
            files.append({"path": path.relative_to(root).as_posix(), "sha256": sha256(path), "pixel_sha256": pixels})
        annotations = set()
        class_counts = {cat: 0 for cat in ids}
        for ann in coco["annotations"]:
            aid, iid, cid = ann.get("id"), ann.get("image_id"), ann.get("category_id")
            require(integer(aid) and aid not in annotations, f"{split}: duplicate/invalid annotation id")
            require(integer(iid) and iid in images and integer(cid) and cid in ids, f"{split}: annotation references missing image/class")
            annotations.add(aid)
            box = ann.get("bbox")
            require(isinstance(box, list) and len(box) == 4 and all(type(x) in (int, float) and math.isfinite(x) for x in box), f"{split}: invalid xywh bbox")
            x, y, w, h = box
            require(x >= 0 and y >= 0 and w > 0 and h > 0 and x+w <= images[iid]["width"] and y+h <= images[iid]["height"], f"{split}: bbox outside image or empty")
            area = ann.get("area")
            require(type(area) in (int, float) and math.isfinite(area) and math.isclose(area, w*h, rel_tol=1e-5), f"{split}: detection area must match bbox")
            require(ann.get("iscrowd", 0) == 0 and not ann.get("ignore", False), f"{split}: crowd/ignored annotations unsupported")
            require(images[iid]["specimen_ids"], f"{split}: annotated image needs physical specimen IDs")
            counts[iid] += 1
            class_counts[cid] += 1
        require(all(class_counts.values()), f"{split}: every registered class needs at least one annotation (coverage, not sufficient statistical evidence)")
        if meta["role"] == "known_defect_detector":
            require(all(image["specimen_ids"] for image in images.values()), "normal defect crops also need specimen_ids")
        split_reports[split] = {"images": len(images), "annotations": len(annotations), "negative_images": sum(v == 0 for v in counts.values()), "class_counts": class_counts}
        files.append({"path": annotation_path.relative_to(root).as_posix(), "sha256": sha256(annotation_path)})
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"schema_version": 1, "dataset_dir": str(root), "metadata": meta, "fingerprint": digest,
            "categories": [{"category_id": cid, "label_index": index, "name": name} for index, (cid, name) in enumerate(reference_categories)],
            "splits": split_reports, "files": files, "scope": "format and declared split leakage checks; not label correctness or product accuracy"}

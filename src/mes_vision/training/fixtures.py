from pathlib import Path

from PIL import Image, ImageDraw

from .data import ROLES, SPLITS, require, write_json


def make_fixture(root: str | Path, role: str):
    """Distinct synthetic splits; values have no physical or acceptance meaning."""
    require(role in ROLES, "invalid fixture role")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=False)
    write_json(root / "dataset.json", {"schema_version": 1, "role": role, "kind": "synthetic", "product_id": "SYNTHETIC_FIXTURE_ONLY"})
    names = ["synthetic_part"] if role == "object_detector" else ["NG03", "NG06"]
    categories = [{"id": index+1, "name": name, "supercategory": "fixture"} for index, name in enumerate(names)]
    for split_index, split in enumerate(SPLITS):
        folder = root / split
        folder.mkdir()
        images, annotations = [], []
        for index in range(2):
            image = Image.new("RGB", (512, 512), (220+split_index*5, 220, 220+index*5))
            draw = ImageDraw.Draw(image)
            left, top = 100+split_index*17+index*9, 130+split_index*13
            if index == 0 or role == "known_defect_detector":
                draw.rectangle((left, top, left+199, top+209), fill="#3266aa")
            if index == 0:
                boxes = [[left, top, 200, 210]]
                if role == "known_defect_detector":
                    draw.line([(left+20, top+20), (left+35, top+50), (left+25, top+80)], fill="#182333", width=4)
                    draw.rectangle((left+100, top+140, left+129, top+159), fill="#9cb9d2")
                    boxes = [[left+18, top+18, 20, 65], [left+100, top+140, 30, 20]]
                for cid, box in enumerate(boxes, 1):
                    annotations.append({"id": len(annotations)+1, "image_id": index+1, "category_id": cid,
                                        "bbox": box, "area": box[2]*box[3], "iscrowd": 0})
            filename = f"synthetic-{index}.png"
            image.save(folder / filename)
            images.append({"id": index+1, "file_name": filename, "width": 512, "height": 512,
                           "specimen_ids": [f"synthetic-{split}-{index}"] if index == 0 or role == "known_defect_detector" else [],
                           "capture_session_id": f"synthetic-session-{split}"})
        write_json(folder / "_annotations.coco.json", {"info": {"description": "Synthetic software execution fixture, not product accuracy"},
                   "licenses": [], "images": images, "annotations": annotations, "categories": categories})
    return root

"""Synthetic plumbing checks only; never product training or accuracy evidence."""
from pathlib import Path
from PIL import Image, ImageDraw
from .collection import Collection, new_object


def make_fixture(root):
    root = Path(root)
    sources = root / "synthetic-sources"
    sources.mkdir(parents=True, exist_ok=False)
    collection = Collection.create(root / "collection", "fixture-part", kind="synthetic")
    for index, split in enumerate(("train", "valid", "test", "challenge")):
        image = Image.new("RGB", (320, 200), (32 + index * 9, 43, 51))
        draw = ImageDraw.Draw(image)
        draw.rectangle((20, 25, 119, 154), fill=(140 + index * 15, 170, 185))
        draw.rectangle((175, 30, 289, 159), fill=(170, 125 + index * 15, 150))
        draw.line((195, 50, 215, 95), fill=(20, 20, 20), width=3)
        draw.rectangle((235, 115, 249, 129), fill=(230, 230, 230))
        path = sources / (split + ".png")
        image.save(path)
        identity = collection.import_image(path, "session-" + split)
        record = collection.record(identity)
        normal = new_object(split + "-normal", [20, 25, 120, 155])
        normal["condition"] = "NORMAL"
        defect = new_object(split + "-ng", [175, 30, 290, 160])
        defect.update(condition="KNOWN_NG", defects=[
            {"id": split + "-crack", "code": "NG03", "bbox": [190, 45, 220, 100], "note": "합성 균열"},
            {"id": split + "-surface", "code": "NG06", "bbox": [235, 115, 250, 130], "note": "합성 표면 표시"}])
        if split == "challenge":
            defect.update(condition="UNKNOWN_NG", note="코드에 없는 합성 관찰 사례", defects=[])
        record["objects"] = [normal, defect]
        collection.save_record(record)
        collection.assign_session(record["capture_session_id"], split)
        collection.review(identity, "SYNTHETIC_FIXTURE")
    return collection

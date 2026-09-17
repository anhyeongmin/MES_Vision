"""Independently verify every exported RGB crop, mask, annotation and split."""
from pathlib import Path
import argparse
import hashlib
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from mes_vision.training.data import read_json as read,write_json as write,sha256 as sha,require,validate_dataset


def main(root):
    package = root.resolve()/"package"
    ledger = read(package/"crop-ledger.json")
    report = read(package/"preparation.json")
    require(validate_dataset(package/"defect-crops")["fingerprint"] == report["fingerprint"],"Primary data changed")
    annotations, images = {}, {}
    group_splits, source_splits, pixels_splits = {}, {}, {}
    class_counts = {}
    for scope,splits in (("defect-crops",("train","valid","test")),("crop-stress",("valid","test"))):
        for split in splits:
            data = read(package/scope/split/"_annotations.coco.json")
            names = {r["id"]:r["file_name"] for r in data["images"]}
            require(len(names) == len(data["images"]),"Duplicate image id")
            require([c["name"] for c in data["categories"]] == [f"NG{i:02d}" for i in range(1,7)],"Category mapping changed")
            for im in data["images"]:
                key = f"{scope}/{split}/{im['file_name']}"
                images[key] = im
                annotations[key] = []
                previous = group_splits.setdefault(im["capture_session_id"],split)
                require(previous == split,"Crop sibling or paired condition crossed splits")
                if im["source_origin"] == "v2":
                    require(split == "train" and im["source_render_id"].startswith("train-"),"Old evaluation reused")
                else:
                    require(im["source_render_id"].startswith("reinforce-"+split+"-"),"New render assigned to wrong split")
            ids = set()
            for ann in data["annotations"]:
                require(ann["id"] not in ids,"Duplicate annotation id")
                ids.add(ann["id"])
                key = f"{scope}/{split}/{names[ann['image_id']]}"
                annotations[key].append(ann)
    require(len(ledger) == len(images) and {r["path"] for r in ledger} == set(images),"Ledger and COCO images differ")
    current = None
    count = 0
    sources = set()
    for row in ledger:
        path = package/row["path"]
        mask_path = package/row["mask_path"]
        require(sha(path) == row["sha256"] and sha(mask_path) == row["mask_sha256"],"Exported file changed")
        source = Path(row["source_path"])
        if current != source:
            require(sha(source) == row["source_sha256"] and sha(Path(row["source_mask_path"])) == row["source_mask_sha256"],"Original file changed")
            with Image.open(source) as im:
                source_rgb = np.array(im)
            with Image.open(row["source_mask_path"]) as im:
                source_mask = np.array(im)
            current = source
        sources.add(str(source))
        require(source_splits.setdefault(str(source),row["split"]) == row["split"],"Source image reused across splits")
        with Image.open(path) as im:
            require(im.mode == "RGB", "RGB required")
            actual = np.array(im)
        with Image.open(mask_path) as im:
            actual_mask = np.array(im)
        left,top,right,bottom = row["parameters"]["bounds"]
        require(np.array_equal(actual,source_rgb[top:bottom,left:right]),"Pixel crop mismatch")
        require(np.array_equal(actual_mask,source_mask[top:bottom,left:right]),"Mask crop mismatch")
        require(np.count_nonzero(actual_mask) == np.count_nonzero(source_mask),"Defect was cut off")
        digest = hashlib.sha256(str(actual.shape).encode()+actual.tobytes()).hexdigest()
        require(pixels_splits.setdefault(digest,row["split"]) == row["split"],"Duplicate pixels across splits")
        info = images[row["path"]]
        require(actual.shape[:2] == (info["height"],info["width"]),"COCO image size mismatch")
        anns = annotations[row["path"]]
        if row["code"] == "OK":
            require(not anns and not source_mask.any(),"Normal has defect label")
        else:
            require(len(anns) == 1,"Missing or extra defect annotation")
            ys,xs = np.nonzero(source_mask)
            x0,y0 = max(0,int(xs.min())-2),max(0,int(ys.min())-2)
            x1,y1 = min(source_mask.shape[1],int(xs.max())+3),min(source_mask.shape[0],int(ys.max())+3)
            require(anns[0]["bbox"] == [x0-left,y0-top,x1-x0,y1-y0],"Mask-derived annotation mismatch")
            require(anns[0]["category_id"] == int(row["code"][2:]),"Wrong defect class")
            require(anns[0]["area"] == (x1-x0)*(y1-y0),"Wrong annotation area")
            count += 1
        key = "/".join(row["path"].split("/")[:2])
        class_counts.setdefault(key,{})[row["code"]] = class_counts.setdefault(key,{}).get(row["code"],0)+1
    result = dict(status="PASSED",checked_rgb_and_masks=len(ledger),checked_defect_annotations=count,
        original_source_images=len(sources),class_counts=class_counts,old_evaluation_reuse=False,
        pixel_or_group_split_leak=False,source_sha256=sha(Path(__file__)))
    write(package/"verification.json",result)
    print(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",required=True,type=Path)
    main(parser.parse_args().root)

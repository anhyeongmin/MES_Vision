"""Versioned defect crops with source-group isolation and paired augmentation gates."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
from PIL import Image
from .dataset_export import read, write, sha, quality, GATES
from .planning import CODES, fingerprint
from .reinforcement import FOCUS, crop_variant, xyxy
from ..training.data import validate_dataset, require


def group_tasks(plan):
    groups = defaultdict(list)
    for task in plan["tasks"]:
        if task["domain"] == "detail":
            groups[task["group"]].append(task)
    for tasks in groups.values():
        require(len(tasks) == 7 and {t["objects"][0]["code"] for t in tasks} == set(CODES), "Incomplete CAD pair group")
        require(len({t["split"] for t in tasks}) == 1, "Render group crosses splits")
        require(len({fingerprint([t["camera"],t["condition"],{k:v for k,v in t["objects"][0].items() if k != "code"}]) for t in tasks}) == 1,
                "Normal/NG render conditions differ")
    return groups


def crop_labels(rgb, mask, bounds, defects):
    left, top, right, bottom = bounds
    require(all(type(v) is int for v in bounds), "Integer crop bounds required")
    require(0 <= left < right <= rgb.shape[1] and 0 <= top < bottom <= rgb.shape[0], "Crop outside source")
    crop = rgb[top:bottom,left:right].copy()
    cut_mask = mask[top:bottom,left:right].copy()
    require(int(np.count_nonzero(mask)) == int(np.count_nonzero(cut_mask)), "Crop truncates defect mask")
    labels = []
    for label in defects:
        x, y, w, h = label["bbox"]
        require(x >= left and y >= top and x+w <= right and y+h <= bottom, "Crop truncates defect box")
        labels.append(dict(code=label["code"],bbox=[x-left,y-top,w,h]))
    require(bool(np.any(mask)) == bool(labels), "Normal or defect mask disagrees with labels")
    return crop, cut_mask, labels


def export(root, parent):
    root, parent = Path(root).resolve(), Path(parent).resolve()
    final = root / "package"
    require(not final.exists(), "Completed package already exists")
    plan, contract = read(root/"plan.json"), read(root/"contract.json")
    require(sha(root/"plan.json") == contract["plan_sha256"] and sha(root/"geometry.json") == contract["geometry_sha256"], "Render inputs changed")
    project = Path(__file__).resolve().parents[3]
    for path, digest in contract["source_hashes"].items():
        require(sha(project/path) == digest, "Generator source changed")
    old_plan, old_contract = read(parent/"plan.json"), read(parent/"contract.json")
    require(sha(parent/"plan.json") == contract["parent_plan_sha256"], "Parent plan changed")
    require(sha(parent/"geometry.json") == old_contract["geometry_sha256"], "Parent geometry changed")
    original = validate_dataset(parent/"training/detail-defects")
    geometry = {k:np.asarray(v) for k,v in read(root/"geometry.json").items()}
    groups = []
    for tasks in group_tasks(old_plan).values():
        if tasks[0]["split"] == "train":
            groups.append(("v2",parent,tasks,old_contract["plan_sha256"]))
    for tasks in group_tasks(plan).values():
        groups.append(("v3",root,tasks,contract["plan_sha256"]))
    qualities, rejected, eligible = {}, {}, []
    for index,(origin,source,tasks,ph) in enumerate(groups):
        key = origin+":"+tasks[0]["group"]
        values = [quality(source,t,geometry,ph) for t in tasks]
        if any(not v["accepted"] for v in values):
            rejected[key] = [dict(id=v["id"],reason=v["reason"]) for v in values if not v["accepted"]]
            continue
        object_boxes = [v["objects"][0]["bbox"] for v in values]
        defect_boxes = [d["bbox"] for v in values for d in v["defects"]]
        try:
            perturbations = [crop_variant(key,i,object_boxes,defect_boxes) for i in range(4)] if tasks[0]["split"] == "train" else []
            stress = [crop_variant(key,i,object_boxes,defect_boxes,stress=True) for i in range(3)] if tasks[0]["split"] != "train" else []
        except ValueError as exc:
            rejected[key] = [dict(reason=str(exc))]
            continue
        qualities[key] = values
        eligible.append((origin,source,tasks,perturbations,stress))
        if (index+1)%10 == 0:
            print(f"GROUP_QUALITY {index+1}/{len(groups)}",flush=True)
    pending = root/"package.pending"
    pending.mkdir(exist_ok=False)
    categories = [dict(id=i,name=c,supercategory="defect") for i,c in enumerate(CODES[1:],1)]
    coco = {}
    for scope,splits in (("defect-crops",("train","valid","test")),("crop-stress",("valid","test"))):
        for split in splits:
            (pending/scope/split).mkdir(parents=True)
            (pending/"masks"/scope/split).mkdir(parents=True)
            coco[scope,split] = dict(images=[],annotations=[],categories=categories,
                info=dict(description="Known CAD synthetic conditions; no physical accuracy evidence"),licenses=[])
    metadata = dict(schema_version=1,kind="synthetic",role="known_defect_detector",product_id="ASH",
        camera_target=plan["camera_target"],cad_lineage=plan["cad_lineage"],cad_split=plan["cad_split"],
        evaluation_scope=plan["evaluation_scope"],split_unit="source render condition, before augmentation",
        specimen_ids_meaning="synthetic source instances; all crop siblings share one instance ID",
        parent_dataset_fingerprint=original["fingerprint"],generator_plan_sha256=contract["plan_sha256"],
        previous_test_policy=plan["previous_test_policy"],real_validation_required=True,
        input_preprocessing=dict(type="object_crop",train_variants=dict(OK=5,NG01=5,NG04=5,other=3),
            evaluation="one ground-truth crop per new source; perturbation stress set separate",
            augmentation_pairing="identical crop parameters across CAD counterparts; gates use all six defect sites",
            min_object_bbox_coverage=.95,defect_context_pixels=4),generation_gates=GATES)
    write(pending/"defect-crops/dataset.json",metadata)
    write(pending/"crop-stress/dataset.json",{**metadata,"purpose":"paired robustness diagnostic only; not independent additional specimens"})
    ledger, source_hashes, included = [], {}, []
    for origin,source,tasks,perturbations,stress in eligible:
        key = origin+":"+tasks[0]["group"]
        split = tasks[0]["split"]
        require(origin != "v2" or split == "train", "Old evaluation image entered new package")
        included.append(dict(group=key,origin=origin,split=split))
        for task,v in zip(tasks,qualities[key]):
            code = task["objects"][0]["code"]
            source_path = source/"raw"/(task["id"]+".png")
            mask_path = source/"masks"/(task["id"]+".png")
            source_hashes[str(source_path)] = v["image_sha256"]
            require(sha(mask_path) == v["mask_sha256"], "Source mask changed")
            with Image.open(source_path) as image:
                rgb = np.array(image)
            with Image.open(mask_path) as image:
                mask = np.array(image)
            base = [int(math.floor(n)) if i < 2 else int(math.ceil(n)) for i,n in enumerate(xyxy(v["objects"][0]["bbox"]))]
            variants = [("defect-crops","base",dict(bounds=base))]
            if split == "train":
                variants += [("defect-crops",f"j{i+1}",p) for i,p in enumerate(perturbations) if i < 2 or code in FOCUS]
            else:
                variants += [("crop-stress",f"s{i+1}",p) for i,p in enumerate(stress)]
            for scope,name,parameters in variants:
                crop,cut_mask,labels = crop_labels(rgb,mask,parameters["bounds"],v["defects"])
                filename = origin+"-"+task["id"]+"__"+name+".png"
                relative = Path(scope)/split/filename
                mask_relative = Path("masks")/relative
                Image.fromarray(crop).save(pending/relative)
                Image.fromarray(cut_mask).save(pending/mask_relative)
                data = coco[scope,split]
                iid = len(data["images"])+1
                data["images"].append(dict(id=iid,file_name=filename,width=crop.shape[1],height=crop.shape[0],
                    specimen_ids=[f"synthetic:{origin}:{task['id']}:0"],capture_session_id=key,cad_source_ids=[code],
                    source_image_sha256=v["image_sha256"],source_origin=origin,source_render_id=task["id"],
                    augmentation=name,source_crop_xyxy=parameters["bounds"],synthetic=True))
                for label in labels:
                    b = label["bbox"]
                    data["annotations"].append(dict(id=len(data["annotations"])+1,image_id=iid,category_id=CODES.index(label["code"]),bbox=b,area=b[2]*b[3],iscrowd=0))
                ledger.append(dict(path=relative.as_posix(),sha256=sha(pending/relative),mask_path=mask_relative.as_posix(),
                    mask_sha256=sha(pending/mask_relative),source_path=str(source_path),source_sha256=v["image_sha256"],
                    source_mask_path=str(mask_path),source_mask_sha256=v["mask_sha256"],parameters=parameters,group=key,split=split,code=code))
    for (scope,split),data in coco.items():
        write(pending/scope/split/"_annotations.coco.json",data)
    validation = validate_dataset(pending/"defect-crops")
    for path,digest in source_hashes.items():
        require(sha(Path(path)) == digest,"Source changed during export")
    require(validate_dataset(parent/"training/detail-defects")["fingerprint"] == original["fingerprint"], "Original dataset changed")
    write(pending/"crop-ledger.json",ledger)
    report = dict(status="VALIDATED",new_planned_rgb=len(plan["tasks"]),included_groups=included,rejected_groups=rejected,
        original_source_images=len(source_hashes),primary=validation["splits"],fingerprint=validation["fingerprint"],
        stress={split:dict(images=len(coco["crop-stress",split]["images"]),annotations=len(coco["crop-stress",split]["annotations"])) for split in ("valid","test")},
        crop_images=len(ledger),model_trained=False,model_inference_performed=False,shared_cad_across_splits=True,
        old_validation_and_test_included=False,exporter_sha256=sha(Path(__file__)))
    write(pending/"preparation.json",report)
    pending.rename(final)
    write(root/"status.json",dict(status="DATASET_READY",model_trained=False,primary_fingerprint=validation["fingerprint"]))
    return report

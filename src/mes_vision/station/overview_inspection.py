"""Manual full-view comparison: reuse verified detections in the same saved pixels."""
from dataclasses import asdict
from pathlib import Path
from mes_vision.inspection.contracts import Box, Detection, DetectionBatch, ModelRef
from mes_vision.training.data import require, sha256
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.anomaly.features import fingerprint


def frozen_overview(job, frame, bundle, image_kind):
    origin=job['overview_origin']
    directory=Path(origin['snapshot']).resolve()
    manifest, result, _=load_snapshot(directory,expected_digest=origin['snapshot_digest'])
    meta=result['config'].get('saved_photo',{})
    require(meta.get('role')=='overview' and result['mode']=='model_file'
            and not result['frame']['is_live'] and result['robot_commands_enabled'] is False,
            'Only saved overview localization can feed full-view inspection')
    require(meta['model_bundle_digest']==fingerprint(bundle) and manifest['product_id']==bundle['product_id'],
            'Overview model bundle changed')
    require(meta['image_kind']==manifest['kind']==image_kind,'Overview image origin changed')
    require(sha256(directory/'frame.png')==job['sha256'] and
            (frame.width,frame.height)==(result['frame']['width'],result['frame']['height']),
            'Full-view inspection must use the original saved overview pixels')
    require(0<len(result['objects'])<=100,'No overview objects or object limit exceeded')
    detections=[]
    for obj in result['objects']:
        d=obj['detection'];box=Box(**d['box'])
        require(asdict(box.clip(frame.width,frame.height))==obj['effective_box'],
                'Overview box or coordinate space changed')
        require(d['label']==bundle['product_id'],'Unexpected overview object class')
        detections.append(Detection(box,d['score'],d['class_id'],d['label']))
    batch=DetectionBatch(frame.frame_id,(frame.width,frame.height),tuple(detections),
        ModelRef(**result['detector']),candidate_threshold=result['config']['detector_candidate_threshold'])
    return batch,result['objects']


def object_mapping(source, output):
    require(len(source)==len(output),'Full-view inspection lost an object')
    pairs=[]
    for old,new in zip(source,output):
        require(old['effective_box']==asdict(new.effective_box) and old['detection']==asdict(new.detection),
                'Full-view object ordering or coordinates changed')
        pairs.append(dict(source_object_id=old['object_id'],object_id=new.object_id))
    require(len({p['source_object_id'] for p in pairs})==len(pairs),'Duplicate overview object identity')
    return pairs

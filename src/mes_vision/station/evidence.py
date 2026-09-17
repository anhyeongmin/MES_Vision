"""Immutable camera captures and detail snapshots with explicit cycle links."""
from dataclasses import asdict
from pathlib import Path
import hashlib
from uuid import uuid4
from PIL import Image
from mes_vision.training.data import require,sha256,write_json
from mes_vision.vlm.snapshots import save_snapshot
from mes_vision.anomaly.features import fingerprint
from mes_vision.inputs.camera_identity import matches_camera


def save_capture(root,frame,*,camera_serial,acquired_at):
    require(matches_camera(frame,camera_serial),'Original live camera capture required')
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    folder=root/uuid4().hex; staging=root/('.'+folder.name+'-writing'); staging.mkdir()
    Image.fromarray(frame.rgb).save(staging/'frame.png')
    value={'frame_id':frame.frame_id,'camera_session':frame.session_id,'camera_serial':camera_serial,
        'sequence':frame.sequence,'image_size':[frame.width,frame.height],'acquired_at':acquired_at,
        'path':str((folder/'frame.png').resolve()),'sha256':sha256(staging/'frame.png'),
        'rgb_sha256':hashlib.sha256(frame.rgb.tobytes()).hexdigest(),'frame_metadata':frame.metadata()}
    write_json(staging/'capture.json',value); staging.rename(folder)
    return value


def save_detail(root,frame,output,*,normal_export=None,criteria=None):
    require(output['detail_frame_id']==frame.frame_id,'Detail output and camera frame differ')
    result=output['inspection']
    if result is None:
        require(output['decision']=='REVIEW' and output['reason'],'Missing inspection must remain REVIEW')
        return {'cycle_id':output['cycle_id'],'target_id':output['target_id'],
                'detail_frame_id':frame.frame_id,'decision':'REVIEW','review_reason':output['reason']}
    directory=Path(root)/result.run_id
    manifest=save_snapshot(directory,frame,result,kind='real',normal_export=normal_export,criteria=criteria)
    require(len(result.objects)==1,'Detail output must contain one object')
    return {'cycle_id':output['cycle_id'],'target_id':output['target_id'],'detail_frame_id':frame.frame_id,
            'snapshot_path':str(directory.resolve()),'snapshot_digest':fingerprint(manifest),
            'object_id':result.objects[0].object_id}

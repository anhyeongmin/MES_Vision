"""Freeze installed capture settings into a single immutable cycle profile."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from mes_vision.robot import load_profile,Pose
from mes_vision.training.data import require,sha256
from .sequence import valid_profile
from .motion import capture_path
from mes_vision.inputs.camera_identity import matches_camera,camera_driver


def build_scan_profile(settings,equipment,*,product_id,product_version,overview_model_digest,
                       detail_policy_digest,frame,max_objects,expected_count,auto_sort,robot_connection_epoch):
    calibration=settings.calibrated(); v=settings.value
    from mes_vision.operation.acquisition import inspection_camera, verify_frame
    c=inspection_camera(equipment['camera']); verify_frame(frame,equipment['camera'])
    require(isinstance(robot_connection_epoch,str) and robot_connection_epoch.strip(),'Confirmed robot connection identity required')
    require(matches_camera(frame,c['serial'],camera_driver(c))
            and frame.coordinate_space=='input_rgb_pixels' and not frame.transformations
            and (frame.width,frame.height)==(c['width'],c['height']),'Connected original camera frame required')
    path=Path(equipment['robot']['profile']); robot_profile=load_profile(path); w=robot_profile.workspace
    require(w is not None,'Registered robot workspace required')
    profile={'schema_version':1,'mode':'overview_detail','profile_id':uuid4().hex,
        'validation_reference':v['validation_reference'],'camera_serial':c['serial'],'mount_revision':c['mount_revision'],
        'camera_session':frame.session_id,'camera_driver':camera_driver(c),'calibration_kind':'robot_mounted_overview_detail',
        'calibration_digest':calibration.digest,'overview_model_digest':overview_model_digest,
        'detail_policy_digest':detail_policy_digest,'product_id':product_id,'product_version':product_version,
        'equipment_version':equipment['version'],'station_settings_version':v['version'],'robot_profile_sha256':sha256(path),
        'robot_connection_epoch':robot_connection_epoch,
        'acquisition_identity':frame.acquisition_identity,
        'image_size':[frame.width,frame.height],'settle_seconds':v['settle_seconds'],
        'action_timeout_seconds':v['action_timeout_seconds'],'pose_tolerance_mm':equipment['robot']['pose_tolerance_mm'],
        'rotation_tolerance_deg':equipment['robot']['rotation_tolerance_deg'],
        'max_objects':max_objects,'expected_count':expected_count,'auto_sort':auto_sort,
        'motion_bounds':{k:[getattr(w,k+'_min'),getattr(w,k+'_max')] for k in ('x','y','z','r')},
        'overview_pose':deepcopy(calibration.value['overview_pose'])}
    valid_profile(profile)
    capture_path(Pose(**profile['overview_pose']),profile['overview_pose'],profile,calibration,robot_profile,equipment)
    return profile

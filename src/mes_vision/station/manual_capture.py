"""Manual arrival captures. No robot coordinates or motion authority."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from mes_vision.training.data import read_json, sha256, require, write_json


class Rectifier:
    def __init__(self, path):
        self.path = Path(path)
        data = read_json(self.path)
        self.digest = sha256(self.path)
        self.size = tuple(data['image_size'])
        k, d = np.asarray(data['K'], dtype=float), np.asarray(data['D'], dtype=float)
        require(k.shape == (3, 3) and np.isfinite(k).all() and np.isfinite(d).all(), '보정 파일 오류')
        self.maps = cv2.initUndistortRectifyMap(k, d, None, k, self.size, cv2.CV_32FC1)

    def apply(self, rgb, square=True):
        require((rgb.shape[1], rgb.shape[0]) == self.size, '보정 해상도와 영상 해상도가 다릅니다. 1280×720 설정을 확인하세요.')
        full = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        w, h = self.size
        side = min(w, h)
        box = ((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2) if square else (0, 0, w, h)
        x1, y1, x2, y2 = box
        return full, np.ascontiguousarray(full[y1:y2, x1:x2]), box


def fresh_after(frame, received, not_before, now):
    return frame is not None and not_before <= received <= now and now-received <= .3


def save_capture(folder, frame, rectifier, square, metadata):
    require(not frame.transformations, '이미 보정된 영상은 다시 보정하지 않습니다.')
    require(frame.acquisition_identity is None, '이미 보정된 영상은 다시 보정하지 않습니다.')
    full, view, box = rectifier.apply(frame.rgb, square)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    for name, rgb in [('raw.png', frame.rgb), ('rectified-full.png', full), ('inspection.png', view)]:
        Image.fromarray(rgb).save(folder/name)
    write_json(folder/'capture.json', dict(metadata, frame=frame.metadata(),
        calibration_sha256=rectifier.digest, calibration_file=str(rectifier.path),
        crop_xyxy=list(box), coordinate_space='rectified_full_frame_then_crop_pixels',
        robot_commands_enabled=False, association='operator_selected_not_robot_verified',
        files={name: sha256(folder/name) for name in ('raw.png', 'rectified-full.png', 'inspection.png')}))
    return folder/'inspection.png'

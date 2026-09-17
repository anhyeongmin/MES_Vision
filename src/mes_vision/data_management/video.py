"""Sample local videos into reviewed collections; never generate ground truth."""
from copy import deepcopy
import math
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from uuid import uuid4

import cv2

from .collection import Collection
from mes_vision.training.data import require, sha256


def import_video(collection, path, session_id, *, interval_seconds=1.0, max_frames=120):
    require(isinstance(session_id, str) and session_id.strip(), '촬영 회차 ID가 필요합니다')
    require(type(interval_seconds) in (int, float) and math.isfinite(interval_seconds)
            and interval_seconds >= 0.1, '추출 간격은 0.1초 이상이어야 합니다')
    require(type(max_frames) is int and 1 <= max_frames <= 1000, '추출 장수는 1~1000이어야 합니다')
    source = Path(path).resolve()
    require(source.is_file(), '영상 파일을 찾을 수 없습니다')
    session_id = session_id.strip()
    digest = sha256(source)
    require(not any(r.get('video_source', {}).get('sha256') == digest
                    for r in collection.data['records']), '이미 가져온 영상입니다. 기존 프레임을 사용하세요')
    split = next((r['split'] for r in collection.data['records']
                  if r['capture_session_id'] == session_id and not r['excluded']), 'unassigned')
    # Build off to the side so decoder/import failures do not add partial records.
    with TemporaryDirectory(prefix='.video-import-', dir=collection.root) as temp:
        stage = Collection.create(Path(temp) / 'collection', collection.data['product_id'],
                                  kind=collection.data['kind'])
        video_name = uuid4().hex + source.suffix.lower()
        owned = stage.root / 'originals' / video_name
        shutil.copyfile(source, owned)
        require(sha256(owned) == digest, '복사 중 원본 영상이 변경되었습니다')
        capture = cv2.VideoCapture(str(owned))
        try:
            require(capture.isOpened(), '영상을 열 수 없습니다')
            fps = capture.get(cv2.CAP_PROP_FPS)
            require(math.isfinite(fps) and 0 < fps <= 1000, '영상 FPS를 확인할 수 없습니다')
            step = max(1, round(fps * interval_seconds))
            index, imported = 0, 0
            while imported < max_frames:
                success = capture.grab()
                if not success:
                    break
                if index % step == 0:
                    success, bgr = capture.retrieve()
                    require(success and bgr is not None, '영상 프레임을 읽을 수 없습니다')
                    frame_path = Path(temp) / f'frame-{index:08d}.png'
                    require(cv2.imwrite(str(frame_path), bgr), '프레임 저장 실패')
                    identity = stage.import_image(frame_path, session_id)
                    record = next(r for r in stage.data['records'] if r['id'] == identity)
                    record.update(source_name=f'{source.name} / frame {index}', source_uri=source.as_uri(), split=split)
                    record['video_source'] = {
                        'original': f'originals/{video_name}', 'sha256': digest,
                        'frame_index': index, 'nominal_fps': fps,
                        'estimated_seconds': index / fps,
                        'time_basis': 'frame_index/nominal_fps; not sensor time; VFR timing approximate',
                        'sampling_step_frames': step, 'requested_interval_seconds': interval_seconds,
                    }
                    imported += 1
                index += 1
            require(imported > 0, '추출 가능한 프레임이 없습니다')
        finally:
            capture.release()
        updated = deepcopy(collection.data)
        updated['records'].extend(stage.data['records'])
        # UUID filenames preserve existing files. A concurrent commit failure may
        # leave unreferenced files, as with ordinary image import; no labels lost.
        for folder in ('originals', 'images'):
            for file in (stage.root / folder).iterdir():
                destination = collection.root / folder / file.name
                require(not destination.exists(), '수집 파일 이름 충돌')
                shutil.move(str(file), str(destination))
        collection._commit(updated)
    return {'frames': imported, 'limit_reached': imported == max_frames,
            'sampling_step_frames': step, 'video_sha256': digest}

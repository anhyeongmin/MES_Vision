"""Guided raw capture, screen target, asynchronous solve, explicit profile activation."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QImage,QPixmap
from PySide6.QtWidgets import QDialog,QLabel,QVBoxLayout,QHBoxLayout,QCheckBox,QLineEdit
from .widgets import button
from .image_view import ImagePanel
from .lens_calibration import board_image,detect,solve
from mes_vision.training.data import require,write_json,read_json,sha256


class BoardDialog(QDialog):
    def __init__(self,parent):
        super().__init__(parent);self.setWindowTitle('체크보드 · Esc로 닫기')
        layout=QVBoxLayout(self);self.label=QLabel();self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet('background:white');layout.setContentsMargins(0,0,0,0);layout.addWidget(self.label)
        a=board_image();self.picture=QPixmap.fromImage(QImage(a.data,a.shape[1],a.shape[0],a.strides[0],QImage.Format_Grayscale8).copy())
        self.resize(1000,750)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.label.setPixmap(self.picture.scaled(self.size(),Qt.KeepAspectRatio,Qt.FastTransformation))


class LensCalibrationDialog(QDialog):
    def __init__(self,studio):
        super().__init__(studio);self.studio=studio;self.camera=studio.camera
        self.settings=deepcopy(studio.equipment['camera']);self.session=None
        self.samples=[];self.receipts=[];self.last_id=None;self.result=None;self.path=None
        self.future=None;self.kind=None;self.closed=False;self.board=None
        self.pool=ThreadPoolExecutor(max_workers=1);self.folder=Path(studio.owner.root)/'artifacts/camera-calibration'/('screen-'+datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid4().hex[:6])
        self.setWindowTitle('카메라 재보정');self.resize(1000,820)
        lay=QVBoxLayout(self)
        text=QLabel('평면 모니터에 표시하세요. 초점을 먼저 맞추고 이후 고정하세요.\n'
            '패턴 전체와 흰 테두리를 포함해 중앙·상하좌우, 가까이·멀리, 기울어진 구도를 18장 이상 촬영하세요.\n'
            '모니터 하나라면 5초 후 수집을 누르세요. 패턴이 뜨는 동안 카메라 구도를 잡으면 자동 촬영합니다. 별도 모니터에서는 패턴 창을 옮겨 사용할 수 있습니다.');text.setWordWrap(True);lay.addWidget(text)
        self.focus=QLineEdit();self.focus.setPlaceholderText('프로필 이름 / 초점 위치 메모 (필수)');lay.addWidget(self.focus)
        self.confirm=QCheckBox('평면 화면이며 표시 칸의 실제 가로·세로 길이가 같고, 초점을 고정했습니다.');lay.addWidget(self.confirm)
        row=QHBoxLayout();lay.addLayout(row)
        row.addWidget(button('체크보드 띄우기',self.show_board))
        row.addWidget(button('5초 후 수집',lambda:self.guarded(self.delayed_capture)))
        self.capture=button('현재 구도 수집',lambda:self.guarded(self.collect));row.addWidget(self.capture)
        self.calculate=button('계산·검증',lambda:self.guarded(self.calculate_profile));row.addWidget(self.calculate)
        self.activate=button('저장한 보정값 적용',lambda:self.guarded(self.apply_profile));self.activate.setEnabled(False);row.addWidget(self.activate)
        images=QHBoxLayout();lay.addLayout(images,1)
        self.raw=ImagePanel(title='원본 · 보정 전',square=False);self.corrected=ImagePanel(title='새 보정값 미리보기',square=False)
        images.addWidget(self.raw);images.addWidget(self.corrected)
        self.status=QLabel('수집 0장 · 사진 원본을 별도 폴더에 보존합니다.');self.status.setWordWrap(True);lay.addWidget(self.status)
        self.fov_label=QLabel('최종 1:1 검사 영상의 추정 화각: 보정 계산 후 표시');self.fov_label.setWordWrap(True);lay.addWidget(self.fov_label)
        lay.addWidget(button('닫기',self.close))
        self.timer=QTimer(self);self.timer.timeout.connect(self.poll);self.timer.start(100)
        self.delay=QTimer(self);self.delay.setSingleShot(True);self.delay.timeout.connect(lambda:self.guarded(self.collect))

    def guarded(self,fn):
        try:fn()
        except Exception as exc:self.status.setText(str(exc))

    def show_board(self):
        if self.board is None:self.board=BoardDialog(self)
        self.board.show();self.board.raise_()

    def delayed_capture(self):
        require(self.future is None and not self.delay.isActive(),'수집 대기 또는 처리 중입니다.')
        require(self.confirm.isChecked() and self.focus.text().strip(),'화면과 초점 확인 및 프로필 이름을 입력하세요.')
        self.show_board();self.delay.start(5000)
        self.status.setText('5초 후 자동 촬영합니다. 패턴 전체가 보이도록 카메라 구도를 잡으세요.')

    def frame(self):
        require(self.camera is self.studio.camera and self.camera is not None,'카메라가 변경됐습니다. 재보정 창을 다시 여세요.')
        frame,received=self.camera.get_latest_raw()
        require(frame is not None and 0<=time.monotonic()-received<.5,'최신 카메라 원본 영상이 없습니다.')
        require(not frame.transformations and frame.acquisition_identity is None,'재보정에는 보정 전 원본이 필요합니다.')
        require((frame.width,frame.height)==(self.settings['width'],self.settings['height']),'촬영 해상도가 변경됐습니다.')
        require(self.session is None or frame.session_id==self.session,'촬영 중 카메라가 재연결됐습니다. 처음부터 다시 수집하세요.')
        return frame

    def collect(self):
        require(self.future is None,'처리 중입니다.')
        require(self.confirm.isChecked() and self.focus.text().strip(),'화면과 초점 확인 및 프로필 이름을 입력하세요.')
        f=self.frame();require(f.frame_id!=self.last_id,'새 프레임을 기다리세요.')
        self.session=f.session_id;self.pending_rgb=f.rgb.copy();self.pending_id=f.frame_id
        self.future=self.pool.submit(detect,self.pending_rgb);self.kind='detect';self.capture.setEnabled(False)

    def calculate_profile(self):
        require(self.future is None,'처리 중입니다.')
        self.frame();self.result=None;self.activate.setEnabled(False)
        self.future=self.pool.submit(solve,list(self.samples),(self.settings['width'],self.settings['height']));self.kind='solve'
        self.capture.setEnabled(False);self.calculate.setEnabled(False);self.status.setText('보정값 계산 및 별도 사진 검증 중…')

    def poll(self):
        if self.closed:return
        if self.future is not None and self.future.done():
            task=self.future;self.future=None
            try:
                value=task.result()
                if self.kind=='detect':
                    require(all(np.sqrt(np.mean((value-c)**2))>8 for c in self.samples),'이미 수집한 구도와 너무 비슷합니다. 위치·각도를 바꾸세요.')
                    self.folder.mkdir(parents=True,exist_ok=True);path=self.folder/f'raw-{len(self.samples):03d}.png';Image.fromarray(self.pending_rgb).save(path)
                    self.samples.append(value);self.last_id=self.pending_id
                    self.receipts.append(dict(file=path.name,sha256=sha256(path),frame_id=self.last_id,corners=value.tolist()))
                    write_json(self.folder/'capture.json',dict(camera=self.settings,focus_note=self.focus.text(),samples=self.receipts))
                    self.result=None;self.activate.setEnabled(False);self.fov_label.setText('사진 추가됨 · 화각을 다시 계산하세요.');self.status.setText(f'수집 {len(self.samples)}장 · 18장 이상, 화면 위치와 기울기를 고르게 바꾸세요.')
                else:
                    value.update(camera_serial=self.settings['serial'],focus_note=self.focus.text(),source='screen_board',capture_manifest_sha256=sha256(self.folder/'capture.json'))
                    self.path=self.folder/('calibration-'+uuid4().hex[:8]+'.json');write_json(self.path,value);self.result=value
                    f=value['cropped_fov'];w,h=f['image_size']
                    text=f"최종 검사 영상 {w}×{h} · 추정 화각: 가로 {f['h_degrees']:.2f}° / 세로 {f['v_degrees']:.2f}° / 대각 {f['d_degrees']:.2f}°"
                    text+='\n대각은 두 대각선 중 큰 값 · '+('품질 점검 통과' if value['quality_passed'] else '품질 점검 실패 · 참고용')
                    if f['includes_invalid_border']:text+=f" · 유효 픽셀 {f['valid_pixel_fraction']*100:.1f}%: 검은 테두리를 포함한 기하학적 화각입니다."
                    self.fov_label.setText(text)
                    self.activate.setEnabled(value['quality_passed'])
                    self.status.setText(f"계산 RMS {value['train_rms_px']:.3f}px / 검증 최대 {max(value['held_out_errors_px']):.3f}px · "+('품질 점검 통과. 보정 전후를 확인한 뒤 적용하세요. 로봇 좌표·ROI는 재확인이 필요합니다.' if value['quality_passed'] else '품질 점검 실패. 반사·초점·촬영 구도를 확인하고 다시 수집하세요.'))
            except Exception as exc:self.status.setText(str(exc))
            if self.kind=='detect' and self.board:self.board.hide()
            self.capture.setEnabled(True);self.calculate.setEnabled(True)
        try:
            f=self.frame();self.raw.canvas.set_rgb(f.rgb)
            if self.result is not None:
                import cv2
                rgb=cv2.undistort(f.rgb,np.array(self.result['K']),np.array(self.result['D']))
                self.corrected.canvas.set_rgb(rgb)
        except Exception:pass

    def apply_profile(self):
        require(self.result is not None and self.result['quality_passed'],'검증을 통과한 보정값이 없습니다.')
        require(read_json(self.path)==self.result,'저장된 보정 파일이 변경됐습니다. 다시 계산하세요.')
        self.frame();require(self.confirm.isChecked(),'초점 고정 상태를 확인하세요.')
        owner=self.studio.owner;root=Path(owner.root)
        require(owner.equipment['camera']['serial']==self.settings['serial'],'장비 선택이 변경됐습니다.')
        from .acquisition import acquisition_config_path
        config=acquisition_config_path(root);old=config.read_bytes() if config.exists() else None
        if old is not None:(self.folder/'previous-acquisition.json').write_bytes(old)
        write_json(self.folder/'previous-equipment.json',owner.equipment)
        spec=dict(kind='undistorted_center_square_v1',sensor_size=self.result['image_size'],calibration_file=str(self.path.resolve()),calibration_sha256=sha256(self.path),camera_serial=self.settings['serial'])
        temp=config.with_suffix('.new');write_json(temp,dict(schema_version=1,processing=spec));temp.replace(config)
        try:
            from .acquisition import configure_equipment
            e=deepcopy(owner.equipment)
            for k in ('width','height','fps','pixel_format'):e['camera'][k]=self.settings[k]
            e['camera']['acquisition_revision']='acq-'+uuid4().hex[:16]
            e=configure_equipment(root,e);owner.equipment=owner.store.save_equipment(e)
        except Exception:
            if old is not None:config.write_bytes(old)
            else:config.unlink(missing_ok=True)
            raise
        self.studio.equipment=deepcopy(owner.equipment);owner.refresh_equipment();self.studio.saved=True
        self.studio.reopen();self.close()
        self.studio.message.setText('새 보정값 적용 · 카메라 재연결 중. ROI·로봇 좌표 보정과 모델 성능을 다시 확인하세요.')

    def closeEvent(self,event):
        self.closed=True;self.timer.stop();self.delay.stop();self.pool.shutdown(wait=False,cancel_futures=True)
        if self.board:self.board.close()
        event.accept()
    def reject(self):self.close()

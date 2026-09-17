"""Camera setup and labelled video collection in a bounded camera process."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
import time
from PySide6.QtCore import QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QLabel,QComboBox,
    QCheckBox,QSpinBox,QLineEdit,QTabWidget,QWidget,QMessageBox,QScrollArea,QApplication)
from mes_vision.i18n import tr
from mes_vision.qt_i18n import ui_text
from .widgets import button,Canvas
from .camera import CameraProcess
from .capture_controls import validate_controls


class CaptureDialog(QDialog):
    def __init__(self, owner, *, collect=False):
        super().__init__(owner); self.owner=owner; self.camera=None; self.closing=False
        self.pending={}; self.ranges={}; self.recording=False; self.record_started=None; self.last_frame=None
        self.applied=None; self.loading=False; self.current='overview'; self.saved=False
        self.equipment=deepcopy(owner.equipment); c=self.equipment['camera']
        base={k:c.get(k) for k in ('auto_exposure','exposure','gain')}
        self.profiles=deepcopy(c.get('capture_profiles',{'overview':base,'detail':base}))
        self.root=Path(owner.root)/'collections'/'recordings'
        self.reconnect=owner.camera is not None
        ui_text(self.setWindowTitle,tr('촬영 설정 · 학습데이터 수집')); self.resize(1000,800)
        layout=QVBoxLayout(self)
        connection=QHBoxLayout(); self.mode=QComboBox()
        from .camera_modes import UVC_PRESETS
        for fmt,sizes in UVC_PRESETS.items():
            if fmt!='MJPG': continue
            for (width,height),rates in sizes.items():
                for fps in rates: self.mode.addItem(f'{width} × {height} · {fmt} · {fps} fps',(fmt,width,height,fps))
        self.mode.setCurrentIndex(max(0,self.mode.findData(('MJPG',c['width'],c['height'],30))))
        connection.addWidget(self.mode,1); self.reconnect_button=button(tr('선택 모드로 재연결'),self.reopen)
        connection.addWidget(self.reconnect_button); layout.addLayout(connection)
        self.canvas=Canvas(); self.canvas.setMinimumHeight(220); layout.addWidget(self.canvas,1)
        self.tabs=QTabWidget(); scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(self.tabs)
        scroll.setMinimumHeight(250); layout.addWidget(scroll)
        setup=QWidget(); form=QFormLayout(setup); ui_text(self.tabs.addTab,setup,tr('촬영 설정'))
        self.profile=QComboBox()
        for label,key in [('전체 촬영','overview'),('상세 촬영','detail')]: ui_text(self.profile.addItem,tr(label),key)
        ui_text(form.addRow,tr('촬영 조건'),self.profile)
        self.auto=ui_text(QCheckBox,tr('자동 노출')); ui_text(form.addRow,self.auto)
        self.exposure=QSpinBox(); self.gain=QSpinBox(); self.use_gain=ui_text(QCheckBox,tr('게인 수동 설정'))
        ui_text(form.addRow,tr('노출'),self.exposure); self.use_gain.hide(); ui_text(form.addRow,tr('게인'),self.gain)
        self.exposure.setKeyboardTracking(False); self.gain.setKeyboardTracking(False)
        self.range_note=QLabel(); self.range_note.setWordWrap(True); ui_text(form.addRow,self.range_note)
        self.apply_button=button(tr('미리보기에 적용'),self.apply); ui_text(form.addRow,self.apply_button)
        self.save_button=button(tr('전체·상세 촬영 조건 저장'),self.save_profiles); ui_text(form.addRow,self.save_button)
        self.calibrate_button=button(tr('카메라 재보정 · 체크보드'),self.open_calibration); form.addRow(self.calibrate_button)
        collect_page=QWidget(); f=QFormLayout(collect_page); ui_text(self.tabs.addTab,collect_page,tr('학습데이터 수집'))
        self.product=QComboBox()
        for product in owner.store.products(): self.product.addItem(product['name'],product['id'])
        if owner.product: self.product.setCurrentIndex(self.product.findData(owner.product['id']))
        ui_text(f.addRow,tr('품목'),self.product)
        self.label=QComboBox()
        for code,name in [('OK','정상'),('NG01','누락'),('NG02','추가 재료'),('NG03','균열'),('NG04','벽 변형'),('NG05','구멍 불량'),('NG06','표면 불량'),('MIXED','여러 종류 혼합'),('UNKNOWN','미확인')]:
            ui_text(self.label.addItem,code+' · '+tr(name),code)
        ui_text(f.addRow,tr('물체 분류'),self.label)
        self.purpose=QComboBox()
        for label,key in [('학습용','train'),('독립 평가용','evaluation')]: ui_text(self.purpose.addItem,tr(label),key)
        ui_text(f.addRow,tr('수집 목적'),self.purpose)
        self.specimen=QLineEdit(); ui_text(self.specimen.setPlaceholderText,tr('같은 실물은 같은 번호 사용 · 선택 사항'))
        ui_text(f.addRow,tr('실물 번호'),self.specimen)
        self.note=QLineEdit(); ui_text(f.addRow,tr('촬영 메모'),self.note)
        self.active_profile=QLabel(); ui_text(f.addRow,tr('현재 촬영 조건'),self.active_profile)
        note=ui_text(QLabel,tr('영상 분류는 선택한 물체 종류입니다. 불량 영역 라벨은 추후 검토가 필요합니다. 원본 영상을 저장하며 왜곡 보정은 데이터 준비 시 적용합니다.'))
        note.setWordWrap(True); ui_text(f.addRow,note)
        row=QHBoxLayout(); self.record_button=button(tr('녹화 시작'),self.start_record); self.stop_button=button(tr('녹화 중단 · 저장'),self.stop_record)
        row.addWidget(self.record_button); row.addWidget(self.stop_button); f.addRow(row)
        self.elapsed=QLabel(); f.addRow(self.elapsed)
        f.addRow(button(tr('수집 폴더 열기'),self.open_folder))
        self.message=QLabel(); self.message.setWordWrap(True); layout.addWidget(self.message)
        layout.addWidget(button(tr('닫기'),self.close))
        self.profile.currentIndexChanged.connect(self.switch_profile)
        self.mode.currentIndexChanged.connect(self.changed)
        for widget in (self.auto,self.use_gain): widget.toggled.connect(self.changed)
        for widget in (self.exposure,self.gain): widget.valueChanged.connect(self.changed)
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(40)
        self.tabs.setCurrentIndex(1 if collect else 0)
        self.disable_controls(); self.message.setText(tr('카메라 연결 준비 중…'))
        available=QApplication.primaryScreen().availableGeometry()
        self.resize(min(1000,available.width()-40),min(800,available.height()-60))
        if owner.camera: owner.disconnect_camera()
        QTimer.singleShot(0,self.connect_when_ready)

    def open_calibration(self):
        try:
            if self.recording or self.pending or self.closing or not self.camera or not self.ranges:
                raise ValueError(tr('녹화를 중단하고 카메라 연결이 완료된 뒤 재보정하세요.'))
            from .lens_calibration_dialog import LensCalibrationDialog
            dialog=LensCalibrationDialog(self)
            try: dialog.exec()
            finally: dialog.deleteLater()
        except Exception as exc: self.message.setText(str(exc))

    def open_folder(self):
        self.root.mkdir(parents=True,exist_ok=True); QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.root)))

    def disable_controls(self):
        for w in (self.auto,self.exposure,self.gain,self.use_gain,self.apply_button,self.save_button,self.record_button,self.stop_button): w.setEnabled(False)

    def connect_when_ready(self):
        if self.closing: return
        if self.owner.camera is not None: QTimer.singleShot(100,self.connect_when_ready); return
        if self.camera is not None and not self.camera.disposed: QTimer.singleShot(100,self.connect_when_ready); return
        try:
            fmt,width,height,fps=self.mode.currentData()
            self.equipment['camera'].update(pixel_format=fmt,width=width,height=height,fps=fps)
            self.camera=CameraProcess(self.equipment['camera'])
            self.camera.connected.connect(lambda _:self.send('ranges',{},'ranges'))
            self.camera.command_finished.connect(self.reply); self.camera.failed.connect(self.failed)
            self.camera.finished.connect(self.camera_finished); self.camera.start()
        except Exception as exc: self.failed(str(exc))

    def reopen(self):
        if self.recording or self.pending or self.closing: return
        self.ranges={}; self.applied=None; self.last_frame=None; self.disable_controls()
        if self.camera and not self.camera.disposed: self.camera.stopping.set()
        self.message.setText(tr('카메라 연결 준비 중…')); QTimer.singleShot(0,self.connect_when_ready)

    def send(self,kind,payload,action):
        token=self.camera.command(kind,payload); self.pending[token]=(action,time.monotonic())
        self.changed(); return token

    def reply(self,event):
        action,_=self.pending.pop(event['id'],(None,None))
        if action is None: return
        if 'error' in event:
            if action=='stop': self.recording=False; self.record_started=None
            self.message.setText(event['error']); self.changed(); return
        result=event['result']
        if action=='ranges':
            self.ranges=result
            for values in self.profiles.values():
                if values.get('gain') is None and result.get('gain',{}).get('manual'):
                    values['gain']=result['gain']['default']
            self.load_profile(); self.message.setText(tr('지원 범위를 확인했습니다. 조건을 적용한 뒤 녹화하세요.'))
        elif action=='apply':
            self.applied=(self.current,deepcopy(result['values'])); self.profiles[self.current]=deepcopy(result['values'])
            self.message.setText(tr('촬영 조건을 적용했습니다. 영상에서 밝기와 불량 부위를 확인하세요.'))
        elif action=='start':
            self.recording=True; self.record_started=time.monotonic(); self.message.setText(tr('녹화 중: ')+result['folder'])
        elif action=='stop':
            self.recording=False; self.record_started=None; self.message.setText(tr('영상과 촬영 정보를 저장했습니다: ')+result['folder'])
        self.changed()

    def values(self):
        return {'auto_exposure':self.auto.isChecked(),'exposure':None if self.auto.isChecked() else self.exposure.value(),
                'gain':self.gain.value() if self.use_gain.isChecked() else None}

    def load_profile(self):
        self.loading=True; v=self.profiles[self.current]
        self.auto.setChecked(v['auto_exposure']); self.use_gain.setChecked(self.ranges.get('gain',{}).get('manual',False))
        descriptions=[]
        for key,w in [('exposure',self.exposure),('gain',self.gain)]:
            bounds=self.ranges.get(key,{})
            if 'min' in bounds:
                w.setRange(bounds['min'],bounds['max']); w.setSingleStep(bounds['step'])
                w.setValue(v.get(key) if v.get(key) is not None else bounds['default'])
                descriptions.append(key+f": {bounds['min']} ~ {bounds['max']} / {bounds['step']}")
            else: descriptions.append(key+': '+tr('지원 범위 확인 불가'))
        self.range_note.setText(' | '.join(descriptions)); self.loading=False; self.changed()

    def switch_profile(self):
        self.current=self.profile.currentData(); self.load_profile()

    def changed(self,*_):
        if self.loading: return
        c=self.equipment['camera']
        connected_mode=(c.get('pixel_format','MJPG'),c['width'],c['height'],c['fps'])
        idle=not self.pending and not self.recording and not self.closing and bool(self.ranges) and self.mode.currentData()==connected_mode
        self.mode.setEnabled(not self.recording and not self.pending and not self.closing)
        self.reconnect_button.setEnabled(self.mode.isEnabled())
        bounds=self.ranges.get('exposure',{})
        self.auto.setEnabled(idle and bounds.get('auto',False) and bounds.get('manual',False))
        self.exposure.setEnabled(idle and bounds.get('manual',False) and not self.auto.isChecked())
        self.use_gain.setEnabled(False)
        self.gain.setEnabled(idle and self.ranges.get('gain',{}).get('manual',False))
        for w in (self.profile,self.product,self.label,self.purpose,self.specimen,self.note): w.setEnabled(not self.recording and not self.pending and not self.closing)
        valid=False
        try: validate_controls(self.values(),self.ranges); valid=True
        except ValueError: pass
        self.apply_button.setEnabled(idle and valid); self.save_button.setEnabled(idle and self.applied is not None)
        self.record_button.setEnabled(idle and valid and self.product.currentData() is not None and self.applied==(self.current,self.values()))
        self.stop_button.setEnabled(self.recording and not self.pending and not self.closing)
        self.active_profile.setText(self.profile.currentText()+(' · '+tr('적용됨') if self.applied==(self.current,self.values()) else ' · '+tr('적용 필요')))

    def apply(self):
        try:
            validate_controls(self.values(),self.ranges); self.applied=None
            self.send('controls',self.values(),'apply')
        except Exception as exc: self.message.setText(str(exc))

    def save_profiles(self):
        try:
            if self.applied!=(self.current,self.values()): raise ValueError(tr('변경한 조건을 먼저 적용하세요.'))
            for values in self.profiles.values(): validate_controls(values,self.ranges)
            e=deepcopy(self.owner.equipment); e['camera']['capture_profiles']=deepcopy(self.profiles)
            for key in ('pixel_format','width','height','fps'): e['camera'][key]=self.equipment['camera'][key]
            e['camera'].update(self.profiles['overview'])
            from .settings_ui import assign_acquisition_revision
            assign_acquisition_revision(self.owner.equipment['camera'],e['camera'],'acq-'+uuid4().hex[:16])
            self.owner.equipment=self.owner.store.save_equipment(e); self.equipment=deepcopy(self.owner.equipment)
            self.owner.refresh_equipment(); self.saved=True
            self.message.setText(tr('전체·상세 조건을 저장했습니다. 다음 연결과 촬영에 적용됩니다.'))
        except Exception as exc: self.message.setText(str(exc))

    def start_record(self):
        try:
            if self.applied!=(self.current,self.values()): raise ValueError(tr('촬영 조건을 먼저 적용하세요.'))
            metadata={'product_id':self.product.currentData(),'product_name':self.product.currentText(),
                'label':self.label.currentData(),'purpose':self.purpose.currentData(),'specimen_id':self.specimen.text().strip(),
                'note':self.note.text().strip(),'capture_profile':self.current,'session_id':uuid4().hex,
                'equipment_version':self.equipment['version']}
            self.send('record_start',{'root':str(self.root),'metadata':metadata},'start')
        except Exception as exc: self.message.setText(str(exc))

    def stop_record(self):
        try: self.send('record_stop',{},'stop')
        except Exception as exc: self.failed(str(exc))

    def failed(self,message):
        self.pending.clear(); self.recording=False; self.ranges={}; self.message.setText(message); self.disable_controls()

    def tick(self):
        if self.pending and time.monotonic()-min(v[1] for v in self.pending.values())>15:
            self.failed(tr('카메라 요청 시간이 초과됐습니다. 창을 닫고 다시 연결하세요.'))
            if self.camera: self.camera.stopping.set()
        if self.camera:
            frame,_=self.camera.get_latest()
            if frame is not None and frame.frame_id!=self.last_frame:
                self.canvas.set_rgb(frame.rgb); self.last_frame=frame.frame_id
        if self.record_started and self.recording: self.elapsed.setText(f'{time.monotonic()-self.record_started:.1f} s')

    def closeEvent(self,event):
        if self.recording or self.pending:
            self.message.setText(tr('녹화를 중단하고 저장이 끝난 뒤 닫으세요.')); event.ignore(); return
        self.closing=True; self.disable_controls()
        if self.camera and not self.camera.disposed:
            self.camera.stopping.set(); event.ignore(); return
        self.timer.stop(); event.accept()

    def reject(self): self.close()

    def camera_finished(self):
        if self.closing:
            self.close()
            if self.reconnect: self.owner.guarded(self.owner.connect_camera)

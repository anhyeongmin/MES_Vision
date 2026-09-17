"""Operator registration of measured moving-camera geometry and capture timing."""
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from PySide6.QtWidgets import QDialog,QVBoxLayout,QLabel,QLineEdit,QDialogButtonBox,QMessageBox
from mes_vision.i18n import tr,trf
from mes_vision.qt_i18n import ui_text
from mes_vision.training.data import require
from mes_vision.operation.widgets import PathField,button
from mes_vision.operation.settings_ui import SettingsPages,optional_number,optional_value
from .settings import StationSettings
from .calibration import MountedCalibration,build_bundle,save_bundle


def note(layout,text):
    label=ui_text(QLabel,tr(text)); label.setWordWrap(True); layout.addWidget(label); return label


def field(form,label,widget):
    ui_text(form.addRow,tr(label),widget); return widget


def controls(dialog,layout):
    box=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel)
    ui_text(box.button(QDialogButtonBox.Save).setText,tr('저장'))
    ui_text(box.button(QDialogButtonBox.Cancel).setText,tr('취소'))
    box.accepted.connect(dialog.save); box.rejected.connect(dialog.reject); layout.addWidget(box)


class StationSettingsDialog(QDialog):
    def __init__(self,runtime,parent=None):
        super().__init__(parent); self.runtime=Path(runtime); self.settings=StationSettings(runtime)
        ui_text(self.setWindowTitle,tr('로봇 촬영 설정')); self.resize(740,600)
        layout=QVBoxLayout(self)
        note(layout,'전체 촬영 → 위치 검출 → 물체별 근접 촬영 순서에 사용할 설정입니다. 저장만으로 장치는 움직이지 않습니다.')
        pages=SettingsPages(layout); pages.advanced.hide(); form=pages.form(tr('촬영 조건'))
        self.bundle=field(form,'촬영·집기 보정 파일',PathField(self.settings.value['calibration_bundle']))
        form.addRow(button(tr('촬영·집기 보정 만들기'),self.create_bundle))
        self.summary=QLabel(); self.summary.setWordWrap(True); form.addRow(self.summary)
        self.bundle.edit.textChanged.connect(self.describe)
        v=self.settings.value
        self.settle=field(form,'이동 후 안정 시간(초)',optional_number(v['settle_seconds'],.1,10,3))
        self.timeout=field(form,'단계 제한 시간(초)',optional_number(v['action_timeout_seconds'],1,300,2))
        self.reference=field(form,'촬영 동작 검증 기록',QLineEdit(v['validation_reference']))
        timing=pages.form(tr('프레임 시점 검증'))
        info=ui_text(QLabel,tr('선택한 해상도·노출 조건에서 노출부터 영상 수신까지의 최대 지연을 측정해 등록하세요. 이 값이 없으면 로봇 이동 후 촬영을 시작할 수 없습니다.'))
        info.setWordWrap(True); timing.addRow(info)
        self.latency=field(timing,'검증된 영상 지연 상한(초)',optional_number(v['max_frame_age_seconds'],.001,2,3))
        self.timing_ref=field(timing,'영상 지연 검증 기록',QLineEdit(v['timing_validation_reference']))
        note(layout,'장비 설치 전에는 미등록 상태로 저장할 수 있습니다. 전체 촬영 위치·장착·렌즈·해상도·검사면 높이가 달라지면 보정을 다시 확인하세요.')
        controls(self,layout); self.describe()

    def describe(self):
        if not self.bundle.value(): ui_text(self.summary.setText,tr('촬영·집기 보정 미등록')); return
        try:
            c=MountedCalibration.load(self.bundle.value()); v=c.value; p=v['overview_pose']
            ui_text(self.summary.setText,trf('품목: {product}\n전체 촬영 X / Y / Z / R: {x} / {y} / {z} / {r}\n근접 촬영 Z: {detail} · 집기 Z: {pick} · 이동 Z: {travel}',
                product=c.product_id,**p,detail=v['detail_z'],pick=v['pick_z'],travel=v['travel_z']))
        except Exception:
            ui_text(self.summary.setText,tr('보정 파일을 확인할 수 없습니다. 검증된 파일을 다시 선택하세요.'))

    def create_bundle(self):
        dialog=CalibrationBundleDialog(self.runtime,self)
        try:
            if dialog.exec(): self.bundle.edit.setText(str(dialog.path))
        finally: dialog.deleteLater()

    def collect(self):
        value=deepcopy(self.settings.value)
        value.update(calibration_bundle=self.bundle.value(),settle_seconds=optional_value(self.settle),
            action_timeout_seconds=optional_value(self.timeout),validation_reference=self.reference.text().strip(),
            max_frame_age_seconds=optional_value(self.latency),timing_validation_reference=self.timing_ref.text().strip())
        return value

    def save(self):
        try: self.settings.save(self.collect()); self.accept()
        except Exception as exc: QMessageBox.warning(self,tr('설정 저장 실패'),str(exc))


class CalibrationBundleDialog(QDialog):
    def __init__(self,runtime,parent=None):
        super().__init__(parent); self.runtime=Path(runtime); self.path=None
        ui_text(self.setWindowTitle,tr('촬영·집기 보정 만들기')); self.resize(760,680)
        layout=QVBoxLayout(self)
        note(layout,'같은 전체사진의 기준점을 사용해 카메라 중심을 맞춘 위치와 집게를 맞춘 위치를 각각 측정한 보정 파일이 필요합니다. 각 보정은 별도의 검증점을 통과해야 합니다.')
        pages=SettingsPages(layout); pages.advanced.hide(); maps=pages.form(tr('측정된 좌표 보정'))
        self.capture=field(maps,'전체사진 → 카메라 중심 보정',PathField())
        self.pick=field(maps,'전체사진 → 집기점 보정',PathField())
        self.version=field(maps,'보정 버전',QLineEdit())
        self.mount=field(maps,'카메라 장착 부위 기록',QLineEdit())
        self.reference=field(maps,'전체 이동 경로·촬영 검증 기록',QLineEdit())
        poses=pages.form(tr('촬영·이동 자세'))
        self.numbers={}
        for key,label,low,high in (
            ('x','전체 촬영 X(mm)',-1000,1000),('y','전체 촬영 Y(mm)',-1000,1000),
            ('z','전체 촬영 Z(mm)',-1000,1000),('r','전체 촬영 R(도)',-360,360),
            ('detail_z','근접 촬영 Z(mm)',-1000,1000),('detail_r','근접 촬영 R(도)',-360,360),
            ('pick_z','집기 Z(mm)',-1000,1000),('pick_r','집기 R(도)',-360,360),
            ('travel_z','이동 Z(mm)',-1000,1000),('u','물체 내 집기 가로 비율',.001,.999),
            ('v','물체 내 집기 세로 비율',.001,.999)):
            self.numbers[key]=field(poses,label,optional_number(None,low,high,3))
        self.grasp_version=field(poses,'집기 기준 버전',QLineEdit())
        note(layout,'수치는 로봇 도구 좌표 기준입니다. 이 입력 범위는 장비의 이동 가능 범위가 아닙니다. 등록된 로봇 운전 범위와 검증된 이동 경로를 실행 전에 확인합니다.')
        controls(self,layout)

    def collect(self):
        values={key:optional_value(widget) for key,widget in self.numbers.items()}
        require(all(x is not None for x in values.values()) and self.capture.value() and self.pick.value(),tr('측정된 두 보정 파일과 모든 촬영·집기 수치를 등록하세요.'))
        return build_bundle(self.capture.value(),self.pick.value(),version=self.version.text().strip(),
            overview_pose={k:values[k] for k in ('x','y','z','r')},
            **{k:values[k] for k in ('detail_z','detail_r','pick_z','pick_r','travel_z')},
            grasp_uv=[values['u'],values['v']],grasp_version=self.grasp_version.text().strip(),
            mount_link=self.mount.text().strip(),validation_reference=self.reference.text().strip())

    def save(self):
        try:
            value=self.collect(); folder=self.runtime/'station-calibrations'; folder.mkdir(parents=True,exist_ok=True)
            path=folder/(uuid4().hex+'.json'); save_bundle(path,value); self.path=path; self.accept()
        except Exception as exc: QMessageBox.warning(self,tr('보정 등록 실패'),str(exc))

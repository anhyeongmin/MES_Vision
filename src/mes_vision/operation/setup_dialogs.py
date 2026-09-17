from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,QWidget,QLineEdit,QComboBox,
    QLabel,QCheckBox,QTableWidgetItem,QAbstractItemView,QDialogButtonBox,QInputDialog,QFileDialog)
from mes_vision.training.data import require,write_json,read_json
from .widgets import button,number,integer,PathField,Canvas,table
from .device_controls import DeviceControls, DeviceSelector
from PySide6.QtCore import QTimer
from .settings_ui import SettingsPages,add_review_controls,assign_acquisition_revision


class EquipmentDialog(DeviceControls,QDialog):
    def __init__(self,equipment,frame=None,parent=None,*,discover=False):
        super().__init__(parent); self.value=deepcopy(equipment); self.original=deepcopy(equipment); ui_text(self.setWindowTitle, tr('장비 · 작업 환경 설정')); self.resize(960,810)
        layout=QVBoxLayout(self); self.init_discovery(layout); self.pages=SettingsPages(layout); form=self.pages.form
        self.revision_token='acq-'+uuid4().hex[:16]
        c=equipment["camera"]; camera=form(tr('카메라'))
        self.driver=QComboBox(); ui_text(self.driver.addItem,tr('U20CAM / USB 카메라'),'uvc'); ui_text(self.driver.addItem,'D405','d405')
        self.driver.setCurrentIndex(self.driver.findData(c.get('driver','d405')))
        ui_text(camera.addRow,tr('카메라 종류'),self.driver)
        self.pixel_format=QComboBox(); ui_text(self.pixel_format.addItem,'MJPEG','MJPG'); ui_text(self.pixel_format.addItem,'YUY2','YUY2')
        self.pixel_format.setCurrentIndex(self.pixel_format.findData(c.get('pixel_format','MJPG')))
        self.pixel_format.setEnabled(self.driver.currentData()=='uvc'); ui_text(camera.addRow,tr('영상 전송 형식'),self.pixel_format)
        self.serial=DeviceSelector(c["serial"]); ui_text(camera.addRow, tr('카메라 장치'),self.serial)
        from .camera_modes import COLOR_PRESETS
        self.resolution=QComboBox(); self.fps=QComboBox()
        current=(c['width'],c['height'])
        for size in COLOR_PRESETS: ui_text(self.resolution.addItem, f'{size[0]} × {size[1]}',f'{size[0]}x{size[1]}')
        if current not in COLOR_PRESETS:
            ui_text(self.resolution.addItem, f'{current[0]} × {current[1]} · '+tr('기존 설정 · 장치 확인 필요'),f'{current[0]}x{current[1]}')
        self.resolution.setCurrentIndex(self.resolution.findData(f'{current[0]}x{current[1]}'))
        self.update_frame_rates(c['fps'])
        self.resolution.currentIndexChanged.connect(lambda _:self.update_frame_rates())
        ui_text(camera.addRow, tr('영상 해상도'),self.resolution); ui_text(camera.addRow, tr('초당 프레임'),self.fps)
        self.mode_hint=ui_text(QLabel); self.mode_hint.setWordWrap(True); ui_text(camera.addRow, self.mode_hint)
        self.serial.currentTextChanged.connect(lambda _:self.refresh_camera_modes())
        self.refresh_camera_modes()
        self.auto_exposure=ui_text(QCheckBox, tr('자동 노출')); self.auto_exposure.setChecked(c["auto_exposure"]); ui_text(camera.addRow, self.auto_exposure)
        self.exposure=number(c["exposure"],-20,200000); ui_text(camera.addRow, tr('수동 노출(장치 단위)'),self.exposure)
        self.driver.currentIndexChanged.connect(self.change_camera_driver)
        self.pixel_format.currentIndexChanged.connect(self.change_pixel_format)
        camera_advanced=form(tr("카메라 설치 · 고급"),advanced=True)
        self.camera_text={}
        for key,label in (("mount_revision",tr('카메라 설치 버전')),("acquisition_revision",tr('촬영 조건 버전')),("robot_base_id",tr('로봇 기준 좌표계 이름')),("tool_frame_id",tr('집기 도구 좌표계 이름'))):
            field=QLineEdit(c[key]); self.camera_text[key]=field; ui_text(camera_advanced.addRow, label,field)
        self.camera_text['acquisition_revision'].setReadOnly(True)
        ui_text(self.camera_text['acquisition_revision'].setPlaceholderText, tr('촬영 설정 변경 시 자동 생성'))
        ui_text(camera_advanced.addRow, ui_text(QLabel, tr('설치 위치를 바꾸면 설치 버전을 새로 입력하고 보정을 다시 확인하세요.')))
        ui_text(camera.addRow, ui_text(QLabel, tr('촬영 조건 변경 시 번호가 자동 갱신됩니다. 기존 좌표 보정은 다시 확인해야 합니다.')))
        self.auto_exposure.toggled.connect(lambda checked:self.exposure.setEnabled(not checked)); self.exposure.setEnabled(not self.auto_exposure.isChecked())
        workspace=form(tr('작업 영역'))
        self.mm_width=number(equipment["workspace"]["width_mm"]); self.mm_height=number(equipment["workspace"]["height_mm"])
        ui_text(workspace.addRow, tr('작업 공간 가로(mm)'),self.mm_width); ui_text(workspace.addRow, tr('작업 공간 세로(mm)'),self.mm_height)
        self.workspace_ref=QLineEdit(equipment["workspace"]["validation_reference"]); ui_text(workspace.addRow, tr('영역 검증 기록'),self.workspace_ref)
        self.canvas=Canvas()
        if frame: self.canvas.set_rgb(frame.rgb)
        self.canvas.roi=self.value["workspace"]["roi"]; self.canvas.excluded=self.value["workspace"]["excluded"]
        self.canvas.rectangle_selected.connect(self.set_region); ui_text(workspace.addRow, self.canvas)
        actions=QWidget(); h=QHBoxLayout(actions)
        h.addWidget(button(tr('검사 영역 그리기'),lambda:self.set_draw("roi"))); h.addWidget(button(tr('제외 영역 추가'),lambda:self.set_draw("exclude")))
        h.addWidget(button(tr('영역 초기화'),self.clear_regions)); ui_text(workspace.addRow, actions)
        tracking=form(tr('추적 · 고급 설정'),advanced=True); self.tracking={}
        for key,label,low,high in (("stable_seconds",tr('안정 확인 시간(초)'),.1,10),("max_gap",tr('추적 연결 최대 간격(초)'),.1,10),
                ("motion_px",tr('이동 판별 기준(px)'),.1,100),("appearance_delta",tr('외관 변경 기준'),.01,1)):
            field=number(equipment["tracking"][key],low,high,3); self.tracking[key]=field; ui_text(tracking.addRow, label,field)
        robot=form("Dobot")
        self.port=DeviceSelector(equipment["robot"]["port"]); ui_text(robot.addRow, tr('통신 포트'),self.port)
        port_hint=ui_text(QLabel, tr('통신 포트 목록만으로 Dobot을 식별할 수 없습니다. 실제 연결된 로봇의 포트를 선택하세요.')); port_hint.setWordWrap(True); ui_text(robot.addRow, port_hint)
        robot_advanced=form(tr("로봇 운전 · 고급"),advanced=True)
        self.di=integer(equipment["robot"]["input_address"] or 0,0,20); ui_text(self.di.setSpecialValueText, tr('미등록')); ui_text(robot_advanced.addRow, tr('집기 확인 디지털 입력'),self.di)
        self.level=QComboBox(); ui_text(self.level.addItems, [tr('LOW일 때 집기 확인'),tr('HIGH일 때 집기 확인')]); self.level.setCurrentIndex(equipment["robot"]["holding_level"]); ui_text(robot_advanced.addRow, tr('센서 신호'),self.level)
        self.pose_tol=number(equipment["robot"]["pose_tolerance_mm"],.01,10,3); self.r_tol=number(equipment["robot"]["rotation_tolerance_deg"],.01,10,3)
        ui_text(robot_advanced.addRow, tr('도달 위치 허용 오차(mm)'),self.pose_tol); ui_text(robot_advanced.addRow, tr('방향 허용 오차(도)'),self.r_tol)
        self.speed=number(equipment["robot"].get("speed_ratio"),0,100); self.acceleration=number(equipment["robot"].get("acceleration_ratio"),0,100)
        self.motion_ref=QLineEdit(equipment["robot"].get("motion_validation_reference",""))
        ui_text(robot_advanced.addRow, tr('검증된 이동 속도 비율(%)'),self.speed); ui_text(robot_advanced.addRow, tr('검증된 가속도 비율(%)'),self.acceleration); ui_text(robot_advanced.addRow, tr('속도 검증 기록'),self.motion_ref)
        self.profile=PathField(equipment["robot"]["profile"],filter=tr('로봇 설정 (*.json)')); ui_text(robot.addRow, tr('검증된 로봇 운전 설정'),self.profile)
        self.calibration=PathField(equipment["calibration"],filter=tr('좌표 보정 (*.json)')); ui_text(robot.addRow, tr('활성 좌표 보정'),self.calibration)
        add_review_controls(self,layout,self.original,self.collect,'equipment')
        self.message=ui_text(QLabel); self.message.setWordWrap(True); layout.addWidget(self.message)
        box=QDialogButtonBox(QDialogButtonBox.Save|QDialogButtonBox.Cancel); box.accepted.connect(self.save); box.rejected.connect(self.reject); layout.addWidget(box); self.save_box=box
        ui_text(box.button(QDialogButtonBox.Save).setText,tr('저장'))
        ui_text(box.button(QDialogButtonBox.Cancel).setText,tr('취소'))
        if discover: QTimer.singleShot(0,self.search_devices)
    def set_draw(self,mode): self.canvas.edit_mode=mode
    def selected_resolution(self):
        value=self.resolution.currentData()
        return tuple(map(int,value.split('x'))) if value else None
    def set_region(self,region):
        self.change_preview.hide()
        if region["mode"]=="roi": self.value["workspace"]["roi"]=region["polygon"]
        else: self.value["workspace"]["excluded"].append(region["polygon"])
        self.canvas.roi=self.value["workspace"]["roi"]; self.canvas.excluded=self.value["workspace"]["excluded"]; self.canvas.update()
    def clear_regions(self):
        self.change_preview.hide()
        self.value["workspace"]["roi"]=[]; self.value["workspace"]["excluded"]=[]
        self.canvas.roi=[]; self.canvas.excluded=[]; self.canvas.update()
    def collect(self):
        value=deepcopy(self.value); old=self.original["camera"]; c=value["camera"]
        size=self.selected_resolution(); rate=self.fps.currentData()
        require(size is not None and rate is not None,tr('해상도와 프레임 속도를 선택하세요.'))
        c.update(driver=self.driver.currentData(),pixel_format=self.pixel_format.currentData(),serial=self.serial.identifier(),width=size[0],height=size[1],fps=rate,
            auto_exposure=self.auto_exposure.isChecked(),exposure=None if self.auto_exposure.isChecked() else self.exposure.value())
        c.update({k:w.text().strip() for k,w in self.camera_text.items()})
        assign_acquisition_revision(old,c,self.revision_token)
        ui_text(self.camera_text['acquisition_revision'].setText, c['acquisition_revision'])
        value["workspace"].update(width_mm=self.mm_width.value() or None,height_mm=self.mm_height.value() or None,validation_reference=self.workspace_ref.text().strip())
        value["tracking"]={k:w.value() for k,w in self.tracking.items()}
        value["robot"].update(port=self.port.identifier(),input_address=self.di.value() or None,holding_level=self.level.currentIndex(),
            pose_tolerance_mm=self.pose_tol.value(),rotation_tolerance_deg=self.r_tol.value(),profile=self.profile.value(),
            speed_ratio=self.speed.value() or None,acceleration_ratio=self.acceleration.value() or None,motion_validation_reference=self.motion_ref.text().strip())
        value["calibration"]=self.calibration.value()
        return value
    def save(self):
        try:
            require(not self.discovery_read.busy,tr('장치를 검색하고 있습니다…'))
            require(self.selected_mode_supported(),tr('선택한 해상도·FPS는 검색된 장치에서 지원되지 않습니다. 지원 모드를 선택하세요.'))
            candidate=self.collect()
            c=candidate['camera']
            if candidate["calibration"]:
                from mes_vision.calibration import load
                from .robot_service import geometry_context
                from .acquisition import inspection_camera
                image_camera=inspection_camera(c)
                calibration=load(Path(candidate["calibration"])); context=geometry_context(candidate,{"width":image_camera["width"],"height":image_camera["height"],"transformations":[]})
                require(calibration.ready and calibration.data["specification"]["kind"]=="real",tr('검증된 실물 보정이 필요합니다.'))
                from mes_vision.calibration import spec_from_dict
                require(spec_from_dict(calibration.data["specification"]).context==context,tr('현재 촬영·설치 조건과 보정이 다릅니다. 기존 보정을 해제하고 다시 보정하세요.'))
            from .quality import validate_equipment
            validate_equipment(candidate); self.value=candidate; self.accept()
        except Exception as exc: ui_text(self.message.setText, str(exc))


class GeometryDialog(QDialog):
    def __init__(self,product_id,parent=None):
        super().__init__(parent); self.product_id=product_id; self.value=None; ui_text(self.setWindowTitle, tr('형상·누락 검사 기준')); self.resize(950,590)
        layout=QVBoxLayout(self); f=QFormLayout(); layout.addLayout(f)
        self.version=QLineEdit(); self.reference=QLineEdit(); self.threshold=integer(128,0,255)
        self.dark=ui_text(QCheckBox, tr('어두운 영역을 특징으로 검사')); self.dark.setChecked(True)
        for label,w in ((tr('기준 버전'),self.version),(tr('실물 검증 기록'),self.reference),(tr('밝기 경계값'),self.threshold)): ui_text(f.addRow, label,w)
        ui_text(f.addRow, self.dark)
        layout.addWidget(ui_text(QLabel, tr('물체 이미지 내부의 상대 영역(0~1)과 특징 면적 비율을 지정합니다. 검증된 방향·정렬 조건에서 사용하세요.')))
        self.table=table([tr('특징 이름'),tr('불량 코드'),tr('왼쪽'),tr('위'),tr('오른쪽'),tr('아래'),tr('면적 하한'),tr('면적 상한')])
        self.table.setEditTriggers(QAbstractItemView.AllEditTriggers); layout.addWidget(self.table)
        layout.addWidget(button(tr('검사 특징 추가'),self.add))
        self.message=ui_text(QLabel); layout.addWidget(self.message); layout.addWidget(button(tr('기준 등록'),self.save))
    def add(self):
        row=self.table.rowCount(); self.table.insertRow(row)
        for i,value in enumerate(("","NG01","0","0","1","1","0","1")): self.table.setItem(row,i,ui_text(QTableWidgetItem, value))
    def save(self):
        try:
            features=[]
            for row in range(self.table.rowCount()):
                values=[self.table.item(row,c).text().strip() for c in range(8)]
                require(values[0],tr('특징 이름을 입력하세요.'))
                features.append({"name":values[0],"code":values[1],"region":list(map(float,values[2:6])),"min_fraction":float(values[6]),"max_fraction":float(values[7])})
            require(self.version.text().strip() and self.reference.text().strip() and features,tr('버전·검증 기록·검사 특징이 필요합니다.'))
            self.value={"version":self.version.text().strip(),"validation_reference":self.reference.text().strip(),"product_id":self.product_id,"kind":"real",
                "binary_threshold":self.threshold.value(),"foreground_dark":self.dark.isChecked(),"features":features,"aspect_range":None}
            for f in features:
                a,b,c,d=f["region"]; require(0<=a<c<=1 and 0<=b<d<=1 and 0<=f["min_fraction"]<=f["max_fraction"]<=1,tr('특징 영역·비율 범위를 확인하세요.'))
            self.accept()
        except Exception as exc: ui_text(self.message.setText, str(exc))


class CalibrationDialog(QDialog):
    def __init__(self,product,equipment,frame,camera_info,runtime,parent=None):
        super().__init__(parent); self.product=product; self.equipment=equipment; self.frame=frame; self.camera_info=camera_info; self.runtime=Path(runtime)
        self.calibration=None; self.path=None; ui_text(self.setWindowTitle, tr('실측 좌표 보정')); self.resize(1150,900)
        layout=QVBoxLayout(self); self.canvas=Canvas(); self.canvas.set_rgb(frame.rgb); self.canvas.edit_mode="point"
        self.canvas.roi=equipment["workspace"]["roi"]; self.canvas.point_selected.connect(self.add_point); layout.addWidget(self.canvas,1)
        row=QHBoxLayout(); self.role=QComboBox(); ui_text(self.role.addItems, [tr('계산점'),tr('독립 확인점')]); row.addWidget(self.role)
        self.measurement_session=QLineEdit(); ui_text(self.measurement_session.setPlaceholderText, tr('실제 측정 회차 이름 · 계산점/확인점은 다른 회차')); row.addWidget(self.measurement_session,1); layout.addLayout(row)
        self.points=table([tr('점'),tr('용도'),tr('측정 회차'),tr('영상 X'),tr('영상 Y'),tr('로봇 X(mm)'),tr('로봇 Y(mm)')])
        self.points.setEditTriggers(QAbstractItemView.AllEditTriggers); self.points.setMaximumHeight(210); self.points.cellChanged.connect(self.invalidate); layout.addWidget(self.points)
        f=QFormLayout(); layout.addLayout(f); self.version=QLineEdit(); self.reference=QLineEdit()
        self.plane=number(product["grasp"]["plane_z_mm"],-1000,1000,3); self.max_error=number(0,0,100,3); self.rms_error=number(0,0,100,3)
        for label,w in ((tr('보정 버전'),self.version),(tr('검사면 높이(mm)'),self.plane),(tr('최대 허용 오차(mm)'),self.max_error),(tr('확인점 RMS 허용(mm)'),self.rms_error),(tr('실측 검증 기록'),self.reference)): ui_text(f.addRow, label,w)
        self.version.textChanged.connect(self.invalidate)
        for w in (self.plane,self.max_error,self.rms_error): w.valueChanged.connect(self.invalidate)
        actions=QHBoxLayout(); actions.addWidget(button(tr('선택점 삭제'),self.remove_point)); actions.addWidget(button(tr('보정 계산'),self.calculate))
        actions.addWidget(button(tr('검증 기록으로 등록'),self.register)); layout.addLayout(actions)
        self.message=ui_text(QLabel, tr('영상에서 측정점을 클릭하고 대응하는 로봇 XY를 입력하세요. 계산점 6개 이상, 독립 확인점 4개 이상이 필요합니다.'))
        self.message.setWordWrap(True); layout.addWidget(self.message)
    def add_point(self,p):
        row=self.points.rowCount(); self.points.insertRow(row)
        values=[f"P{uuid4().hex[:6]}","fit" if self.role.currentIndex()==0 else "check",self.measurement_session.text().strip(),f"{p[0]:.3f}",f"{p[1]:.3f}","",""]
        for i,value in enumerate(values): self.points.setItem(row,i,ui_text(QTableWidgetItem, value))
    def remove_point(self):
        if self.points.currentRow()>=0: self.points.removeRow(self.points.currentRow()); self.invalidate()
    def invalidate(self,*_): self.calibration=None
    def calculate(self):
        try:
            from mes_vision.calibration import Specification,Lens,Pair,Limits,fit
            from .robot_service import geometry_context
            c=self.camera_info; model=c["distortion"]
            if model=="distortion.none": lens=Lens("none_verified","D405 SDK intrinsics: "+c["serial"]+" / "+c["firmware"])
            elif model=="distortion.brown_conrady":
                lens=Lens("opencv_brown5","D405 SDK intrinsics: "+c["serial"],((c["fx"],0.,c["ppx"]),(0.,c["fy"],c["ppy"]),(0.,0.,1.)),tuple(c["coeffs"]))
            else: raise ValueError(tr('현재 렌즈 모델은 자동 변환 대상이 아닙니다. 확인된 OpenCV 보정 파일로 보정점을 등록해야 합니다: ')+model)
            pairs=[]
            for r in range(self.points.rowCount()):
                cells=[self.points.item(r,i).text().strip() for i in range(7)]
                pairs.append(Pair(*cells[:3],tuple(map(float,cells[3:5])),tuple(map(float,cells[5:7]))))
            spec=Specification(self.version.text().strip(),self.product["id"],"real",geometry_context(self.equipment,self.frame.metadata()),lens,
                self.plane.value(),0.,tuple(tuple(v) for v in self.equipment["workspace"]["roi"]),
                Limits(self.max_error.value(),self.max_error.value(),self.rms_error.value(),.7),tuple(pairs))
            self.calibration=fit(spec); m=self.calibration.data["metrics"]
            ui_text(self.message.setText, trf('계산점 최대 {v0:.4f} mm · 확인점 최대 {v1:.4f} mm · RMS {v2:.4f} mm\n', v0=m['fit_max_mm'], v1=m['check_max_mm'], v2=m['check_rmse_mm'])
                +(tr('수치 기준 통과 · 검증 기록 등록 전') if not self.calibration.data["failures"] else tr('기준 미달: ')+text_join(', ', self.calibration.data["failures"])))
        except Exception as exc: ui_text(self.message.setText, str(exc))
    def register(self):
        try:
            require(self.calibration is not None,tr('보정을 먼저 계산하세요.'))
            accepted=self.calibration.accept(self.reference.text().strip()); directory=self.runtime/"calibrations"; directory.mkdir(parents=True,exist_ok=True)
            self.path=str(directory/(uuid4().hex+".json")); accepted.save(self.path); self.accept()
        except Exception as exc: ui_text(self.message.setText, str(exc))


class RobotProfileDialog(QDialog):
    def __init__(self,product,equipment,runtime,parent=None):
        super().__init__(parent); self.product=product; self.equipment=equipment; self.runtime=Path(runtime); self.path=None
        ui_text(self.setWindowTitle, tr('Dobot 운전 기준')); self.resize(700,800); layout=QVBoxLayout(self); f=QFormLayout(); layout.addLayout(f)
        self.version=QLineEdit(); self.reference=QLineEdit(); ui_text(f.addRow, tr('설정 버전'),self.version); ui_text(f.addRow, tr('실물 검증 기록'),self.reference)
        self.fields={}
        for key,label in (("x_min",tr('작업 X 최소')),("x_max",tr('작업 X 최대')),("y_min",tr('작업 Y 최소')),("y_max",tr('작업 Y 최대')),
            ("z_min",tr('작업 Z 최소')),("z_max",tr('작업 Z 최대')),("r_min",tr('방향 최소')),("r_max",tr('방향 최대')),("travel",tr('이동 높이 Z'))):
            w=number(0,-1000,1000,3); self.fields[key]=w; ui_text(f.addRow, label,w)
        self.destinations={}
        for verdict in ("OK","NG"):
            h=QHBoxLayout(); widgets=[number(0,-1000,1000,3) for _ in range(4)]; self.destinations[verdict]=widgets
            for label,w in zip(("X","Y","Z",tr('방향')),widgets): h.addWidget(ui_text(QLabel, label)); h.addWidget(w)
            ui_text(f.addRow, verdict+tr(' 분류 위치'),h)
        self.tool=QComboBox(); ui_text(self.tool.addItems, [tr('흡착'),tr('그리퍼')]); ui_text(f.addRow, tr('집기 도구'),self.tool)
        self.age=number(2,.1,10); self.timeout=number(10,.1,300); ui_text(f.addRow, tr('검사 좌표 유효 시간(초)'),self.age); ui_text(f.addRow, tr('동작별 제한 시간(초)'),self.timeout)
        self.message=ui_text(QLabel, tr('독립 집기 확인 센서와 검증된 좌표·집기 기준이 필요합니다. 등록 후 로봇 상태 확인을 거쳐 운전을 준비합니다.'))
        self.message.setWordWrap(True); layout.addWidget(self.message); layout.addWidget(button(tr('검증 기록으로 등록'),self.save))
    def save(self):
        try:
            from mes_vision.robot import RobotProfile,Workspace,Pose
            from mes_vision.calibration import load
            require(self.equipment["calibration"] and self.equipment["robot"]["input_address"],tr('좌표 보정과 집기 확인 입력을 먼저 등록하세요.'))
            calibration=load(Path(self.equipment["calibration"])); g=self.product["grasp"]
            require(calibration.ready and calibration.data["specification"]["kind"]=="real" and calibration.data["specification"]["product_id"]==self.product["id"],tr('이 품목의 검증된 실측 보정이 필요합니다.'))
            profile=RobotProfile(self.version.text().strip(),self.product["id"],"real",True,self.reference.text().strip(),calibration.identity,
                g["version"],Workspace(**{k:w.value() for k,w in self.fields.items() if k!="travel"}),g["plane_z_mm"],self.fields["travel"].value(),
                {k:Pose(*(w.value() for w in fields)) for k,fields in self.destinations.items()},"suction" if self.tool.currentIndex()==0 else "gripper",
                "digital_input",self.age.value(),self.timeout.value(),False)
            require(g["validation_reference"],tr('품목 집기 기준의 검증 기록을 입력하세요.'))
            directory=self.runtime/"robot-profiles"; directory.mkdir(parents=True,exist_ok=True)
            self.path=str(directory/(uuid4().hex+".json")); write_json(Path(self.path),asdict(profile)); self.accept()
        except Exception as exc: ui_text(self.message.setText, str(exc))

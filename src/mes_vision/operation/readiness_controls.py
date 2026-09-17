from mes_vision.theme import color as theme_color
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
"""Pre-start checks and navigation, without arming cameras, models or motion."""
from copy import deepcopy
import json
import math
import os
from pathlib import Path
import tempfile
import time
from uuid import uuid4
from filelock import FileLock

from PySide6.QtCore import Qt,QSize
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QVBoxLayout, QLabel, QPushButton, QScrollArea, QSizePolicy
from mes_vision.i18n import tr, trf
from mes_vision.training.data import require
from mes_vision.ui_refresh import BackgroundRead
from .catalog import readiness
from .storage import check_free_space


def check_storage(store):
    """Check actual image-directory and SQLite writes without keeping probe records."""
    with FileLock(str(store.root/'storage.lock'),timeout=0):
        check_free_space(store)
        folder = store.root/'inspections'
        folder.mkdir(exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix='.readiness-', dir=folder) as stream:
            stream.write(b'MES storage readiness\n'); stream.flush(); os.fsync(stream.fileno())
        with store.connect() as db:
            db.execute('INSERT INTO preferences(key,value) VALUES(?,?)', ('_readiness_'+uuid4().hex, 'probe'))
            db.rollback()


def check_coordinates(product, equipment, geometry):
    from mes_vision.calibration import load, spec_from_dict
    from .robot_service import geometry_context
    require(product is not None, tr('품목을 적용하세요.'))
    require(equipment['calibration'], tr('좌표 보정이 필요합니다.'))
    calibration = load(Path(equipment['calibration']))
    spec = spec_from_dict(calibration.data['specification'])
    require(calibration.ready and spec.kind == 'real' and spec.product_id == product['id'],
            tr('이 품목의 검증된 실측 보정이 필요합니다.'))
    require(spec.context == geometry_context(equipment, geometry), tr('현재 촬영·설치 조건과 좌표 보정이 다릅니다.'))
    z = product['grasp']['plane_z_mm']
    require(type(z) in {int,float} and math.isfinite(z) and abs(z-spec.plane_z_mm)<=spec.plane_tolerance_mm,
            tr('검사면 높이와 좌표 보정이 다릅니다.'))
    return calibration.identity


def check_robot_setup(product, equipment, calibration_id):
    from mes_vision.robot import load_profile
    require(calibration_id is not None, tr('좌표 보정을 먼저 확인하세요.'))
    require(equipment['robot']['profile'], tr('로봇 운전 설정을 등록하세요.'))
    profile = load_profile(Path(equipment['robot']['profile']))
    grasp = product['grasp']
    require(profile.validated and profile.kind == 'real' and profile.product_id == product['id']
            and profile.calibration_version == calibration_id and profile.grasp_policy_version == grasp['version'],
            tr('품목·보정과 로봇 운전 설정이 다릅니다.'))
    require(grasp['version'] and grasp['validation_reference'] and 0<grasp['u']<1 and 0<grasp['v']<1,
            tr('품목 집기 기준을 확인하세요.'))
    require(equipment['robot']['input_address'] is not None and equipment['robot']['motion_validation_reference'],
            tr('집기 확인 입력과 로봇 운전 검증 기록이 필요합니다.'))


def probe_readiness(store, product, equipment, geometry):
    result = {'settings': [], 'storage': None, 'coordinates': None, 'robot': None}
    try: result['settings'] = readiness(product,equipment,verify_files=False)
    except Exception as exc: result['settings'] = [str(exc)]
    try: check_storage(store)
    except Exception as exc: result['storage'] = str(exc)
    calibration_id = None
    try: calibration_id = check_coordinates(product,equipment,geometry)
    except Exception as exc: result['coordinates'] = str(exc)
    try: check_robot_setup(product,equipment,calibration_id)
    except Exception as exc: result['robot'] = str(exc)
    return result


class ReadinessCard(QPushButton):
    def heightForWidth(self,width):
        if not hasattr(self,'heading'): return 48
        width=max(40,width-20)
        return 18+sum(max(label.fontMetrics().height(),label.heightForWidth(width)) for label in (self.heading,self.detail))
    def sizeHint(self): return QSize(320,self.heightForWidth(320))
    def minimumSizeHint(self): return QSize(80,48)


class ReadinessControls:
    def make_readiness_controls(self, layout):
        self.readiness_read = BackgroundRead(self)
        self.readiness_read.succeeded.connect(self.readiness_received)
        self.readiness_read.failed.connect(self.readiness_failed)
        self.readiness_key = None; self.readiness_report = None; self.readiness_at = 0.; self.readiness_due = 0.
        self.readiness_paint_key = None
        self.readiness_error = None
        header = QHBoxLayout()
        self.readiness_toggle = ui_text(QPushButton, tr('검사 준비 상태'))
        self.readiness_toggle.setCheckable(True); self.readiness_toggle.setChecked(True)
        header.addWidget(self.readiness_toggle)
        self.readiness_summary = ui_text(QLabel, tr('준비 상태를 확인하고 있습니다.'))
        self.readiness_summary.setWordWrap(True); header.addWidget(self.readiness_summary,1)
        layout.addLayout(header)
        self.readiness_panel = QScrollArea(); self.readiness_panel.setWidgetResizable(True)
        self.readiness_panel.setFrameShape(QScrollArea.NoFrame); self.readiness_panel.setMaximumHeight(210)
        cards=QWidget(); self.readiness_panel.setWidget(cards)
        grid = QGridLayout(cards); grid.setContentsMargins(0,0,0,0); self.readiness_grid=grid
        self.readiness_buttons = {}
        titles = [('product','품목 · 검사 조건'),('models','검사 모델'),('camera','카메라 영상'),
                  ('coordinates','좌표 보정'),('storage','검사 원본 저장'),('robot','로봇 자동 분류')]
        for index,(key,title) in enumerate(titles):
            control=ReadinessCard(); control.setObjectName('readiness_'+key)
            control.setMinimumHeight(48)
            policy=QSizePolicy(QSizePolicy.Expanding,QSizePolicy.Preferred); policy.setHeightForWidth(True); control.setSizePolicy(policy)
            labels=QVBoxLayout(control); labels.setContentsMargins(10,6,10,6); labels.setSpacing(2)
            control.heading=ui_text(QLabel, tr(title)+'  ›'); control.detail=ui_text(QLabel)
            for label in (control.heading,control.detail):
                label.setWordWrap(True); label.setAttribute(Qt.WA_TransparentForMouseEvents)
                label.setSizePolicy(QSizePolicy.Ignored,QSizePolicy.Preferred); labels.addWidget(label)
            control.clicked.connect(lambda checked=False,key=key:self.readiness_navigate(key))
            grid.addWidget(control,index//2,index%2); self.readiness_buttons[key]=(control,title)
        grid.setColumnStretch(0,1); grid.setColumnStretch(1,1)
        layout.addWidget(self.readiness_panel)
        self.readiness_toggle.toggled.connect(self.readiness_panel.setVisible)
        self.resize_readiness()

    def resizeEvent(self,event):
        super().resizeEvent(event)
        if hasattr(self,'readiness_grid'): self.resize_readiness()

    def resize_readiness(self):
        columns=3 if self.width()>=1300 else 2
        if getattr(self,'readiness_columns',None)!=columns:
            self.readiness_columns=columns
            while self.readiness_grid.count(): self.readiness_grid.takeAt(0)
            for index,(button,_) in enumerate(self.readiness_buttons.values()):
                self.readiness_grid.addWidget(button,index//columns,index%columns)
            for index in range(3): self.readiness_grid.setColumnStretch(index,1 if index<columns else 0)
        self.readiness_panel.setFixedHeight(210 if self.height()>=850 else 140)

    def readiness_inputs(self):
        camera=self.equipment['camera']
        geometry={'width':camera['width'],'height':camera['height'],'transformations':[]}
        if self.frame is not None:
            metadata=self.frame.metadata()
            geometry={key:metadata[key] for key in geometry}
        return json.dumps((self.product,self.equipment,geometry),sort_keys=True,ensure_ascii=False),geometry

    def readiness_tick(self,now):
        key,geometry=self.readiness_inputs()
        if key!=self.readiness_key:
            self.readiness_read.invalidate(); self.readiness_key=key; self.readiness_report=None; self.readiness_due=0.
            self.readiness_error=None
        if now>=self.readiness_due and not self.readiness_read.busy:
            product,equipment=deepcopy(self.product),deepcopy(self.equipment)
            self.readiness_due=now+5.
            self.readiness_read.start(lambda:(key,probe_readiness(self.store,product,equipment,geometry)))
        # Only inexpensive live state is checked on the existing watchdog tick.
        self.render_readiness(now)

    def readiness_received(self,value):
        key,report=value
        current,_=self.readiness_inputs()
        if self.closing or key!=current or key!=self.readiness_key: return
        self.readiness_report=report; self.readiness_error=None; self.readiness_at=time.monotonic(); self.readiness_due=self.readiness_at+5.
        self.render_readiness(self.readiness_at)

    def readiness_failed(self,message):
        if self.closing: return
        self.readiness_error=message
        self.readiness_report=None; self.readiness_due=time.monotonic()+5.
        ui_text(self.readiness_summary.setText, tr('준비 상태 조회 실패: ')+message)
        self.start.setEnabled(False)

    def readiness_status(self,now):
        key,_=self.readiness_inputs()
        report=self.readiness_report if key==self.readiness_key and now-self.readiness_at<=15 else None
        waiting=tr('확인 중'); issues=[]
        product_ok=bool(self.product and self.product.get('active'))
        camera_ok=bool(self.camera_info and self.frame is not None and 0<=now-self.frame_received<.5)
        model_ok=bool(self.engine is not None and self.engine_ready and not self.stop_at)
        auto=self.auto.isChecked()
        live_robot=bool(self.robot and self.robot_state['state'] in {'IDLE','RECAPTURE'}
                        and not self.robot_state.get('busy') and self.robot_seen_at and 0<=now-self.robot_seen_at<5)
        states={
            'product':(product_ok, tr('품목 적용됨') if product_ok else tr('사용할 품목을 적용하세요.')),
            'models':(model_ok, tr('모델 준비 완료') if model_ok else tr('모델 준비가 필요합니다.')),
            'camera':(camera_ok,tr('새 영상 수신 중') if camera_ok else tr('카메라의 새 영상이 필요합니다.')),
            'coordinates':(False,waiting), 'storage':(False,waiting),'robot':(False,waiting)}
        if not product_ok: issues.append(tr('사용할 품목을 적용하세요.'))
        if not model_ok: issues.append(tr('모델 준비가 필요합니다.'))
        if not camera_ok: issues.append(tr('카메라의 새 영상이 필요합니다.'))
        if report is None: issues.append(tr('준비 상태 조회 실패: ')+self.readiness_error if self.readiness_error else tr('준비 상태를 확인하고 있습니다.'))
        else:
            if report['settings']:
                states['product']=(False,tr('검사 조건 확인 필요'))
                issues.extend(tr(v) for v in report['settings'])
            states['storage']=(report['storage'] is None,tr('저장 가능') if report['storage'] is None else tr('저장 상태 확인 필요'))
            if report['storage']: issues.append(tr('저장 상태 확인 필요')+': '+tr(report['storage']))
            states['coordinates']=(report['coordinates'] is None,tr('현재 조건과 보정 일치') if report['coordinates'] is None else tr('분류 전 보정 필요'))
            robot_ok=live_robot and report['robot'] is None and report['coordinates'] is None
            states['robot']=(robot_ok,tr('분류 준비 완료') if robot_ok else tr('분류 준비 확인 필요'))
            if auto:
                if report['coordinates']: issues.append(tr('좌표 보정')+': '+tr(report['coordinates']))
                if report['robot']: issues.append(tr('로봇 운전 설정')+': '+tr(report['robot']))
                if not live_robot: issues.append(tr('로봇 준비 확인 후 자동 분류를 켜세요.'))
        if not auto:
            for k in ('coordinates','robot'):
                states[k]=(states[k][0],states[k][1]+' · '+tr('자동 분류 시 필수'))
        if self.faults.active: issues.insert(0,tr('미해결 운전 오류를 확인하세요.'))
        if self.robot_pending or self.pick_block: issues.insert(0,tr('로봇 동작 또는 집기 위치 확인을 먼저 완료하세요.'))
        if self.running: issues.insert(0,tr('검사가 이미 진행 중입니다.'))
        return states,list(dict.fromkeys(issues)),report

    def render_readiness(self,now):
        states,issues,report=self.readiness_status(now)
        self.start.setEnabled(not issues and not self.closing)
        summary=(tr('연속 검사 중') if self.running else tr('검사 시작 가능') if not issues
                 else tr('시작 전 확인: ')+issues[0]+(trf(' · 추가 {v0}건',v0=len(issues)-1) if len(issues)>1 else ''))
        paint_key=(states,issues,report,summary,self.auto.isChecked())
        if paint_key==self.readiness_paint_key: return
        self.readiness_paint_key=deepcopy(paint_key)
        ui_text(self.readiness_summary.setText, summary); ui_text(self.readiness_summary.setToolTip, text_join('\n', issues))
        ui_text(self.start.setToolTip, text_join('\n', issues) if issues else tr('검사 시작 가능'))
        for key,(control,title) in self.readiness_buttons.items():
            ok,text=states[key]
            optional=key in {'coordinates','robot'} and not self.auto.isChecked()
            color=theme_color('success' if ok else 'muted' if optional else 'warning')
            control.setStyleSheet(f'text-align:left;color:{color};padding:6px 10px;')
            ui_text(control.detail.setText, text); control.setAccessibleName(tr(title)+': '+text)
            details=[]
            if report:
                details=report['settings'] if key=='product' else [report.get(key)] if key in {'storage','coordinates','robot'} else []
            ui_text(control.setToolTip, text_join('\n', [tr(title),text,*[tr(v) for v in details if v],tr('클릭하면 해당 설정으로 이동합니다.')]))

    def readiness_navigate(self,key):
        targets={'product':(1,'products_table'),'models':(1,'model_prepare_button'),
                 'camera':(2,'camera_connect_button'),'coordinates':(2,'calibration_button'),
                 'storage':(3,'storage_button'),'robot':(2,'robot_ready_button')}
        page,name=targets[key]; self.nav.setCurrentRow(page)
        target=getattr(self,name,None)
        if target is not None: target.setFocus(Qt.OtherFocusReason)

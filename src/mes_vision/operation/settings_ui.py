from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
"""Simple/advanced settings views, draft comparison and read-only saved versions."""
import json
from datetime import datetime
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QFormLayout,QTabWidget,QScrollArea,
    QCheckBox,QPlainTextEdit,QDialog,QDialogButtonBox,QLabel,QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox)
from mes_vision.i18n import tr,trf
from .widgets import button,table,number
from mes_vision.ui_refresh import update_table


LABELS={'id':'품목 ID','name':'품목 이름','active':'사용 상태','camera':'카메라','serial':'카메라 장치',
    'width':'영상 가로','height':'영상 세로','fps':'초당 프레임','auto_exposure':'자동 노출','exposure':'수동 노출(장치 단위)',
    'mount_revision':'카메라 설치 버전','acquisition_revision':'촬영 조건 버전','robot_base_id':'로봇 기준 좌표계 이름','tool_frame_id':'집기 도구 좌표계 이름',
    'workspace':'작업 영역','width_mm':'작업 공간 가로(mm)','height_mm':'작업 공간 세로(mm)','roi':'검사 영역','excluded':'제외 영역',
    'validation_reference':'검증 기록','tracking':'추적 · 고급 설정','stable_seconds':'안정 확인 시간(초)','max_gap':'추적 연결 최대 간격(초)',
    'motion_px':'이동 판별 기준(px)','appearance_delta':'외관 변경 기준','calibration':'활성 좌표 보정','robot':'Dobot',
    'port':'통신 포트','baudrate':'통신 속도','input_address':'집기 확인 디지털 입력','holding_level':'센서 신호',
    'pose_tolerance_mm':'도달 위치 허용 오차(mm)','rotation_tolerance_deg':'방향 허용 오차(도)','speed_ratio':'검증된 이동 속도 비율(%)',
    'acceleration_ratio':'검증된 가속도 비율(%)','motion_validation_reference':'속도 검증 기록','profile':'검증된 로봇 운전 설정',
    'auto_sort':'자동 분류','objects':'물체 검출 모델','defects':'불량 검출 모델','anomaly':'이상 탐지','bank':'정상 특징 폴더',
    'criteria':'이상 점수 기준','policy':'활성 판정 기준 파일','geometry':'형상·누락 검사 기준','normal_reference':'검토된 정상 참조 폴더',
    'vlm_criteria':'품목 검사 기준 설명','vlm_policy':'자동 전달 조건','count_mode':'수량 조건','expected_count':'고정 수량',
    'quality':'촬영 · 판정 기준','version':'버전','blur_min':'선명도 하한','brightness_min':'평균 밝기 하한','brightness_max':'평균 밝기 상한',
    'grasp':'집기 기준','u':'물체 내 가로 비율','v':'물체 내 세로 비율','plane_z_mm':'검사면 높이 Z(mm)','rotation_deg':'집기 방향(도)',
    'workspace_id':'작업 환경','weights':'모델 파일','sha256':'파일 식별값','class_names':'검사 분류','threshold':'검출 하한','class_codes':'불량 코드'}


def changes(before,after,path=()):
    rows=[]
    if isinstance(before,dict) or isinstance(after,dict):
        before=before if isinstance(before,dict) else {}; after=after if isinstance(after,dict) else {}
        for key in sorted(set(before)|set(after)):
            if not path and key=='version': continue
            rows.extend(changes(before.get(key),after.get(key),path+(key,)))
    elif before!=after:
        rows.append((text_join(' / ', (tr(LABELS.get(k,k)) for k in path)),before,after))
    return rows


def optional_number(value,low,high,decimals=2):
    sentinel=low-10**(-decimals); widget=number(sentinel if value is None else value,sentinel,high,decimals)
    ui_text(widget.setSpecialValueText, tr('미등록')); return widget


def optional_value(widget): return None if widget.value()==widget.minimum() else widget.value()


def assign_acquisition_revision(old,new,token):
    keys=('driver','pixel_format','serial','width','height','fps','auto_exposure','exposure','gain','capture_profiles')
    defaults={'driver':'d405','pixel_format':'MJPG'}
    changed=any(old.get(key,defaults.get(key))!=new.get(key,defaults.get(key)) for key in keys)
    new['acquisition_revision']=token if changed else old['acquisition_revision']


def display(value):
    if value is None or value=='': return tr('미등록')
    if value is True: return tr('예')
    if value is False: return tr('아니요')
    if isinstance(value,(dict,list)): return json.dumps(value,ensure_ascii=False)
    values={'free':'자유 수량','fixed':'고정 수량','review_visual':'외관 판단 보류만 자동 분석',
            'ng_and_review':'NG와 외관 판단 보류 분석','manual':'수동 요청만'}
    return tr(values.get(str(value),str(value)))


def describe(before,after):
    rows=changes(before,after)
    return text_join('\n\n', (name+'\n'+display(old)+' → '+display(new) for name,old,new in rows)) or tr('변경된 설정이 없습니다.')


class SettingsPages(QWidget):
    def __init__(self,layout):
        super().__init__(); outer=QVBoxLayout(self); outer.setContentsMargins(0,0,0,0)
        self.advanced=ui_text(QCheckBox, tr('고급 설정 표시')); outer.addWidget(self.advanced)
        self.tabs=QTabWidget(); outer.addWidget(self.tabs,1); self.advanced_pages=[]
        self.advanced.toggled.connect(self.toggle); layout.addWidget(self,1)

    def form(self,name,advanced=False):
        content=QWidget(); form=QFormLayout(content); form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(content)
        index=ui_text(self.tabs.addTab, scroll,name)
        if advanced: self.advanced_pages.append(index); self.tabs.setTabVisible(index,self.advanced.isChecked())
        return form

    def toggle(self,checked):
        if not checked and self.tabs.currentIndex() in self.advanced_pages: self.tabs.setCurrentIndex(0)
        for index in self.advanced_pages: self.tabs.setTabVisible(index,checked)


def add_review_controls(dialog,layout,original,collect,kind,identity=None):
    row=QHBoxLayout(); layout.addLayout(row)
    preview=QPlainTextEdit(); preview.setReadOnly(True); preview.setMaximumHeight(150); preview.hide(); layout.addWidget(preview)
    dialog.change_preview=preview
    # A preview is a draft snapshot. Hide it as soon as an input changes.
    for cls,signal in ((QLineEdit,'textChanged'),(QComboBox,'currentIndexChanged'),(QDoubleSpinBox,'valueChanged'),
                       (QSpinBox,'valueChanged'),(QCheckBox,'toggled'),(QPlainTextEdit,'textChanged')):
        for widget in dialog.findChildren(cls):
            if widget is not preview: getattr(widget,signal).connect(lambda *_:preview.hide())
    def show():
        try:
            text=describe(original,collect())
            if kind=='equipment': text=tr('촬영 조건 변경 시 번호가 자동 갱신됩니다. 기존 좌표 보정은 다시 확인해야 합니다.')+'\n\n'+text
            ui_text(preview.setPlainText, text); preview.show()
        except Exception as exc: ui_text(preview.setPlainText, str(exc)); preview.show()
    row.addWidget(button(tr('변경 내용 확인'),show))
    row.addWidget(button(tr('변경 내용 접기'),preview.hide))
    store=getattr(dialog.parent(),'store',None)
    if store is not None:
        def history():
            view=SettingsHistoryDialog(store,kind,identity,dialog)
            try: view.exec()
            finally: view.deleteLater()
        row.addWidget(button(tr('저장된 설정 이력'),history))


class SettingsHistoryDialog(QDialog):
    def __init__(self,store,kind,identity=None,parent=None):
        super().__init__(parent); self.store=store; self.kind=kind; self.identity=identity; self.page=0; self.rows=[]
        if kind not in ('equipment','product'): raise ValueError('invalid settings history kind')
        ui_text(self.setWindowTitle, tr('저장된 설정 이력')); self.resize(900,650); layout=QVBoxLayout(self)
        label=ui_text(QLabel, tr('저장된 버전과 직전 버전의 차이를 표시합니다. 현재 설정을 복원하거나 적용하지 않습니다.')); label.setWordWrap(True); layout.addWidget(label)
        self.table=table([tr('버전'),tr('저장 시각')]); self.table.setMaximumHeight(200); layout.addWidget(self.table)
        self.details=QPlainTextEdit(); self.details.setReadOnly(True); layout.addWidget(self.details,1)
        row=QHBoxLayout(); self.previous=button(tr('이전'),lambda:self.move(-1)); self.next=button(tr('다음'),lambda:self.move(1))
        row.addWidget(self.previous); row.addWidget(self.next); layout.addLayout(row)
        self.table.itemSelectionChanged.connect(self.select)
        box=QDialogButtonBox(QDialogButtonBox.Close); box.rejected.connect(self.reject); layout.addWidget(box); self.load()

    def move(self,delta): self.page=max(0,self.page+delta); self.load()

    def load(self):
        try:
            with self.store.connect() as db:
                if self.kind=='equipment': rows=db.execute('SELECT version,data,created FROM equipment ORDER BY version DESC LIMIT 51 OFFSET ?',(self.page*50,)).fetchall()
                else: rows=db.execute('SELECT version,data,created FROM products WHERE id=? ORDER BY version DESC LIMIT 51 OFFSET ?',(self.identity,self.page*50)).fetchall()
            self.rows=[dict(row) for row in rows[:50]]; self.previous.setEnabled(self.page>0); self.next.setEnabled(len(rows)>50)
            update_table(self.table,[(r['version'],((str(r['version']),''),(datetime.fromtimestamp(r['created']).strftime('%Y-%m-%d %H:%M:%S'),''))) for r in self.rows],select_first=True)
            if self.rows: self.select()
            else: ui_text(self.details.setPlainText, tr('저장된 설정이 없습니다.'))
        except Exception as exc: ui_text(self.details.setPlainText, str(exc))

    def select(self):
        index=self.table.currentRow()
        if not 0<=index<len(self.rows): return
        try:
            row=self.rows[index]
            with self.store.connect() as db:
                if self.kind=='equipment': previous=db.execute('SELECT data FROM equipment WHERE version<? ORDER BY version DESC LIMIT 1',(row['version'],)).fetchone()
                else: previous=db.execute('SELECT data FROM products WHERE id=? AND version<? ORDER BY version DESC LIMIT 1',(self.identity,row['version'])).fetchone()
            ui_text(self.details.setPlainText, describe(json.loads(previous[0]) if previous else {},json.loads(row['data'])))
        except Exception as exc: ui_text(self.details.setPlainText, str(exc))

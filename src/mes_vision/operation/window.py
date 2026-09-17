from mes_vision.qt_i18n import ui_text, bound_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
"""Production operator shell. Hardware and trained products are explicitly registered."""
import re
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
from uuid import uuid4

from PySide6.QtCore import Qt,QTimer,QUrl,QByteArray
from PySide6.QtGui import QDesktopServices,QPixmap,QFont,QShortcut,QKeySequence
from PySide6.QtWidgets import (QMainWindow,QWidget,QVBoxLayout,QHBoxLayout,QLabel,QListWidget,QStackedWidget,
    QSplitter,QPlainTextEdit,QTableWidgetItem,QCheckBox,QInputDialog,QMessageBox,QLineEdit,QComboBox,QApplication,QTableWidget,QSizePolicy)
from mes_vision.training.data import require
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.snapshots import basic_reasons
from mes_vision.vlm.viewer import AnalysisViewer,WorkerThread
from mes_vision.theme import METRICS
from .catalog import OperationStore,new_product,readiness
from .widgets import Canvas,Task,button,table,STATES
from .shell import Sidebar,PageHeader,ContextBar,VerdictBanner,ViewArea,panel,styled
from .camera import D405Camera,CameraProcess
from .engine import EngineProcess
from .robot_service import RobotThread
from .tracking import overlap
from .product_dialog import ProductDialog
from .setup_dialogs import EquipmentDialog,CalibrationDialog,RobotProfileDialog
from .history_controls import HistoryControls
from .recovery_controls import RecoveryControls
from .faults import FaultJournal,fault_code
from .preferences import Preferences
from mes_vision.ui_refresh import BackgroundRead
from .readiness_controls import ReadinessControls
from .responsive import ResponsiveWindow,FlowLayout,scroll_page


class DesktopWindow(ResponsiveWindow,ReadinessControls,RecoveryControls,HistoryControls,QMainWindow):
    def __init__(self,root,runtime,*,start_worker=True):
        super().__init__(); self.root=Path(root); self.store=OperationStore(runtime); self.runtime=self.store.root
        self.preferences=Preferences(self.runtime)
        interrupted=self.store.recover_sessions(); self.faults=FaultJournal(self.runtime)
        self.faults.recover_from_store(self.store)
        if interrupted:
            entry,new=self.faults.raise_fault("INTERRUPTED_SESSION",tr('이전 검사가 정상 종료되지 않았습니다. 장치 상태와 미완료 기록을 확인하세요.'))
            if new:
                try: self.store.event("OPERATOR_FAULT",entry)
                except Exception: pass
        self.equipment=self.store.equipment(); self.product=None; self.vlm_worker_enabled=start_worker
        self.vlm_probe_at=0.; self.cached_vlm_enabled=False; self.vlm_unavailable=False
        self.vlm_read=BackgroundRead(self)
        self.vlm_read.succeeded.connect(self.apply_vlm_status); self.vlm_read.failed.connect(self.vlm_read_failed)
        self.camera=None; self.camera_info=None; self.frame=None; self.frame_received=0.; self.engine=None; self.engine_ready=False
        self.robot=None; self.robot_state={"state":"DISCONNECTED","busy":False}; self.robot_pending=False; self.pick_block=None
        self.robot_seen_at=0.
        self.session=None; self.generation=0; self.running=False; self.tracks=[]; self.track_updated=0.; self.selected_id=None
        self.record_key=None; self.record=None; self.tasks=[]; self.children=[]; self.closing=False; self.stop_at=None; self.checked_at=0.
        self.queue=AnalysisQueue(self.runtime/"vlm"); self.worker=None; self.last_history=0.; self.applied_message=tr('품목을 선택하고 모델을 준비하세요.')
        ui_text(self.setWindowTitle, tr('MES Vision · 검사 운영')); self.resize(1440,900); self.setMinimumSize(960,600)
        # Shell: full-height sidebar on the left, page column on the right. The sidebar
        # must be built first so the page column can sit beside it.
        central=QWidget(); central.setObjectName("central"); self.setCentralWidget(central)
        shell=QHBoxLayout(central); shell.setContentsMargins(0,0,0,0); shell.setSpacing(0)
        # self.nav stays a QListWidget so the responsive mixin, the compact selector and
        # every existing currentRowChanged connection keep working unchanged. It is parked
        # in a hidden host and mirrored by the visible grouped sidebar.
        self._nav_host=QWidget(); self._nav_host.hide()
        self.nav=QListWidget(self._nav_host)
        ui_text(self.nav.addItems, [tr('1  실시간 검사'),tr('2  품목 설정'),tr('3  장비 · 보정'),tr('4  검사 이력'),tr('5  VLM 추가 분석')])
        self.nav.setObjectName('mainNavigation')
        self.sidebar=Sidebar()
        for group_title,entries in [(tr('운영'),[(tr('실시간 검사'),'F1')]),
                                    (tr('설정'),[(tr('품목 설정'),'F2'),(tr('장비 · 보정'),'F3')]),
                                    (tr('기록 · 분석'),[(tr('검사 이력'),'F4'),(tr('VLM 추가 분석'),'F5')])]:
            group_label,_=self.sidebar.add_group(group_title,entries); ui_text(group_label.setText,group_title)
        self.sidebar.action.setText(tr('모델 준비')); self.sidebar.action_clicked.connect(lambda:self.guarded(self.prepare_engine))
        self.sidebar.current_changed.connect(self.nav.setCurrentRow)
        self.nav.currentRowChanged.connect(self.sidebar.set_current)
        self.nav.currentRowChanged.connect(self.sync_page_header)
        self.brand_logo=self.sidebar.symbol
        shell.addWidget(self.sidebar)
        self.page_column=QWidget(); shell.addWidget(self.page_column,1)
        outer=QVBoxLayout(self.page_column)
        outer.setContentsMargins(24,22,24,16); outer.setSpacing(round(METRICS['layout_gap']))
        # Page header: kicker / title on the left, session controls on the right,
        # production context strip underneath (MonoFactory MES header structure).
        self.header=PageHeader(); header=self.header.actions
        self.compact_logo=QLabel(); self.compact_logo.setObjectName("compactBrandLogo"); self.compact_logo.setFixedSize(48,54)
        header.addWidget(self.compact_logo); self.compact_logo.hide()
        self.title_label=self.header.title
        self.active_label=ui_text(QLabel, tr('선택된 품목 없음')); self.active_label.setObjectName("statusPill")
        self.active_label.setMinimumWidth(0); self.active_label.setSizePolicy(QSizePolicy.Maximum,QSizePolicy.Preferred)
        header.addWidget(self.active_label)
        self.vlm=button("VLM OFF",self.toggle_vlm); self.vlm.setCheckable(True); self.vlm.setObjectName("statusPill"); header.addWidget(self.vlm)
        self.settings_button=button(tr("환경설정"),lambda:self.guarded(self.open_preferences)); self.settings_button.setObjectName("openPreferences"); header.addWidget(self.settings_button)
        self.theme_button=button("Light",lambda:self.guarded(self.toggle_theme)); self.theme_button.setObjectName("themeToggle"); header.addWidget(self.theme_button)
        self.global_stop=button(tr('로봇 정지'),self.stop_robot); self.global_stop.setObjectName('globalRobotStop'); header.addWidget(self.global_stop)
        self.context=ContextBar(["품목","버전","카메라","로봇","모델"]); self.header.body.addWidget(self.context)
        outer.addWidget(self.header); self.status=ui_text(QLabel, tr('장비 · 보정에서 카메라를 연결하세요.')); self.status.setWordWrap(True); outer.addWidget(self.status)
        self.make_recovery_controls(outer)
        self.pages=QStackedWidget(); outer.addWidget(self.pages,1)
        self.make_live(); self.make_products(); self.make_equipment(); self.make_history()
        self.analysis=AnalysisViewer(self.queue,self.root,run_worker=False); self.analysis.setWindowFlags(Qt.Widget); self.pages.addWidget(scroll_page(self.analysis))
        self.analysis.enabled_changed.connect(self.apply_vlm_status)
        self.init_compact_navigation(self.header.actions)
        self.nav.currentRowChanged.connect(self.pages.setCurrentIndex); self.nav.setCurrentRow(self.preferences.value["start_page"])
        # Legacy labels wrap; the shell's own labels (object names) manage their own
        # wrapping, so the blanket rule must not re-wrap the banner or context strip.
        for label in self.findChildren(QLabel):
            if label is not self.title_label and not label.objectName(): label.setWordWrap(True)
        self.apply_preferences()
        restored=False
        if self.preferences.value["remember_window"] and self.preferences.value["window_geometry"]:
            restored=self.restoreGeometry(QByteArray.fromHex(self.preferences.value["window_geometry"].encode("ascii")))
        # Operator stations run the window filling the screen. A remembered geometry still
        # wins so the preference keeps meaning; F11 switches to borderless full screen.
        if not restored: self.setWindowState(self.windowState()|Qt.WindowMaximized)
        self.fullscreen_shortcut=QShortcut(QKeySequence(Qt.Key_F11),self)
        self.fullscreen_shortcut.activated.connect(self.toggle_fullscreen)
        self.exit_fullscreen_shortcut=QShortcut(QKeySequence(Qt.Key_Escape),self)
        self.exit_fullscreen_shortcut.activated.connect(self.leave_fullscreen)
        # The sidebar prints F1-F5 next to each entry, so the keys have to do what the
        # labels claim. setCurrentRow drives the sidebar, the page stack and the header
        # together, which is the same path a click takes.
        self.page_shortcuts=[]
        for index in range(len(self.PAGES)):
            shortcut=QShortcut(QKeySequence(getattr(Qt,'Key_F%d'%(index+1))),self)
            shortcut.activated.connect(lambda index=index:self.nav.setCurrentRow(index))
            self.page_shortcuts.append(shortcut)
        self.update_responsive_layout()
        QTimer.singleShot(0,self.fit_available_screen)
        self.refresh_products(); self.refresh_equipment(); self.refresh_history()
        if start_worker:
            self.worker=WorkerThread(self.queue,self.root,allow_synthetic=False,backend="qwen")
            self.worker.failed.connect(self.vlm_worker_failed); self.worker.start()
        self.timer=QTimer(self); self.timer.timeout.connect(self.safe_tick); self.timer.start(40)
        if self.preferences.warning: self.notice(tr("환경설정 파일을 읽지 못해 기본값을 사용합니다. 저장하면 기존 파일을 별도로 보존합니다."))

    PAGES=[('실시간 검사','카메라 영상에서 물체를 추적하고 판정한 뒤 결과에 따라 자동 분류합니다.'),
           ('품목 설정','품목별 모델·판정 기준·집기 기준을 등록합니다.'),
           ('장비 · 보정','카메라와 로봇을 연결하고 픽셀·로봇 좌표를 보정합니다.'),
           ('검사 이력','저장된 검사 기록을 조건으로 조회합니다.'),
           ('VLM 추가 분석','기본 판정과 별도로 불량 근거 설명을 생성합니다.')]

    @staticmethod
    def nav_label(text):
        """Navigation entries carry a leading order number; the shell shows the name only."""
        return re.sub(r'^\s*\d+\s+','',str(text))

    def refresh_nav_labels(self):
        """Mirror the QListWidget labels onto the visible sidebar.

        Subclasses rename navigation entries after construction (StationWindow renames
        page 0), so this is called again whenever that happens.
        """
        for index,item in enumerate(self.sidebar.items):
            if index<self.nav.count(): item.setText(self.nav_label(bound_text(self.nav.item(index))))
        self.sync_page_header(self.nav.currentRow())

    def sync_page_header(self,index):
        if not 0<=index<len(self.PAGES): return
        title,description=self.PAGES[index]
        if 0<=index<self.nav.count():
            label=self.nav_label(bound_text(self.nav.item(index)))
            if label: title=label
        self.header.set_page('MES Vision',title,tr(description))

    def apply_preferences(self):
        from mes_vision.i18n import language
        from mes_vision.qt_i18n import apply_language, bound_text
        if self.preferences.value['language'] != language():
            apply_language(QApplication.instance(), self.preferences.value['language'])
        scale=self.preferences.value["text_scale"]/100
        from mes_vision.theme import apply_theme, logo_path, stylesheet
        mode=self.preferences.value['theme']; apply_theme(mode,scale)
        self.setStyleSheet(stylesheet(mode,scale))
        self.theme_button.setText('Dark' if mode=='dark' else 'Light')
        pixmap=QPixmap(str(logo_path(mode)))
        self.sidebar.apply_mode(mode,scale)
        self.compact_logo.setPixmap(pixmap.scaled(round(48*scale),round(54*scale),Qt.KeepAspectRatio,Qt.SmoothTransformation))
        font=QFont(QApplication.font()); font.setPointSizeF(10*scale); QApplication.setFont(font)
        self.sync_page_header(self.nav.currentRow())
        self.update_responsive_layout()
        if not hasattr(self,"table_base_widths"): self.table_base_widths={}
        for widget in self.findChildren(QTableWidget):
            for column in range(widget.columnCount()):
                header=widget.horizontalHeaderItem(column)
                if header:
                    base=self.table_base_widths.setdefault((widget,column),widget.columnWidth(column))
                    widget.setColumnWidth(column,max(round(base*scale),widget.fontMetrics().horizontalAdvance(header.text())+28))
                    ui_text(header.setToolTip, bound_text(header))

    def toggle_fullscreen(self):
        if self.isFullScreen(): self.showMaximized()
        else: self.showFullScreen()

    def leave_fullscreen(self):
        if self.isFullScreen(): self.showMaximized()

    def toggle_theme(self):
        preferences=Preferences(self.runtime)
        value=deepcopy(preferences.value)
        value['theme']='dark' if value['theme']=='light' else 'light'
        preferences.save(value)
        self.preferences=preferences; self.apply_preferences()

    def open_preferences(self):
        from .preferences_dialog import PreferencesDialog
        preferences=Preferences(self.runtime)
        dialog=PreferencesDialog(preferences,self.runtime,self)
        if dialog.exec():
            self.preferences=preferences; self.apply_preferences()
            self.notice(tr("환경설정을 저장했습니다."))
        dialog.deleteLater()

    def open_capture_studio(self,collect=False):
        from .capture_dialog import CaptureDialog
        self.require_idle()
        require(self.equipment['camera'].get('driver')=='uvc',tr('U20CAM / USB 카메라를 선택하세요.'))
        require(self.equipment['camera']['serial'],tr('장비 설정에서 카메라를 먼저 선택하세요.'))
        dialog=CaptureDialog(self,collect=collect)
        self.children.append(dialog)
        try: dialog.exec()
        finally: self.children.remove(dialog); dialog.deleteLater()

    def page(self):
        page=QWidget(); layout=QVBoxLayout(page); self.pages.addWidget(scroll_page(page)); return layout
    def notice(self,message):
        ui_text(self.status.setText, message if isinstance(message,str) else str(message))
        try: self.store.event("OPERATOR_NOTICE",{"message":str(message)},session=self.session)
        except Exception:
            if self.running and hasattr(self,"fault_label"): self.trip_fault("STORAGE_FAILURE",tr('운영 기록을 저장할 수 없습니다. 저장 공간과 권한을 확인하세요.'))
            else: ui_text(self.status.setText, str(message)+tr(' · 기록 저장 불가: 저장 공간과 권한을 확인하세요.'))
    def error(self,message):
        self.notice(message); QMessageBox.warning(self,tr('설정 확인'),str(message))
    def guarded(self,fn):
        try: return fn()
        except Exception as exc: self.error(exc)
    def task(self,fn,done):
        job=Task(fn); self.tasks.append(job); job.succeeded.connect(lambda result:self.guarded(lambda:done(result))); job.failed.connect(self.error)
        job.finished.connect(lambda:self.tasks.remove(job) if job in self.tasks else None); job.start()
    def require_idle(self,*,camera=False,robot=False):
        require(self.engine is None and not self.tasks,tr('모델을 해제하고 진행 중인 설정 작업이 끝난 뒤 변경하세요.'))
        require(not self.robot_pending and not self.robot_state.get("busy"),tr('로봇 동작 완료 후 변경하세요.'))
        require(not camera or self.camera is None,tr('카메라 연결을 해제한 뒤 촬영 설정을 변경하세요.'))
        require(not robot or self.robot is None,tr('로봇 연결을 해제한 뒤 운전 설정을 변경하세요.'))

    def make_live(self):
        out=self.page()
        # --- toolbar -------------------------------------------------------
        row=FlowLayout()
        self.start=button(tr('검사 시작'),lambda:self.guarded(self.begin)); self.start.setObjectName('inspectionStart')
        self.pause=button(tr('검사 중단'),self.pause_inspection); self.pause.setObjectName('dangerButton')
        self.unload=button(tr('모델 해제'),self.unload_engine)
        row.addWidget(self.start); row.addWidget(self.pause); row.addWidget(self.unload)
        self.auto=ui_text(QCheckBox, tr('자동 분류')); self.auto.toggled.connect(self.auto_changed); row.addWidget(self.auto)
        row.addWidget(button(tr('현재 영상 수집'),lambda:self.guarded(self.capture_training)))
        toolbar=styled(QWidget()); toolbar.setObjectName('toolbar')
        toolbar_layout=QVBoxLayout(toolbar); toolbar_layout.setContentsMargins(12,9,12,9); toolbar_layout.addLayout(row)

        self.live_state=ui_text(QLabel); self.live_state.setWordWrap(True); out.addWidget(self.live_state)
        self.make_readiness_controls(out)

        # --- image panes: full shot on the left, live and detail stacked -----
        body=QHBoxLayout(); body.setSpacing(round(METRICS['layout_gap'])); out.addLayout(body,1)
        frame,frame_layout=panel(); frame_layout.addWidget(toolbar)
        self.verdict=VerdictBanner(); frame_layout.addWidget(self.verdict)
        from .image_view import ImagePanel
        self.live_image=ImagePanel(title=tr('저장된 전체사진'),meta=tr('중앙 1:1 크롭'))
        self.canvas=self.live_image.canvas; self.canvas.selected.connect(self.select_track)
        self.live_view=ImagePanel(title=tr('왜곡보정 실시간 영상'),compact=True)
        self.detail_image=ImagePanel(title=tr('선택 물체의 상세사진'),compact=True)
        self.detail_canvas=self.detail_image.canvas
        self.views=ViewArea(self.live_image,self.live_view,self.detail_image)
        frame_layout.addWidget(self.views,1)
        # The card is width-locked to its square panes, so it takes no stretch; the
        # judgement column beside it absorbs whatever width is left.
        body.addWidget(frame)

        # --- judgement column ------------------------------------------------
        column=QVBoxLayout(); column.setSpacing(round(METRICS['layout_gap']))
        result,result_layout=panel()
        head=styled(QWidget()); head.setObjectName('panelHead')
        head_layout=QHBoxLayout(head); head_layout.setContentsMargins(12,8,12,8)
        head_title=ui_text(QLabel, tr('선택 물체 판정')); head_title.setObjectName('panelTitle')
        head_layout.addWidget(head_title); head_layout.addStretch(1)
        result_layout.addWidget(head)
        inner=QWidget(); layout=QVBoxLayout(inner); layout.setContentsMargins(12,10,12,10); layout.setSpacing(8)
        self.selection=ui_text(QLabel, tr('화면의 물체를 선택하세요.')); self.selection.setWordWrap(True); layout.addWidget(self.selection)
        self.live_details=QPlainTextEdit(); self.live_details.setReadOnly(True); layout.addWidget(self.live_details,1)
        actions=QHBoxLayout(); actions.setSpacing(7)
        self.recheck_button=button(tr('선택 물체 재검사'),lambda:self.guarded(self.recheck)); actions.addWidget(self.recheck_button)
        self.pick_button=button(tr('선택 물체 분류'),lambda:self.guarded(self.pick_selected))
        self.pick_button.setObjectName('primaryButton'); actions.addWidget(self.pick_button)
        layout.addLayout(actions)
        layout.addWidget(button(tr('선택 물체 검사 원본 보기'),lambda:self.guarded(self.open_selected_record)))
        result_layout.addWidget(inner,1)
        column.addWidget(result,1)
        self.counts=ui_text(QLabel, tr('검사한 물체 0  ·  재검사 0')); self.counts.setObjectName('statusPill')
        column.addWidget(self.counts)
        holder=QWidget(); holder.setLayout(column)
        # The image panes cannot use extra width (they are square), so the judgement
        # column absorbs all of it. A maximum here would make QBoxLayout centre the
        # column and leave dead space on both sides, so only a minimum is set.
        scale=self.preferences.value["text_scale"]/100
        holder.setMinimumWidth(round(430*scale))
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.judgement_column=holder
        body.addWidget(holder,1)

        out.addWidget(ui_text(QLabel, tr('검사 중단 시 영상·추적은 유지됩니다. 이미 시작한 로봇 동작은 완료되며, 즉시 중단은 ‘로봇 정지’를 사용하세요.')))

    def make_products(self):
        out=self.page(); out.addWidget(ui_text(QLabel, tr('품목별 모델·판정 기준·집기 기준을 등록합니다. 저장할 때마다 새 버전이 생성됩니다.')))
        row=FlowLayout()
        for label,fn in [(tr('품목 추가'),lambda:self.edit_product("new")),(tr('수정'),lambda:self.edit_product("edit")),(tr('복사'),lambda:self.edit_product("copy")),
                         (tr('사용 / 비활성'),self.toggle_product),(tr('선택 품목 적용'),self.apply_product),(tr('모델 준비'),self.prepare_engine),(tr('데이터 · 학습'),self.open_training)]:
            control=button(label,lambda checked=False,fn=fn:self.guarded(fn)); row.addWidget(control)
            if fn==self.prepare_engine: self.model_prepare_button=control
        self.products_actions=row
        out.addLayout(row); self.products_table=table([tr('품목'),tr('버전'),tr('사용'),tr('물체 모델'),tr('판정 기준')]); out.addWidget(self.products_table,1)
        self.product_details=QPlainTextEdit(); self.product_details.setReadOnly(True); self.product_details.setMaximumHeight(235); out.addWidget(self.product_details)
        self.products_table.itemSelectionChanged.connect(self.product_selection)
    def refresh_products(self):
        self.products=self.store.products(); self.products_table.setRowCount(len(self.products))
        for i,p in enumerate(self.products):
            values=[p["name"],str(p["version"]),tr('사용') if p["active"] else tr('비활성'),tr('등록') if p["objects"] else tr('미등록'),tr('등록') if p["policy"] else tr('미등록')]
            for j,v in enumerate(values): self.products_table.setItem(i,j,ui_text(QTableWidgetItem, v))
        if self.products: self.products_table.selectRow(0)
    def selected_product(self):
        index=self.products_table.currentRow(); require(0<=index<len(self.products),tr('품목을 선택하세요.')); return self.products[index]
    def product_selection(self):
        if self.products_table.currentRow()<0: return
        p=self.selected_product(); problems=readiness(p,self.equipment,verify_files=False)
        ui_text(self.product_details.setPlainText, trf('{v0} · 버전 {v1}\n품목 ID: {v2}\n\n', v0=p['name'], v1=p['version'], v2=p['id'])+(text_join('\n', problems) if problems else tr('설정 등록됨 · 모델 준비 시 파일과 모델 일치를 확인합니다.')))
    def edit_product(self,mode):
        self.require_idle(); p=new_product() if mode=="new" else deepcopy(self.selected_product())
        if mode=="copy":
            p.update(id=uuid4().hex,name=p["name"]+tr(' 복사'),version=0,policy=None,anomaly=None,geometry=None,normal_reference=None)
        dialog=ProductDialog(p,self.equipment,self.root,self.runtime,self)
        if dialog.exec(): self.store.save_product(dialog.value); self.refresh_products()
    def toggle_product(self):
        self.require_idle(); p=deepcopy(self.selected_product()); p["active"]=not p["active"]; self.store.save_product(p); self.refresh_products()
        if self.product and self.product["id"]==p["id"]: self.product=None; ui_text(self.active_label.setText, tr('품목을 다시 적용하세요.'))
    def apply_product(self):
        self.require_idle(); p=deepcopy(self.selected_product()); require(p["active"],tr('비활성 품목은 적용할 수 없습니다.'))
        self.product=p; ui_text(self.active_label.setText, trf('{v0}  ·  버전 {v1}', v0=p['name'], v1=p['version'])); self.notice(tr('품목 적용됨. 장비 설정을 확인하고 모델을 준비하세요.'))
    def capture_training(self):
        require(self.camera_info and self.frame and time.monotonic()-self.frame_received<.5,tr('카메라의 새 영상이 필요합니다.'))
        require(self.product is not None,tr('수집할 품목을 먼저 적용하세요.'))
        frame=self.frame; product=deepcopy(self.product); equipment=deepcopy(self.equipment)
        def save():
            from PIL import Image
            from mes_vision.training.data import write_json
            folder=self.runtime/"captures"/uuid4().hex; folder.mkdir(parents=True)
            Image.fromarray(frame.rgb).save(folder/"frame.png")
            write_json(folder/"capture.json",{"product":product,"equipment":equipment,"frame":frame.metadata()})
            return folder
        self.task(save,lambda path:self.notice(tr('영상 수집됨. 데이터·학습의 라벨링 도구에서 사진을 가져오세요: ')+str(path)))
    def prepare_engine(self):
        require(not self.faults.active,tr('오류 확인 · 재개 준비를 완료한 뒤 모델을 준비하세요.'))
        self.require_idle(); require(self.product is not None,tr('품목을 적용하세요.'))
        require(not any(getattr(w,"busy",False) for w in self.children),tr('학습이 끝난 뒤 모델을 준비하세요.'))
        p=deepcopy(self.product); e=deepcopy(self.equipment); self.notice(tr('등록 파일과 판정 기준을 확인하고 있습니다.'))
        def done(problems):
            if self.closing: return
            if problems: self.error(text_join('\n', problems)); return
            self.session=self.store.start_session(p,e)
            try: self.engine=EngineProcess(self.root,self.runtime,p,e,self.session)
            except Exception as exc:
                self.store.end_session(self.session,"INTERRUPTED"); self.session=None; self.trip_fault("ENGINE_START",str(exc)); return
            self.checked_at=time.monotonic(); self.engine_ready=False; self.applied_message=tr('모델을 불러오고 첫 추론을 준비하고 있습니다.')
        self.task(lambda:readiness(p,e),done)
    def unload_engine(self):
        self.pause_inspection(); self.auto.setChecked(False)
        if self.engine: self.engine.stop(); self.stop_at=time.monotonic(); self.engine_ready=False
    def finish_engine(self,state="STOPPED"):
        if self.engine: self.engine.dispose()
        self.engine=None; self.engine_ready=False; self.stop_at=None; self.tracks=[]; self.canvas.tracks=[]; self.canvas.update()
        if self.session:
            session=self.session; self.session=None
            try: self.store.end_session(session,state)
            except Exception as exc: self.trip_fault("STORAGE_FAILURE",tr('검사 회차 종료 기록 실패: ')+str(exc))
        self.applied_message=tr('모델이 해제됐습니다. 품목 설정에서 다시 준비할 수 있습니다.')

    def make_equipment(self):
        out=self.page(); out.addWidget(ui_text(QLabel, tr('설치 조건과 작업 영역을 등록하고 실제 장치에 연결합니다. 연결만으로 로봇은 움직이지 않습니다.')))
        row=FlowLayout()
        for label,fn in [(tr('장치 검색'),self.discover),(tr('장비 설정'),self.edit_equipment),(tr('로봇 촬영 설정'),self.station_settings),(tr('카메라 연결'),self.connect_camera),(tr('카메라 해제'),self.disconnect_camera)]:
            control=button(label,lambda checked=False,fn=fn:self.guarded(fn)); row.addWidget(control)
            if fn==self.connect_camera: self.camera_connect_button=control
        out.addLayout(row)
        self.equipment_details=QPlainTextEdit(); self.equipment_details.setReadOnly(True)
        # Eight lines of summary text: size it to the content instead of letting it
        # stretch into a page-tall empty box.
        scale=self.preferences.value["text_scale"]/100
        self.equipment_details.setMaximumHeight(round(200*scale))
        out.addWidget(self.equipment_details)
        out.addStretch(1)
        row=FlowLayout()
        for label,fn in [(tr('좌표 보정'),self.calibrate),(tr('로봇 운전 설정'),self.robot_profile),(tr('Dobot 연결'),self.connect_robot),(tr('로봇 해제'),self.disconnect_robot),
                         (tr('로봇 준비 확인'),self.recover_robot),(tr('로봇 정지'),self.stop_robot)]:
            control=button(label,lambda checked=False,fn=fn:self.guarded(fn)); row.addWidget(control)
            if fn==self.calibrate: self.calibration_button=control
            if fn==self.recover_robot: self.robot_ready_button=control
        out.addLayout(row); self.robot_label=ui_text(QLabel, tr('로봇 미연결')); self.robot_label.setWordWrap(True); out.addWidget(self.robot_label)
        out.addWidget(ui_text(QLabel, tr('좌표 보정은 실제 측정점과 별도의 검증점이 필요합니다. 집기 높이·이동 높이·분류 위치·물체 감지 입력을 등록한 뒤 운전 준비를 확인하세요.')))
    def refresh_equipment(self):
        e=self.equipment; c=e["camera"]; w=e["workspace"]
        ui_text(self.equipment_details.setPlainText, trf('장비 설정 버전: {v0}\n카메라: {v1} · {v2} × {v3} · {v4} fps\n카메라 설치: {v5} / 촬영 조건: {v6}\n작업 공간: {v7} × {v8} mm\n검사 영역: {v9} · 제외 영역 {v10}개\n좌표 보정: {v11}\nDobot: {v12}\n로봇 운전 설정: {v13}', v0=e['version'], v1=c['serial'] or tr('미등록'), v2=c['width'], v3=c['height'], v4=c['fps'], v5=c['mount_revision'] or tr('미등록'), v6=c['acquisition_revision'] or tr('미등록'), v7=w['width_mm'] or tr('미등록'), v8=w['height_mm'] or tr('미등록'), v9=tr('등록') if w['roi'] else tr('미등록'), v10=len(w['excluded']), v11=e['calibration'] or tr('미등록'), v12=e['robot']['port'] or tr('미등록'), v13=e['robot']['profile'] or tr('미등록')))
        self.canvas.roi=w["roi"]; self.canvas.excluded=w["excluded"]; self.canvas.update()
    def discover(self):
        self.edit_equipment()
    def edit_equipment(self):
        self.require_idle(robot=True)
        # A frozen latest frame remains available for drawing after camera disconnection.
        self.require_idle(camera=True)
        dialog=EquipmentDialog(self.equipment,self.frame,self,discover=True)
        if dialog.exec(): self.equipment=self.store.save_equipment(dialog.value); self.camera_info=None; self.refresh_equipment()
    def station_settings(self):
        from mes_vision.station.settings_dialog import StationSettingsDialog
        self.require_idle(robot=True)
        dialog=StationSettingsDialog(self.runtime,self)
        try:
            if dialog.exec(): self.notice(tr('로봇 촬영 설정을 저장했습니다.'))
        finally: dialog.deleteLater()
    def connect_camera(self):
        require(self.camera is None,tr('카메라가 이미 연결 중이거나 연결돼 있습니다.'))
        require(self.equipment["camera"]["serial"],tr('장치 검색 후 장비 설정에서 카메라를 선택하세요.'))
        self.camera_info=None; self.camera=CameraProcess(self.equipment["camera"]); self.camera.connected.connect(self.camera_connected)
        self.camera.failed.connect(self.camera_failed); self.camera.finished.connect(self.camera_finished)
        try: self.camera.start()
        except Exception as exc: self.camera=None; self.trip_fault("CAMERA_START",tr('카메라 작업 시작 실패: ')+str(exc)); return
        self.notice(tr('카메라 연결 중…'))
    def camera_connected(self,info): self.camera_info=info; self.notice(tr('카메라 영상 연결됨. 검사 시작 전에는 판정하지 않습니다.'))
    def camera_failed(self,message): self.camera_info=None; self.trip_fault("CAMERA_FAILURE",message)
    def camera_finished(self):
        self.camera=None; self.camera_info=None; self.pause_inspection(); self.canvas.tracks=[]; self.canvas.update()
    def disconnect_camera(self):
        if self.robot_pending or self.pick_block: self.trip_fault("CAMERA_DURING_MOTION",tr('로봇 작업 중 카메라를 해제했습니다. 장치와 물체 상태를 확인하세요.'))
        self.pause_inspection()
        if self.camera: self.camera.stopping.set(); self.camera_info=None
    def calibrate(self):
        self.require_idle(robot=True); require(self.product and self.frame and self.camera_info,tr('품목을 적용하고 카메라 영상을 연결하세요.'))
        dialog=CalibrationDialog(self.product,self.equipment,self.frame,self.camera_info,self.runtime,self)
        if dialog.exec():
            e=deepcopy(self.equipment); e["calibration"]=str(dialog.path); e["robot"]["profile"]=None
            self.equipment=self.store.save_equipment(e); self.refresh_equipment()
    def robot_profile(self):
        self.require_idle(robot=True); require(self.product,tr('품목을 적용하세요.'))
        dialog=RobotProfileDialog(self.product,self.equipment,self.runtime,self)
        if dialog.exec():
            e=deepcopy(self.equipment); e["robot"]["profile"]=str(dialog.path); self.equipment=self.store.save_equipment(e); self.refresh_equipment()
    def connect_robot(self):
        require(self.robot is None,tr('로봇 연결이 이미 있습니다.')); require(self.equipment["robot"]["port"],tr('Dobot 포트를 등록하세요.'))
        self.robot=RobotThread(self.runtime/"robot",self.equipment,self.store); self.robot_seen_at=time.monotonic(); self.robot.changed.connect(self.robot_changed)
        self.robot.failed.connect(self.robot_failed); self.robot.completed.connect(self.robot_completed); self.robot.finished.connect(self.robot_finished); self.robot.start()
    def disconnect_robot(self):
        if self.robot_pending or self.pick_block: self.trip_fault("ROBOT_DISCONNECT",tr('진행 중인 로봇 연결을 해제했습니다. 장치와 물체 상태를 확인하세요.'))
        self.auto.setChecked(False)
        if self.robot: self.robot.stop_motion(); self.robot.stopping.set()
    def robot_finished(self): self.robot=None; self.robot_pending=False; self.robot_state={"state":"DISCONNECTED","busy":False}
    def robot_changed(self,data):
        self.robot_seen_at=time.monotonic()
        self.robot_state=data; status=data.get("status",{}); pose=status.get("pose")
        if not data.get("busy") and data['state'] in {"STOPPED","FAULT","RECOVERY","DISCONNECTED"}: self.robot_pending=False; self.pick_block=None
        ui_text(self.robot_label.setText, trf('로봇 상태: {v0}  ·  위치: {v1}  ·  물체 감지: {v2}', v0=data['state'], v1=pose or tr('확인 불가'), v2=status.get('holding', tr('확인 불가'))))
    def robot_failed(self,message):
        self.trip_fault("ROBOT_FAILURE",tr('로봇: ')+message)
    def recover_robot(self):
        require(self.robot is not None,tr('로봇을 연결하세요.'))
        ref,ok=QInputDialog.getText(self,tr('로봇 준비 확인'),tr('물체 없음·이동 높이 대기 위치·작업 영역 확인 기록 (확인자 / 메모)'))
        if ok and ref.strip(): self.robot.request("recover",reference=ref.strip())
    def stop_robot(self):
        self.auto.setChecked(False); self.pause_inspection(); self.pick_block=None
        if self.robot: self.robot.stop_motion(); self.notice(tr('로봇 정지 요청됨. 장비 상태 확인 후 준비 확인이 필요합니다.'))
    def auto_changed(self,checked):
        if checked:
            try:
                require(not self.faults.active,tr('미해결 오류를 확인한 뒤 자동 분류를 켜세요.'))
                require(self.robot and self.robot_state["state"] in {"IDLE","RECAPTURE"},tr('로봇 준비 확인 후 자동 분류를 켜세요.'))
                require(self.engine_ready,tr('모델 준비가 필요합니다.'))
                _,problems,_=self.readiness_status(time.monotonic())
                problems=[p for p in problems if p!=tr('검사가 이미 진행 중입니다.')]
                require(not problems,text_join('\n', problems))
            except Exception as exc: self.auto.setChecked(False); self.error(exc)
    def robot_completed(self,event):
        self.robot_pending=False
        if event["state"]!="RECAPTURE": self.trip_fault("ROBOT_OUTCOME",tr('로봇 동작 결과가 확정되지 않았습니다. 장치와 물체 상태를 확인하세요.')); return
        if self.pick_block: self.pick_block.update(finished=time.monotonic(),empty_observations=0)
        self.notice(tr('분류 동작 확인됨. 집기 위치가 비워졌는지 새 영상으로 확인 중입니다.'))

    def begin(self):
        require(not self.faults.active,tr('미해결 오류를 확인한 뒤 재개하세요.'))
        require(self.engine and self.engine_ready,tr('품목 설정에서 모델을 준비하세요.'))
        require(self.camera_info and self.frame and time.monotonic()-self.frame_received<.5,tr('카메라의 새 영상이 필요합니다.'))
        require(not self.robot_pending and self.pick_block is None,tr('로봇 동작 또는 집기 위치 확인을 먼저 완료하세요.'))
        _,problems,_=self.readiness_status(time.monotonic())
        require(not problems,text_join('\n', problems))
        self.running=True; self.control_enabled(True); self.notice(tr('연속 검사 중 · 새 물체와 재검사 대상이 안정되면 판정합니다.'))
    def control_enabled(self,value,*,reset_all=False):
        self.generation+=1
        for t in self.tracks:
            if reset_all or t.get("result") is None or t.get("missing"): t.update(result=None,status="WAITING")
        if self.engine:
            try: self.engine.command("enable",value=value,generation=self.generation,reset_all=reset_all)
            except Exception as exc: self.trip_fault("ENGINE_CONTROL",tr('검사 제어 전달 실패: ')+str(exc))
        self.record_key=None; self.select_track(self.selected_id)
    def pause_inspection(self):
        self.running=False
        if self.engine: self.control_enabled(False)
        self.start.setEnabled(False)
    def select_track(self,identity):
        self.selected_id=identity; self.canvas.selected_id=identity; self.canvas.update()
        t=next((t for t in self.tracks if t["track_id"]==identity),None)
        from .status_hooks import update_verdict
        update_verdict(self,None,identity)
        if not t:
            ui_text(self.selection.setText, tr('선택 물체가 없거나 추적이 종료됐습니다.')); ui_text(self.live_details.setPlainText, tr('현재 물체를 선택하세요. 이전 검사는 검사 이력에서 확인할 수 있습니다.')); self.record_key=None; return
        ui_text(self.selection.setText, trf('{v0}  ·  {v1}  ·  추적 버전 {v2}', v0=identity.split(':')[-1], v1=STATES.get(t['status'], t['status']), v2=t['revision']))
        result=t.get("result"); key=result["object_id"] if result else None
        if key and key!=self.record_key:
            try:
                row,manifest,run,obj=self.store.open_record(key); ui_text(self.live_details.setPlainText, self.reason_text(obj)); self.record_key=key
            except Exception as exc: ui_text(self.live_details.setPlainText, tr('검사 원본 확인 실패: ')+str(exc)); self.record_key=None
        elif not key: ui_text(self.live_details.setPlainText, tr('유효한 현재 판정이 없습니다. 안정된 영상에서 검사를 완료하면 근거를 표시합니다.')); self.record_key=None
        update_verdict(self,t.get("status") if key and self.record_key==key else None,identity,self.live_details.toPlainText().split("\n")[0])
    @staticmethod
    def reason_text(obj):
        from mes_vision.decision.display import decision_lines
        from mes_vision.vlm.viewer import CHECK_LABELS
        lines=decision_lines(obj)
        assessed={c["check_id"]:c for c in obj.get("decision_details",{}).get("checks",[])}
        for r in basic_reasons(obj):
            assessment=assessed.get(r["check_id"],{}); status=assessment.get("status",r["status"])
            label=tr('확인 필요') if status=="REVIEW" else CHECK_LABELS.get(status,status)
            if assessment and not assessment.get("required") and r["status"]=="NOT_RUN": label=tr('등록 기준에서 제외')
            lines.append(tr(r["name"])+": "+label)
            from .finding_display import finding_caption
            from mes_vision.inspection.finding_scores import score_band
            for f in r["findings"]:
                if score_band(f)!="HIDDEN":lines.append("  "+finding_caption(f))
            if r["raw_score"] is not None: lines.append(trf('  점수: {v0:.4g}', v0=r['raw_score']))
            if r["criteria_version"]: lines.append(tr('  기준: ')+r["criteria_version"])
        return text_join('\n', lines)
    def current_track(self):
        t=next((t for t in self.tracks if t["track_id"]==self.selected_id),None); require(t is not None,tr('화면의 물체를 선택하세요.')); return t
    def recheck(self):
        require(self.running and not self.robot_pending and self.pick_block is None,tr('검사 중이며 로봇 대기 상태일 때 재검사할 수 있습니다.'))
        t=self.current_track(); self.engine.command("recheck",track_id=t["track_id"]); t.update(result=None,status="RECHECK"); self.select_track(t["track_id"])
    def open_selected_record(self):
        t=self.current_track(); require(t.get("result"),tr('선택 물체의 현재 검사 결과가 없습니다.'))
        self.show_record(t["result"]["object_id"]); self.nav.setCurrentRow(3)
    def pick_selected(self): self.pick(self.current_track())
    def pick(self,t):
        require(not self.faults.active,tr('미해결 오류가 있어 로봇 작업을 요청할 수 없습니다.'))
        require(self.running and self.engine_ready and self.camera_info,tr('검사 실행 중에만 새 로봇 작업을 요청할 수 있습니다.'))
        require(time.monotonic()-self.track_updated<.5 and time.monotonic()-self.frame_received<.5,tr('현재 물체 위치를 다시 확인하세요.'))
        require(not self.robot_pending and self.pick_block is None and self.robot and self.robot_state["state"] in {"IDLE","RECAPTURE"},tr('로봇이 분류 준비 상태가 아닙니다.'))
        require(t["status"] in {"OK","NG"} and t.get("result"),tr('유효한 현재 OK/NG 판정이 필요합니다.'))
        row,manifest,result,obj=self.store.open_record(t["result"]["object_id"])
        captured=deepcopy(t); self.robot_pending=True; self.pick_block={"box":t["box"],"finished":None,"empty_observations":0}
        self.control_enabled(False,reset_all=True)
        try: self.robot.request("pick",result=result,track=captured,product=self.product,session=self.session)
        except Exception: self.robot_pending=False; self.pick_block=None; self.running=False; raise

    def make_history(self):
        out=self.page(); self.make_history_controls(out)
        self.history_table=table([tr('검사 시각'),tr('품목'),tr('물체 ID'),tr('판정'),tr('검사 버전'),tr('재검사'),tr('검토'),tr('보존')]); self.history_table.setMaximumHeight(220); out.addWidget(self.history_table)
        for i,width in enumerate((170,160,100,70,85,85,80)): self.history_table.setColumnWidth(i,width)
        self.history_table.itemSelectionChanged.connect(self.history_selected)
        from .image_view import ImagePanel
        panes=QSplitter(); self.saved_image=ImagePanel(); self.saved_canvas=self.saved_image.canvas; panes.addWidget(self.saved_image)
        self.saved_image.layout().addWidget(button(tr('이미지 확대 · 정상 비교'),lambda:self.guarded(self.open_evidence)))
        self.history_details=QPlainTextEdit(); self.history_details.setReadOnly(True); panes.addWidget(self.history_details); out.addWidget(panes,1)
        panes.setStretchFactor(0,3); panes.setStretchFactor(1,2); panes.setSizes([650,450])
        self.saved_canvas.placeholder=tr('검사 이력을 선택하면 저장한 원본이 표시됩니다.')
        row=FlowLayout()
        for label,fn in [(tr('검토 기록'),self.add_review),(tr('별도 보존'),self.hold_record),(tr('VLM 분석 요청'),self.request_analysis),(tr('VLM 결과 보기'),self.show_analysis),(tr('원본 폴더'),self.open_record_folder)]:
            row.addWidget(button(label,lambda checked=False,fn=fn:self.guarded(fn)))
        out.addLayout(row); out.addWidget(ui_text(QLabel, tr('저장된 검사 시점의 원본과 근거입니다. 과거 이력에서는 로봇을 움직일 수 없습니다.')))
    def history_selected(self):
        i=self.history_table.currentRow()
        if 0<=i<len(self.history_rows): self.guarded(lambda:self.show_record(self.history_rows[i]["object_id"]))
    def show_record(self,identity):
        changed=self.record is None or self.record[0]['object_id']!=identity
        self.record=self.store.open_record(identity); row,manifest,result,obj=self.record
        self.saved_canvas.set_file(Path(row["path"])/"frame.png")
        if changed: self.saved_canvas.fit()
        b=obj["effective_box"]; b=[b[k] for k in ("x1","y1","x2","y2")] if isinstance(b,dict) else b
        self.saved_canvas.tracks=[{"track_id":row["track_id"],"box":b,"status":obj["final_decision"]}]; self.saved_canvas.selected_id=row["track_id"]; self.saved_canvas.update()
        import json
        with self.store.connect() as db:
            events=[dict(r) for r in db.execute("SELECT type,data FROM events WHERE track_id=? AND type LIKE 'ROBOT_%' ORDER BY id",(row["track_id"],))]
        motion=[]
        for event in events:
            data=json.loads(event["data"])
            if data.get("object_id")!=identity: continue
            motion.append(tr('분류 요청 기록됨') if event["type"]=="ROBOT_PLAN" else tr('집기·놓기 센서 확인 완료') if data.get("state")=="RECAPTURE" else tr('동작 상태 확인 필요: ')+data.get("state",""))
        review=json.loads(row['review']) if row['review'] else None
        review_text=(f"{review.get('operator','')} · {review.get('note','')}" if review else tr('없음'))
        ui_text(self.history_details.setPlainText, self.reason_text(obj)+trf('\n\n운전 회차: {v0}\n추적 ID: {v1}\n검사 ID: {v2}\n품목 버전: {v3} / 장비 버전: {v4}\n이전 검사: {v5}\n검토 기록: {v6}\n로봇 기록: {v7}', v0=row['session'], v1=row['track_id'], v2=row['run_id'], v3=result['config'].get('product_version'), v4=result['config'].get('equipment_version'), v5=row['prior_object_id'] or tr('없음'), v6=review_text, v7=text_join(' / ', motion) or tr('없음')))
    def open_evidence(self):
        require(self.record,tr('검사 이력을 선택하세요.'))
        from .evidence_dialog import show_evidence
        row=self.record[0]
        show_evidence(self,row['path'],row['object_id'],row['digest'])
    def add_review(self):
        require(self.record,tr('검사 이력을 선택하세요.'))
        operator,ok=QInputDialog.getText(self,tr('검토 기록'),tr('확인자'))
        if not ok: return
        note,ok=QInputDialog.getMultiLineText(self,tr('검토 기록'),tr('검토 내용 (기본 판정은 유지됩니다)'))
        if ok: self.store.review(self.record[0]["object_id"],operator,note); self.show_record(self.record[0]["object_id"])
    def request_analysis(self):
        require(self.record,tr('검사 이력을 선택하세요.')); require(self.queue.enabled(),tr('VLM을 ON으로 설정하세요.'))
        self.show_record(self.record[0]["object_id"])
        row=self.record[0]; self.queue.enqueue(row["path"],row["object_id"],asdict(GenerationConfig())); self.analysis.refresh(); self.notice(tr('VLM 추가 분석 요청됨.'))
    def show_analysis(self):
        require(self.record,tr('검사 이력을 선택하세요.')); identity=self.record[0]["object_id"]
        self.analysis.refresh()
        for i,row in enumerate(self.analysis.rows):
            if row["object_id"]==identity: self.analysis.table.selectRow(i); break
        else: self.notice(tr('이 물체의 VLM 요청이 없습니다. 기본 검사 근거는 검사 이력에 표시돼 있습니다.')); return
        self.nav.setCurrentRow(4)
    def open_record_folder(self):
        require(self.record,tr('검사 이력을 선택하세요.')); self.show_record(self.record[0]["object_id"])
        QDesktopServices.openUrl(QUrl.fromLocalFile(self.record[0]["path"]))
    def toggle_vlm(self,checked):
        if checked and self.vlm_worker_enabled and (self.worker is None or not self.worker.isRunning()):
            self.worker=WorkerThread(self.queue,self.root,allow_synthetic=False,backend="qwen"); self.worker.failed.connect(self.vlm_worker_failed); self.worker.start()
        def change():
            self.queue.set_enabled(bool(checked)); self.apply_vlm_status(bool(checked))
        self.guarded(change); self.analysis.refresh()
    def vlm_worker_failed(self,message):
        self.analysis.refresh_read.invalidate()
        try: self.queue.set_enabled(False); self.apply_vlm_status(False)
        except Exception: pass
        self.notice(tr('VLM 추가 분석을 중단했습니다. 기본 검사는 유지됩니다. 원인 확인 후 상단 VLM 버튼으로 다시 켜세요: ')+message)
    def open_training(self):
        self.require_idle(); require(not self.queue.enabled(),tr('데이터 학습 도구를 열기 전에 VLM을 OFF로 설정하세요.'))
        from .training_dialog import TrainingDialog
        dialog=TrainingDialog(self.root,self.runtime,self); self.children.append(dialog); dialog.show()

    def handle_engine(self,event):
        kind=event["type"]
        if kind=="ready":
            if self.faults.active or self.engine and getattr(self.engine,"stopping",False): return
            self.engine_ready=True; self.applied_message=tr('모델 준비됨 · 추적 중. 검사 시작을 누르면 판정합니다.')
            self.control_enabled(False)
            if not event["concurrent_vlm"]: self.applied_message+=tr(' VLM은 모델 해제 후 GPU를 사용할 수 있습니다.')
        elif kind=="error":
            self.trip_fault(event.get("code",fault_code(event["message"])),tr('검사 처리 오류: ')+event["message"])
            try: self.store.event("ENGINE_ERROR",event,session=self.session)
            except Exception: pass
        elif kind=="inspection" and event["generation"]==self.generation:
            c=event["counts"]; ui_text(self.counts.setText, trf('검사한 물체 {v0}  ·  정상 {v1}  ·  불량 {v2}  ·  보류 {v3}  ·  재검사 {v4}', v0=c['OK'] + c['NG'] + c['REVIEW'], v1=c['OK'], v2=c['NG'], v3=c['REVIEW'], v4=c['reinspections']))
            self.refresh_history()
        elif kind=="tracks" and event["generation"]==self.generation:
            self.tracks=event["tracks"]; self.track_updated=event["updated"]
            self.canvas.tracks=[t for t in self.tracks if not t["missing"]]; self.canvas.update(); self.select_track(self.selected_id)
            self.applied_message=trf('{v0}  ·  물체 {v1}개  ·  처리 {v2:.0f} ms', v0=tr('연속 검사 중') if self.running else tr('판정 중단 · 추적 유지'), v1=len(self.canvas.tracks), v2=event['loop_ms'])
            if self.pick_block and self.pick_block["finished"] and event["updated"]>self.pick_block["finished"]:
                clear=not any(overlap(t["box"],self.pick_block["box"])>.1 for t in self.canvas.tracks)
                self.pick_block["empty_observations"]=self.pick_block["empty_observations"]+1 if clear else 0
                if self.pick_block["empty_observations"]>=3:
                    self.pick_block=None
                    if self.running: self.control_enabled(True)
                elif time.monotonic()-self.pick_block["finished"]>5:
                    self.auto.setChecked(False); self.running=False; self.notice(tr('집기 위치의 물체가 계속 검출됩니다. 상태 확인 후 재시작하세요.')); self.pick_block=None
            if self.auto.isChecked() and self.running and not self.robot_pending and self.pick_block is None:
                candidate=next((t for t in self.tracks if t["status"] in {"OK","NG"} and not t["missing"]),None)
                if candidate:
                    try: self.pick(candidate)
                    except Exception as exc: self.auto.setChecked(False); self.notice(tr('자동 분류 대기: ')+str(exc))
    def tick(self):
        now=time.monotonic()
        if self.closing:
            if self.engine and now-self.close_at>3: self.finish_engine()
            if not self.camera and not self.robot and not self.engine and not self.tasks and not self.vlm_read.busy and not self.analysis.refresh_read.busy and not self.readiness_read.busy and (not self.worker or not self.worker.isRunning()): self.close()
            return
        if self.camera:
            frame,received=self.camera.get_latest()
            if frame is not None and (self.frame is None or frame.frame_id!=self.frame.frame_id):
                self.frame=frame; self.frame_received=received; self.canvas.set_rgb(frame.rgb)
                if self.engine and not self.engine.stopping: self.engine.frame(frame,received)
        if self.running and self.camera_info and now-self.frame_received>2:
            self.trip_fault("CAMERA_STALE",tr('새 카메라 영상이 없어 검사를 중단했습니다. 연결을 확인하세요.'))
        if self.robot and self.robot_seen_at and now-self.robot_seen_at>5 and not any(e['code']=='ROBOT_STATUS_TIMEOUT' for e in self.faults.active):
            self.trip_fault("ROBOT_STATUS_TIMEOUT",tr('로봇 상태 응답이 5초 동안 없습니다. 정지를 요청했으며 실제 장치와 연결 상태를 확인해야 합니다.'))
        if self.engine:
            try:
                for event in self.engine.poll(): self.handle_engine(event)
                watchdog=self.engine.watchdog_error(now) if hasattr(self.engine,"watchdog_error") else None
                if not self.engine.process.is_alive():
                    expected=self.engine.stopping
                    if not expected: self.trip_fault("ENGINE_EXIT",tr('검사 프로세스가 예기치 않게 종료됐습니다. GPU와 오류 기록을 확인하세요.'))
                    self.finish_engine("STOPPED" if expected else "INTERRUPTED")
                elif self.stop_at and now-self.stop_at>3: self.finish_engine()
                elif watchdog: self.trip_fault("ENGINE_TIMEOUT",watchdog)
                elif not self.engine_ready and not self.stop_at and now-self.checked_at>180:
                    self.trip_fault("ENGINE_TIMEOUT",tr('모델 준비 시간이 초과됐습니다. 모델 파일과 GPU 상태를 확인하세요.'))
            except Exception as exc: self.trip_fault(fault_code(exc),tr('검사 작업 연결 오류: ')+str(exc))
        if self.tracks and now-self.track_updated>max(1.,self.equipment["tracking"]["max_gap"]):
            for t in self.tracks: t.update(result=None,status="LOST")
            self.canvas.tracks=[]; self.canvas.update(); self.select_track(self.selected_id)
        if now>=self.vlm_probe_at and not self.vlm_read.busy:
            self.vlm_probe_at=now+1.
            self.vlm_read.start(self.queue.enabled)
        self.readiness_tick(now)
        self.pause.setEnabled(self.running); self.unload.setEnabled(self.engine is not None)
        self.recheck_button.setEnabled(self.running and self.selected_id is not None and not self.robot_pending and not self.pick_block)
        self.pick_button.setEnabled(self.running and self.selected_id is not None and self.robot is not None and not self.robot_pending and not self.pick_block)
        camera_label=tr('영상 수신 중') if self.camera_info and now-self.frame_received<2 else tr('카메라 미연결 / 영상 대기')
        ui_text(self.live_state.setText, camera_label+"  ·  "+self.applied_message)
        if self.pages.currentIndex()==3 and now-self.last_history>2: self.refresh_history()
        from .status_hooks import update_context
        update_context(self)
    def apply_vlm_status(self,enabled):
        # Local ON/OFF actions supersede any older read already in flight.
        self.vlm_read.invalidate(); self.vlm_probe_at=time.monotonic()+1.
        self.cached_vlm_enabled=bool(enabled); self.vlm_unavailable=False
        self.vlm.blockSignals(True); self.vlm.setChecked(bool(enabled))
        ui_text(self.vlm.setText, "VLM ON" if enabled else "VLM OFF"); self.vlm.blockSignals(False)

    def vlm_read_failed(self,message):
        if not self.vlm_unavailable: self.vlm_worker_failed(tr('VLM 기록 접근 실패: ')+message)
        self.cached_vlm_enabled=False; self.vlm_unavailable=True; self.vlm_probe_at=time.monotonic()+5.
        self.vlm.blockSignals(True); self.vlm.setChecked(False); ui_text(self.vlm.setText, tr('VLM 기록 오류')); self.vlm.blockSignals(False)

    def safe_tick(self):
        try: self.tick()
        except Exception as exc:
            self.trip_fault(fault_code(exc),tr('운영 처리 오류로 새 검사를 중단했습니다: ')+str(exc))
    def closeEvent(self,event):
        from .evidence_dialog import close_evidence
        close_evidence(self); close_evidence(self.analysis)
        if self.closing and not self.camera and not self.robot and not self.engine and not self.tasks and not self.vlm_read.busy and not self.analysis.refresh_read.busy and not self.readiness_read.busy and (not self.worker or not self.worker.isRunning()):
            try: self.preferences.save_geometry(bytes(self.saveGeometry().toHex()).decode("ascii"))
            except Exception as exc: self.notice(tr("환경설정을 저장하지 못했습니다: ")+str(exc))
            self.analysis.timer.stop(); self.timer.stop(); event.accept(); return
        if any(getattr(w,"busy",False) for w in self.children):
            event.ignore(); self.notice(tr('진행 중인 학습 작업을 중단하거나 완료한 뒤 종료하세요.')); return
        event.ignore()
        if self.closing:
            QTimer.singleShot(50,self.close); return
        self.closing=True; self.readiness_read.invalidate(); self.vlm_read.invalidate(); self.analysis.stop_refresh(); self.close_at=time.monotonic(); self.pause_inspection(); self.disconnect_camera(); self.disconnect_robot()
        if self.engine: self.engine.stop()
        if self.worker: self.worker.stop_event.set()
        for child in self.children: child.close()
        self.notice(tr('장치와 작업을 종료하고 기록을 저장하고 있습니다.'))

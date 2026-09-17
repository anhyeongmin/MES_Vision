"""Production overview/detail inspection screen; the legacy tracker is not run."""
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import json,time
from PySide6.QtCore import Qt,QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QSplitter,QPlainTextEdit,QCheckBox,QDialog,QTableWidgetItem,QScrollArea,QFrame,QSizePolicy
from mes_vision.operation.window import DesktopWindow
from mes_vision.operation.widgets import button,table
from mes_vision.operation.image_view import ImagePanel
from mes_vision.operation.responsive import FlowLayout
from mes_vision.operation.shell import VerdictBanner,ViewArea,panel,styled
from mes_vision.theme import METRICS
from mes_vision.operation.catalog import readiness
from mes_vision.operation.engine import vlm_eligible
from mes_vision.operation.storage import check_free_space
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.ui_refresh import BackgroundRead
from mes_vision.training.data import require
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr,trf,text_join
from .sequence import ScanJournal,TERMINAL
from .settings import StationSettings
from .profile import build_scan_profile
from .process_camera import ProcessCameraPort
from .robot_port import StationRobotPort
from .recipe import load_recipe
from .worker import StationProcess,read_capture
from .runtime import ScanRuntime

STATES={'CREATED':'검사 준비','MOVE_OVERVIEW':'전체 촬영 위치로 이동','SETTLING':'진동 안정 대기',
    'CAPTURE_OVERVIEW':'전체사진 촬영','DETECT_OVERVIEW':'물체 위치 검출','MAP_TARGETS':'로봇 좌표 변환',
    'MOVE_DETAIL':'근접 촬영 위치로 이동','CAPTURE_DETAIL':'상세사진 촬영','INSPECT_DETAIL':'상세 검사',
    'WAITING':'검사 대기','MOVING':'촬영 위치로 이동','INSPECTING':'검사 중','OK':'정상','NG':'불량','REVIEW':'보류',
    'SORT_OBJECT':'분류 이송 중','SORTING':'분류 이송 중','SORTED':'이송 완료',
    'SORT_UNCONFIRMED':'이송 확인 필요',
    'COMPLETED':'검사 완료','CANCELLED':'검사 중단','FAILED':'검사 실패','INTERRUPTED':'중단된 검사'}
REASONS={'OVERVIEW_OVERLAP':'물체가 겹쳐 있어 개별 위치를 확정할 수 없습니다.',
    'OVERVIEW_WORKSPACE':'물체가 검증된 전체 촬영 영역을 벗어났습니다.',
    'DETAIL_TARGET_UNCONFIRMED':'상세사진에서 선택한 물체 하나를 확실하게 식별하지 못했습니다.',
    'DETAIL_IMAGE_QUALITY':'상세사진의 선명도 또는 밝기가 검사 기준에 맞지 않습니다.',
    'DETAIL_WORKSPACE':'물체가 상세 검사 영역을 벗어났습니다.'}


class InspectionPage(QScrollArea):
    """Fit splitters to available height; scroll only below the usable minimum.

QScrollArea's default height-for-width calculation otherwise expands nested
splitters to preferred heights and pushes reasons below a laptop viewport.
"""
    def __init__(self,content):
        super().__init__(); self.setFrameShape(QFrame.NoFrame); self.setWidgetResizable(False); self.setWidget(content)
        self.setMinimumSize(0,0)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        content=self.widget(); minimum=content.minimumSizeHint()
        content.resize(max(self.viewport().width(),minimum.width()),max(self.viewport().height(),minimum.height()))


class StationWindow(DesktopWindow):
    PAGES=DesktopWindow.PAGES+[('로봇 제어 · 티칭','조인트·XYZR 수동 제어와 위치 티칭을 진행합니다.')]

    def __init__(self,root,runtime,*,start_worker=True):
        self.scan=None; self.capture_port=None; self.display_data=None; self.rendered=None; self.overview_key=None; self.detail_key=None
        self.model_info=None; self.recipe=None; self.preview=None; self.ui_probe_at=0.; self.history_due=0.; self.observed_revision=None
        self.scan_journal=ScanJournal(runtime)
        # Production launcher already holds desktop.lock before constructing this window.
        interrupted=self.scan_journal.recover_interrupted()
        with self.scan_journal.connect() as db:
            db.execute('CREATE TABLE IF NOT EXISTS station_vlm_jobs(cycle_id TEXT,target_id TEXT,job_id TEXT,PRIMARY KEY(cycle_id,target_id))')
        super().__init__(root,runtime,start_worker=start_worker)
        from mes_vision.operation.acquisition import configure_equipment
        self.equipment=configure_equipment(self.root,self.equipment)
        self.refresh_equipment()
        from .teaching_dialog import TeachingDialog
        self.robot_panel=TeachingDialog(self.runtime,self.equipment,self,embedded=True)
        self.pages.addWidget(self.robot_panel)
        self.nav.addItem('6  로봇 제어 · 티칭'); self.compact_nav.addItem('6  로봇 제어 · 티칭')
        self.sidebar.add_group('로봇',[('로봇 제어 · 티칭','F6')])
        self.nav.currentRowChanged.connect(self.robot_page_changed)
        ui_text(self.nav.item(0).setText,tr('1  검사 운영')); ui_text(self.compact_nav.setItemText,0,tr('1  검사 운영'))
        self.refresh_nav_labels()
        if interrupted: self.trip_fault('INTERRUPTED_STATION',tr('중단된 회차는 다시 실행하지 않습니다. 장치 확인 후 새 전체사진으로 검사하세요.'))

    def make_live(self):
        content=QWidget(); out=QVBoxLayout(content); out.setContentsMargins(0,0,0,0)
        out.setSpacing(round(METRICS['layout_gap'])); self.pages.addWidget(InspectionPage(content))

        # --- toolbar -------------------------------------------------------
        row=FlowLayout()
        self.start=button(tr('검사 시작'),lambda:self.guarded(self.begin)); self.start.setObjectName('inspectionStart')
        self.pause=button(tr('검사 중단'),self.pause_inspection); self.pause.setObjectName('dangerButton')
        self.unload=button(tr('모델 해제'),self.unload_engine)
        self.sort_enabled=ui_text(QCheckBox,tr('검사 후 자동 분류'))
        ui_text(self.sort_enabled.setToolTip,tr('모든 상세 검사 후 정상·불량을 이송합니다. 보류 물체는 이동하지 않습니다.'))
        for w in (self.start,self.pause,self.unload,button(tr('카메라 영상'),self.open_preview),
                  button(tr('현재 영상 수집'),lambda:self.guarded(self.capture_training)),
                  button(tr('학습데이터 수집'),lambda:self.guarded(lambda:self.open_capture_studio(collect=True)))): row.addWidget(w)
        row.addWidget(self.sort_enabled)
        toolbar=styled(QWidget()); toolbar.setObjectName('toolbar')
        toolbar_layout=QVBoxLayout(toolbar); toolbar_layout.setContentsMargins(12,9,12,9); toolbar_layout.addLayout(row)

        self.live_state=QLabel(); self.live_state.setWordWrap(True); out.addWidget(self.live_state)
        self.readiness_read=BackgroundRead(self)
        self.readiness_toggle=button('',lambda:None); self.readiness_toggle.setCheckable(True); self.readiness_toggle.hide()
        self.auto=QCheckBox(); self.auto.hide()  # Legacy recovery controls require these references; no legacy actions are connected.
        self.recheck_button=button('',lambda:None); self.recheck_button.hide(); self.pick_button=button('',lambda:None); self.pick_button.hide()

        # --- image panes: full shot on the left, live and detail stacked ------
        body=QHBoxLayout(); body.setSpacing(round(METRICS['layout_gap'])); out.addLayout(body,1)
        frame,frame_layout=panel(); frame_layout.addWidget(toolbar)
        self.verdict=VerdictBanner(); frame_layout.addWidget(self.verdict)
        self.live_image=ImagePanel(title=tr('저장된 전체사진'),meta=tr('중앙 1:1 크롭'))
        self.canvas=self.live_image.canvas
        self.canvas.placeholder=tr('전체사진이 없습니다.\n검사 시작 후 촬영한 전체사진을 표시합니다.')
        self.canvas.selected.connect(self.select_track)
        # Titled '원본' because nothing rectifies this frame yet. When the backend wraps
        # it in station.manual_capture.Rectifier, change this string to '왜곡보정 실시간 영상'
        # (already present in all four locales).
        self.live_view=ImagePanel(title=tr('실시간 영상 · 원본'),compact=True)
        self.live_view.canvas.placeholder=tr('카메라 영상을 연결하면 표시합니다.')
        self.detail_image=ImagePanel(title=tr('선택 물체의 상세사진'),compact=True)
        self.detail_image.canvas.placeholder=tr('선택한 물체의 상세사진이 여기에 표시됩니다.')
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
        head_title=ui_text(QLabel,tr('선택 물체 판정')); head_title.setObjectName('panelTitle')
        head_layout.addWidget(head_title); head_layout.addStretch(1); result_layout.addWidget(head)
        inner=QWidget(); detail=QVBoxLayout(inner); detail.setContentsMargins(12,10,12,10); detail.setSpacing(8)
        self.selection=ui_text(QLabel,tr('화면의 물체를 선택하세요.')); self.selection.setWordWrap(True); detail.addWidget(self.selection)
        detail.addWidget(ui_text(QLabel,tr('기본 판정과 이유')))
        self.live_details=QPlainTextEdit(); self.live_details.setReadOnly(True); detail.addWidget(self.live_details,1)
        detail.addWidget(ui_text(QLabel,tr('VLM 보조 설명')))
        self.vlm_details=QPlainTextEdit(); self.vlm_details.setReadOnly(True); detail.addWidget(self.vlm_details,1)
        detail.addWidget(button(tr('VLM 설명 요청'),lambda:self.guarded(self.request_selected_vlm)))
        result_layout.addWidget(inner,1); column.addWidget(result,1)

        objects,objects_layout=panel()
        objects_head=styled(QWidget()); objects_head.setObjectName('panelHead')
        objects_head_layout=QHBoxLayout(objects_head); objects_head_layout.setContentsMargins(12,8,12,8)
        objects_title=ui_text(QLabel,tr('물체 목록')); objects_title.setObjectName('panelTitle')
        objects_head_layout.addWidget(objects_title); objects_head_layout.addStretch(1)
        objects_layout.addWidget(objects_head)
        self.object_table=table([tr('물체 ID'),tr('판정'),tr('진행 상태')])
        self.object_table.itemSelectionChanged.connect(self.select_row)
        objects_layout.addWidget(self.object_table,1); column.addWidget(objects,1)

        self.counts=QLabel(); self.counts.setObjectName('statusPill'); column.addWidget(self.counts)
        holder=QWidget(); holder.setLayout(column)
        # The image panes cannot use extra width (they are square), so the judgement
        # column absorbs all of it. A maximum here would make QBoxLayout centre the
        # column and leave dead space on both sides, so only a minimum is set.
        scale=self.preferences.value["text_scale"]/100
        holder.setMinimumWidth(round(430*scale))
        holder.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.judgement_column=holder
        body.addWidget(holder,1)

        note=ui_text(QLabel,tr('전체 촬영 → 위치 검출 → 물체별 근접 촬영·판정. 검사 중에는 물체를 움직이지 마세요. 중단하면 로봇 정지를 요청하고 영상 연결은 유지합니다.'))
        note.setWordWrap(True); out.addWidget(note)

    def make_products(self):
        super().make_products()
        # These belong in the action row with the other product buttons; added to the
        # page layout directly they stretch to the full page width.
        self.products_actions.addWidget(button(tr('전체·상세 검사 설정'),lambda:self.guarded(self.edit_recipe)))
        self.products_actions.addWidget(button(tr('저장사진 검사'),lambda:self.guarded(self.open_photo_inspection)))
        self.products_actions.addWidget(button(tr('수동 도착 검사'),lambda:self.guarded(self.open_manual_inspection)))
    def open_manual_inspection(self):
        from .manual_dialog import ManualInspectionDialog
        self.require_idle(robot=True)
        require(self.camera is not None and not self.camera.stopping.is_set(),tr('장비 설정에서 U20CAM 카메라를 연결하세요.'))
        require(self.equipment['camera'].get('driver')=='uvc',tr('U20CAM 보정 데이터를 사용하는 수동 검사입니다.'))
        dialog=ManualInspectionDialog(self.root,self.runtime,self.equipment['camera'],self,camera=self.camera)
        self.children.append(dialog)
        try: dialog.exec()
        finally: dialog.deleteLater(); self.children.remove(dialog)
    def open_photo_inspection(self):
        from .photo_dialog import PhotoInspectionDialog
        self.require_idle()
        require(self.engine is None,tr('모델을 해제한 뒤 저장사진 검사를 시작하세요.'))
        dialog=PhotoInspectionDialog(self.root,self.runtime,self)
        self.children.append(dialog); dialog.exec(); dialog.deleteLater(); self.children.remove(dialog)
    def edit_recipe(self):
        from .recipe_dialog import RecipeDialog
        self.require_idle(); require(self.product is not None,tr('품목을 적용하세요.'))
        dialog=RecipeDialog(self.runtime,self.product,self.equipment,self)
        try:
            if dialog.exec(): self.notice(tr('전체·상세 검사 설정을 저장했습니다.'))
        finally: dialog.deleteLater()
    def edit_equipment(self):
        from .equipment_dialog import StationEquipmentDialog
        self.require_idle(robot=True,camera=True)
        dialog=StationEquipmentDialog(self.equipment,self.frame,self,discover=True)
        try:
            if dialog.exec():
                self.equipment=dialog.persist(self.store); self.camera_info=None; self.refresh_equipment()
        finally: dialog.deleteLater()
    def robot_profile(self):
        from .robot_profile_dialog import StationRobotProfileDialog
        self.require_idle(robot=True); require(self.product is not None,tr('품목을 적용하세요.'))
        dialog=StationRobotProfileDialog(self.product,self.equipment,self.runtime,self)
        try:
            if dialog.exec():
                e=deepcopy(self.equipment); e['robot']['profile']=str(dialog.path); self.equipment=self.store.save_equipment(e); self.refresh_equipment()
        finally: dialog.deleteLater()
    def make_history(self):
        super().make_history()
        layout=self.pages.widget(3).widget().layout()
        self.cycles_table=table([tr('검사 회차'),tr('진행 상태'),tr('물체 수')]); self.cycles_table.setMaximumHeight(180)
        self.cycles_table.itemDoubleClicked.connect(lambda *_:self.guarded(self.open_cycle))
        self.cycle_toggle=ui_text(QCheckBox,tr('최근 회차 목록 · 두 번 눌러 결과 보기'))
        self.cycles_table.hide(); self.cycle_toggle.toggled.connect(self.cycles_table.setVisible)
        layout.insertWidget(0,self.cycle_toggle); layout.insertWidget(1,self.cycles_table)
        layout.insertWidget(2,button(tr('선택 이력의 전체·상세 결과'),lambda:self.guarded(self.open_history_cycle)))
    def open_history_cycle(self):
        require(self.record and 'data' in self.record[0],tr('전체·상세 검사 이력을 선택하세요.'))
        require(not self.running,tr('검사 중단 후 이전 회차를 확인하세요.'))
        row=self.record[0]; data=self.scan_journal.read(row['session'])
        self.render_cycle(data,force=True); self.select_track(row['object_id']); self.nav.setCurrentRow(0)
    def show_record(self,identity):
        record=self.store.open_record(identity)
        if 'data' not in record[0]: return super().show_record(identity)
        self.record=record; row,manifest,result,obj=record; data=json.loads(row['data']); target=data['target']
        capture=target.get('detail') or data['overview']; self.saved_canvas.set_rgb(read_capture(capture).rgb)
        self.saved_canvas.tracks=[]
        if obj:
            from mes_vision.operation.finding_display import object_overlays
            self.saved_canvas.tracks=object_overlays(obj,identity=identity)
            reason=self.reason_text(obj)
        else:
            reason=tr(REASONS.get((target.get('result') or {}).get('review_reason'),'')) or tr('상세 검사 결과가 없습니다. 전체사진의 위치 확인 사유를 확인하세요.')
            if not target.get('detail'):
                box=target['overview']['box']; self.saved_canvas.tracks=[{'track_id':identity,'box':[box[k] for k in ('x1','y1','x2','y2')],'status':'REVIEW'}]
                code=target['overview'].get('reason'); reason=text_join('\n',[reason,tr(REASONS.get(code,code or ''))])
        self.saved_canvas.selected_id=identity; self.saved_canvas.fit(); self.saved_canvas.update()
        review=json.loads(row['review']) if row['review'] else None
        ui_text(self.history_details.setPlainText,text_join('\n',[reason,
            trf('검사 회차: {id}',id=row['session']),trf('물체 ID: {id}',id=identity),
            trf('이송 상태: {state}',state=tr(STATES.get(target['status'],target['status']))),
            trf('검토 기록: {note}',note=(review['operator']+' · '+review['note']) if review else tr('없음'))]))
    def open_evidence(self):
        if not self.record or 'data' not in self.record[0]: return super().open_evidence()
        row=self.record[0]
        if row['path']:
            from mes_vision.operation.evidence_dialog import show_evidence
            return show_evidence(self,row['path'],row['source_object_id'],row['digest'])
        data=json.loads(row['data']); dialog=QDialog(self); dialog.resize(900,600); ui_text(dialog.setWindowTitle,tr('저장된 전체사진'))
        panel=ImagePanel(); QVBoxLayout(dialog).addWidget(panel); panel.canvas.set_rgb(read_capture(data['target'].get('detail') or data['overview']).rgb)
        dialog.exec(); dialog.deleteLater()
    def request_analysis(self):
        if not self.record or 'data' not in self.record[0]: return super().request_analysis()
        self.show_record(self.record[0]['object_id']); row=self.record[0]
        require(row['path'] and row['source_object_id'],tr('확인된 상세 검사 근거가 있어야 VLM 분석을 요청할 수 있습니다.'))
        require(self.queue.enabled(),tr('VLM을 ON으로 설정하세요.'))
        job=self.queue.enqueue(row['path'],row['source_object_id'],asdict(GenerationConfig()))
        with self.scan_journal.connect() as db:
            db.execute('INSERT OR REPLACE INTO station_vlm_jobs VALUES(?,?,?)',(row['session'],row['object_id'],job))
        self.analysis.refresh()
        self.notice(tr('VLM 추가 분석 요청됨.'))
    def add_review(self):
        super().add_review(); self.refresh_history()
    def show_analysis(self):
        if not self.record or 'data' not in self.record[0]: return super().show_analysis()
        identity=self.record[0]['source_object_id']; self.analysis.refresh()
        for i,row in enumerate(self.analysis.rows):
            if row['object_id']==identity: self.analysis.table.selectRow(i); self.nav.setCurrentRow(4); return
        self.notice(tr('이 물체의 VLM 요청이 없습니다. 기본 검사 근거는 검사 이력에 표시돼 있습니다.'))
    def open_record_folder(self):
        if not self.record or 'data' not in self.record[0]: return super().open_record_folder()
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl
        self.show_record(self.record[0]['object_id']); row=self.record[0]; data=json.loads(row['data'])
        path=row['path'] or str(Path((data['target'].get('detail') or data['overview'])['path']).parent)
        QDesktopServices.openUrl(QUrl.fromLocalFile(path))
    def refresh_history(self):
        super().refresh_history()
        if not hasattr(self,'cycles_table'): return
        with self.scan_journal.connect() as db:
            rows=db.execute('SELECT id,state,json_array_length(json_extract(data,\'$.targets\')) FROM cycles ORDER BY rowid DESC LIMIT 100').fetchall()
        self.cycle_rows=rows; self.cycles_table.setRowCount(len(rows))
        for i,(identity,state,count) in enumerate(rows):
            for j,value in enumerate((identity,tr(STATES.get(state,state)),str(count))): self.cycles_table.setItem(i,j,ui_text(QTableWidgetItem,value))
        if not self.running and self.display_data:
            current=self.scan_journal.read(self.display_data['id'])
            if current['revision']!=self.display_data['revision']:
                self.overview_key=None; self.shown_frame=None; self.detail_key=None
                self.render_cycle(current,force=True)
    def open_cycle(self):
        require(not self.running,tr('검사 중단 후 이전 회차를 확인하세요.'))
        i=self.cycles_table.currentRow(); require(0<=i<len(self.cycle_rows),tr('검사 회차를 선택하세요.'))
        self.render_cycle(self.scan_journal.read(self.cycle_rows[i][0]),force=True); self.nav.setCurrentRow(0)
    def open_preview(self):
        if self.preview is not None: self.preview.show(); self.preview.raise_(); return
        dialog=QDialog(self); ui_text(dialog.setWindowTitle,tr('카메라 영상')); dialog.resize(760,570)
        layout=QVBoxLayout(dialog); panel=ImagePanel(); layout.addWidget(panel)
        self.preview=dialog; self.preview_panel=panel; self.children.append(dialog); dialog.show()
    def prepare_engine(self):
        self.require_idle(); require(not self.faults.active,tr('미해결 오류를 확인한 뒤 재개하세요.'))
        require(self.product is not None,tr('품목을 적용하세요.'))
        p=deepcopy(self.product); e=deepcopy(self.equipment)
        def validate():
            problems=readiness(p,e)
            require(not problems,'\n'.join(str(x) for x in problems))
            recipe=load_recipe(self.runtime,p,e); check_free_space(self.store); return recipe
        def ready(recipe):
            if self.closing: return
            self.recipe=recipe; self.model_info=None; self.engine=StationProcess(self.root,self.runtime,p,e,recipe)
            self.engine_ready=False; self.checked_at=time.monotonic(); self.applied_message=tr('전체·상세 모델을 준비하고 있습니다.')
        self.task(validate,ready)
    def finish_engine(self,state='STOPPED'):
        if self.engine: self.engine.dispose()
        self.engine=None; self.engine_ready=False; self.model_info=None; self.stop_at=None
        self.applied_message=tr('모델이 해제됐습니다. 품목 설정에서 다시 준비할 수 있습니다.')
    def open_teaching(self):
        self.nav.setCurrentRow(5)

    def manual_robot_active(self):
        return getattr(getattr(self,'robot_panel',None),'worker',None) is not None

    def robot_page_changed(self,index):
        if index!=5 and self.manual_robot_active(): self.robot_panel.stop()

    def require_idle(self,*,camera=False,robot=False):
        super().require_idle(camera=camera,robot=robot)
        if camera or robot:
            require(not self.manual_robot_active(),'왼쪽 로봇 제어 탭에서 연결을 해제한 후 설정을 바꾸세요.')

    def stop_robot(self):
        if self.manual_robot_active(): self.robot_panel.stop()
        super().stop_robot()

    def connect_robot(self):
        require(not self.manual_robot_active(),'로봇 제어 탭에서 연결을 해제한 후 자동운전용 연결을 하세요.')
        require(self.robot is None and self.equipment['robot']['port'],tr('Dobot 포트를 등록하세요.'))
        self.robot=StationRobotPort(self.runtime,self.equipment); self.robot_seen_at=time.monotonic()
        self.robot.changed.connect(self.robot_changed); self.robot.failed.connect(self.robot_failed)
        self.robot.completed.connect(self.robot_completed); self.robot.finished.connect(self.robot_finished); self.robot.start()
    def begin(self):
        require(not self.manual_robot_active(),'로봇 제어 탭 연결 중에는 자동 검사를 시작할 수 없습니다.')
        require(not self.running and not self.faults.active,tr('미해결 오류를 확인한 뒤 재개하세요.'))
        require(self.engine_ready and self.engine and not self.engine.stopping,tr('품목 설정에서 모델을 준비하세요.'))
        require(self.camera and self.camera_info and self.frame and time.monotonic()-self.frame_received<.5,tr('카메라의 새 영상이 필요합니다.'))
        require(self.robot and self.robot_state['state'] in {'IDLE','RECAPTURE'} and not self.robot_state.get('busy')
            and time.monotonic()-self.robot_seen_at<1,tr('로봇 준비 확인을 완료하세요.'))
        require(not self.capture_port or not self.capture_port.busy,tr('이전 원본 저장이 끝날 때까지 기다리세요.'))
        settings=StationSettings(self.runtime); calibration=settings.calibrated()
        profile=build_scan_profile(settings,self.equipment,product_id=self.product['id'],product_version=self.product['version'],
            overview_model_digest=self.model_info['overview_model_digest'],detail_policy_digest=self.model_info['detail_policy_digest'],
            frame=self.frame,max_objects=100,expected_count=self.product['expected_count'] if self.product['count_mode']=='fixed' else None,
            auto_sort=self.sort_enabled.isChecked(),robot_connection_epoch=self.robot_state['status']['connection_epoch'])
        if self.capture_port: self.capture_port.close()
        self.capture_port=ProcessCameraPort(self.camera,self.engine,settings.value)
        self.scan=ScanRuntime(self.scan_journal,self.capture_port,self.robot,self.engine,calibration,self.equipment)
        self.running=True; self.scan.start(profile,time.monotonic()); self.rendered=None
        self.render_cycle(self.scan.sequence.data,force=True)
    def pause_inspection(self):
        was_running=self.running; self.running=False; self.start.setEnabled(False)
        if self.scan and self.scan.active:
            try: self.scan.stop()
            except Exception as exc: self.notice(str(exc))
        elif was_running and self.robot: self.robot.stop_motion()
        if self.scan and self.scan.sequence.data: self.render_cycle(self.scan.sequence.data)
        if self.engine and self.engine.stopping: self.engine_ready=False; self.stop_at=time.monotonic()
    def camera_finished(self):
        self.camera=None; self.camera_info=None
        if self.running: self.trip_fault('CAMERA_FAILURE',tr('카메라 연결이 종료되어 검사를 중단했습니다.'))
    def robot_completed(self,event):
        if self.scan:
            accepted=self.scan.complete(event,time.monotonic())
            if not accepted and self.scan.sequence.data and self.scan.sequence.data['state']=='FAILED':
                self.trip_fault('STATION_FAILURE',self.scan.sequence.data['error'])
    def trip_fault(self,code,message,*,stop_engine=True):
        # Preserve already saved overview and baseline verdicts when revoking motion.
        self.running=False
        if self.robot:
            try: self.robot.stop_motion()
            except Exception: pass
        if self.scan:
            try: self.scan.stop(str(message))
            except Exception: pass
        if self.engine and stop_engine: self.engine.stop(); self.engine_ready=False; self.stop_at=time.monotonic()
        entry,new=self.faults.raise_fault(code,message)
        if new:
            try: self.store.event('OPERATOR_FAULT',entry)
            except Exception: pass
        self.status.setText(str(message)); self.start.setEnabled(False); self.refresh_faults()
    def handle_engine(self,event):
        if event['type']=='ready' and self.engine and not self.engine.stopping:
            self.model_info=event; self.engine_ready=True; self.applied_message=tr('모델 준비됨 · 검사 시작을 기다립니다.')
        elif event['type']=='error': self.trip_fault('ENGINE_FAILURE',event['message'])
        elif event['type']=='completed' and self.scan:
            if self.capture_port: self.capture_port.finished(event['request'])
            accepted=self.scan.complete(event,time.monotonic())
            if accepted and event['request']['kind']=='inspect_detail':
                identity=event['request']['target_id']; target=next(t for t in self.scan.sequence.data['targets'] if t['id']==identity)
                try: self.enqueue_vlm(self.scan.sequence.data,target,automatic=True)
                except Exception as exc: self.notice(tr('VLM 추가 설명 요청 실패: ')+str(exc))
            elif self.scan.sequence.data and self.scan.sequence.data['state']=='FAILED': self.trip_fault('STATION_FAILURE',self.scan.sequence.data['error'])
    def enqueue_vlm(self,cycle,target,*,automatic):
        if not self.queue.enabled(): return
        receipt=target.get('result') or {}
        if not receipt.get('snapshot_path'): return
        _,inspection,_=load_snapshot(receipt['snapshot_path'],expected_digest=receipt['snapshot_digest'])
        obj=inspection['objects'][0]
        if automatic:
            from mes_vision.decision.io import run_from_dict
            if not vlm_eligible(run_from_dict(inspection).objects[0],self.product['vlm_policy']): return
        job=self.queue.enqueue(receipt['snapshot_path'],obj['object_id'],asdict(GenerationConfig()))
        with self.scan_journal.connect() as db:
            db.execute('INSERT OR REPLACE INTO station_vlm_jobs VALUES(?,?,?)',(cycle['id'],target['id'],job))
    def request_selected_vlm(self):
        require(self.queue.enabled(),tr('VLM을 ON으로 설정하세요.')); target=self.selected_target()
        require(target and target.get('result',{}).get('snapshot_path'),tr('상세 검사 결과가 저장된 물체를 선택하세요.'))
        self.enqueue_vlm(self.display_data,target,automatic=False); self.refresh_selected_vlm()
    def selected_target(self):
        return next((x for x in (self.display_data or {}).get('targets',[]) if x['id']==self.selected_id),None)
    def select_row(self):
        row=self.object_table.currentRow(); targets=(self.display_data or {}).get('targets',[])
        if 0<=row<len(targets): self.select_track(targets[row]['id'])
    def select_track(self,identity):
        self.selected_id=identity; self.canvas.selected_id=identity; self.canvas.update()
        target=self.selected_target(); detail=(target or {}).get('detail'); receipt=(target or {}).get('result') or {}
        if target:
            index=next(i for i,t in enumerate(self.display_data['targets']) if t['id']==identity)
            if self.object_table.currentRow()!=index:
                self.object_table.blockSignals(True); self.object_table.selectRow(index); self.object_table.blockSignals(False)
        key=(identity,detail.get('frame_id') if detail else None,receipt.get('snapshot_digest'),receipt.get('review_reason'),(target or {}).get('status'))
        if key!=self.detail_key:
            from mes_vision.operation.status_hooks import update_verdict
            update_verdict(self,None,identity)
            self.live_details.clear(); self.vlm_details.clear()
            self.detail_key=None; self.detail_image.canvas.pixmap=QPixmap(); self.detail_image.canvas.tracks=[]; self.detail_image.canvas.update()
            if target:
                ui_text(self.selection.setText,trf('선택 물체: {id}',id=identity.split(':')[-1]))
                if detail: self.detail_image.canvas.set_rgb(read_capture(detail).rgb)
                if receipt.get('snapshot_path'):
                    _,inspection,_=load_snapshot(receipt['snapshot_path'],expected_digest=receipt['snapshot_digest'])
                    ui_text(self.live_details.setPlainText,self.reason_text(inspection['objects'][0]))
                    obj=inspection['objects'][0]
                    from mes_vision.operation.finding_display import object_overlays
                    self.detail_image.canvas.tracks=object_overlays(obj,identity=identity)
                    self.detail_image.canvas.update()
                else:
                    reason=receipt.get('review_reason') or target.get('reason') or target['overview'].get('reason') or ''
                    ui_text(self.live_details.setPlainText,tr(STATES.get(target['status'],target['status']))+'\n'+tr(REASONS.get(reason,reason)))
            else:
                ui_text(self.selection.setText,tr('화면의 물체를 선택하세요.')); self.live_details.clear()
            self.detail_key=key
        from mes_vision.operation.status_hooks import update_verdict
        update_verdict(self,(target or {}).get('decision'),identity,self.live_details.toPlainText().split('\n')[0])
        self.refresh_selected_vlm()
    def refresh_selected_vlm(self):
        target=self.selected_target()
        if not target: self.vlm_details.clear(); return
        with self.scan_journal.connect() as db:
            row=db.execute('SELECT job_id FROM station_vlm_jobs WHERE cycle_id=? AND target_id=?',(self.display_data['id'],target['id'])).fetchone()
        if not row:
            ui_text(self.vlm_details.setPlainText,tr('VLM 설명이 없어도 기본 판정과 이유를 확인할 수 있습니다.')); return
        job=self.queue.get(row[0]); result=job['result']
        if result:
            # The queue's snapshot/object binding is rechecked before exposing text on this target.
            receipt=target.get('result') or {}
            require(job['object_id']==receipt.get('object_id') and job['snapshot_digest']==receipt.get('snapshot_digest'),'VLM target binding changed')
            ui_text(self.vlm_details.setPlainText,tr('보조 설명 · 기본 판정을 변경하지 않습니다.')+'\n'+result['analysis']['observation'])
        else:
            from mes_vision.vlm.viewer import STATE_LABELS
            ui_text(self.vlm_details.setPlainText,trf('VLM 처리 상태: {state}',state=STATE_LABELS.get(job['state'],job['state'])))
    def render_cycle(self,data,*,force=False):
        key=(data['id'],data['revision'])
        if not force and key==self.rendered: return
        self.rendered=key; self.display_data=deepcopy(data)
        if self.overview_key!=data['id']:
            self.overview_key=data['id']; self.canvas.pixmap=QPixmap(); self.canvas.tracks=[]; self.canvas.update(); self.selected_id=None; self.detail_key=None
        overview=data.get('overview')
        if overview and getattr(self,'shown_frame',None)!=overview['frame_id']:
            self.canvas.set_rgb(read_capture(overview).rgb); self.shown_frame=overview['frame_id']
        targets=data['targets']; self.canvas.tracks=[{'track_id':t['id'],'box':[t['overview']['box'][k] for k in ('x1','y1','x2','y2')],
            'status':t.get('decision') or 'INSPECTING'} for t in targets]; self.canvas.update()
        self.object_table.blockSignals(True); self.object_table.setRowCount(len(targets))
        for i,t in enumerate(targets):
            for j,value in enumerate((t['id'].split(':')[-1],tr(STATES.get(t.get('decision'), '검사 대기')),tr(STATES.get(t['status'],t['status'])))):
                self.object_table.setItem(i,j,ui_text(QTableWidgetItem,value))
        self.object_table.blockSignals(False)
        if self.selected_id not in {t['id'] for t in targets}: self.selected_id=targets[0]['id'] if targets else None
        self.select_track(self.selected_id)
        done=sum(t.get('decision') in {'OK','NG','REVIEW'} for t in targets)
        ui_text(self.counts.setText,trf('검사 완료 {done} / {total} · 정상 {ok} · 불량 {ng} · 보류 {review}',done=done,total=len(targets),
            ok=sum(t.get('decision')=='OK' for t in targets),ng=sum(t.get('decision')=='NG' for t in targets),review=sum(t.get('decision')=='REVIEW' for t in targets)))
        self.applied_message=tr(STATES.get(data['state'],data['state']))
        if data['state'] in TERMINAL: self.running=False; self.refresh_history()
    def tick(self):
        now=time.monotonic()
        self.sort_enabled.setEnabled(not self.running)
        if self.camera:
            frame,received=self.camera.get_latest()
            if frame is not None and (self.frame is None or frame.frame_id!=self.frame.frame_id):
                self.frame=frame; self.frame_received=received
                # Acquisition supplies inspection pixels; injected/raw frames remain labelled raw.
                self.live_view.canvas.set_rgb(frame.rgb)
                ui_text(self.live_view.title_label.setText,tr('왜곡보정 실시간 영상') if frame.acquisition_identity else tr('실시간 영상 · 원본'))
                if self.preview and self.preview.isVisible(): self.preview_panel.canvas.set_rgb(frame.rgb)
        if self.engine:
            for event in self.engine.poll(): self.handle_engine(event)
            if self.engine.stopping:
                if not self.engine.process.is_alive() or now-self.engine.stop_at>3: self.finish_engine()
            else:
                error=self.engine.watchdog_error(now)
                if error: self.trip_fault('ENGINE_FAILURE',error)
        if self.capture_port:
            if self.scan: self.scan.tick(now)
            else: self.capture_port.tick(now)
        if self.scan and self.scan.sequence.data:
            data=self.scan.sequence.data
            revision=(data['id'],data['revision'])
            if revision!=self.observed_revision:
                self.observed_revision=revision; self.render_cycle(data)
            if data['state']=='FAILED' and not any(x['code']=='STATION_FAILURE' for x in self.faults.active): self.trip_fault('STATION_FAILURE',data['error'])
        if self.running:
            p=self.scan.sequence.data['profile']
            if not self.camera_info or not self.frame or now-self.frame_received>2 or self.frame.session_id!=p['camera_session']:
                self.trip_fault('CAMERA_STALE',tr('카메라 영상이 변경되어 새 전체사진이 필요합니다.'))
            elif not self.robot or now-self.robot_seen_at>5 or self.robot_state.get('status',{}).get('connection_epoch')!=p['robot_connection_epoch']:
                self.trip_fault('ROBOT_STATUS_TIMEOUT',tr('로봇 연결이 변경되어 새 전체사진이 필요합니다.'))
        if self.closing:
            if self.capture_port and not self.capture_port.busy: self.capture_port.close(); self.capture_port=None
            if not self.capture_port: super().tick()
            return
        if now>=self.vlm_probe_at and not self.vlm_read.busy:
            self.vlm_probe_at=now+1; self.vlm_read.start(self.queue.enabled)
        if now>=self.ui_probe_at:
            self.ui_probe_at=now+1
            try: self.refresh_selected_vlm()
            except Exception as exc: self.vlm_details.setPlainText(str(exc))
        self.start.setEnabled(not self.running and self.engine_ready and not self.faults.active and bool(self.camera_info)
            and bool(self.robot) and self.robot_state['state'] in {'IDLE','RECAPTURE'} and not (self.capture_port and self.capture_port.busy))
        self.pause.setEnabled(self.running); self.unload.setEnabled(self.engine is not None)
        ui_text(self.live_state.setText,self.applied_message)
        from mes_vision.operation.status_hooks import update_context
        update_context(self)
    def closeEvent(self,event):
        if self.manual_robot_active():
            self.robot_panel.disconnect_robot(); event.ignore(); QTimer.singleShot(100,self.close); return
        if getattr(self,'robot_panel',None) is not None: self.robot_panel.timer.stop()
        if self.capture_port and self.capture_port.busy and self.closing:
            event.ignore(); return
        super().closeEvent(event)

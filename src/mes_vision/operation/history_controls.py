from mes_vision.qt_i18n import ui_text, bound_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
"""Search controls shared by the operator history page."""
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import time
from PySide6.QtCore import QDateTime,QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QHBoxLayout,QGridLayout,QLineEdit,QComboBox,QLabel,QCheckBox,
    QDateTimeEdit,QFileDialog,QInputDialog,QTableWidgetItem,QWidget,QVBoxLayout,QScrollArea)
from mes_vision.training.data import require
from mes_vision.vlm.viewer import WorkerThread
from .storage import StorageService,Search
from .widgets import button,Task,STATES
from .responsive import FlowLayout
from mes_vision.ui_refresh import update_table


def combo(items):
    widget=QComboBox()
    for label,value in items: ui_text(widget.addItem, label,value)
    return widget


class HistoryControls:
    def make_history_controls(self,out):
        self.history_page=0; self.history_rows=[]; self.history_busy=False; self.history_pending=False; self.history_serial=0
        self.history_search=Search(as_of=time.time()); self.history_result=None
        row=QHBoxLayout(); self.history_query=QLineEdit(); ui_text(self.history_query.setPlaceholderText, tr('물체·검사·회차 ID 또는 품목 이름'))
        self.history_filter=combo([(tr('전체 판정'),""),(tr('정상'),"OK"),(tr('불량'),"NG"),(tr('보류'),"REVIEW"),(tr('불량·보류'),"ATTENTION")])
        row.addWidget(self.history_query,1); row.addWidget(self.history_filter); row.addWidget(button(tr('조회'),lambda:self.guarded(self.search_history)))
        row.addWidget(button(tr('미검토 이상'),lambda:self.guarded(self.find_unreviewed)))
        row.addWidget(button(tr('조건 초기화'),lambda:self.guarded(self.reset_history_filters))); out.addLayout(row)
        self.history_advanced_toggle=ui_text(QCheckBox, tr('상세 검색 조건')); out.addWidget(self.history_advanced_toggle)
        advanced_content=QWidget(); advanced_layout=QVBoxLayout(advanced_content)
        self.history_advanced=QScrollArea(); self.history_advanced.setWidgetResizable(True); self.history_advanced.setWidget(advanced_content)
        self.history_advanced.setMaximumHeight(220); self.history_advanced.hide(); out.addWidget(self.history_advanced)
        self.history_advanced_toggle.toggled.connect(self.history_advanced.setVisible)
        grid=QGridLayout(); self.history_product=QLineEdit(); ui_text(self.history_product.setPlaceholderText, tr('품목 이름 / ID'))
        self.history_defect=combo([(tr('전체 불량'),""),(tr('누락'),"NG01"),(tr('돌출·버'),"NG02"),(tr('균열'),"NG03"),(tr('변형'),"NG04"),(tr('구멍 불량'),"NG05"),(tr('표면 불량'),"NG06"),(tr('미등록 이상'),"NG_UNKNOWN")])
        self.history_review=combo([(tr('전체 검토'),""),(tr('미검토'),"unreviewed"),(tr('검토 완료'),"reviewed")])
        self.history_vlm=combo([(tr('전체 VLM'),"")]+[(label,value) for label,value in ((tr('미요청'),"NONE"),(tr('대기'),"PENDING"),(tr('분석 중'),"RUNNING"),(tr('분석 완료'),"COMPLETED"),(tr('실패'),"FAILED"),(tr('취소'),"CANCELLED"),(tr('후순위 대기'),"DEFERRED"))])
        self.history_robot=combo([(tr('전체 로봇'),""),(tr('미요청'),"NONE"),(tr('동작 요청'),"REQUESTED"),(tr('집기·놓기 확인'),"RECAPTURE"),(tr('오류'),"FAULT"),(tr('중단'),"STOPPED"),(tr('복구 필요'),"RECOVERY")])
        self.history_storage=combo([(tr('전체 보존'),""),(tr('운영 원본'),"active"),(tr('외부 보관'),"archived"),(tr('별도 보존'),"held")])
        for i,(label,widget) in enumerate(((tr('품목'),self.history_product),(tr('불량 유형'),self.history_defect),(tr('검토'),self.history_review),('VLM',self.history_vlm),(tr('로봇'),self.history_robot),(tr('보존'),self.history_storage))):
            grid.addWidget(ui_text(QLabel, label),i//2,(i%2)*2); grid.addWidget(widget,i//2,(i%2)*2+1)
        advanced_layout.addLayout(grid)
        self.history_period=ui_text(QCheckBox, tr('기간 지정')); self.history_since=QDateTimeEdit(QDateTime.currentDateTime().addDays(-30)); self.history_until=QDateTimeEdit(QDateTime.currentDateTime().addDays(1))
        for widget in (self.history_since,self.history_until): widget.setCalendarPopup(True); widget.setDisplayFormat("yyyy-MM-dd HH:mm"); widget.setEnabled(False)
        self.history_period.toggled.connect(self.history_since.setEnabled); self.history_period.toggled.connect(self.history_until.setEnabled)
        period=FlowLayout(); period.addWidget(self.history_period); period.addWidget(self.history_since); period.addWidget(ui_text(QLabel, tr('이상 ~ 미만'))); period.addWidget(self.history_until); advanced_layout.addLayout(period)
        self.history_applied=ui_text(QLabel); self.history_applied.setWordWrap(True); out.addWidget(self.history_applied)
        self.history_dirty=ui_text(QLabel); self.history_dirty.setWordWrap(True); out.addWidget(self.history_dirty)
        for field in (self.history_query,self.history_product):
            field.textChanged.connect(self.history_draft_changed); field.returnPressed.connect(lambda:self.guarded(self.search_history))
        for field in (self.history_filter,self.history_defect,self.history_review,self.history_vlm,self.history_robot,self.history_storage): field.currentIndexChanged.connect(self.history_draft_changed)
        self.history_period.toggled.connect(self.history_draft_changed)
        self.history_since.dateTimeChanged.connect(self.history_draft_changed); self.history_until.dateTimeChanged.connect(self.history_draft_changed)
        self.history_summary=ui_text(QLabel); self.history_summary.setWordWrap(True); out.addWidget(self.history_summary)
        row=FlowLayout(); self.history_prev=button(tr('이전'),lambda:self.history_move(-1)); self.history_next=button(tr('다음'),lambda:self.history_move(1)); self.history_paging=ui_text(QLabel)
        row.addWidget(self.history_prev); row.addWidget(self.history_paging); row.addWidget(self.history_next); row.addStretch()
        self.export_images=ui_text(QCheckBox, tr('원본 포함')); self.export_images.setChecked(True); row.addWidget(self.export_images)
        row.addWidget(button(tr('검색 전체 내보내기'),lambda:self.guarded(self.export_history)))
        self.storage_button=button(tr('저장 · 백업 관리'),lambda:self.guarded(self.open_storage)); row.addWidget(self.storage_button); out.addLayout(row)

    def history_inputs(self):
        return dict(text=self.history_query.text().strip(),product=self.history_product.text().strip(),decision=self.history_filter.currentData(),
            defect=self.history_defect.currentData(),review=self.history_review.currentData(),vlm=self.history_vlm.currentData(),robot=self.history_robot.currentData(),archived=self.history_storage.currentData(),
            since=self.history_since.dateTime().toSecsSinceEpoch() if self.history_period.isChecked() else 0.,until=self.history_until.dateTime().toSecsSinceEpoch() if self.history_period.isChecked() else 32503680000.)

    def history_draft_changed(self,*_):
        dirty=any(getattr(self.history_search,key)!=value for key,value in self.history_inputs().items())
        ui_text(self.history_dirty.setText, tr('검색 조건이 변경되었습니다. 조회를 누르면 적용됩니다.') if dirty else '')
        self.history_dirty.setVisible(dirty)

    def clear_history_selection(self):
        self.record=None; self.history_table.clearSelection(); self.history_table.setCurrentCell(-1,-1)
        self.saved_canvas.pixmap=QPixmap(); self.saved_canvas.tracks=[]; self.saved_canvas.selected_id=None; self.saved_canvas.fit()
        ui_text(self.history_details.setPlainText, tr('검사 이력을 선택하면 저장한 원본이 표시됩니다.'))

    def search_history(self):
        candidate=Search(**self.history_inputs(),as_of=time.time())
        self.history_search=candidate; self.history_page=0; self.history_serial+=1
        self.clear_history_selection(); self.history_draft_changed(); self.refresh_history()

    def reset_history_filters(self):
        ui_text(self.history_query.clear); ui_text(self.history_product.clear); self.history_period.setChecked(False)
        for field in (self.history_filter,self.history_defect,self.history_review,self.history_vlm,self.history_robot,self.history_storage): field.setCurrentIndex(0)
        self.search_history()

    def show_applied_history_filters(self,filters):
        parts=[]
        if filters['text']: parts.append(tr('검색어')+': '+filters['text'])
        if filters['product']: parts.append(tr('품목')+': '+filters['product'])
        for key,field in (('decision',self.history_filter),('defect',self.history_defect),('review',self.history_review),('vlm',self.history_vlm),('robot',self.history_robot),('archived',self.history_storage)):
            if filters[key]: parts.append(bound_text(field,'setItemText',field.findData(filters[key])))
        if filters['since'] or filters['until']!=32503680000.:
            parts.append(datetime.fromtimestamp(filters['since']).strftime('%Y-%m-%d %H:%M')+' ~ '+datetime.fromtimestamp(filters['until']).strftime('%Y-%m-%d %H:%M'))
        full=text_join(' · ', parts) or tr('전체 조건')
        ui_text(self.history_applied.setText, tr('적용된 검색: ')+(full if len(full)<=180 else full[:180]+'…')); ui_text(self.history_applied.setToolTip, full)
        self.history_draft_changed()

    def find_unreviewed(self):
        # Both unknown NG and uncertain objects can need review; keep known NG visible too.
        for w in (self.history_filter,self.history_defect,self.history_vlm,self.history_robot,self.history_storage): w.setCurrentIndex(0)
        self.history_filter.setCurrentIndex(4)
        self.history_review.setCurrentIndex(1); ui_text(self.history_query.clear); ui_text(self.history_product.clear); self.history_period.setChecked(False)
        self.search_history()

    def history_move(self,delta):
        self.history_page=max(0,self.history_page+delta); self.history_serial+=1; self.clear_history_selection(); self.guarded(self.refresh_history)

    def refresh_history(self):
        if getattr(self,'closing',False): return
        self.last_history=time.monotonic()
        if self.history_busy: self.history_pending=True; return
        service=StorageService(self.store); filters=self.history_search; page=self.history_page; serial=self.history_serial
        if page==0: filters=replace(filters,as_of=time.time()); self.history_search=filters
        if not hasattr(self,"timer"):
            self.render_history(service.search(filters,page=page)); return
        self.history_busy=True; self.history_pending=False
        if self.history_result is None or getattr(self,"history_rendered_serial",None)!=serial:
            ui_text(self.history_summary.setText, tr('검사 이력을 조회하고 있습니다…'))
        job=Task(lambda:service.search(filters,page=page)); self.tasks.append(job)
        job.succeeded.connect(lambda result:self.render_history(result) if serial==self.history_serial else None)
        job.failed.connect(lambda message:ui_text(self.history_summary.setText, tr('이력 조회 실패: ')+message) if serial==self.history_serial else None)
        def finished():
            self.history_busy=False
            if job in self.tasks: self.tasks.remove(job)
            if self.history_pending and not self.closing: QTimer.singleShot(0,self.refresh_history)
        job.finished.connect(finished); job.start()

    def render_history(self,result):
        self.history_rendered_serial=self.history_serial
        self.history_result=result; self.history_rows=result["rows"]; summary=result["summary"]
        self.show_applied_history_filters(result['filters'])
        ui_text(self.history_summary.setText, trf('검사 {v0:,}건 · 물체 ID {v1:,}개 · 재검사 {v2:,}건  |  정상 {v3:,} / 불량 {v4:,} / 보류 {v5:,}', v0=summary['inspections'], v1=summary['objects'], v2=summary['reinspections'], v3=summary['ok'], v4=summary['ng'], v5=summary['review'])
            + (trf(' · 색인 갱신 필요 {v0:,}건 (불량 종류 검색에서 누락될 수 있음)', v0=result['index_missing']) if result['index_missing'] else ""))
        pages=max(1,(summary["inspections"]+result["page_size"]-1)//result["page_size"])
        ui_text(self.history_paging.setText, trf('{v0} / {v1} 페이지 · 100건씩', v0=result['page'] + 1, v1=pages))
        
        if not self.history_rows: ui_text(self.history_summary.setText, tr('조건에 맞는 검사 이력이 없습니다.'))
        self.history_prev.setEnabled(result["page"]>0); self.history_next.setEnabled(result["page"]+1<pages)
        selected=self.record[0]["object_id"] if self.record else None
        display=[]
        for i,r in enumerate(self.history_rows):
            vals=[datetime.fromtimestamp(r["created"]).strftime("%Y-%m-%d %H:%M:%S"),r["product_name"],r["track_id"].split(":")[-1],tr(STATES.get(r["decision"],r["decision"])),str(r["revision"]),tr('재검사') if r["prior_object_id"] else tr('첫 검사'),tr('완료') if r["review"] else tr('미검토'),tr('별도 보존') if r["hold_note"] else tr('보관') if r["archived"] else tr('운영')]
            display.append((r['object_id'],tuple((value,r['track_id']+'\n'+r['object_id'] if column==2 else value) for column,value in enumerate(vals))))
        update_table(self.history_table,display,selected_key=selected)

    def hold_record(self):
        require(self.record,tr('검사 이력을 선택하세요.'))
        row=self.record[0]; service=StorageService(self.store)
        with self.store.connect() as db: existing=db.execute("SELECT hold_note FROM capture_storage WHERE run_id=?",(row["run_id"],)).fetchone()
        note,ok=QInputDialog.getText(self,tr('별도 보존'),tr('보존 사유 (비워서 저장하면 해제, 같은 원본의 모든 물체에 적용)'),text=(existing[0] or "") if existing else "")
        if ok: service.hold(row["run_id"],note.strip() or None); self.refresh_history()

    def export_history(self):
        require(not self.history_busy and self.history_result and self.history_rendered_serial==self.history_serial,tr('조회가 완료된 뒤 내보내세요.'))
        filters=Search(**self.history_result["filters"]); images=self.export_images.isChecked(); service=StorageService(self.store)
        destination=QFileDialog.getSaveFileName(self,tr('새 내보내기 폴더 이름'),str(self.runtime.parent/("MES-report-"+datetime.now().strftime("%Y%m%d-%H%M%S"))),tr('폴더 이름 (*)'),options=QFileDialog.DontConfirmOverwrite)[0]
        if destination:
            self.notice(tr('검색 조건 전체를 CSV·JSON')+(tr('·원본 이미지') if images else "")+tr('로 내보내고 있습니다.'))
            self.task(lambda:service.export(destination,filters,images=images),lambda result:self.notice(trf('{v0:,}건 내보내기 완료: {v1}', v0=result['inspections'], v1=result['path'])))

    def open_storage(self):
        self.require_idle(camera=True,robot=True); require(not self.queue.enabled(),tr('저장 관리 전에 VLM을 OFF로 설정하세요.'))
        require(not any(getattr(w,"busy",False) for w in self.children),tr('진행 중인 학습·설정 작업이 끝난 뒤 저장 관리를 여세요.'))
        require(not getattr(self,"maintenance_waiting",False),tr('VLM 작업 종료를 기다리고 있습니다.'))
        if self.worker and self.worker.isRunning():
            self.maintenance_waiting=True; self.centralWidget().setEnabled(False); self.notice(tr('저장 관리를 위해 VLM 작업을 종료하고 있습니다.'))
            self.worker.finished.connect(self.storage_worker_finished); self.worker.stop_event.set()
        else: self.show_storage_dialog()

    def storage_worker_finished(self):
        self.worker.finished.disconnect(self.storage_worker_finished); self.maintenance_waiting=False; self.centralWidget().setEnabled(True)
        if not self.closing: self.guarded(self.show_storage_dialog)

    def show_storage_dialog(self):
        from .storage_dialog import StorageDialog
        # Let an in-flight storage write probe release its maintenance lease first.
        if self.readiness_read.busy:
            if not getattr(self,'storage_readiness_wait',False):
                self.storage_readiness_wait=True
                def resume():
                    if self.closing: self.storage_readiness_wait=False; return
                    if self.readiness_read.busy: QTimer.singleShot(50,resume); return
                    self.storage_readiness_wait=False; self.guarded(self.show_storage_dialog)
                QTimer.singleShot(50,resume)
            return
        self.require_idle(camera=True,robot=True)
        restart=self.worker is not None
        dialog=StorageDialog(self.store,self.root,self); self.children.append(dialog); self.timer.stop(); self.analysis.timer.stop()
        try: dialog.exec()
        finally:
            self.children.remove(dialog); dialog.deleteLater()
            if not self.closing:
                self.readiness_report=None; self.readiness_due=0.
                if restart:
                    self.worker=WorkerThread(self.queue,self.root,allow_synthetic=False,backend="qwen")
                    self.worker.failed.connect(self.vlm_worker_failed); self.worker.start()
                self.timer.start(40); self.analysis.timer.start(500); self.refresh_history()
                if self.record: self.guarded(lambda:self.show_record(self.record[0]["object_id"]))

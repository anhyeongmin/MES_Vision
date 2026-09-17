from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
from pathlib import Path
import threading
import time

from PySide6.QtCore import Qt, QTimer, QThread, Signal
from PySide6.QtGui import QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QAbstractItemView, QSplitter, QPlainTextEdit, QMessageBox)

from .snapshots import load_snapshot, object_record
from .worker import AnalysisWorker
from mes_vision.ui_refresh import BackgroundRead, update_table
from .progress import overall, job_label, selected_text

STATE_LABELS = {"PENDING": tr('분석 대기'), "RUNNING": tr('분석 중'), "COMPLETED": tr('분석 완료'), "FAILED": tr('분석 실패'),
                "CANCELLED": tr('분석 취소'), "DEFERRED": tr('대기열 가득 참'), "SKIPPED_DISABLED": tr('OFF로 미요청')}
CHECK_LABELS = {"PASS": tr('통과'), "FAIL": tr('불합격 근거'), "UNCERTAIN": tr('보류'), "ERROR": tr('검사 오류'), "NOT_RUN": tr('미실행')}
DECISIONS = {None: tr('판정 연결 전'), "OK": tr('정상'), "NG": tr('불량'), "REVIEW": tr('보류')}


class WorkerThread(QThread):
    failed = Signal(str)
    def __init__(self, queue, project_root, **options):
        super().__init__()
        self.stop_event = threading.Event()
        self.worker = AnalysisWorker(queue, project_root, stop_event=self.stop_event, **options)
    def run(self):
        try: self.worker.run()
        except Exception as exc: self.failed.emit(str(exc))


class AnalysisViewer(QMainWindow):
    """Stored inspection evidence and optional background analysis."""
    enabled_changed = Signal(bool)
    def __init__(self, queue, project_root, *, run_worker=True, allow_synthetic=False, backend="qwen"):
        super().__init__()
        self.queue, self.project_root = queue, Path(project_root)
        ui_text(self.setWindowTitle, tr('MES Vision · 검사 근거와 VLM 추가 분석'))
        self.resize(1240, 820)
        self.rows = []
        self.worker_thread = None
        self.closing = False
        self.selected_key = None
        self.render_version = None
        self.refresh_read = BackgroundRead(self)
        self.refresh_read.succeeded.connect(self.read_refreshed)
        self.refresh_read.failed.connect(self.refresh_failed)
        self.refresh_error = None
        self.runtime_state = None
        self.queue_enabled = False
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        header = QHBoxLayout()
        self.toggle = ui_text(QPushButton)
        self.toggle.setCheckable(True)
        self.toggle.setMinimumWidth(160)
        self.toggle.clicked.connect(self.toggle_vlm)
        header.addWidget(self.toggle)
        self.summary = ui_text(QLabel)
        header.addWidget(self.summary, 1)
        self.retry = ui_text(QPushButton, tr('선택 분석 다시 요청'))
        self.retry.clicked.connect(self.retry_selected)
        header.addWidget(self.retry)
        self.cancel = ui_text(QPushButton, tr('선택 분석 취소'))
        self.cancel.clicked.connect(self.cancel_selected)
        header.addWidget(self.cancel)
        outer.addLayout(header)
        self.progress = ui_text(QLabel)
        self.progress.setObjectName("vlmProgress")
        self.progress.setWordWrap(True)
        self.progress.setStyleSheet("padding:10px; font-weight:600;")
        outer.addWidget(self.progress)
        outer.addWidget(ui_text(QLabel, tr('기본 검사 근거는 즉시 확인할 수 있습니다. VLM은 저장한 자료에 대한 추가 의견이며 기본 판정을 변경하지 않습니다.')))
        self.table = QTableWidget(0, 4)
        ui_text(self.table.setHorizontalHeaderLabels, [tr('물체'), tr('기본 판정'), tr('VLM 상태'), tr('추가 관찰')])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setColumnWidth(0, 180); self.table.setColumnWidth(1, 130); self.table.setColumnWidth(2, 160)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.itemSelectionChanged.connect(self.select_row)
        self.table.setMaximumHeight(240)
        outer.addWidget(self.table)
        panes = QSplitter()
        outer.addWidget(panes, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(ui_text(QLabel, tr('검사한 물체 · 기본 검사 근거 위치')))
        self.picture = ui_text(QLabel, tr('물체를 선택하세요'))
        self.picture.setAlignment(Qt.AlignCenter)
        self.picture.setMinimumSize(300, 230)
        left_layout.addWidget(self.picture, 1)
        self.evidence_button=ui_text(QPushButton, tr('이미지 확대 · 정상 비교'))
        self.evidence_button.setEnabled(False); self.evidence_button.clicked.connect(self.open_evidence)
        left_layout.addWidget(self.evidence_button)
        self.reference_state = ui_text(QLabel)
        left_layout.addWidget(self.reference_state)
        panes.addWidget(left)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(ui_text(QLabel, tr('기본 검사 근거')))
        self.reasons = QPlainTextEdit()
        self.reasons.setReadOnly(True)
        right_layout.addWidget(self.reasons, 1)
        right_layout.addWidget(ui_text(QLabel, tr('VLM 추가 분석')))
        self.selected_progress = ui_text(QLabel)
        self.selected_progress.setObjectName("vlmSelectedProgress")
        self.selected_progress.setWordWrap(True)
        right_layout.addWidget(self.selected_progress)
        self.analysis = QPlainTextEdit()
        self.analysis.setReadOnly(True)
        right_layout.addWidget(self.analysis, 1)
        panes.addWidget(right)
        panes.setSizes([430, 770])
        self.status = ui_text(QLabel, tr('저장된 분석 요청을 표시합니다.'))
        outer.addWidget(self.status)
        self.setStyleSheet("QPushButton {padding: 8px;}")
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_refresh)
        self.timer.start(500)
        self.refresh()
        if run_worker:
            self.worker_thread = WorkerThread(queue, project_root, allow_synthetic=allow_synthetic, backend=backend)
            self.worker_thread.failed.connect(lambda error: ui_text(self.status.setText, tr('분석 작업자: ') + error))
            self.worker_thread.finished.connect(self.worker_finished)
            self.worker_thread.start()

    def error(self, message):
        QMessageBox.warning(self, tr('추가 분석'), str(message))

    def toggle_vlm(self, checked):
        try:
            self.queue.set_enabled(bool(checked))
            self.enabled_changed.emit(bool(checked))
            ui_text(self.status.setText, tr('VLM ON · 새 요청을 분석합니다. 취소된 요청은 자동 재개하지 않습니다.') if checked
                                else tr('VLM OFF · 대기 요청 취소 및 진행 중 분석 중단 요청. 기본 검사와 기존 이력은 유지됩니다.'))
            self.refresh()
        except Exception as exc: self.error(exc)

    def retry_selected(self):
        if self.selected_key is None: return
        try: self.queue.retry(self.selected_key); self.refresh()
        except Exception as exc: self.error(exc)

    def cancel_selected(self):
        if self.selected_key is None: return
        try: self.queue.cancel(self.selected_key); self.refresh()
        except Exception as exc: self.error(exc)

    def read_refresh(self):
        return self.queue.summary(), self.queue.list(), self.queue.runtime()

    def poll_refresh(self):
        if self.isVisible() and not self.closing:
            self.refresh_read.start(self.read_refresh)

    def showEvent(self, event):
        super().showEvent(event)
        # A page that was hidden must not display an old queue until the next timer.
        self.poll_refresh()

    def hideEvent(self, event):
        self.refresh_read.invalidate()
        super().hideEvent(event)

    def stop_refresh(self):
        self.timer.stop()
        self.refresh_read.invalidate()

    def refresh(self):
        # Explicit navigation/actions retain immediate results; timers use background reads.
        self.refresh_read.invalidate()
        try: self.apply_refresh(self.read_refresh(), explicit=True)
        except Exception as exc: self.refresh_failed(str(exc))

    def refresh_failed(self, message):
        if self.closing: return
        self.refresh_error = tr('이력 읽기 오류: ') + message
        ui_text(self.status.setText, self.refresh_error)
        ui_text(self.progress.setText, tr('상태 조회 실패 · 표시된 결과는 마지막 조회 기준입니다.'))
        ui_text(self.selected_progress.setText, tr('상태 조회 실패 · 표시된 결과는 마지막 조회 기준입니다.'))

    def read_refreshed(self, data):
        try: self.apply_refresh(data)
        except Exception as exc: self.refresh_failed(str(exc))

    def apply_refresh(self, data, *, explicit=False):
        if self.closing or (not explicit and not self.isVisible()): return
        summary, rows, self.runtime_state = data
        self.queue_enabled = summary["enabled"]
        now = time.time()
        ui_text(self.progress.setText, overall(summary, self.runtime_state, now))
        self.enabled_changed.emit(summary["enabled"])
        if self.refresh_error is not None:
            if self.status.text() == self.refresh_error:
                ui_text(self.status.setText, tr('저장된 분석 요청을 표시합니다.'))
            self.refresh_error = None
        enabled = summary["enabled"]
        text = "VLM ON" if enabled else "VLM OFF"
        self.toggle.setChecked(enabled)
        if self.toggle.text() != text:
            ui_text(self.toggle.setText, text)
            self.toggle.setProperty("vlmEnabled",enabled)
        counts = summary["counts"]
        text = trf('대기 {v0} · 처리 중 {v1} · 완료 {v2} · 실패 {v3} · 적체 {v4}', v0=counts['PENDING'], v1=counts['RUNNING'], v2=counts['COMPLETED'], v3=counts['FAILED'], v4=counts['DEFERRED'])
        if self.summary.text() != text: ui_text(self.summary.setText, text)
        self.rows = rows
        display = []
        for row in rows:
            result = row["result"] if row["state"] == "COMPLETED" else None
            observation = result["analysis"]["observation"] if result else ""
            state = job_label(row, self.runtime_state, enabled, now)
            values = [row["object_id"].split(":")[-1], DECISIONS.get(row["payload"]["base_decision"], tr('미확인')), state, observation]
            display.append((row["id"], tuple((value, row["object_id"] if column == 0 else value) for column, value in enumerate(values))))
        update_table(self.table, display, selected_key=self.selected_key, select_first=True)
        if rows:
            self.select_row()
        else:
            self.selected_key = self.render_version = None
            self.evidence_button.setEnabled(False)
            ui_text(self.picture.clear); ui_text(self.reference_state.clear); ui_text(self.reasons.clear); ui_text(self.analysis.clear)
            ui_text(self.selected_progress.clear)

    def select_row(self):
        index = self.table.currentRow()
        if index < 0 or index >= len(self.rows): return
        row = self.rows[index]
        self.selected_key = row["id"]
        ui_text(self.selected_progress.setText, selected_text(row, self.runtime_state, self.queue_enabled, time.time()))
        if row["state"] != "COMPLETED":
            text = (job_label(row, self.runtime_state, self.queue_enabled, time.time())
                    + ("\n" + row["error"] if row["error"] else "")
                    + tr('\n기본 검사 근거는 위에서 확인할 수 있습니다.'))
            if self.analysis.toPlainText() != text: ui_text(self.analysis.setPlainText, text)
        version = (row["id"], row["updated"], row["cancel_requested"], row["snapshot_path"], row["snapshot_digest"])
        if version == self.render_version: return
        self.render_version = version
        self.evidence_button.setEnabled(False)
        try:
            manifest, inspection, _ = load_snapshot(row["snapshot_path"], expected_digest=row["snapshot_digest"])
            record = object_record(manifest, inspection, row["object_id"])
            source = next(o for o in manifest["objects"] if o["object_id"] == row["object_id"])
            pixmap = QPixmap(str(Path(row["snapshot_path"]) / source["file"]))
            painter = QPainter(pixmap)
            painter.setPen(QPen(QColor("#e34c36"), 2))
            for check in record["checks"]:
                if check["check_id"] == "vlm": continue
                for finding in check["findings"]:
                    box = finding["crop_box"]
                    if box:
                        painter.drawRect(int(box["x1"]), int(box["y1"]), int(box["x2"]-box["x1"]), int(box["y2"]-box["y1"]))
            painter.end()
            self.picture.setPixmap(pixmap.scaled(400, 380, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.evidence_button.setEnabled(True)
            ui_text(self.reference_state.setText, (tr('합성 시험 자료 · ') if manifest["kind"] == "synthetic" else tr('실물 자료 · '))
                                        + (tr('정상 참조 있음') if manifest["reference"] else tr('정상 참조 미설정')))
            from mes_vision.decision.display import decision_lines
            lines = decision_lines(record)
            for reason in row["payload"]["basic_reasons"]:
                lines.append(tr(reason['name'])+': '+CHECK_LABELS[reason['status']])
                for finding in reason["findings"]:
                    lines.append('  '+(finding['defect_code'] or tr('미분류 관찰'))+' · '+finding['label'])
                if reason["raw_score"] is not None: lines.append(trf('  검사 점수: {v0:.4g}', v0=reason['raw_score']))
                if reason["criteria_version"]: lines.append(trf('  적용 기준: {v0}', v0=reason['criteria_version']))
            ui_text(self.reasons.setPlainText, text_join('\n', lines))
            if row["state"] == "COMPLETED":
                result = row["result"]
                note = tr('추가 검토 의견 있음') if result["analysis"]["needs_review"] else tr('추가 검토 의견 없음 — 합격 판정은 아님')
                limitations = {"NORMAL_REFERENCE_UNCONFIGURED": tr('정상 참조가 없습니다.'), "INSPECTION_CRITERIA_UNCONFIGURED": tr('검사 기준 설명이 미설정입니다.'),
                               "SYNTHETIC_NOT_PRODUCT_VALIDATION": tr('합성 시험 결과입니다.')}
                ui_text(self.analysis.setPlainText, result["analysis"]["observation"] + "\n\n" + note + "\n" + text_join('\n', (limitations.get(v, v) for v in result["limitations"])))
        except Exception as exc:
            ui_text(self.picture.clear)
            ui_text(self.reasons.setPlainText, tr('저장 근거를 확인할 수 없습니다: ') + str(exc))
            ui_text(self.analysis.clear)

    def open_evidence(self):
        row=next((row for row in self.rows if row['id']==self.selected_key),None)
        if row is None: return
        try:
            from mes_vision.operation.evidence_dialog import show_evidence
            show_evidence(self,row['snapshot_path'],row['object_id'],row['snapshot_digest'])
        except Exception as exc: self.error(str(exc))

    def worker_finished(self):
        if self.closing: self.close()

    def closeEvent(self, event):
        from mes_vision.operation.evidence_dialog import close_evidence
        close_evidence(self)
        self.stop_refresh()
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.closing = True
            self.worker_thread.stop_event.set()
            ui_text(self.status.setText, tr('분석 작업을 정리하고 종료하고 있습니다…'))
            event.ignore()
        elif self.refresh_read.busy:
            self.closing = True
            event.ignore()
            QTimer.singleShot(50, self.close)
        else:
            event.accept()

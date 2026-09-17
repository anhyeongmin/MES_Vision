from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
import sys
import time
from uuid import uuid4

from PySide6.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, Signal
from PySide6.QtGui import QColor, QDesktopServices, QFont, QPainter, QPen, QPixmap
from PySide6.QtCore import QUrl
from PySide6.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QMainWindow, QMessageBox, QPlainTextEdit, QPushButton,
    QSpinBox, QSplitter, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from mes_vision.decision.display import decision_lines
from mes_vision.decision.io import run_from_dict
from mes_vision.training.data import read_json, write_json
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.snapshots import basic_reasons
from mes_vision.vlm.viewer import AnalysisViewer, WorkerThread
from .service import AppStore, load_recipe


COLORS = {"OK": "#16876b", "NG": "#d44042", "REVIEW": "#b77812", None: "#b77812"}
VERDICTS = {"OK": "OK · 정상", "NG": "NG · 불량", "REVIEW": "보류", None: "보류"}
DEFECTS = {"NG01": "누락", "NG02": "돌출·버", "NG03": "균열", "NG04": "변형", "NG05": "구멍 결함", "NG06": "표면 결함", "NG_UNKNOWN": "미등록 이상"}
CHECK_STATES = {"PASS": "통과", "FAIL": "불합격", "UNCERTAIN": "확인 필요", "ERROR": "오류", "NOT_RUN": "미실행"}
MESSAGES = {"CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED": "불량 후보를 판정 규칙에서 재평가했습니다.",
    "SYNTHETIC_TEST_FAILURE": "오류 처리 확인을 위해 만든 모의 실패입니다.",
    "ANOMALY_SCORE_IN_REVIEW_BAND": "정상·불량 사이의 점수 구간으로 확인이 필요합니다.",
    "IMAGE_QUALITY_NOT_CONFIGURED": "촬영 품질 기준이 미설정입니다.", "ROI_NOT_CONFIGURED": "검사 영역이 미설정입니다.",
    "OBJECT_MODEL_NOT_PRODUCT_TRAINED": "현재 검출 모델은 제품 전용으로 학습되지 않았습니다.",
    "ZERO_DETECTIONS_NOT_CONFIRMED_EMPTY": "물체 미검출만으로 빈 작업대라고 확정할 수 없습니다.",
    "EXPECTED_COUNT_NOT_CONFIGURED": "예상 물체 수가 미설정입니다."}


def button(text, slot):
    item = ui_text(QPushButton, text); item.clicked.connect(slot)
    return item


def table(headers):
    item = QTableWidget(0, len(headers)); ui_text(item.setHorizontalHeaderLabels, headers)
    item.setSelectionBehavior(QAbstractItemView.SelectRows); item.setSelectionMode(QAbstractItemView.SingleSelection)
    item.setEditTriggers(QAbstractItemView.NoEditTriggers); item.verticalHeader().hide()
    item.horizontalHeader().setStretchLastSection(True)
    return item


class FrameView(QWidget):
    def __init__(self):
        super().__init__()
        self.pixmap = QPixmap(); self.objects = []; self.selected = -1
        self.setMinimumSize(390, 230)

    def set_frame(self, path, objects):
        self.pixmap = QPixmap(str(path)); self.objects = objects; self.selected = -1; self.update()

    def paintEvent(self, event):
        p = QPainter(self); p.fillRect(self.rect(), QColor("#e8eef5"))
        if self.pixmap.isNull():
            p.setPen(QColor("#52627a"))
            p.drawText(self.rect(), Qt.AlignCenter, "검사 시작을 누르면 세 물체의 합성 예제가 표시됩니다.\n\n실제 사진은 ‘사진 모델 시험’에서 선택하세요.")
            return
        scale = min((self.width()-32)/self.pixmap.width(), (self.height()-48)/self.pixmap.height())
        x = (self.width()-self.pixmap.width()*scale)/2; y = (self.height()-self.pixmap.height()*scale)/2
        p.translate(x, y); p.scale(scale, scale); p.drawPixmap(0, 0, self.pixmap)
        for i, obj in enumerate(self.objects):
            b = obj["effective_box"]; color = QColor(COLORS[obj["final_decision"]])
            p.setPen(QPen(color, (4 if self.selected == i else 2)/scale))
            p.drawRect(int(b["x1"]), int(b["y1"]), int(b["x2"]-b["x1"]), int(b["y2"]-b["y1"]))
            p.setFont(QFont("Malgun Gothic", max(7, int(11/scale))))
            p.drawText(int(b["x1"]), max(12, int(b["y1"]-6)), f"{i+1:02d}  {VERDICTS[obj['final_decision']]}")
            for check in obj["checks"]:
                if check["check_id"] == "vlm": continue
                for finding in check["findings"]:
                    f = finding["original_box"]
                    if f:
                        p.setPen(QPen(QColor("#f4ce48"), 2/scale))
                        p.drawRect(int(f["x1"]), int(f["y1"]), int(f["x2"]-f["x1"]), int(f["y2"]-f["y1"]))


class DesktopWindow(QMainWindow):
    inspection_finished = Signal(str)

    def __init__(self, project_root, runtime, *, run_vlm_worker=True, vlm_backend="qwen"):
        super().__init__()
        self.root = Path(project_root).resolve(); self.store = AppStore(runtime)
        self.settings = self.store.settings(); self.queue = AnalysisQueue(self.store.root / "vlm")
        self.process = None; self.cancelled = False; self.closing = False; self.worker = None
        self.current = None; self.current_path = None; self.history_only = True; self.observed = None
        self.children_windows = []; self.controller = None; self.scene = None; self.robot_active = False
        self.recipe = self.settings["recipe"]; self.history_rows = []; self.last_error = None
        ui_text(self.setWindowTitle, "MES Vision · 검사 프로그램"); self.resize(1380, 900); self.setMinimumSize(1100, 740)
        central = QWidget(); self.setCentralWidget(central); outer = QHBoxLayout(central)
        self.navigation = QListWidget(); self.navigation.setObjectName("navigation"); self.navigation.setFixedWidth(177)
        ui_text(self.navigation.addItems, ["01   검사", "02   품목 설정", "03   장비 · 보정", "04   검사 이력", "05   VLM 분석"])
        outer.addWidget(self.navigation)
        main = QVBoxLayout(); outer.addLayout(main, 1)
        header = QHBoxLayout(); title = ui_text(QLabel, "MES VISION"); title.setObjectName("brand"); header.addWidget(title)
        header.addWidget(ui_text(QLabel, "로컬 검사 프로그램"), 1)
        self.vlm_toggle = button("VLM OFF", self.toggle_vlm); self.vlm_toggle.setCheckable(True); header.addWidget(self.vlm_toggle)
        main.addLayout(header)
        self.banner = ui_text(QLabel, "준비 모드 · D405 미연결 · Dobot 실물 운전 미구현"); self.banner.setObjectName("banner"); main.addWidget(self.banner)
        self.pages = QStackedWidget(); main.addWidget(self.pages, 1)
        self.make_inspection(); self.make_settings(); self.make_devices(); self.make_history()
        self.vlm_view = AnalysisViewer(self.queue, self.root, run_worker=False)
        self.vlm_view.setWindowFlags(Qt.Widget); self.pages.addWidget(self.vlm_view)
        self.navigation.currentRowChanged.connect(self.pages.setCurrentIndex); self.navigation.setCurrentRow(0)
        self.status = ui_text(QLabel, "합성 예제부터 시작할 수 있습니다. VLM은 기본 OFF입니다.")
        self.status.setWordWrap(True); main.addWidget(self.status)
        self.setStyleSheet("""
            QMainWindow, QWidget { background:#f5f7fb; color:#172c45; font-family:'Malgun Gothic'; font-size:13px; }
            QLabel#brand { font-size:23px; font-weight:700; color:#17375e; padding:8px 0; }
            QLabel#banner { background:#e7edf7; color:#36587e; border-radius:6px; padding:11px; }
            QListWidget#navigation {background:#162c48; color:#dce7f5; border:0; border-radius:8px; padding:10px 3px;}
            QListWidget#navigation::item {height:55px; padding-left:7px;}
            QListWidget#navigation::item:selected {background:#31547e; border-radius:5px; color:white;}
            QPushButton {background:white; border:1px solid #cbd5e2; border-radius:5px; padding:9px 14px;}
            QPushButton:hover {background:#e4edf9;} QPushButton:disabled {color:#98a5b6; background:#edf0f5;}
            QPushButton#primary {background:#245b9b; color:white; border:0; font-weight:600;}
            QPushButton#stop {color:#b63137; border:1px solid #dbafb0;}
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {background:white; border:1px solid #cbd5e2; padding:6px; border-radius:4px;}
            QPlainTextEdit, QTableWidget {background:white; border:1px solid #d6deea; selection-background-color:#dfeafb; selection-color:#172c45;}
            QHeaderView::section {background:#eaf0f8; border:0; padding:8px; font-weight:600;}
        """)
        self.refresh_history(); self.refresh_vlm()
        self.timer = QTimer(self); self.timer.timeout.connect(self.refresh_vlm); self.timer.start(500)
        self.robot_timer = QTimer(self); self.robot_timer.timeout.connect(self.tick_robot); self.robot_timer.setInterval(100)
        self.job_timeout = QTimer(self); self.job_timeout.setSingleShot(True)
        self.job_timeout.timeout.connect(lambda: self.stop_inspection("검사 제한 시간 180초 초과"))
        if run_vlm_worker:
            self.worker = WorkerThread(self.queue, self.root, allow_synthetic=True, backend=vlm_backend)
            self.worker.failed.connect(lambda error: self.error("VLM 작업자 오류: " + error))
            self.worker.finished.connect(self.finish_close); self.worker.start()

    def page(self):
        page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0, 14, 0, 0)
        self.pages.addWidget(page); return page, layout

    def make_inspection(self):
        _, layout = self.page(); actions = QHBoxLayout()
        self.mode = QComboBox(); ui_text(self.mode.addItem, "합성 예제 · 소프트웨어 시험", "synthetic"); ui_text(self.mode.addItem, "사진 모델 시험 · RF-DETR", "file")
        self.mode.currentIndexChanged.connect(self.mode_changed); actions.addWidget(self.mode)
        self.image_path = QLineEdit(); self.image_path.setReadOnly(True); ui_text(self.image_path.setPlaceholderText, "사진 모델 시험에서 이미지 선택")
        self.image_path.setEnabled(False)
        actions.addWidget(self.image_path, 1); self.image_button = button("사진 선택", self.choose_image); self.image_button.setEnabled(False); actions.addWidget(self.image_button)
        self.start_button = button("검사 시작", self.start_inspection); self.start_button.setObjectName("primary"); actions.addWidget(self.start_button)
        self.stop_button = button("검사 중단", lambda: self.stop_inspection()); self.stop_button.setObjectName("stop"); self.stop_button.setEnabled(False); actions.addWidget(self.stop_button)
        layout.addLayout(actions)
        self.mode_note = ui_text(QLabel, "합성 도형 · 고정된 시험 결과 · 품목 설정과 무관한 fixture-part 예제입니다."); self.mode_note.setWordWrap(True); layout.addWidget(self.mode_note)
        self.counts = ui_text(QLabel, "검사 대기     |     OK —     NG —     보류 —"); self.counts.setStyleSheet("font-size:20px; font-weight:600; padding:14px 0;"); layout.addWidget(self.counts)
        split = QSplitter(); layout.addWidget(split, 1)
        left = QWidget(); ll = QVBoxLayout(left); ll.setContentsMargins(0, 0, 10, 0)
        self.frame_view = FrameView(); ll.addWidget(self.frame_view, 1)
        self.frame_caption = ui_text(QLabel, "검사 이미지 · 원본 픽셀 좌표"); self.frame_caption.setWordWrap(True); ll.addWidget(self.frame_caption)
        self.objects = table(["물체", "판정", "불량 유형", "검출 점수"])
        self.objects.setColumnWidth(0, 75); self.objects.setColumnWidth(1, 100); self.objects.setColumnWidth(2, 190)
        self.objects.setMaximumHeight(185); self.objects.itemSelectionChanged.connect(self.select_object); ll.addWidget(self.objects)
        split.addWidget(left)
        right = QWidget(); rl = QVBoxLayout(right); rl.setContentsMargins(3, 0, 0, 0)
        self.detail_title = ui_text(QLabel, "물체별 기본 판정 근거"); self.detail_title.setStyleSheet("font-size:17px; font-weight:600;"); rl.addWidget(self.detail_title)
        self.reasons = QPlainTextEdit(); self.reasons.setReadOnly(True); ui_text(self.reasons.setPlaceholderText, "물체를 선택하면 불량 유형·검사 상태·보류 사유가 표시됩니다."); rl.addWidget(self.reasons, 1)
        self.xy = ui_text(QLabel, "로봇 XY: 보정 미등록"); self.xy.setWordWrap(True); rl.addWidget(self.xy)
        rl.addWidget(ui_text(QLabel, "VLM은 저장 후 별도 분석하며 기본 판정을 변경하지 않습니다."))
        rl.addWidget(button("선택 물체 VLM 분석 열기", self.open_selected_vlm))
        self.robot_button = button("선택 물체 모의 집기", self.start_robot); self.robot_button.setEnabled(False); rl.addWidget(self.robot_button)
        self.robot_stop = button("모의 로봇 정지", self.stop_robot); self.robot_stop.setObjectName("stop"); self.robot_stop.setEnabled(False); rl.addWidget(self.robot_stop)
        self.robot_label = ui_text(QLabel, "모의 로봇 대기 · 실물 명령 없음"); self.robot_label.setWordWrap(True); rl.addWidget(self.robot_label)
        split.addWidget(right); split.setSizes([700, 400])

    def make_settings(self):
        self.settings_page, layout = self.page()
        layout.addWidget(ui_text(QLabel, "실물 품목 설정 · 합성 예제에는 적용되지 않습니다."))
        form = QFormLayout(); layout.addLayout(form)
        self.product = QLineEdit(self.settings["product_id"]); ui_text(form.addRow, "품목 ID", self.product)
        self.width_mm = QDoubleSpinBox(); self.height_mm = QDoubleSpinBox()
        for field, key in ((self.width_mm, "workspace_width_mm"), (self.height_mm, "workspace_height_mm")):
            field.setRange(0, 10000); field.setSuffix(" mm"); ui_text(field.setSpecialValueText, "미설정"); field.setValue(self.settings[key])
        ui_text(form.addRow, "작업 공간 가로", self.width_mm); ui_text(form.addRow, "작업 공간 세로", self.height_mm)
        self.expected = QSpinBox(); self.expected.setRange(0, 100); ui_text(self.expected.setSpecialValueText, "미설정"); self.expected.setValue(self.settings["expected_count"]); ui_text(form.addRow, "예상 물체 수", self.expected)
        self.threshold = QDoubleSpinBox(); self.threshold.setRange(0, 1); self.threshold.setSingleStep(.05); self.threshold.setValue(self.settings["threshold"]); ui_text(form.addRow, "일반 검출 후보 임계값", self.threshold)
        layout.addWidget(ui_text(QLabel, "작업 공간 크기는 보정값이 아닙니다. 실물 XY는 카메라·설치·평면 높이 보정 후 사용할 수 있습니다."))
        row = QHBoxLayout(); row.addWidget(button("학습 모델 설정 가져오기", self.import_recipe)); row.addWidget(button("일반 RF-DETR로 초기화", self.clear_recipe)); row.addStretch(); layout.addLayout(row)
        self.recipe_info = QPlainTextEdit(); self.recipe_info.setReadOnly(True); layout.addWidget(self.recipe_info, 1); self.render_recipe()
        layout.addWidget(ui_text(QLabel, "파일 경로·해시·품목·클래스를 확인합니다. 사진 시험은 촬영 품질/작업 영역 검증 전이므로 최종 판정이 보류될 수 있습니다."))
        self.save_settings_button = button("품목 설정 저장", self.save_settings); self.save_settings_button.setObjectName("primary"); layout.addWidget(self.save_settings_button)

    def make_devices(self):
        _, layout = self.page()
        device_table = table(["구성", "현재 상태", "사용 가능한 기능"])
        for i, values in enumerate([
            ("D405 카메라", "미연결 · SDK 연결 미구현", "사진 입력 시험"),
            ("Dobot Magician", "실물 운전 미구현", "선택 물체 모의 집기·정지·복구"),
            ("실물 좌표 보정", "미등록", "측정점 파일 편집·계산·검증·저장"),
            ("합성 좌표 보정", "시험 전용", "검사 예제와 연결 · 실제 좌표로 사용 불가")]):
            device_table.insertRow(i)
            for j, value in enumerate(values): device_table.setItem(i, j, ui_text(QTableWidgetItem, value))
        device_table.setColumnWidth(0, 170); device_table.setColumnWidth(1, 255); device_table.setMaximumHeight(220); layout.addWidget(device_table)
        layout.addWidget(button("합성 보정 예제 열기", self.open_calibration_demo))
        layout.addWidget(button("실측 보정점 파일 열기", self.open_calibration_file))
        layout.addWidget(button("데이터 라벨링 도구 열기", self.open_labeler))
        layout.addWidget(button("모의 로봇 상태 확인 후 복구", self.recover_robot))
        layout.addWidget(ui_text(QLabel, "복구는 모의 장비의 빈 집게 상태를 재설정합니다. 이전 명령을 재개하지 않으며 새 검사가 필요합니다."))
        layout.addWidget(button("학습·데이터 준비 안내 열기", lambda: self.open_local(self.root / "docs")))
        layout.addStretch()

    def make_history(self):
        _, layout = self.page(); row = QHBoxLayout()
        self.history_search = QLineEdit(); ui_text(self.history_search.setPlaceholderText, "품목 또는 검사 ID 검색 · 최근 500건")
        self.history_search.textChanged.connect(self.refresh_history); row.addWidget(self.history_search, 1)
        row.addWidget(button("새로고침", self.refresh_history)); row.addWidget(button("선택 검사 열기", self.open_history)); layout.addLayout(row)
        self.history = table(["저장 시각", "자료 구분", "품목", "물체 수", "종합 판정", "검사 ID"])
        for col, width in enumerate((160, 125, 160, 80, 100)): self.history.setColumnWidth(col, width)
        self.history.cellDoubleClicked.connect(lambda *_: self.open_history()); layout.addWidget(self.history, 1)
        layout.addWidget(ui_text(QLabel, "저장 원본과 판정 근거를 다시 엽니다. 과거 검사 이력으로 로봇을 실행할 수 없습니다."))
        layout.addWidget(button("저장 폴더 열기", lambda: self.open_local(self.store.root)))

    def error(self, error):
        self.last_error = str(error); ui_text(self.status.setText, "오류: " + str(error))

    def mode_changed(self):
        file_mode = self.mode.currentData() == "file"
        self.image_button.setEnabled(file_mode and self.process is None)
        self.image_path.setEnabled(file_mode)
        if not file_mode: ui_text(self.image_path.clear)
        ui_text(self.mode_note.setText, "실제 RF-DETR 단건 시험 · 매 검사 모델 로딩 포함 · 연속 운전 속도 최적화 전입니다." if file_mode
            else "합성 도형 · 고정된 시험 결과 · 품목 설정과 무관한 fixture-part 예제입니다.")
        # A mode/source switch invalidates the live scene, while preserving its visible evidence.
        self.history_only = True; self.select_object()

    def choose_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "검사할 사진", "", "이미지 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)")
        if path:
            ui_text(self.image_path.setText, path); self.history_only = True; self.select_object()

    def save_settings(self):
        try:
            if self.process or self.robot_active: raise ValueError("검사·동작 종료 후 설정을 저장하세요.")
            value = dict(product_id=self.product.text().strip(), workspace_width_mm=self.width_mm.value(),
                workspace_height_mm=self.height_mm.value(), expected_count=self.expected.value(), threshold=self.threshold.value(), recipe=self.recipe)
            self.store.save_settings(value); self.settings = value; self.history_only = True; self.select_object()
            ui_text(self.status.setText, "품목 설정을 저장했습니다. 다음 사진 검사부터 적용합니다.")
        except Exception as exc: self.error(exc)

    def import_recipe(self):
        path, _ = QFileDialog.getOpenFileName(self, "학습 모델 설정", "", "JSON (*.json)")
        if not path: return
        try:
            value = load_recipe(path); self.recipe = value; ui_text(self.product.setText, value["product_id"]); self.render_recipe()
            ui_text(self.status.setText, "모델 설정을 읽었습니다. ‘품목 설정 저장’을 눌러 적용하세요.")
        except Exception as exc: self.error(exc)

    def clear_recipe(self): self.recipe = None; self.render_recipe()

    def render_recipe(self):
        import json
        ui_text(self.recipe_info.setPlainText, json.dumps(self.recipe, ensure_ascii=False, indent=2) if self.recipe else
            "일반 RF-DETR Small (COCO)\n\n물체·불량 제품 전용 학습 가중치: 미등록\n정상 특징 메모리: 미등록\n품목별 판정 기준: 미등록\n\n사진에서 일반 물체를 찾는 모델 연결 시험이 가능합니다.\n제품의 OK/NG 정확도 검증은 실물 학습 후 진행합니다.")

    def set_busy(self, busy):
        self.start_button.setEnabled(not busy and not self.closing)
        self.stop_button.setEnabled(busy and not self.robot_active)
        self.mode.setEnabled(not busy); self.image_button.setEnabled(not busy and self.mode.currentData() == "file")
        self.settings_page.setEnabled(not busy); self.select_object()

    def start_inspection(self):
        if self.process is not None or self.robot_active or self.closing: return
        try:
            mode = self.mode.currentData()
            if mode == "file" and not Path(self.image_path.text()).is_file(): raise ValueError("먼저 검사할 사진을 선택하세요.")
            job = self.store.root / "requests" / (uuid4().hex + ".json")
            output = self.store.root / "inspections" / job.stem
            job.parent.mkdir(parents=True, exist_ok=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            write_json(job, {"mode": mode, "settings": self.settings, "image": self.image_path.text(),
                "queue_root": str(self.queue.root), "output": str(output)})
            process = QProcess(self); self.process = process; self.cancelled = False; self.last_error = None
            self.history_only = True; self.set_busy(True)
            ui_text(self.counts.setText, "검사 처리 중 · 아래는 이전 결과입니다." if self.current else "검사 처리 중…")
            process.setWorkingDirectory(str(self.root))
            env = QProcessEnvironment.systemEnvironment(); env.insert("PYTHONIOENCODING", "utf-8"); env.insert("HF_HUB_OFFLINE", "1")
            process.setProcessEnvironment(env); process.setProcessChannelMode(QProcess.MergedChannels)
            self.log_file = (job.parent / (job.stem + ".log")).open("wb")
            process.readyReadStandardOutput.connect(self.drain_log)
            process.finished.connect(lambda code, status: self.job_finished(process, output, code))
            process.errorOccurred.connect(lambda error: self.process_error(process, error))
            executable = self.root / ".venv/Scripts/python.exe"
            if not executable.exists(): executable = Path(sys.executable)
            process.start(str(executable), [str(self.root / "scripts/desktop.py"), "--inspect-job", str(job)])
            self.job_timeout.start(180000); ui_text(self.status.setText, "검사 처리 중… 화면을 계속 사용할 수 있습니다. 완료되면 원본과 근거를 저장합니다.")
        except Exception as exc:
            if self.process is not None and self.process.state() == QProcess.NotRunning:
                self.process.deleteLater(); self.process = None
            self.set_busy(False); self.error(exc)

    def drain_log(self):
        if self.process is not None:
            data = bytes(self.process.readAllStandardOutput())
            if data and getattr(self, "log_file", None) and not self.log_file.closed:
                self.log_file.write(data); self.log_file.flush()

    def process_error(self, process, error):
        if error == QProcess.FailedToStart and self.process is process:
            self.error("검사 프로세스를 시작하지 못했습니다: " + process.errorString())
            self.job_finished(process, None, -1)

    def stop_inspection(self, reason="사용자가 검사를 중단했습니다."):
        if self.process is None: return
        self.cancelled = True; self.process.kill(); ui_text(self.status.setText, reason + " 결과를 게시하지 않습니다.")

    def job_finished(self, process, output, code):
        if self.process is not process: return
        self.job_timeout.stop(); self.drain_log()
        if getattr(self, "log_file", None): self.log_file.close()
        self.process = None; process.deleteLater()
        try:
            if not self.cancelled and code == 0:
                manifest, result = self.store.register(output)
                self.show_result(output, manifest, result, history=False)
                ui_text(self.status.setText, "검사 완료 · 원본/물체 이미지와 기본 판정 근거를 저장했습니다.")
                try:
                    from mes_vision.vlm.backend import GenerationConfig
                    for obj in result["objects"]: self.queue.enqueue(output, obj["object_id"], asdict(GenerationConfig()))
                except Exception as exc: self.error("기본 검사는 저장됐지만 VLM 요청 등록에 실패했습니다: " + str(exc))
                self.refresh_history(); self.vlm_view.refresh(); self.inspection_finished.emit(result["run_id"])
            elif not self.cancelled:
                error_file = output.parent / (output.name + "-error.json") if output else None
                detail = read_json(error_file)["error"] if error_file and error_file.exists() else f"종료 코드 {code}. requests 폴더의 기록을 확인하세요."
                self.error("검사 실패 · " + detail)
            if self.cancelled or code != 0:
                ui_text(self.counts.setText, ("검사 중단" if self.cancelled else "검사 실패") + (" · 아래는 이전 저장 결과입니다." if self.current else " · 새 결과 없음"))
        except Exception as exc: self.error(exc)
        self.set_busy(False); self.finish_close()

    def show_result(self, path, manifest, result, *, history):
        self.current_path = Path(path); self.current = result; self.history_only = history
        self.observed = time.monotonic() if not history else None
        self.frame_view.set_frame(self.current_path / "frame.png", result["objects"])
        counts = {k: sum(o["final_decision"] == k for o in result["objects"]) for k in ("OK", "NG", "REVIEW")}
        ui_text(self.counts.setText, f"{'저장 이력' if history else '검사 완료'} · {VERDICTS[result['final_decision']]}    |    OK {counts['OK']}    NG {counts['NG']}    보류 {counts['REVIEW']}")
        kind = "합성 시험 자료" if manifest["kind"] == "synthetic" else "사진 모델 시험"
        ui_text(self.frame_caption.setText, f"{kind} · {manifest['product_id']} · {result['frame']['width']} × {result['frame']['height']} px\n검사 ID: {result['run_id']}")
        self.objects.blockSignals(True); self.objects.setRowCount(len(result["objects"]))
        for i, obj in enumerate(result["objects"]):
            codes = obj["decision_details"].get("defect_codes", [])
            values = [f"{i+1:02d}", VERDICTS[obj["final_decision"]], text_join(' · ', (DEFECTS.get(c, c) for c in codes)) or "—", f"{obj['detection']['score']:.3f}"]
            for j, value in enumerate(values):
                cell = ui_text(QTableWidgetItem, value); cell.setForeground(QColor(COLORS[obj["final_decision"]])); self.objects.setItem(i, j, cell)
        self.objects.blockSignals(False)
        if result["objects"]: self.objects.selectRow(0)
        else:
            ui_text(self.detail_title.setText, "검사 결과 · 물체 미검출")
            ui_text(self.reasons.setPlainText, "물체를 검출하지 못했습니다. 합격으로 처리하지 않습니다.\n\n" + text_join('\n', (MESSAGES.get(v, v) for v in result["issues"])))
        self.select_object()

    def selected_object(self):
        index = self.objects.currentRow()
        return self.current["objects"][index] if self.current and 0 <= index < len(self.current["objects"]) else None

    def select_object(self):
        if not hasattr(self, "objects"): return
        obj = self.selected_object(); self.frame_view.selected = self.objects.currentRow(); self.frame_view.update()
        eligible = bool(obj and self.current["mode"] == "simulation" and not self.history_only and not self.process
            and not self.robot_active and obj["final_decision"] in {"OK", "NG"})
        self.robot_button.setEnabled(eligible)
        ui_text(self.xy.setText, "로봇 XY: 실물 보정 미등록")
        if not obj: return
        ui_text(self.detail_title.setText, f"물체 {self.objects.currentRow()+1:02d} · {VERDICTS[obj['final_decision']]}")
        lines = decision_lines(obj)
        for reason in basic_reasons(obj):
            lines.append(f"\n{reason['name']}: {CHECK_STATES.get(reason['status'], reason['status'])}")
            for finding in reason["findings"]: lines.append(f"  {finding['defect_code'] or '미분류'} · {finding['label']}")
            if reason["raw_score"] is not None: lines.append(f"  검사 점수: {reason['raw_score']:.4g}")
            lines.extend("  " + MESSAGES.get(v, v) for v in reason["messages"])
        ui_text(self.reasons.setPlainText, text_join('\n', lines))
        if self.current["mode"] == "simulation":
            try:
                target, _, _ = self.synthetic_target(obj)
                ui_text(self.xy.setText, f"합성 보정 XY: {target.x_mm:.2f}, {target.y_mm:.2f} mm\n시험 전용 · 중심 집기 가정 · 실물 사용 불가")
            except Exception as exc: ui_text(self.xy.setText, "합성 좌표 변환 불가: " + str(exc))

    def synthetic_target(self, obj):
        from mes_vision.calibration import fit, map_target
        from mes_vision.calibration.fixtures import make_spec
        from mes_vision.robot.fixtures import make_fixture
        if not hasattr(self, "synthetic_calibration"):
            self.synthetic_spec = make_spec(); self.synthetic_calibration = fit(self.synthetic_spec).accept("SYNTHETIC-DESKTOP-TEST-ONLY")
            self.synthetic_profile = replace(make_fixture()["profile"], calibration_version=self.synthetic_calibration.identity,
                pick_z_mm=self.synthetic_spec.plane_z_mm, frame_max_age_seconds=60., command_timeout_seconds=3.)
        result = run_from_dict(self.current); box = obj["effective_box"]
        target = map_target(self.synthetic_calibration, result, obj["object_id"],
            ((box["x1"]+box["x2"])/2, (box["y1"]+box["y2"])/2), self.synthetic_spec.context,
            plane_z_mm=self.synthetic_spec.plane_z_mm, r_deg=0, grasp_policy_version=self.synthetic_profile.grasp_policy_version)
        return target, result, self.synthetic_profile

    def start_robot(self):
        try:
            if not self.robot_button.isEnabled(): raise ValueError("새 합성 검사에서 OK 또는 NG 물체를 선택하세요.")
            from mes_vision.robot import RobotController, SceneStamp, build_plan
            from mes_vision.robot.simulator import SimulatedMagician
            from mes_vision.robot.contracts import Pose
            obj = self.selected_object(); target, result, profile = self.synthetic_target(obj)
            if self.controller is None: self.controller = RobotController(self.store.root / "robot", SimulatedMagician(Pose(0, 0, 100, 0)))
            self.scene = SceneStamp(result.run_id, result.frame["frame_id"], 1, self.observed,
                profile.calibration_version, self.controller.adapter.epoch)
            plan = build_plan(result, obj["object_id"], target, profile, self.scene, now=time.monotonic())
            self.controller.start(plan, self.scene, now=time.monotonic())
            self.robot_active = True; self.set_busy(True); self.robot_stop.setEnabled(True); self.robot_timer.start()
        except Exception as exc: self.error("모의 집기 불가: " + str(exc))

    def tick_robot(self):
        if not self.controller or not self.robot_active: return
        try:
            state = self.controller.tick(self.scene, now=time.monotonic())
            ui_text(self.robot_label.setText, "모의 로봇: " + state)
            if state in {"RECAPTURE", "STOPPED", "FAULT", "RECOVERY"}: self.end_robot()
        except Exception as exc:
            self.stop_robot(); self.error(exc)

    def end_robot(self):
        self.robot_timer.stop(); self.robot_active = False; self.history_only = True
        self.robot_stop.setEnabled(False); self.set_busy(False)
        ui_text(self.status.setText, "모의 동작 종료 · " + self.controller.state + " · 다음 동작에는 새 검사가 필요합니다.")

    def stop_robot(self):
        if self.controller and self.robot_active:
            try: self.controller.stop("DESKTOP_USER_STOP")
            except Exception as exc: self.error(exc)
            finally:
                ui_text(self.robot_label.setText, "모의 로봇: " + self.controller.state); self.end_robot()

    def recover_robot(self):
        try:
            if self.robot_active or self.process: raise ValueError("진행 중인 작업을 먼저 중단하세요.")
            if self.controller is None:
                from mes_vision.robot import RobotController
                from mes_vision.robot.simulator import SimulatedMagician
                from mes_vision.robot.contracts import Pose
                self.controller = RobotController(self.store.root / "robot", SimulatedMagician(Pose(0, 0, 100, 0)))
            if self.controller.state not in {"STOPPED", "FAULT", "RECOVERY"}: raise ValueError("복구가 필요한 상태가 아닙니다.")
            self.controller.adapter.reconcile_empty()
            # Only the synthetic adapter's pose is reset; no physical command is dispatched.
            from mes_vision.robot.contracts import Pose
            self.controller.adapter.pose = Pose(0, 0, 100, 0)
            self.controller.recover(confirmation_reference="SYNTHETIC-DESKTOP-USER-RESET")
            self.history_only = True; self.select_object(); ui_text(self.robot_label.setText, "모의 로봇: RECAPTURE")
            ui_text(self.status.setText, "모의 장비 상태를 복구했습니다. 새 검사를 실행하세요.")
        except Exception as exc: self.error(exc)

    def refresh_history(self, *_):
        try:
            self.history_rows = self.store.list(self.history_search.text()); self.history.setRowCount(len(self.history_rows))
            for i, row in enumerate(self.history_rows):
                values = [datetime.fromtimestamp(row["created"]).strftime("%m-%d %H:%M:%S"),
                    "합성 시험" if row["kind"] == "synthetic" else "사진 모델 시험", row["product"], str(row["objects"]), VERDICTS[row["verdict"]], row["run_id"]]
                for j, value in enumerate(values): self.history.setItem(i, j, ui_text(QTableWidgetItem, value))
        except Exception as exc:
            if hasattr(self, "status"): self.error(exc)

    def open_history(self):
        try:
            if self.process or self.robot_active: raise ValueError("진행 중인 작업 종료 후 이력을 여세요.")
            index = self.history.currentRow()
            if index < 0: return
            path, manifest, result = self.store.open(self.history_rows[index]["run_id"])
            self.show_result(path, manifest, result, history=True); self.navigation.setCurrentRow(0)
            ui_text(self.status.setText, "저장 이력 조회 · 원본 무결성 확인 완료 · 모의 집기 비활성")
        except Exception as exc: self.error(exc)

    def toggle_vlm(self, checked):
        try:
            self.queue.set_enabled(bool(checked)); self.refresh_vlm(); self.vlm_view.refresh()
            ui_text(self.status.setText, "VLM ON · 새 요청을 뒤에서 분석합니다." if checked else "VLM OFF · 대기/진행 분석을 취소합니다. 기본 근거와 완료 이력은 보존합니다.")
        except Exception as exc: self.error(exc)

    def open_selected_vlm(self):
        try:
            obj = self.selected_object()
            if obj:
                match = next((r for r in self.queue.list() if r["object_id"] == obj["object_id"]), None)
                if match: self.vlm_view.selected_key = match["id"]
                else:
                    ui_text(self.status.setText, "선택한 물체의 VLM 요청이 없습니다. 기본 검사 근거는 현재 화면에서 확인할 수 있습니다.")
                    return
            self.vlm_view.refresh(); self.navigation.setCurrentRow(4)
        except Exception as exc: self.error(exc)

    def refresh_vlm(self):
        try:
            enabled = self.queue.enabled(); self.vlm_toggle.setChecked(enabled); ui_text(self.vlm_toggle.setText, "VLM ON" if enabled else "VLM OFF")
            self.vlm_toggle.setStyleSheet("background:#d5eee3;" if enabled else "background:#e6ebf2;")
        except Exception as exc:
            if hasattr(self, "status"): self.error(exc)

    def keep_window(self, window):
        window.setAttribute(Qt.WA_DeleteOnClose); self.children_windows.append(window)
        window.destroyed.connect(lambda: self.children_windows.remove(window) if window in self.children_windows else None)
        window.show()

    def open_calibration_demo(self):
        from mes_vision.calibration.fixtures import make_spec
        from mes_vision.calibration.viewer import CalibrationViewer
        self.keep_window(CalibrationViewer(asdict(make_spec())))

    def open_calibration_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "보정 측정점 파일", "", "JSON (*.json)")
        if not path: return
        try:
            from mes_vision.calibration.viewer import CalibrationViewer
            self.keep_window(CalibrationViewer(read_json(path)))
        except Exception as exc: self.error(exc)

    def open_labeler(self):
        try:
            from mes_vision.data_management.labeler import Labeler
            self.keep_window(Labeler(None))
        except Exception as exc: self.error(exc)

    def open_local(self, path):
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))): self.error("열지 못했습니다: " + str(path))

    def finish_close(self):
        if self.closing and self.process is None and (not self.worker or not self.worker.isRunning()):
            QTimer.singleShot(0, self.close)

    def closeEvent(self, event):
        # Let annotation/calibration windows handle their own unsaved-change prompts first.
        for window in list(self.children_windows):
            if not window.close(): event.ignore(); return
        self.closing = True
        if self.robot_active: self.stop_robot()
        if self.process: self.stop_inspection("프로그램 종료 요청")
        if self.worker and self.worker.isRunning(): self.worker.stop_event.set()
        if self.process or (self.worker and self.worker.isRunning()):
            ui_text(self.status.setText, "검사와 추가 분석을 정리하고 종료하고 있습니다…"); event.ignore(); return
        if self.controller: self.controller.close()
        self.timer.stop(); self.vlm_view.timer.stop(); self.robot_timer.stop(); event.accept()

from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
"""Native point editor and calibration viewer; full operator UI remains step13."""
from copy import deepcopy
from pathlib import Path
from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QPolygonF
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QTableWidget,
    QTableWidgetItem, QDoubleSpinBox, QPlainTextEdit, QMessageBox, QFileDialog, QInputDialog)
from .core import spec_from_dict, fit
from mes_vision.training.data import require


class PointPlot(QWidget):
    def __init__(self, data):
        super().__init__(); self.data = data; self.setMinimumHeight(240)
    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#f2f6fa"))
        w, h = self.data["context"]["image_size"]
        scale = min((self.width()-70)/w, (self.height()-55)/h)
        def point(p): return QPointF(30+p[0]*scale, 25+p[1]*scale)
        painter.setPen(QPen(QColor("#60758a"), 1))
        painter.drawText(30, 17, "원본 픽셀 · (0,0) 왼쪽 위 · X → / Y ↓")
        painter.drawRect(30, 25, int(w*scale), int(h*scale))
        painter.setPen(QPen(QColor("#94a4b5"), 2, Qt.DashLine))
        painter.drawPolygon(QPolygonF([point(p) for p in self.data["application_polygon"]]))
        for item in self.data["pairs"]:
            painter.setPen(QPen(QColor("#287cbd" if item["role"] == "fit" else "#bb7526"), 3))
            p = point(item["pixel"]); painter.drawEllipse(p, 4, 4)
            painter.drawText(p+QPointF(6, 13), item["point_id"])
        painter.end()


class CalibrationViewer(QMainWindow):
    def __init__(self, specification):
        super().__init__()
        spec_from_dict(specification)
        self.input = deepcopy(specification); self.calibration = None
        ui_text(self.setWindowTitle, "MES Vision · 평면 좌표 보정")
        self.resize(1140, 870)
        body = QWidget(); self.setCentralWidget(body); outer = QVBoxLayout(body)
        kind = "합성 시험 자료" if self.input["kind"] == "synthetic" else "실물 측정 자료"
        c = self.input["context"]
        outer.addWidget(ui_text(QLabel, f"{kind} · 카메라 {c['camera_id']} · 검사면 Z {self.input['plane_z_mm']} mm · {c['image_size'][0]} × {c['image_size'][1]} px"))
        self.plot = PointPlot(self.input); outer.addWidget(self.plot)
        self.table = QTableWidget(len(self.input["pairs"]), 7)
        ui_text(self.table.setHorizontalHeaderLabels, ["점 ID", "용도 fit/check", "측정 회차", "원본 X(px)", "원본 Y(px)", "로봇 X(mm)", "로봇 Y(mm)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        for column, width in enumerate((90, 110, 220, 100, 100, 185)): self.table.setColumnWidth(column, width)
        for row, p in enumerate(self.input["pairs"]):
            values = [p["point_id"], p["role"], p["measurement_session"], *p["pixel"], *p["robot_xy_mm"]]
            for column, value in enumerate(values): self.table.setItem(row, column, ui_text(QTableWidgetItem, str(value)))
        self.table.cellChanged.connect(self.invalidate)
        self.table.setMaximumHeight(230); outer.addWidget(self.table)
        actions = QHBoxLayout()
        self.fit_button = ui_text(QPushButton, "보정 계산 · 오차 확인"); self.fit_button.clicked.connect(self.calculate_fit); actions.addWidget(self.fit_button)
        self.accept_button = ui_text(QPushButton, "검증 기록 연결"); self.accept_button.clicked.connect(self.ask_accept); actions.addWidget(self.accept_button)
        self.save_button = ui_text(QPushButton, "보정 파일 저장"); self.save_button.clicked.connect(self.ask_save); actions.addWidget(self.save_button)
        outer.addLayout(actions)
        self.summary = QPlainTextEdit(); self.summary.setReadOnly(True); outer.addWidget(self.summary, 1)
        mapping = QHBoxLayout()
        self.pixel_x = QDoubleSpinBox(); self.pixel_y = QDoubleSpinBox()
        for spin, dimension in ((self.pixel_x, c["image_size"][0]), (self.pixel_y, c["image_size"][1])):
            spin.setDecimals(3); spin.setRange(0, dimension-1); spin.setValue(dimension/2)
            spin.valueChanged.connect(self.map_point)
        mapping.addWidget(ui_text(QLabel, "변환할 원본 픽셀 X / Y")); mapping.addWidget(self.pixel_x); mapping.addWidget(self.pixel_y)
        self.map_button = ui_text(QPushButton, "로봇 XY 확인"); self.map_button.clicked.connect(self.map_point); mapping.addWidget(self.map_button)
        outer.addLayout(mapping)
        self.mapping = ui_text(QLabel, "보정 계산과 검증 기록 연결이 필요합니다."); outer.addWidget(self.mapping)
        outer.addWidget(ui_text(QLabel, "픽셀과 mm는 별도 좌표계입니다. 이 창에서는 로봇 명령을 보내지 않습니다."))
        self.invalidate()

    def error(self, message): QMessageBox.warning(self, "좌표 보정", str(message))

    def invalidate(self, *args):
        self.calibration = None
        ui_text(self.summary.setPlainText, "대응점 계산 전 · 수정 후에는 재계산과 검증 기록 연결이 필요합니다.")
        ui_text(self.mapping.setText, "변환 미설정")
        self.accept_button.setEnabled(False); self.save_button.setEnabled(False); self.map_button.setEnabled(False)

    def calculate_fit(self):
        self.invalidate()
        try:
            data = deepcopy(self.input); data["pairs"] = []
            for row in range(self.table.rowCount()):
                values = [self.table.item(row, column).text().strip() for column in range(7)]
                data["pairs"].append(dict(point_id=values[0], role=values[1], measurement_session=values[2],
                                          pixel=[float(v) for v in values[3:5]], robot_xy_mm=[float(v) for v in values[5:7]]))
            self.calibration = fit(spec_from_dict(data))
            self.input = data; self.plot.data = data; self.plot.update()
            metrics = self.calibration.data["metrics"]
            failed = self.calibration.data["failures"]
            lines = ["허용 오차/범위 검사 실패" if failed else "수치 검사 통과 · 검증 기록 연결 전",
                f"계산점 최대 오차: {metrics['fit_max_mm']:.6f} mm",
                f"독립 확인점 최대 오차: {metrics['check_max_mm']:.6f} mm / RMS: {metrics['check_rmse_mm']:.6f} mm",
                f"확인점 영역 비율: {metrics['check_coverage']:.1%}"]
            if failed: lines.append("확인 필요: " + text_join(', ', failed))
            ui_text(self.summary.setPlainText, text_join('\n', lines))
            self.accept_button.setEnabled(not failed); self.save_button.setEnabled(True)
        except Exception as exc: self.error(exc)

    def accept_reference(self, reference):
        require(self.calibration is not None, "먼저 보정을 계산하세요")
        self.calibration = self.calibration.accept(reference)
        ui_text(self.summary.setPlainText, self.summary.toPlainText().replace("수치 검사 통과 · 검증 기록 연결 전", "수치 검사 통과 · 검증 기록 연결 완료"))
        self.summary.appendPlainText("검증 기록 연결됨: " + reference + "\n현재 조건에서 좌표 변환 가능 · 로봇 운전은 별도 설정 필요")
        self.accept_button.setEnabled(False); self.map_button.setEnabled(True)
        self.map_point()

    def ask_accept(self):
        reference, ok = QInputDialog.getText(self, "검증 기록 연결", "검토한 측정·허용 오차 검증 기록의 이름 또는 경로")
        if ok:
            try: self.accept_reference(reference)
            except Exception as exc: self.error(exc)

    def save_to(self, path):
        require(self.calibration is not None, "계산 결과가 없습니다")
        self.calibration.save(Path(path))

    def ask_save(self):
        path, _ = QFileDialog.getSaveFileName(self, "새 보정 파일 저장", "calibration.json", "JSON (*.json)")
        if path:
            try: self.save_to(path)
            except Exception as exc: self.error(exc)

    def map_point(self, *args):
        if self.calibration is None or not self.calibration.ready:
            ui_text(self.mapping.setText, "변환 미설정 · 계산과 검증 기록 연결 필요"); return
        try:
            spec = spec_from_dict(self.input)
            xy = self.calibration.map_xy((self.pixel_x.value(), self.pixel_y.value()), spec.context, plane_z_mm=spec.plane_z_mm, kind=spec.kind)
            ui_text(self.mapping.setText, f"로봇 X = {xy[0]:.4f} mm / Y = {xy[1]:.4f} mm · 검사면 Z = {spec.plane_z_mm} mm")
        except Exception as exc: ui_text(self.mapping.setText, "좌표 변환 차단: " + str(exc))

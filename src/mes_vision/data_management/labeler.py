"""Focused desktop annotation tool; independent from the future inspection UI."""
from __future__ import annotations
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf

from copy import deepcopy
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, Signal, QThread, QRectF
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QPushButton, QLabel, QListWidget, QComboBox, QLineEdit, QPlainTextEdit, QCheckBox,
    QSplitter, QGraphicsView, QGraphicsScene, QGraphicsItem, QFileDialog,
    QInputDialog, QMessageBox)

from .collection import Collection, CONDITIONS, CODES, SPLITS, new_object, validate_record
from .export import export_collection

CONDITION_LABELS = [tr('미검토'), tr('정상'), tr('알려진 불량'), tr('미등록 불량'), tr('판독 불확실')]
SPLIT_LABELS = [tr('미지정'), tr('학습'), tr('검증'), tr('시험'), tr('미등록·불확실 별도 보관')]


class Work(QThread):
    succeeded = Signal(object)
    failed = Signal(str)
    def __init__(self, action):
        super().__init__()
        self.action = action
    def run(self):
        try:
            self.succeeded.emit(self.action())
        except Exception as exc:
            self.failed.emit(str(exc))


class Canvas(QGraphicsView):
    box_drawn = Signal(object)
    object_clicked = Signal(str)
    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.Antialiasing)
        self.setBackgroundBrush(QColor("#e8edf3"))
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.record = None
        self.start = None
        self.rubber = None
        self.drawing = False
        self.layers = []
        self.setDragMode(QGraphicsView.ScrollHandDrag)

    def set_drawing(self, enabled):
        self.drawing = enabled
        self.start = None
        if self.rubber is not None:
            ui_text(self.scene().removeItem, self.rubber)
            self.rubber = None
        self.setDragMode(QGraphicsView.NoDrag if enabled else QGraphicsView.ScrollHandDrag)

    def show_record(self, record, image_path=None, selected=None):
        self.record = record
        if image_path is not None:
            ui_text(self.scene().clear)
            self.layers = []
            self.scene().addPixmap(QPixmap(str(image_path)))
            self.setSceneRect(0, 0, record["width"], record["height"])
            self.fitInView(self.sceneRect(), Qt.KeepAspectRatio)
        else:
            for item in self.layers:
                ui_text(self.scene().removeItem, item)
            self.layers = []
        for obj in record["objects"]:
            color = "#1980b5" if obj["id"] != selected else "#00ae75"
            x1, y1, x2, y2 = obj["bbox"]
            pen = QPen(QColor(color), 3)
            pen.setCosmetic(True)
            item = self.scene().addRect(QRectF(x1, y1, x2-x1, y2-y1), pen)
            item.setData(0, obj["id"])
            self.layers.append(item)
            text = self.scene().addSimpleText(obj["specimen_id"])
            text.setBrush(QColor(color))
            text.setPos(x1, y1)
            text.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)
            text.setData(0, obj["id"])
            self.layers.append(text)
            for defect in obj["defects"]:
                if defect["bbox"] is not None:
                    a, b, c, d = defect["bbox"]
                    pen = QPen(QColor("#de5636"), 2, Qt.DashLine)
                    pen.setCosmetic(True)
                    item = self.scene().addRect(QRectF(a, b, c-a, d-b), pen)
                    item.setData(0, obj["id"])
                    self.layers.append(item)

    def point(self, event):
        p = self.mapToScene(event.position().toPoint())
        p.setX(max(0, min(self.record["width"], p.x())))
        p.setY(max(0, min(self.record["height"], p.y())))
        return p

    def mousePressEvent(self, event):
        if self.record and event.button() == Qt.LeftButton and self.drawing:
            self.start = self.point(event)
            self.rubber = self.scene().addRect(QRectF(self.start, self.start), QPen(QColor("#ee793a"), 1))
            event.accept()
            return
        if event.button() == Qt.LeftButton:
            for item in self.items(event.position().toPoint()):
                if item.data(0):
                    self.object_clicked.emit(item.data(0))
                    break
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.start is not None and self.rubber is not None:
            self.rubber.setRect(QRectF(self.start, self.point(event)).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self.start is not None and self.rubber is not None and event.button() == Qt.LeftButton:
            rect = QRectF(self.start, self.point(event)).normalized()
            ui_text(self.scene().removeItem, self.rubber)
            self.start, self.rubber = None, None
            if rect.width() >= 1 and rect.height() >= 1:
                self.box_drawn.emit([rect.left(), rect.top(), rect.right(), rect.bottom()])
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        scale = self.transform().m11()
        factor = 1.2 if event.angleDelta().y() > 0 else 1/1.2
        if .02 < scale*factor < 30:
            self.scale(factor, factor)
        event.accept()


class Labeler(QMainWindow):
    def __init__(self, workspace=None, *, allow_synthetic=True, runtime=None):
        super().__init__()
        self.runtime=Path(runtime) if runtime else Path(__file__).resolve().parents[3]/'artifacts/operation'
        self.allow_synthetic = allow_synthetic
        ui_text(self.setWindowTitle, tr('MES Vision · 데이터 라벨링'))
        self.resize(1380, 900)
        self.collection = None
        self.draft = None
        self.dirty = False
        self.loading = False
        self.undo_stack = []
        self.worker = None
        self.selected = None
        self.central = QWidget()
        self.setCentralWidget(self.central)
        outer = QVBoxLayout(self.central)
        bar = QHBoxLayout()
        for text, callback in [(tr('새 수집 프로젝트'), self.create_project), (tr('프로젝트 열기'), self.open_project),
                               (tr('사진 추가'), self.import_images), (tr('영상 추가'), self.import_video), (tr('저장'), self.save), (tr('검토 완료'), self.review),
                               (tr('되돌리기'), self.undo), (tr('학습 데이터 내보내기'), self.export_dialog)]:
            button = ui_text(QPushButton, text)
            button.clicked.connect(callback)
            bar.addWidget(button)
        outer.addLayout(bar)
        self.heading = ui_text(QLabel, tr('사진을 모으고 물체·불량을 표시하세요. 자동 AI 판정 화면이 아닙니다.'))
        self.heading.setStyleSheet("font-size: 17px; font-weight: 600; padding: 7px;")
        outer.addWidget(self.heading)
        panes = QSplitter()
        outer.addWidget(panes, 1)
        self.captures = QListWidget()
        self.captures.setMinimumWidth(210)
        self.captures.currentRowChanged.connect(self.select_capture)
        panes.addWidget(self.captures)
        middle = QWidget()
        layout = QVBoxLayout(middle)
        tools = QHBoxLayout()
        self.mode = QComboBox()
        ui_text(self.mode.addItems, [tr('선택·이동'), tr('물체 상자 그리기'), tr('불량 상자 그리기'), tr('선택 물체 상자 다시 그리기')])
        self.mode.currentIndexChanged.connect(lambda i: self.canvas.set_drawing(i != 0))
        tools.addWidget(self.mode)
        fit = ui_text(QPushButton, tr('전체 보기'))
        fit.clicked.connect(lambda: self.canvas.fitInView(self.canvas.sceneRect(), Qt.KeepAspectRatio))
        tools.addWidget(fit)
        layout.addLayout(tools)
        sam_row=QHBoxLayout()
        self.sam_object=ui_text(QPushButton,tr('SAM 물체 라벨'));self.sam_object.setObjectName('samObjectLabel')
        self.sam_defect=ui_text(QPushButton,tr('SAM 불량 영역'));self.sam_defect.setObjectName('samDefectLabel')
        self.sam_object.clicked.connect(lambda:self.sam_label(False));self.sam_defect.clicked.connect(lambda:self.sam_label(True))
        sam_row.addWidget(self.sam_object);sam_row.addWidget(self.sam_defect);layout.addLayout(sam_row)
        self.canvas = Canvas()
        self.canvas.box_drawn.connect(self.box_drawn)
        self.canvas.object_clicked.connect(self.select_object_id)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(ui_text(QLabel, tr('휠: 확대/축소 · 선택·이동 모드: 드래그로 이동 · 좌표는 원본 RGB 픽셀')))
        panes.addWidget(middle)
        right = QWidget()
        right.setMinimumWidth(330)
        fields = QVBoxLayout(right)
        self.session = ui_text(QLabel, tr('촬영 회차: —'))
        fields.addWidget(self.session)
        splitrow = QHBoxLayout()
        self.split = QComboBox()
        ui_text(self.split.addItems, SPLIT_LABELS)
        splitrow.addWidget(self.split)
        assign = ui_text(QPushButton, tr('회차 전체에 적용'))
        assign.clicked.connect(self.assign_split)
        splitrow.addWidget(assign)
        fields.addLayout(splitrow)
        self.empty = ui_text(QCheckBox, tr('물체 없는 작업대임을 확인'))
        self.empty.toggled.connect(self.empty_changed)
        fields.addWidget(self.empty)
        self.exclude = ui_text(QPushButton, tr('이 사진 제외 / 제외 해제'))
        self.exclude.clicked.connect(self.exclude_capture)
        fields.addWidget(self.exclude)
        fields.addWidget(ui_text(QLabel, tr('물체 목록')))
        self.objects = QListWidget()
        self.objects.setMaximumHeight(150)
        self.objects.currentRowChanged.connect(self.select_object)
        fields.addWidget(self.objects)
        form = QFormLayout()
        self.specimen = QLineEdit()
        self.condition = QComboBox()
        ui_text(self.condition.addItems, CONDITION_LABELS)
        self.note = QPlainTextEdit()
        self.note.setMaximumHeight(65)
        self.specimen.textChanged.connect(self.edit_fields)
        self.condition.currentIndexChanged.connect(self.edit_fields)
        self.note.textChanged.connect(self.edit_fields)
        ui_text(form.addRow, tr('실물 ID'), self.specimen)
        ui_text(form.addRow, tr('관찰 상태'), self.condition)
        ui_text(form.addRow, tr('관찰 메모'), self.note)
        fields.addLayout(form)
        apply = ui_text(QPushButton, tr('물체 정보 확인'))
        apply.clicked.connect(self.apply_object)
        fields.addWidget(apply)
        delete = ui_text(QPushButton, tr('선택 물체 삭제'))
        delete.clicked.connect(self.delete_object)
        fields.addWidget(delete)
        fields.addWidget(ui_text(QLabel, tr('불량 유형 · 한 물체에 여러 개 표시 가능')))
        self.code = QComboBox()
        ui_text(self.code.addItems, [f"{code} · {tr(label)}" for code, label in CODES.items()])
        fields.addWidget(self.code)
        self.defects = QListWidget()
        self.defects.setMaximumHeight(120)
        fields.addWidget(self.defects)
        global_button = ui_text(QPushButton, tr('영역 없이 불량 관찰 추가'))
        global_button.clicked.connect(self.add_global)
        fields.addWidget(global_button)
        remove = ui_text(QPushButton, tr('선택 불량 삭제'))
        remove.clicked.connect(self.delete_defect)
        fields.addWidget(remove)
        self.reviewer = QLineEdit()
        ui_text(self.reviewer.setPlaceholderText, tr('검토자 이름 (검토 완료 시 필요)'))
        fields.addWidget(self.reviewer)
        fields.addStretch()
        panes.addWidget(right)
        panes.setSizes([230, 790, 360])
        self.status = ui_text(QLabel, tr('프로젝트를 만들거나 열어 주세요.'))
        outer.addWidget(self.status)
        self.setStyleSheet("QPushButton { padding: 6px; } QLineEdit, QComboBox { min-height: 25px; } ")
        if workspace:
            self.load_collection(workspace)

    def error(self, message):
        QMessageBox.warning(self, tr('데이터 확인'), str(message))

    def guard(self, action):
        try:
            return action()
        except Exception as exc:
            self.error(exc)
            return None

    def discard_or_save(self):
        if not self.dirty:
            return True
        choice = QMessageBox.question(self, tr('저장하지 않은 라벨'), tr('변경 내용을 저장할까요?'),
                                      QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, QMessageBox.Save)
        if choice == QMessageBox.Save:
            return self.save()
        return choice == QMessageBox.Discard

    def load_collection(self, root):
        collection = Collection(root)
        if not self.allow_synthetic and collection.data['kind'] != 'real':
            raise ValueError(tr('실물 데이터 프로젝트를 선택하세요.'))
        self.collection = collection
        self.draft, self.selected, self.dirty = None, None, False
        self.refresh_captures()
        ui_text(self.heading.setText, trf('{v0} · {v1} · 수동 라벨링', v0=self.collection.data['product_id'], v1=tr('합성 시험 데이터') if self.collection.data['kind'] == 'synthetic' else tr('실물 데이터')))

    def create_project(self):
        if not self.discard_or_save(): return
        path, _ = QFileDialog.getSaveFileName(self, tr('새 수집 프로젝트 폴더 이름'), "", tr('모든 파일 (*)'))
        if not path: return
        name, ok = QInputDialog.getText(self, tr('품목'), tr('품목 ID'))
        if not ok: return
        kind, ok = QInputDialog.getItem(self, tr('데이터 종류'), tr('데이터 출처'), [tr('실물 사진'), tr('합성 시험 사진')], 0, False) if self.allow_synthetic else (tr('실물 사진'), True)
        if ok:
            def create():
                Collection.create(path, name, kind="real" if kind == tr('실물 사진') else "synthetic")
                self.load_collection(path)
            self.guard(create)

    def open_project(self):
        if not self.discard_or_save(): return
        path, _ = QFileDialog.getOpenFileName(self, tr('수집 프로젝트 열기'), "", tr('수집 프로젝트 (collection.json)'))
        if path: self.guard(lambda: self.load_collection(Path(path).parent))

    def run_work(self, action, finished):
        self.central.setEnabled(False)
        ui_text(self.status.setText, tr('파일을 처리하고 있습니다…'))
        self.worker = Work(action)
        def success(result):
            ui_text(self.status.setText, tr('파일 작업 완료'))
            finished(result)
        def failure(message):
            ui_text(self.status.setText, tr('파일 작업 실패 · 오류 내용을 확인하세요'))
            self.error(message)
        self.worker.succeeded.connect(success)
        self.worker.failed.connect(failure)
        self.worker.finished.connect(lambda: self.central.setEnabled(True))
        self.worker.start()

    def import_images(self):
        if not self.collection or not self.discard_or_save(): return
        paths, _ = QFileDialog.getOpenFileNames(self, tr('사진 추가'), "", tr('사진 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp)'))
        if not paths: return
        session, ok = QInputDialog.getText(self, tr('촬영 회차'), tr('같은 촬영 회차 ID'))
        if not ok: return
        collection = self.collection
        def work():
            errors = []
            for path in paths:
                try: collection.import_image(path, session)
                except Exception as exc: errors.append(f"{Path(path).name}: {exc}")
            return errors
        def done(errors):
            self.dirty = False
            self.refresh_captures()
            ui_text(self.status.setText, trf('사진 추가 완료 · 성공 {v0} · 오류 {v1}', v0=len(paths) - len(errors), v1=len(errors)))
            if errors: self.error(text_join('\n', errors))
        self.run_work(work, done)

    def import_video(self):
        if not self.collection or not self.discard_or_save(): return
        path, _ = QFileDialog.getOpenFileName(self, tr('영상 추가'), '', 'Video (*.mp4 *.avi *.mov *.mkv *.m4v)')
        if not path: return
        session, ok = QInputDialog.getText(self, tr('촬영 회차'), tr('같은 촬영 회차 ID'))
        if not ok: return
        interval, ok = QInputDialog.getDouble(self, tr('영상 추가'), tr('추출 간격(초, 영상 FPS 기준 근사)'), 1.0, 0.1, 3600, 1)
        if not ok: return
        maximum, ok = QInputDialog.getInt(self, tr('영상 추가'), tr('최대 추출 장수'), 120, 1, 1000)
        if not ok: return
        from .video import import_video
        def done(result):
            self.dirty = False
            self.refresh_captures()
            ui_text(self.status.setText, trf('영상에서 {v0}장 추가 · 물체와 불량 라벨을 검토하세요', v0=result['frames']))
        self.run_work(lambda: import_video(self.collection, path, session,
                      interval_seconds=interval, max_frames=maximum), done)

    def refresh_captures(self):
        self.loading = True
        ui_text(self.captures.clear)
        for r in self.collection.data["records"]:
            state = tr('제외') if r["excluded"] else tr('검토 완료') if r["reviewed"] else tr('작성 중')
            ui_text(self.captures.addItem, f"{r['source_name']}\n{r['capture_session_id']} · {state}")
        self.loading = False
        if self.collection.data["records"]:
            self.captures.setCurrentRow(0)
        else:
            ui_text(self.canvas.scene().clear)
            self.canvas.record = None
            ui_text(self.objects.clear)

    def select_capture(self, row):
        if self.loading or row < 0 or not self.collection: return
        if not self.discard_or_save():
            self.loading = True
            self.captures.setCurrentRow(next(i for i, r in enumerate(self.collection.data["records"]) if r["id"] == self.draft["id"]))
            self.loading = False
            return
        self.draft = deepcopy(self.collection.data["records"][row])
        self.selected, self.dirty, self.undo_stack = None, False, []
        self.canvas.set_drawing(self.mode.currentIndex() != 0)
        self.canvas.show_record(self.draft, self.collection.image_path(self.draft))
        self.render()

    def render(self):
        if not self.draft: return
        self.loading = True
        ui_text(self.session.setText, trf('촬영 회차: {v0}', v0=self.draft['capture_session_id']))
        self.split.setCurrentIndex(SPLITS.index(self.draft["split"]))
        self.empty.setChecked(self.draft["empty_confirmed"])
        ui_text(self.objects.clear)
        for obj in self.draft["objects"]:
            ui_text(self.objects.addItem, f"{obj['specimen_id']} · {CONDITION_LABELS[CONDITIONS.index(obj['condition'])]}")
        self.loading = False
        index = next((i for i, o in enumerate(self.draft["objects"]) if o["id"] == self.selected), -1)
        if index >= 0: self.objects.setCurrentRow(index)
        else:
            self.loading = True
            ui_text(self.specimen.clear); ui_text(self.note.clear); ui_text(self.defects.clear)
            self.loading = False
        self.canvas.show_record(self.draft, selected=self.selected)
        for i, record in enumerate(self.collection.data["records"]):
            if record["id"] == self.draft["id"]:
                state = tr('저장 필요') if self.dirty else tr('제외') if record["excluded"] else tr('검토 완료') if record["reviewed"] else tr('작성 중')
                ui_text(self.captures.item(i).setText, f"{record['source_name']}\n{record['capture_session_id']} · {state}")
        ui_text(self.status.setText, trf('{v0} · {v1}×{v2} 원본 · {v3}개 물체', v0=tr('저장 필요') if self.dirty else tr('검토 완료') if self.draft['reviewed'] else tr('작성 중'), v1=self.draft['width'], v2=self.draft['height'], v3=len(self.draft['objects'])))

    def selected_object(self):
        if not self.draft or not self.selected: return None
        return next((o for o in self.draft["objects"] if o["id"] == self.selected), None)

    def select_object_id(self, identity):
        for index, obj in enumerate(self.draft["objects"]):
            if obj["id"] == identity: self.objects.setCurrentRow(index); break

    def select_object(self, index):
        if self.loading or not self.draft or index < 0: return
        obj = self.draft["objects"][index]
        self.selected = obj["id"]
        self.loading = True
        ui_text(self.specimen.setText, obj["specimen_id"])
        self.condition.setCurrentIndex(CONDITIONS.index(obj["condition"]))
        ui_text(self.note.setPlainText, obj["note"])
        ui_text(self.defects.clear)
        for defect in obj["defects"]:
            ui_text(self.defects.addItem, f"{defect['code']} · {tr(CODES[defect['code']])} · {tr('영역') if defect['bbox'] else tr('관찰만')}")
        self.loading = False
        self.canvas.show_record(self.draft, selected=self.selected)

    def edit_fields(self, *_):
        # Keep even unfinished text in the draft; validation happens on save.
        if self.loading or not self.selected_object(): return
        self.undo_stack.append(deepcopy(self.draft))
        self.undo_stack = self.undo_stack[-30:]
        self.selected_object().update(specimen_id=self.specimen.text().strip(),
            condition=CONDITIONS[self.condition.currentIndex()], note=self.note.toPlainText())
        self.draft.update(reviewed=False, reviewer=None, reviewed_at_utc=None)
        self.dirty = True
        ui_text(self.status.setText, tr('저장 필요 · 물체 정보 변경 중'))

    def mutate(self, change):
        if not self.draft: return False
        proposed = deepcopy(self.draft)
        try:
            change(proposed)
            validate_record(proposed)
        except Exception as exc:
            self.error(exc)
            return False
        self.undo_stack.append(deepcopy(self.draft))
        self.undo_stack = self.undo_stack[-30:]
        proposed.update(reviewed=False, reviewer=None, reviewed_at_utc=None)
        self.draft, self.dirty = proposed, True
        self.render()
        return True

    def sam_label(self, defect=False):
        if not self.draft or not self.collection:return
        if defect and not self.selected_object():self.error(tr('먼저 물체를 선택하세요'));return
        from .sam_dialog import SamDialog,retain_proposal
        from PySide6.QtWidgets import QDialog
        dialog=None
        try:
            record=self.collection.record(self.draft['id'])
            dialog=SamDialog(self.collection.image_path(record),record['image_sha256'],self,runtime=self.runtime)
            if dialog.exec()!=QDialog.Accepted:return
            proposal=dialog.accepted_proposal;box=proposal['candidate']['box']
            if defect:
                obj=self.selected_object();a,b,c,d=obj['bbox']
                if not (a<=box[0]<box[2]<=c and b<=box[1]<box[3]<=d):
                    self.error(tr('불량 영역은 선택한 물체 상자 안에 있어야 합니다'));return
                assist=retain_proposal(self.collection,proposal)
                item={'id':uuid4().hex,'code':list(CODES)[self.code.currentIndex()],
                      'bbox':box,'note':'','annotation_assist':assist}
                def change(r):
                    target=next(o for o in r['objects'] if o['id']==self.selected)
                    target['defects'].append(item)
                    if target['condition'] not in {'UNKNOWN_NG','UNCERTAIN'}:target['condition']='KNOWN_NG'
                self.mutate(change)
            else:
                specimen,ok=QInputDialog.getText(self,tr('실물 ID'),tr('이 물체의 실제 개체 ID (다른 사진에서도 같은 ID)'))
                if not ok:return
                obj=new_object(specimen.strip(),box);obj['annotation_assist']=retain_proposal(self.collection,proposal)
                if self.mutate(lambda r:(r['objects'].append(obj),r.update(empty_confirmed=False))):self.select_object_id(obj['id'])
        except Exception as exc:self.error(str(exc))
        finally:
            if dialog is not None:dialog.cleanup();dialog.deleteLater()

    def box_drawn(self, bbox):
        if self.mode.currentIndex() == 1:
            specimen, ok = QInputDialog.getText(self, tr('실물 ID'), tr('이 물체의 실제 개체 ID (다른 사진에서도 같은 ID)'))
            if not ok: return
            obj = new_object(specimen.strip(), bbox)
            if self.mutate(lambda r: (r["objects"].append(obj), r.update(empty_confirmed=False))):
                self.select_object_id(obj["id"])
        elif self.mode.currentIndex() == 2:
            self.add_defect(bbox, "")
        elif self.mode.currentIndex() == 3 and self.selected_object():
            self.mutate(lambda r: next(o for o in r["objects"] if o["id"] == self.selected).update(bbox=bbox))

    def apply_object(self):
        if not self.selected_object(): return
        def confirm():
            validate_record(self.draft)
            self.render()
        self.guard(confirm)

    def add_defect(self, bbox, note):
        if not self.selected_object(): self.error(tr('먼저 물체를 선택하세요')); return
        defect = {"id": uuid4().hex, "code": list(CODES)[self.code.currentIndex()], "bbox": bbox, "note": note}
        def change(r):
            obj = next(o for o in r["objects"] if o["id"] == self.selected)
            obj["defects"].append(defect)
            if obj["condition"] not in {"UNKNOWN_NG", "UNCERTAIN"}: obj["condition"] = "KNOWN_NG"
        self.mutate(change)

    def add_global(self):
        if not self.selected_object(): return
        note, ok = QInputDialog.getText(self, tr('영역 없는 관찰'), tr('누락 위치·형상 등 관찰 설명 (빈 상자를 만들지 않습니다)'))
        if ok: self.add_defect(None, note)

    def delete_object(self):
        if self.selected_object():
            self.mutate(lambda r: r.update(objects=[o for o in r["objects"] if o["id"] != self.selected]))

    def delete_defect(self):
        index = self.defects.currentRow()
        if index < 0 or not self.selected_object(): return
        def change(r):
            obj = next(o for o in r["objects"] if o["id"] == self.selected)
            obj["defects"].pop(index)
            if not obj["defects"] and obj["condition"] == "KNOWN_NG": obj["condition"] = "UNREVIEWED"
        self.mutate(change)

    def empty_changed(self, value):
        if not self.loading and self.draft:
            if not self.mutate(lambda r: r.update(empty_confirmed=value)): self.render()

    def exclude_capture(self):
        if not self.draft: return
        if self.draft["excluded"]:
            self.mutate(lambda r: r.update(excluded=False, exclude_reason=""))
        else:
            reason, ok = QInputDialog.getText(self, tr('사진 제외'), tr('제외 사유 (흔들림·가림 등)'))
            if ok: self.mutate(lambda r: r.update(excluded=True, exclude_reason=reason))

    def undo(self):
        if self.undo_stack:
            self.draft = self.undo_stack.pop()
            self.dirty = True
            self.render()

    def save(self):
        if not self.draft or not self.dirty: return True
        try:
            self.collection.save_record(self.draft)
            self.draft = self.collection.record(self.draft["id"])
            self.dirty, self.undo_stack = False, []
            self.render()
            return True
        except Exception as exc:
            self.error(exc)
            return False

    def review(self):
        if not self.draft or not self.save(): return
        def action():
            self.collection.review(self.draft["id"], self.reviewer.text())
            self.draft = self.collection.record(self.draft["id"])
            self.render()
        self.guard(action)

    def assign_split(self):
        if not self.draft: return
        chosen = SPLITS[self.split.currentIndex()]
        if not self.save(): return
        def action():
            self.collection.assign_session(self.draft["capture_session_id"], chosen)
            self.draft = self.collection.record(self.draft["id"])
            self.render()
        self.guard(action)

    def export_dialog(self):
        if not self.collection or not self.save(): return
        labels = [tr('물체 위치 학습'), tr('불량 상자 학습'), tr('정상 기준 이미지'), tr('미등록·불확실 별도 보관')]
        choice, ok = QInputDialog.getItem(self, tr('내보내기'), tr('목적'), labels, 0, False)
        if not ok: return
        role = ["object_detector", "known_defect_detector", "normal_bank", "challenge"][labels.index(choice)]
        codes = ()
        if role == "known_defect_detector":
            text, ok = QInputDialog.getText(self, tr('불량 학습 클래스'), tr('영역으로 학습할 코드 (쉼표 구분, 예: NG03,NG06)'))
            if not ok: return
            codes = tuple(c.strip() for c in text.split(",") if c.strip())
        path, _ = QFileDialog.getSaveFileName(self, tr('새 내보내기 폴더 이름'), "", tr('모든 파일 (*)'))
        if not path: return
        self.run_work(lambda: export_collection(self.collection, path, role, defect_codes=codes),
                      lambda report: ui_text(self.status.setText, trf('내보내기 완료 · {v0}개 이미지 · {v1}개 제외 기록 · {v2}', v0=len(report['items']), v1=len(report['excluded']), v2=path)))

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.error(tr('파일 작업이 끝난 뒤 닫아 주세요'))
            event.ignore()
        elif self.discard_or_save(): event.accept()
        else: event.ignore()

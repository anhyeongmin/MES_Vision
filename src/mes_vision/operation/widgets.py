from mes_vision.theme import color as theme_color
from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr, trf
from PySide6.QtCore import Qt,Signal,QPointF,QRectF,QThread
from PySide6.QtGui import QColor,QImage,QPainter,QPen,QPixmap,QPolygonF
from PySide6.QtWidgets import QWidget,QPushButton,QTableWidget,QAbstractItemView,QDoubleSpinBox,QSpinBox,QLineEdit,QHBoxLayout,QFileDialog

COLORS={"OK":"#14846b","NG":"#d23c47","REVIEW":"#ad7510","WAITING":"#597084","STABLE":"#347cc3",
        "INSPECTING":"#347cc3","RECHECK":"#ad7510","LOST":"#8b98aa","OCCLUDED":"#ad7510","IMAGE_REVIEW":"#ad7510"}
STATES={"OK":tr('정상'),"NG":tr('불량'),"REVIEW":tr('보류'),"WAITING":tr('안정 대기'),"STABLE":tr('검사 대기'),"INSPECTING":tr('검사 중'),
        "RECHECK":tr('재검사 필요'),"LOST":tr('추적 끊김'),"OCCLUDED":tr('가림 확인'),"IMAGE_REVIEW":tr('촬영 조건 확인'),"UNCERTAIN":tr('식별 불확실')}


def button(text,slot):
    b=ui_text(QPushButton, text); b.clicked.connect(slot); return b
def table(headers):
    t=QTableWidget(0,len(headers)); ui_text(t.setHorizontalHeaderLabels, headers); t.verticalHeader().hide()
    t.setSelectionBehavior(QAbstractItemView.SelectRows); t.setSelectionMode(QAbstractItemView.SingleSelection)
    t.setEditTriggers(QAbstractItemView.NoEditTriggers); t.horizontalHeader().setStretchLastSection(True); return t
def number(value=0,low=0,high=10000,decimals=2):
    spin=QDoubleSpinBox(); spin.setDecimals(decimals); spin.setRange(low,high); spin.setValue(value or 0); return spin
def integer(value=0,low=0,high=100):
    spin=QSpinBox(); spin.setRange(low,high); spin.setValue(value or 0); return spin


class Task(QThread):
    succeeded=Signal(object); failed=Signal(str)
    def __init__(self,fn): super().__init__(); self.fn=fn
    def run(self):
        try: self.succeeded.emit(self.fn())
        except Exception as exc: self.failed.emit(str(exc))


class PathField(QWidget):
    def __init__(self,value=None,*,folder=False,filter=tr('모든 파일 (*)')):
        super().__init__(); self.folder=folder; self.filter=filter
        layout=QHBoxLayout(self); layout.setContentsMargins(0,0,0,0)
        self.edit=QLineEdit(value or ""); self.edit.setReadOnly(True); layout.addWidget(self.edit)
        layout.addWidget(button(tr('선택'),self.choose)); layout.addWidget(button(tr('해제'),self.edit.clear))
    def choose(self):
        path=QFileDialog.getExistingDirectory(self,tr('폴더 선택')) if self.folder else QFileDialog.getOpenFileName(self,tr('파일 선택'),"",self.filter)[0]
        if path: ui_text(self.edit.setText, path)
    def value(self): return self.edit.text().strip() or None


class Canvas(QWidget):
    selected=Signal(str); point_selected=Signal(object); rectangle_selected=Signal(object)
    def __init__(self):
        super().__init__(); self.setMinimumSize(400,260); self.pixmap=QPixmap(); self.tracks=[]; self.selected_id=None
        self.roi=[]; self.excluded=[]; self.edit_mode=None; self.drag_start=None; self.drag_end=None
        self.placeholder=tr('카메라 미연결\n장비 · 보정에서 카메라를 연결하세요.')
    def set_rgb(self,rgb):
        h,w=rgb.shape[:2]; self.pixmap=QPixmap.fromImage(QImage(rgb.data,w,h,rgb.strides[0],QImage.Format_RGB888).copy()); self.update()
    def set_file(self,path): self.pixmap=QPixmap(str(path)); self.update()
    def transform(self):
        if self.pixmap.isNull(): return 1.,0.,0.
        scale=min((self.width()-24)/self.pixmap.width(),(self.height()-24)/self.pixmap.height())
        return scale,(self.width()-self.pixmap.width()*scale)/2,(self.height()-self.pixmap.height()*scale)/2
    def pixel(self,position):
        if self.pixmap.isNull(): return None
        s,x,y=self.transform(); p=((position.x()-x)/s,(position.y()-y)/s)
        return p if 0<=p[0]<self.pixmap.width() and 0<=p[1]<self.pixmap.height() else None
    def mousePressEvent(self,event):
        p=self.pixel(event.position())
        if p is None: return
        if self.edit_mode=="point": self.point_selected.emit(p); return
        if self.edit_mode in {"roi","exclude"}: self.drag_start=p; self.drag_end=p; return
        matches=[t for t in self.tracks if t["box"][0]<=p[0]<t["box"][2] and t["box"][1]<=p[1]<t["box"][3]]
        if matches:
            t=min(matches,key=lambda t:(t["box"][2]-t["box"][0])*(t["box"][3]-t["box"][1]))
            self.selected_id=t["track_id"]; self.selected.emit(t["track_id"]); self.update()
    def mouseMoveEvent(self,event):
        if self.drag_start:
            p=self.pixel(event.position())
            if p: self.drag_end=p; self.update()
    def mouseReleaseEvent(self,event):
        if not self.drag_start: return
        end=self.pixel(event.position()) or self.drag_end; a=self.drag_start; self.drag_start=None; self.drag_end=None
        x1,x2=sorted((a[0],end[0])); y1,y2=sorted((a[1],end[1]))
        if x2-x1>3 and y2-y1>3: self.rectangle_selected.emit({"mode":self.edit_mode,"polygon":[[x1,y1],[x2,y1],[x2,y2],[x1,y2]]})
        self.update()
    def paintEvent(self,event):
        p=QPainter(self); p.fillRect(self.rect(),QColor(theme_color("soft")))
        if self.pixmap.isNull():
            p.setPen(QColor(theme_color("muted"))); p.drawText(self.rect(),Qt.AlignCenter,str(self.placeholder)); return
        scale,x,y=self.transform(); p.translate(x,y); p.scale(scale,scale); p.drawPixmap(0,0,self.pixmap)
        for polygon in [self.roi,*self.excluded]:
            if polygon:
                p.setPen(QPen(QColor("#637ca2"),1/scale,Qt.DashLine)); p.drawPolygon(QPolygonF([QPointF(*v) for v in polygon]))
        for t in self.tracks:
            color=QColor(COLORS.get(t["status"],"#657890")); p.setPen(QPen(color,(4 if t["track_id"]==self.selected_id else 2)/scale))
            b=t["box"]; p.drawRect(QRectF(b[0],b[1],b[2]-b[0],b[3]-b[1]))
            p.drawText(QPointF(b[0],max(14,b[1]-5)),t["track_id"].split(":")[-1]+" · "+str(STATES.get(t["status"],t["status"])))
        if self.drag_start and self.drag_end:
            p.setPen(QPen(QColor("#ee9b21"),2/scale)); p.drawRect(QRectF(QPointF(*self.drag_start),QPointF(*self.drag_end)).normalized())

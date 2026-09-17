from mes_vision.theme import color as theme_color
from mes_vision.qt_i18n import ui_text
"""Image navigation without resampling the stored source or changing image coordinates."""
import math
from PySide6.QtCore import Qt, Signal, QPointF, QRectF
from PySide6.QtGui import QPainter, QColor, QPen, QPolygonF
from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QApplication
from mes_vision.i18n import tr
from .widgets import Canvas, button, COLORS, STATES


class ImageCanvas(Canvas):
    view_changed=Signal()

    def __init__(self):
        super().__init__(); self.setMinimumSize(200,160)
        self.setFocusPolicy(Qt.StrongFocus)
        self._scale=None; self._native=False; self._center=None; self._source=None
        self._press=None; self._moved=False; self.show_overlays=True
        ui_text(self.setToolTip, tr('휠: 확대·축소 · 드래그: 이동 · 클릭: 물체 선택 · F: 화면 맞춤 · 1: 원본 배율'))

    def set_rgb(self,rgb):
        changed=(self.pixmap.width(),self.pixmap.height())!=(rgb.shape[1],rgb.shape[0])
        super().set_rgb(rgb); self._source=None
        if changed: self.fit()

    def set_file(self,path):
        changed=self._source!=str(path)
        super().set_file(path); self._source=str(path)
        if changed: self.fit()

    def fit(self):
        self._scale=None; self._native=False; self._center=None
        self.update(); self.view_changed.emit()

    def center(self,scale):
        w,h=self.pixmap.width(),self.pixmap.height()
        cx,cy=self._center or (w/2,h/2)
        half_w,half_h=self.width()/(2*scale),self.height()/(2*scale)
        return (w/2 if half_w>=w/2 else min(w-half_w,max(half_w,cx)),
                h/2 if half_h>=h/2 else min(h-half_h,max(half_h,cy)))

    def transform(self):
        if self.pixmap.isNull(): return 1.,0.,0.
        if self._scale is None and not self._native: return super().transform()
        scale=1/self.devicePixelRatioF() if self._native else self._scale
        cx,cy=self.center(scale)
        return scale,self.width()/2-cx*scale,self.height()/2-cy*scale

    def zoom(self,factor,anchor=None):
        if self.pixmap.isNull() or not math.isfinite(factor) or factor<=0: return
        anchor=anchor or QPointF(self.width()/2,self.height()/2)
        scale,x,y=self.transform(); px,py=(anchor.x()-x)/scale,(anchor.y()-y)/scale
        self._native=False; self._scale=min(32/self.devicePixelRatioF(),max(.01/self.devicePixelRatioF(),scale*factor))
        self._center=(px-(anchor.x()-self.width()/2)/self._scale,py-(anchor.y()-self.height()/2)/self._scale)
        self._center=self.center(self._scale); self.update(); self.view_changed.emit()

    def native_size(self):
        if self.pixmap.isNull(): return
        scale,x,y=self.transform()
        self._center=((self.width()/2-x)/scale,(self.height()/2-y)/scale)
        self._native=True; self.update(); self.view_changed.emit()

    def wheelEvent(self,event):
        delta=event.angleDelta().y() or event.pixelDelta().y()
        if not self.pixmap.isNull() and delta:
            self.zoom(1.2**max(-6,min(6,delta/120)),event.position()); event.accept()
        else: event.ignore()

    def mousePressEvent(self,event):
        if self.edit_mode is not None: return super().mousePressEvent(event)
        if not self.pixmap.isNull() and event.button() in (Qt.LeftButton,Qt.MiddleButton,Qt.RightButton):
            self.setFocus(); self._press=event.position(); self._last=event.position(); self._moved=False
            event.accept()

    def mouseMoveEvent(self,event):
        if self.edit_mode is not None: return super().mouseMoveEvent(event)
        if self._press is None: return
        if not self._moved and (event.position()-self._press).manhattanLength()<QApplication.startDragDistance(): return
        self._moved=True; scale,x,y=self.transform()
        self._center=((self.width()/2-x)/scale-(event.position().x()-self._last.x())/scale,
                      (self.height()/2-y)/scale-(event.position().y()-self._last.y())/scale)
        self._scale=scale; self._center=self.center(scale); self._last=event.position()
        self.setCursor(Qt.ClosedHandCursor); self.update(); self.view_changed.emit()

    def mouseReleaseEvent(self,event):
        if self.edit_mode is not None: return super().mouseReleaseEvent(event)
        if self._press is None: return
        if not self._moved and event.button()==Qt.LeftButton: super().mousePressEvent(event)
        self._press=None; self.unsetCursor(); event.accept()

    def keyPressEvent(self,event):
        if event.key()==Qt.Key_F: self.fit()
        elif event.key()==Qt.Key_1: self.native_size()
        elif event.key() in (Qt.Key_Plus,Qt.Key_Equal): self.zoom(1.2)
        elif event.key()==Qt.Key_Minus: self.zoom(1/1.2)
        else: return super().keyPressEvent(event)
        event.accept()

    def resizeEvent(self,event):
        super().resizeEvent(event); self.view_changed.emit()

    def paintEvent(self,event):
        if self.pixmap.isNull() or self.edit_mode is not None: return super().paintEvent(event)
        painter=QPainter(self); painter.fillRect(self.rect(),QColor(theme_color('soft')))
        scale,x,y=self.transform(); painter.save(); painter.translate(x,y); painter.scale(scale,scale); painter.drawPixmap(0,0,self.pixmap)
        if self.show_overlays:
            for polygon in [self.roi,*self.excluded]:
                if polygon:
                    painter.setPen(QPen(QColor('#637ca2'),1/scale,Qt.DashLine))
                    painter.drawPolygon(QPolygonF([QPointF(*point) for point in polygon]))
            for track in self.tracks:
                painter.setPen(QPen(QColor(COLORS.get(track['status'],'#657890')),(4 if track['track_id']==self.selected_id else 2)/scale))
                a,b,c,d=track['box']; painter.drawRect(QRectF(a,b,c-a,d-b))
        painter.restore()
        if self.show_overlays:
            # Labels remain readable at large magnifications without covering the defect.
            for track in self.tracks:
                painter.setPen(QColor(COLORS.get(track['status'],'#657890')))
                a,b,_,_=track['box']; label=tr(STATES.get(track['status'],track['status']) or '판정 연결 전')
                caption=track.get('label')
                from mes_vision.i18n import render_text
                painter.drawText(QPointF(x+a*scale,max(painter.fontMetrics().height(),y+b*scale-5)),
                    render_text(caption) if caption is not None else track['track_id'].split(':')[-1]+' · '+label)


class ImagePanel(QWidget):
    def __init__(self,canvas=None):
        super().__init__(); self.canvas=canvas or ImageCanvas()
        layout=QVBoxLayout(self); layout.setContentsMargins(0,0,0,0); layout.addWidget(self.canvas,1)
        controls=QHBoxLayout(); self.fit_button=button(tr('화면 맞춤'),self.canvas.fit)
        self.native_button=button('1:1',self.canvas.native_size)
        ui_text(self.native_button.setToolTip, tr('원본 픽셀 하나를 화면 픽셀 하나로 표시합니다.'))
        self.minus=button('−',lambda:self.canvas.zoom(1/1.2)); self.plus=button('+',lambda:self.canvas.zoom(1.2))
        ui_text(self.minus.setToolTip, tr('축소')); ui_text(self.plus.setToolTip, tr('확대'))
        for widget in (self.fit_button,self.native_button,self.minus,self.plus): controls.addWidget(widget)
        self.scale_label=ui_text(QLabel); controls.addWidget(self.scale_label); controls.addStretch()
        layout.addLayout(controls)
        self.overlays=ui_text(QCheckBox, tr('검사 표시선')); self.overlays.setChecked(True)
        self.overlays.toggled.connect(self.toggle_overlays); layout.addWidget(self.overlays)
        self.canvas.view_changed.connect(self.update_scale); self.update_scale()

    def toggle_overlays(self,checked): self.canvas.show_overlays=checked; self.canvas.update()

    def update_scale(self):
        valid=not self.canvas.pixmap.isNull()
        for widget in (self.fit_button,self.native_button,self.minus,self.plus,self.overlays): widget.setEnabled(valid)
        ui_text(self.scale_label.setText, f'{self.canvas.transform()[0]*self.canvas.devicePixelRatioF()*100:.0f}%' if valid else '—')
        ui_text(self.scale_label.setToolTip, f'{self.canvas.pixmap.width()} × {self.canvas.pixmap.height()}' if valid else '')

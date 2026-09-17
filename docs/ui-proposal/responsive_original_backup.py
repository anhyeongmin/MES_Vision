from mes_vision.qt_i18n import ui_text, bound_text
"""Keep navigation and stop controls accessible on smaller operator displays."""
from PySide6.QtCore import Qt,QRect,QSize,QTimer
from PySide6.QtWidgets import QLayout,QScrollArea,QFrame,QComboBox,QSizePolicy


class FlowLayout(QLayout):
    def __init__(self):
        super().__init__(); self.items=[]; self.setContentsMargins(0,0,0,0); self.setSpacing(6)
    def addItem(self,item): self.items.append(item)
    def count(self): return len(self.items)
    def itemAt(self,index): return self.items[index] if 0<=index<len(self.items) else None
    def takeAt(self,index): return self.items.pop(index) if 0<=index<len(self.items) else None
    def addStretch(self,*_): pass
    def expandingDirections(self): return Qt.Orientation(0)
    def hasHeightForWidth(self): return True
    def heightForWidth(self,width): return self.arrange(QRect(0,0,width,0),False)
    def setGeometry(self,rect): super().setGeometry(rect); self.arrange(rect,True)
    def sizeHint(self): return self.minimumSize()
    def minimumSize(self):
        size=QSize()
        for item in self.items:
            if not item.isEmpty(): size=size.expandedTo(item.minimumSize())
        return size
    def arrange(self,rect,apply):
        x,y=rect.x(),rect.y(); line=0; spacing=self.spacing()
        for item in self.items:
            if item.isEmpty(): continue
            size=item.sizeHint(); width=size.width()
            if x>rect.x() and x+width>rect.x()+rect.width(): x=rect.x(); y+=line+spacing; line=0
            if apply: item.setGeometry(QRect(x,y,width,size.height()))
            x+=width+spacing; line=max(line,size.height())
        return y+line-rect.y()


def scroll_page(content):
    area=QScrollArea(); area.setWidgetResizable(True); area.setFrameShape(QFrame.NoFrame)
    area.setMinimumSize(0,0); area.setWidget(content)
    return area


class ResponsiveWindow:
    def init_compact_navigation(self,header):
        self.compact_nav=QComboBox(); ui_text(self.compact_nav.addItems, [bound_text(self.nav.item(i)) for i in range(self.nav.count())])
        self.compact_nav.setMinimumContentsLength(16)
        self.compact_nav.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        ui_text(self.compact_nav.setToolTip, bound_text(self,'setWindowTitle')); header.insertWidget(1,self.compact_nav)
        self.compact_nav.currentIndexChanged.connect(self.nav.setCurrentRow)
        self.nav.currentRowChanged.connect(self.compact_nav.setCurrentIndex)
        self.compact_nav.setCurrentIndex(self.nav.currentRow()); self.compact_nav.hide()
        self._readiness_user_choice=False
        self.readiness_toggle.clicked.connect(self.remember_readiness_choice)

    def remember_readiness_choice(self,*_): self._readiness_user_choice=True

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.update_responsive_layout()

    def update_responsive_layout(self):
        if not hasattr(self,'compact_nav'): return
        scale=self.preferences.value['text_scale']/100
        compact=self.width()<1300*scale
        self.nav.setVisible(not compact);
        if hasattr(self,"sidebar"): self.sidebar.setVisible(not compact); self.compact_logo.setVisible(compact)
        self.compact_nav.setVisible(compact); self.title_label.setVisible(not compact)
        if not self._readiness_user_choice: self.readiness_toggle.setChecked(not compact)
        if hasattr(self,'history_table'): self.history_table.setMaximumHeight(120 if compact else 220)
        if hasattr(self,'history_advanced'): self.history_advanced.setMaximumHeight(round(220*scale))

    def fit_available_screen(self):
        screen=self.screen()
        if screen is None: return
        available=screen.availableGeometry()
        usable_width=max(1,available.width()-max(0,self.frameGeometry().width()-self.width()))
        usable_height=max(1,available.height()-max(0,self.frameGeometry().height()-self.height()))
        self.setMinimumSize(min(960,usable_width),min(600,usable_height))
        self.resize(min(self.width(),usable_width),min(self.height(),usable_height))
        frame=self.frameGeometry()
        self.move(max(available.left(),min(frame.left(),available.right()-frame.width()+1)),
                  max(available.top(),min(frame.top(),available.bottom()-frame.height()+1)))
        self.update_responsive_layout()

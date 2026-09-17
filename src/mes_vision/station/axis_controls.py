"""Explicit step and absolute axis commands inside the robot sidebar page."""
from copy import deepcopy
import math
import time
from PySide6.QtWidgets import (QWidget,QVBoxLayout,QHBoxLayout,QGridLayout,QLabel,
    QPushButton,QLineEdit,QDoubleSpinBox,QCheckBox,QTabWidget,QMessageBox)
from mes_vision.training.data import require
from mes_vision.robot.contracts import Pose,JointPose
from .teaching import axes_target


class AxisControls(QWidget):
    def __init__(self,host):
        super().__init__(); self.host=host; self.groups={}; self.epoch=None
        layout=QVBoxLayout(self)
        note=QLabel('± 버튼: 한 번 클릭한 만큼 이동 · 목표 수치: 입력 후 이동 버튼으로 실행\n'
                    '축 제어는 이동 높이를 거치지 않는 직접 이동입니다. 빈 집게로 경로를 확인하세요.')
        note.setWordWrap(True); layout.addWidget(note)
        self.tabs=QTabWidget(); layout.addWidget(self.tabs)
        for space,title,keys in (('cartesian','X / Y / Z / R',tuple('xyzr')),('joint','J1 / J2 / J3 / J4',('j1','j2','j3','j4'))):
            page=QWidget(); out=QVBoxLayout(page); grid=QGridLayout()
            group={'keys':keys,'rows':{},'buttons':[]}; self.groups[space]=group
            for col,label in enumerate(('축','현재값','−','+','목표 수치','실행','허용 최소','허용 최대')):
                grid.addWidget(QLabel(label),0,col)
            for i,key in enumerate(keys,1):
                unit='°' if space=='joint' or key=='r' else 'mm'
                live=QLabel('—'); target=QLineEdit(); target.setPlaceholderText(unit)
                lower=QLineEdit(); upper=QLineEdit(); lower.setPlaceholderText('최소'); upper.setPlaceholderText('최대')
                for edit in (target,lower,upper): edit.setMaximumWidth(140)
                minus=QPushButton('−'); plus=QPushButton('+'); go=QPushButton('이 축 이동')
                minus.setAutoRepeat(False); plus.setAutoRepeat(False)
                minus.clicked.connect(lambda checked=False,s=space,k=key:self.host.guard(lambda:self.step(s,k,-1)))
                plus.clicked.connect(lambda checked=False,s=space,k=key:self.host.guard(lambda:self.step(s,k,1)))
                go.clicked.connect(lambda checked=False,s=space,k=key:self.host.guard(lambda:self.absolute(s,k)))
                for col,w in enumerate((QLabel(key.upper()+' ('+unit+')'),live,minus,plus,target,go,lower,upper)):
                    grid.addWidget(w,i,col)
                group['rows'][key]={'live':live,'target':target,'lower':lower,'upper':upper}
                group['buttons'] += [minus,plus,go]
            out.addLayout(grid); row=QHBoxLayout()
            step=QDoubleSpinBox(); step.setRange(.1,5); step.setDecimals(1); step.setValue(1)
            speed=QDoubleSpinBox(); speed.setRange(.1,10); speed.setDecimals(1); speed.setValue(5)
            speed.setSuffix(' °/s' if space=='joint' else ' mm/s (R: °/s)')
            row.addWidget(QLabel('1회 이동량 (°)' if space=='joint' else '1회 이동량 (XYZ: mm / R: °)')); row.addWidget(step)
            row.addWidget(QLabel('속도')); row.addWidget(speed); out.addLayout(row)
            group.update(step=step,speed=speed)
            row=QHBoxLayout(); fill=QPushButton('현재값을 목표칸에 채우기'); all_go=QPushButton('네 축 입력값으로 이동…')
            fill.clicked.connect(lambda checked=False,s=space:self.host.guard(lambda:self.fill(s)))
            all_go.clicked.connect(lambda checked=False,s=space:self.host.guard(lambda:self.absolute(s)))
            row.addWidget(fill); row.addWidget(all_go); out.addLayout(row); group['buttons'] += [fill,all_go]
            info=QLabel('조인트 이동은 직선 경로가 아닙니다. 관절별 허용 범위는 장착물과 실물 가동 범위를 확인해 입력하세요.'
                        if space=='joint' else '도봇 base 좌표 기준 직선 이동입니다. R은 회전각이며 J4 관절각과 구분합니다.')
            info.setWordWrap(True); out.addWidget(info); out.addStretch(1); self.tabs.addTab(page,title)
        self.armed=QCheckBox('빈 집게 · 손과 Unlock 버튼 놓음 · 현재 이동 경로 확인 (직접 제어 허용)')
        layout.insertWidget(1,self.armed)
        self.tabs.currentChanged.connect(lambda _:self.host.stop())

    def observed(self,space):
        require(self.host.ready(),'새 위치 응답을 확인한 후 제어하세요.')
        state=self.host.last_state
        values=state.get('joints') if space=='joint' else state.get('status',{}).get('pose')
        require(values is not None,'현재 축 값을 읽을 수 없습니다.')
        return values

    def number(self,edit,label):
        try:
            value=float(edit.text().strip())
            if not math.isfinite(value): raise ValueError()
            return value
        except ValueError:
            self.host.scroll.ensureWidgetVisible(edit); edit.setFocus(); edit.selectAll()
            raise ValueError(label+'을 숫자로 입력하세요.') from None

    def fill(self,space):
        values=self.observed(space)
        for key,row in self.groups[space]['rows'].items(): row['target'].setText(format(values[key],'.8g'))

    def payload(self,space,**extra):
        current=self.observed(space); require(self.armed.isChecked(),'직접 제어 허용 확인란을 먼저 체크하세요.')
        group=self.groups[space]; bounds={}
        for key,row in group['rows'].items():
            lo=self.number(row['lower'],key.upper()+' 허용 최소값'); hi=self.number(row['upper'],key.upper()+' 허용 최대값')
            require(lo<hi,key.upper()+' 허용 최소값은 최대값보다 작아야 합니다.')
            bounds[key]=[lo,hi]
        return {'space':space,'epoch':self.host.last_state['status']['connection_epoch'],
                'expected':deepcopy(current),'issued_at':time.monotonic(),'bounds':bounds,
                'speed':group['speed'].value(),'empty_tool_confirmed':True,'path_confirmed':True,**extra}

    def validate(self,payload):
        state=self.host.last_state
        return axes_target(payload,Pose(**state['status']['pose']),
                           JointPose(**state['joints']) if state.get('joints') else None,
                           state['status']['connection_epoch'],time.monotonic())

    def step(self,space,key,direction):
        payload=self.payload(space,mode='step',axis=key,delta=direction*self.groups[space]['step'].value())
        self.validate(payload); self.host.request('teach_axes',**payload)

    def absolute(self,space,key=None):
        group=self.groups[space]; keys=[key] if key is not None else group['keys']
        target={k:self.number(group['rows'][k]['target'],k.upper()+' 목표값') for k in keys}
        payload=self.payload(space,mode='absolute',target=target); resolved=self.validate(payload)
        details=' / '.join(f'{k.upper()} {getattr(resolved,k):.2f}' for k in group['keys'])
        text=('조인트 궤적' if space=='joint' else '직선 경로')+'으로 직접 이동합니다.\n'+details+'\n전체 이동 경로를 확인했습니까?'
        if QMessageBox.question(self,'수치 입력 이동',text,QMessageBox.Yes|QMessageBox.No,QMessageBox.No)==QMessageBox.Yes:
            payload['issued_at']=time.monotonic()
            self.host.request('teach_axes',**payload)

    def update_state(self):
        state=self.host.last_state; epoch=state.get('status',{}).get('connection_epoch')
        fresh=state.get('state')=='TEACHING' and time.monotonic()-self.host.received<1.2
        if epoch!=self.epoch or not fresh:
            self.epoch=epoch; self.armed.setChecked(False)
        for space,group in self.groups.items():
            values=state.get('joints') if space=='joint' else state.get('status',{}).get('pose')
            for key,row in group['rows'].items(): row['live'].setText(f'{values[key]:.2f}' if fresh and values else '—')
            for button in group['buttons']: button.setEnabled(self.host.ready() and values is not None)

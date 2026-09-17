"""Robot sidebar panel; dialog wrapper retained for isolated component tests."""
from copy import deepcopy
import json
import math
import time

from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QPainter, QColor
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QComboBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QMessageBox, QDoubleSpinBox, QGroupBox, QScrollArea, QWidget, QHeaderView,QTabWidget)

from mes_vision.operation.image_view import ImagePanel, ImageCanvas
from mes_vision.robot.magician import Magician
from mes_vision.training.data import require
from .robot_port import StationRobotPort
from .teaching import ROLES, CORNERS, TeachingStore


class CrosshairCanvas(ImageCanvas):
    def paintEvent(self,event):
        super().paintEvent(event)
        if self.pixmap.isNull(): return
        # Image center, accounting for zoom/pan through the canvas transform.
        scale,ox,oy=self.transform()
        x=ox+self.pixmap.width()*scale/2; y=oy+self.pixmap.height()*scale/2
        painter=QPainter(self); painter.setPen(QColor('#00ff88'))
        painter.drawLine(int(x-14),int(y),int(x+14),int(y))
        painter.drawLine(int(x),int(y-14),int(x),int(y+14))


class TeachingDialog(QDialog):
    def __init__(self,runtime,equipment,parent=None,*,embedded=False):
        super().__init__(parent)
        self.embedded=embedded
        if embedded: self.setWindowFlags(Qt.Widget)
        self.runtime=runtime; self.equipment=deepcopy(equipment); self.station=parent
        self.worker=None; self.pending=False; self.closing=False; self.received=0
        self.awaiting_status=False
        self.last_state={}; self.draft_key=None
        self.setWindowTitle('Dobot 위치 티칭 · 저장 / 시험 이동'); self.resize(1150,850)
        outer=QVBoxLayout(self)
        scroll=QScrollArea(); scroll.setWidgetResizable(True); body=QWidget(); layout=QVBoxLayout(body)
        self.scroll=scroll
        scroll.setWidget(body); outer.addWidget(scroll)
        note=QLabel('로봇 연결 후 축 제어(± / 수치 입력) 또는 위치 티칭을 선택하세요.\n'
                    '저장은 이동하지 않습니다. HOME은 저장한 대기 자세입니다. 자동 Homing은 하지 않습니다.\n'
                    '직접 축 제어와 저장 위치 이동은 별도입니다. 카메라는 R축과 함께 회전하지 않는 설치입니다.')
        note.setWordWrap(True); layout.addWidget(note)
        row=QHBoxLayout(); self.ports=QComboBox(); row.addWidget(self.ports,1)
        self.refresh_button=self.button('포트 새로고침',self.refresh_ports,row)
        self.connect_button=self.button('로봇 연결',self.connect_robot,row)
        self.disconnect_button=self.button('연결 해제',self.disconnect_robot,row)
        layout.addLayout(row)
        self.identity={}; identity=QGridLayout()
        camera=self.equipment['camera']
        for i,(key,label,default) in enumerate((
            ('installation_id','설치 ID',self.equipment.get('id','main')),
            ('robot_base_id','로봇 기준 ID',camera.get('robot_base_id') or 'base-initial'),
            ('tool_frame_id','집게 기준 ID',camera.get('tool_frame_id') or 'tool-initial'),
            ('mount_revision','카메라 장착 버전',camera.get('mount_revision') or 'mount-initial'))):
            edit=QLineEdit(default); self.identity[key]=edit
            identity.addWidget(QLabel(label),i//2,(i%2)*2); identity.addWidget(edit,i//2,(i%2)*2+1)
        layout.addLayout(identity)
        self.pose_label=QLabel('현재 위치: 미연결'); self.pose_label.setWordWrap(True); layout.addWidget(self.pose_label)
        self.feedback=QLabel('기존 자동운전 프로필 없이 위치를 초안으로 저장할 수 있습니다.'); self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        self.modes=QTabWidget(); layout.addWidget(self.modes)
        teaching_body=QWidget(); layout=QVBoxLayout(teaching_body)
        middle=QHBoxLayout(); self.preview=ImagePanel(canvas=CrosshairCanvas(),title='카메라 / 중심 십자선',compact=True)
        self.preview.setMinimumSize(280,220); middle.addWidget(self.preview,1)
        saves=QGroupBox('현재 위치를 이름별로 저장'); grid=QGridLayout(saves); self.save_buttons=[]
        for i,(role,label) in enumerate(ROLES.items()):
            btn=QPushButton(label+' 저장'); btn.clicked.connect(lambda checked=False,r=role:self.guard(lambda:self.save(r)))
            grid.addWidget(btn,i//2,i%2); self.save_buttons.append(btn)
        self.offset_button=self.button('두 기준점으로 카메라 오프셋 계산',lambda:self.request('teach_offset'),grid,(7,0,1,2))
        middle.addWidget(saves,2); layout.addLayout(middle)
        if parent is not None:
            self.button('카메라 연결 / 영상 확인',self.connect_camera,layout)
        info=QLabel('작업판 모서리는 동일한 집게 기준점으로 등록하세요. 네 점은 자동 이동 허용 영역이 아닙니다.\n'
                    '오프셋 두 위치는 같은 물리 기준점에 맞추세요. ΔZ에는 촬영/집기 높이 차이가 포함됩니다. 자동 보정으로 적용하지 않습니다.')
        info.setWordWrap(True); layout.addWidget(info)
        self.points=QTableWidget(0,7); self.points.setHorizontalHeaderLabels(['위치','X','Y','Z','R','측정 시각 (UTC)','상태'])
        self.points.setSelectionBehavior(QTableWidget.SelectRows); self.points.setSelectionMode(QTableWidget.SingleSelection)
        self.points.setEditTriggers(QTableWidget.NoEditTriggers); self.points.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.points.setMinimumHeight(210); layout.addWidget(self.points)
        row=QHBoxLayout(); self.delete_button=self.button('선택 위치 삭제',self.delete,row)
        self.move_button=self.button('선택 위치로 시험 이동…',self.request_move,row); layout.addLayout(row)
        group=QGroupBox('시험 이동 조건 · 실물로 확인한 범위를 입력'); limits=QGridLayout(group); self.bounds={}
        for i,axis in enumerate('xyzr'):
            lo=QLineEdit(); hi=QLineEdit(); lo.setPlaceholderText('최소'); hi.setPlaceholderText('최대')
            self.bounds[axis]=(lo,hi); limits.addWidget(QLabel(axis.upper()+(' (deg)' if axis=='r' else ' (mm)')),i,0)
            limits.addWidget(lo,i,1); limits.addWidget(hi,i,2)
        self.speed=QDoubleSpinBox(); self.speed.setRange(.1,10); self.speed.setValue(5); self.speed.setSuffix(' mm/s')
        limits.addWidget(QLabel('직선 이동 속도'),4,0); limits.addWidget(self.speed,4,1)
        text=QLabel('현재 XY에서 저장한 이동 높이로 상승 → 목표 XY로 이동 → 목표 Z로 하강.\n'
                    '범위 입력은 충돌·도달 가능성 검증을 대신하지 않습니다. 재연결 후 기존 점은 다시 저장해야 이동할 수 있습니다.')
        text.setWordWrap(True); limits.addWidget(text,5,0,1,3); layout.addWidget(group)
        from .axis_controls import AxisControls
        self.axes=AxisControls(self)
        self.modes.addTab(self.axes,'축 제어'); self.modes.addTab(teaching_body,'위치 티칭 / 카메라')
        self.modes.currentChanged.connect(lambda _:self.stop())
        outer.addWidget(self.feedback)
        bottom=QHBoxLayout(); stop=self.button('로봇 정지',self.stop,bottom); stop.setObjectName('dangerButton')
        if not embedded: self.button('닫기',self.reject,bottom)
        outer.addLayout(bottom)
        self.refresh_ports()
        saved=TeachingStore(runtime,{}).value
        for key,edit in self.identity.items():
            if saved.get('context',{}).get(key): edit.setText(saved['context'][key])
        self.show_draft(saved,None)
        for control in self.findChildren(QPushButton):
            control.setAutoDefault(False); control.setDefault(False)
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(150); self.tick()

    def button(self,text,fn,layout,position=None):
        button=QPushButton(text); button.clicked.connect(lambda checked=False:self.guard(fn))
        if position: layout.addWidget(button,*position)
        else: layout.addWidget(button)
        return button

    def guard(self,fn):
        try: fn()
        except Exception as exc: self.feedback.setText(str(exc))

    def refresh_ports(self):
        require(self.worker is None,'연결을 해제한 후 포트를 바꾸세요.')
        from serial.tools import list_ports
        self.ports.clear()
        for p in list_ports.comports(): self.ports.addItem(f'{p.device} · {p.description}',{'port':p.device,'hardware_id':p.hwid})
        if not self.ports.count(): self.feedback.setText('COM 포트 없음: 도봇 전원·USB·드라이버를 확인하고 새로고침하세요.')

    def connect_robot(self):
        require(self.worker is None,'이미 연결 중입니다.')
        if self.station is not None:
            self.station.require_idle(robot=True)
            require(not self.station.running,'검사를 중단한 후 로봇 제어를 연결하세요.')
            self.equipment=deepcopy(self.station.equipment)
        device=self.ports.currentData(); require(device is not None,'도봇 COM 포트를 선택하세요.')
        context={key:edit.text().strip() for key,edit in self.identity.items()}
        require(all(context.values()),'설치와 기준 ID를 입력하세요.')
        context.update(device); context['camera_rotates_with_r']=False
        context['camera_id']=self.equipment['camera'].get('serial','')
        context['acquisition_revision']=self.equipment['camera'].get('acquisition_revision','')
        self.equipment['robot']['port']=device['port']
        worker=StationRobotPort(self.runtime,self.equipment,teaching_context=context)
        try:
            worker.changed.connect(self.changed); worker.completed.connect(self.completed)
            worker.failed.connect(self.failed); worker.finished.connect(self.worker_finished)
            self.worker=worker; self.last_state={}; self.received=0; self.draft_key=None
            self.feedback.setText('연결 중… 정지와 현재 위치를 확인합니다.'); worker.start()
        except Exception:
            self.worker=None; worker.deleteLater(); self.tick()
            raise
        self.tick()

    def disconnect_robot(self):
        if self.worker: self.worker.stop_motion(); self.worker.stopping.set()

    def connect_camera(self):
        if self.station.camera is None: self.station.connect_camera()

    def changed(self,state):
        self.last_state=state; self.received=time.monotonic(); self.awaiting_status=False
        if 'draft' in state: self.show_draft(state['draft'],state['status']['connection_epoch'])
        self.tick()

    def completed(self,event):
        self.pending=False; self.awaiting_status=True
        self.feedback.setText(event.get('error') or event.get('message','완료')); self.tick()

    def failed(self,message):
        self.pending=False; self.feedback.setText('연결/통신 오류: '+message)

    def worker_finished(self):
        worker=self.worker; self.worker=None; self.pending=False; self.last_state={}; self.tick()
        if worker: worker.deleteLater()
        if self.closing: super().reject()

    def request(self,kind,**payload):
        require(self.worker is not None and self.ready(),'최신 위치를 확인할 수 없습니다. 연결과 정지 상태를 확인하세요.')
        self.worker.request(kind,**payload); self.pending=True; self.tick()

    def save(self,role):
        overwrite=role in self.draft.get('points',{})
        if overwrite and QMessageBox.question(self,'위치 덮어쓰기',ROLES[role]+'을 새로 읽은 현재 위치로 덮어쓸까요?')!=QMessageBox.Yes: return
        self.request('teach_save',role=role,overwrite=overwrite)

    def selected(self):
        row=self.points.currentRow(); require(row>=0,'저장 목록에서 위치를 선택하세요.')
        return self.points.item(row,0).data(Qt.UserRole)

    def delete(self):
        role=self.selected()
        if QMessageBox.question(self,'위치 삭제',ROLES[role]+'을 삭제할까요?')==QMessageBox.Yes: self.request('teach_delete',role=role)

    def read_bounds(self):
        bounds={}
        for axis,fields in self.bounds.items():
            values=[]
            for name,edit in zip(('최소','최대'),fields):
                text=edit.text().strip()
                try:
                    value=float(text)
                    if not math.isfinite(value): raise ValueError()
                except ValueError:
                    self.modes.setCurrentIndex(1)
                    self.scroll.ensureWidgetVisible(edit); edit.setFocus(); edit.selectAll()
                    raise ValueError(f'시험 이동 조건의 {axis.upper()} {name}값을 숫자로 입력하세요. '
                                     '위치 저장에는 이 입력이 필요하지 않습니다.') from None
                values.append(value)
            if values[0]>=values[1]:
                self.modes.setCurrentIndex(1)
                self.scroll.ensureWidgetVisible(fields[0]); fields[0].setFocus()
                raise ValueError(f'{axis.upper()} 최소값은 최대값보다 작아야 합니다.')
            bounds[axis]=values
        return bounds

    def request_move(self):
        role=self.selected(); require(role not in CORNERS and role!='travel','모서리와 이동 높이는 목적지가 아닙니다.')
        require(self.worker is not None and self.ready(),'최신 위치를 확인한 후 이동하세요.')
        status=self.last_state['status']
        from mes_vision.robot.contracts import Pose
        store=TeachingStore(self.runtime,self.worker.teaching_context)
        # Explain stale points and missing travel teaching before asking for limits.
        store.point(role,status['connection_epoch']); store.point('travel',status['connection_epoch'])
        bounds=self.read_bounds()
        path=store.route(role,Pose(**status['pose']),status['connection_epoch'],bounds)
        expected_start=deepcopy(status['pose']); draft_revision=store.value.get('updated_at')
        coords='\n'.join(f'{i+1}. X {p.x:.1f} / Y {p.y:.1f} / Z {p.z:.1f} / R {p.r:.1f}' for i,p in enumerate(path))
        answer=QMessageBox.question(self,'빈 집게 시험 이동',
            f'{ROLES[role]}로 {self.speed.value():.1f} mm/s 직선 이동합니다.\n{coords}\n\n'
            '집게가 비어 있고, 손·Unlock 버튼을 놓았으며, 전체 경로에 카메라·케이블·장애물 간섭이 없습니까?',
            QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer==QMessageBox.Yes:
            self.request('teach_move',role=role,bounds=bounds,speed=self.speed.value(),
                         empty_tool_confirmed=True,path_confirmed=True,expected_start=expected_start,
                         draft_revision=draft_revision)

    def stop(self):
        self.axes.armed.setChecked(False)
        if self.worker: self.worker.stop_motion()
        self.pending=False; self.feedback.setText('정지 요청됨. 실제 장비 정지를 확인하세요.')

    def ready(self):
        return (self.last_state.get('state')=='TEACHING' and time.monotonic()-self.received<1.2
                and not self.last_state.get('busy') and not self.pending and not self.awaiting_status)

    def tick(self):
        connected=self.worker is not None; ready=self.ready()
        self.connect_button.setEnabled(not connected); self.refresh_button.setEnabled(not connected)
        self.ports.setEnabled(not connected); self.disconnect_button.setEnabled(connected)
        for edit in self.identity.values(): edit.setEnabled(not connected)
        for btn in [*self.save_buttons,self.move_button,self.delete_button,self.offset_button]: btn.setEnabled(ready)
        self.axes.update_state()
        status=self.last_state.get('status',{}); pose=status.get('pose')
        if pose and time.monotonic()-self.received<1.2 and self.last_state.get('state')=='TEACHING':
            self.pose_label.setText('현재 위치: '+ ' · '.join(f'{k.upper()} {pose[k]:.1f}' for k in 'xyzr')
                +'  |  '+('이동 중' if self.last_state.get('busy') else '수동 제어 대기')+'  |  '+self.last_state.get('observed_at',''))
        else: self.pose_label.setText('현재 위치: 미연결 또는 응답 지연 (이전 좌표 사용 불가)')
        if self.station is not None:
            frame=getattr(self.station,'frame',None)
            if frame is not None and time.monotonic()-self.station.frame_received<1:
                self.preview.canvas.set_rgb(frame.rgb)
                h,w=frame.rgb.shape[:2]; c=self.equipment['camera']
                self.preview.set_title('카메라 / 영상 중심',f"설정 {c['width']}×{c['height']} / 현재 영상 {w}×{h}")
            else: self.preview.set_title('카메라 / 새 영상 없음','카메라 연결을 확인하세요.')

    def show_draft(self,value,epoch):
        key=json.dumps(value,sort_keys=True)+str(epoch)
        if key==self.draft_key: return
        self.draft_key=key; self.draft=value; self.points.setRowCount(len(value['points']))
        for row,(role,p) in enumerate(value['points'].items()):
            valid=p.get('epoch')==epoch and self.worker is not None and p.get('context')==self.worker.teaching_context
            values=[ROLES.get(role,role),*[f"{p['pose'][k]:.2f}" for k in 'xyzr'],p['observed_at'],
                    '초안 / 이번 연결' if valid else '초안 / 재티칭 필요']
            for col,text in enumerate(values):
                item=QTableWidgetItem(text); item.setData(Qt.UserRole,role); self.points.setItem(row,col,item)

    def reject(self):
        if self.embedded:
            self.stop(); return
        if self.worker:
            self.closing=True; self.disconnect_robot(); self.feedback.setText('로봇 연결 종료 중…'); return
        super().reject()

    def closeEvent(self,event):
        if self.worker: event.ignore(); self.reject()
        else: super().closeEvent(event)

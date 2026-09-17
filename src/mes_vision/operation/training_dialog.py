from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import tr, trf
from pathlib import Path
from uuid import uuid4
import sys
from PySide6.QtCore import QProcess,QProcessEnvironment,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog,QVBoxLayout,QFormLayout,QHBoxLayout,QComboBox,QLabel,QPlainTextEdit
from mes_vision.training.config import JobConfig
from mes_vision.training.data import read_json,write_json,require
from .widgets import button,PathField,integer


class TrainingDialog(QDialog):
    def __init__(self,root,runtime,parent=None):
        super().__init__(parent); self.root=Path(root); self.runtime=Path(runtime); self.busy=False; self.output=None; self.action=None; self.labelers=[]
        ui_text(self.setWindowTitle, tr('데이터 · 모델 학습')); self.resize(1000,780); layout=QVBoxLayout(self)
        layout.addWidget(ui_text(QLabel, tr('실물 데이터 준비 → 라벨 검토·내보내기 → 학습 → 별도 평가 → 품목 설정에서 모델 등록')))
        layout.addWidget(button(tr('데이터 수집·라벨링 열기'),self.labeler))
        f=QFormLayout(); layout.addLayout(f); self.role=QComboBox(); ui_text(self.role.addItems, [tr('물체 검출'),tr('불량 검출'),tr('정상 특징 구축')]); ui_text(f.addRow, tr('작업 종류'),self.role)
        self.dataset=PathField(folder=True); ui_text(f.addRow, tr('검토 완료 데이터 내보내기 폴더'),self.dataset)
        self.epochs=integer(100,1,10000); self.batch=integer(2,1,32); self.accum=integer(4,1,64)
        ui_text(f.addRow, tr('전체 학습 횟수'),self.epochs); ui_text(f.addRow, tr('한 번에 처리할 이미지'),self.batch); ui_text(f.addRow, tr('누적 횟수'),self.accum)
        self.previous=PathField(folder=True); ui_text(f.addRow, tr('재개할 이전 학습 폴더 (선택)'),self.previous)
        self.evaluation=PathField(folder=True); ui_text(f.addRow, tr('평가할 학습 결과 폴더'),self.evaluation)
        row=QHBoxLayout()
        for label,fn in [(tr('데이터 확인'),lambda:self.launch("validate")),(tr('학습 시작'),lambda:self.launch("train")),(tr('별도 평가'),lambda:self.launch("evaluate")),
                         (tr('저장 후 중단'),self.stop),(tr('작업 강제 종료'),self.cancel),(tr('최근 결과 폴더'),self.open_output)]:
            row.addWidget(button(label,lambda checked=False,fn=fn:self.guard(fn)))
        layout.addLayout(row); self.message=ui_text(QLabel); self.message.setWordWrap(True); layout.addWidget(self.message)
        self.log=QPlainTextEdit(); self.log.setReadOnly(True); self.log.setMaximumBlockCount(2500); layout.addWidget(self.log,1)
        self.process=QProcess(self); self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.read_output); self.process.finished.connect(self.finished)
        self.process.errorOccurred.connect(self.process_error)
    def guard(self,fn):
        try: fn()
        except Exception as exc: ui_text(self.message.setText, str(exc))
    def labeler(self):
        from mes_vision.data_management.labeler import Labeler
        window=Labeler(None,allow_synthetic=False,runtime=self.runtime); self.labelers.append(window); window.show()
    def launch(self,action):
        require(not self.busy,tr('현재 작업이 끝난 뒤 실행하세요.'))
        parent=self.parent()
        require(parent.engine is None and not parent.queue.enabled(),tr('모델을 해제하고 VLM을 OFF로 설정하세요.'))
        role=["object_detector","known_defect_detector","normal"][self.role.currentIndex()]
        job=self.runtime/"training-jobs"/uuid4().hex; job.mkdir(parents=True); self.output=job/"result"
        if action=="evaluate":
            require(self.evaluation.value(),tr('평가할 학습 결과를 선택하세요.'))
            previous=read_json(Path(self.evaluation.value())/"run.json"); require(not previous["config"]["synthetic"],tr('실물 학습 결과를 선택하세요.'))
            script="training.py"; arguments=["evaluate","--run",self.evaluation.value(),"--output",str(self.output),"--split","test"]
        elif role=="normal":
            require(action=="train",tr('정상 특징은 검토된 정상 내보내기 폴더를 선택한 뒤 학습 시작을 누르세요.'))
            require(self.dataset.value(),tr('정상 참조 내보내기 폴더를 선택하세요.'))
            script="anomaly.py"; arguments=["build","--normal-export",self.dataset.value(),"--output",str(self.output)]
        else:
            require(self.dataset.value(),tr('학습 데이터 폴더를 선택하세요.'))
            path=self.root/"configs/training"/("objects.json" if role=="object_detector" else "defects.json")
            template=JobConfig.load(path).snapshot(); template.update(dataset_dir=self.dataset.value(),epochs=self.epochs.value(),batch_size=self.batch.value(),grad_accum_steps=self.accum.value(),synthetic=False)
            config=job/"config.json"; JobConfig(**template); write_json(config,template)
            script="training.py"; arguments=[action,"--config",str(config)]
            if action=="train":
                arguments.extend(["--output",str(self.output)])
                if self.previous.value(): arguments.extend(["--resume-from",self.previous.value()])
        self.action=action; self.busy=True; ui_text(self.log.clear); ui_text(self.message.setText, tr('작업 시작 중…'))
        environment=QProcessEnvironment.systemEnvironment(); environment.insert("PYTHONUTF8","1"); environment.insert("PYTHONUNBUFFERED","1")
        self.process.setProcessEnvironment(environment); self.process.setWorkingDirectory(str(self.root))
        executable=Path(sys.executable)
        if executable.name.lower()=="pythonw.exe": executable=executable.with_name("python.exe")
        self.process.start(str(executable),[str(self.root/"scripts/operation_job.py"),"--runtime",str(self.runtime),"--script",script,"--",*arguments])
    def read_output(self): self.log.appendPlainText(bytes(self.process.readAllStandardOutput()).decode("utf-8",errors="replace"))
    def finished(self,code,status):
        self.read_output(); self.busy=False; ui_text(self.message.setText, tr('작업 완료. 결과를 검토한 뒤 품목 설정에서 등록하세요.') if code==0 else tr('작업이 완료되지 않았습니다. 아래 오류와 저장된 상태를 확인하세요.'))
        if code==0 and self.action=="train": ui_text(self.evaluation.edit.setText, str(self.output))
    def process_error(self,error):
        if self.process.state()==QProcess.NotRunning: self.busy=False
        ui_text(self.message.setText, tr('학습 프로세스 오류: ')+self.process.errorString())
    def stop(self):
        require(self.busy and self.action=="train" and self.output and (self.output/"run.json").exists(),tr('학습 초기화가 끝난 뒤 저장 후 중단할 수 있습니다.'))
        (self.output/"STOP").touch(exist_ok=True); ui_text(self.message.setText, tr('현재 학습 회차를 저장한 뒤 중단하도록 요청했습니다.'))
    def cancel(self):
        if self.busy: self.process.kill(); ui_text(self.message.setText, tr('작업 종료 요청됨. 완료되지 않은 결과는 등록하지 마세요.'))
    def open_output(self):
        require(self.output is not None,tr('실행한 작업이 없습니다.')); QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.output.parent)))
    def closeEvent(self,event):
        if self.busy: event.ignore(); ui_text(self.message.setText, tr('작업 중단 또는 완료 후 닫으세요.')); return
        for window in self.labelers:
            if not window.close(): event.ignore(); return
        event.accept()

from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
from mes_vision.i18n import tr, trf
"""Operator initiated storage maintenance; long file operations run off the UI thread."""
from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
from PySide6.QtCore import Qt,QTimer,QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (QDialog,QVBoxLayout,QHBoxLayout,QFormLayout,QWidget,QLabel,
    QPlainTextEdit,QCheckBox,QFileDialog,QMessageBox,QProgressBar,QTableWidgetItem)
from .widgets import Task,button,integer,number,PathField,table
from .maintenance import Maintenance


def gib(size): return f"{size/1024**3:,.2f} GiB"


class StorageDialog(QDialog):
    def __init__(self,store,application_root,parent=None):
        super().__init__(parent); self.service=Maintenance(store); self.application_root=Path(application_root)
        self.busy=False; self.job=None; self.preview=None; self.restored=None
        ui_text(self.setWindowTitle, tr('저장 · 보관 · 백업')); self.resize(980,790); self.setWindowModality(Qt.ApplicationModal)
        outer=QVBoxLayout(self); self.body=QWidget(); layout=QVBoxLayout(self.body); outer.addWidget(self.body)
        heading=ui_text(QLabel, tr('검사 원본과 판정 근거를 함께 보존합니다. 정리는 미리보기를 확인한 뒤 직접 실행합니다.'))
        heading.setWordWrap(True); layout.addWidget(heading)
        self.usage_label=ui_text(QLabel, tr('용량 확인 전')); self.usage_label.setWordWrap(True); layout.addWidget(self.usage_label)
        row=QHBoxLayout(); row.addWidget(button(tr('용량 확인'),lambda:self.run(self.service.usage,self.show_usage)))
        row.addWidget(button(tr('이전 기록 색인 갱신'),lambda:self.run(self.service.rebuild_index)))
        row.addWidget(button(tr('중단된 보관 작업 복구'),lambda:self.run(self.service.repair_archive))); layout.addLayout(row)
        form=QFormLayout(); policy=self.service.policy(); self.days={}
        for key,label in (("days_ok",tr('정상 원본 보존일')),("days_ng",tr('불량 원본 보존일')),("days_review",tr('보류 원본 보존일')),("orphan_days",tr('미등록 저장물 보존일'))):
            self.days[key]=integer(policy[key],1,36500); ui_text(form.addRow, label,self.days[key])
        self.capacity=number(policy["max_active_gib"],.01,1000000); self.free=number(policy["min_free_gib"],.01,1000000)
        ui_text(form.addRow, tr('원본 용량 알림 기준 (GiB)'),self.capacity); ui_text(form.addRow, tr('검사 저장 최소 여유 (GiB)'),self.free)
        self.protect=ui_text(QCheckBox, tr('미검토 보류·미등록 이상 보호')); self.protect.setChecked(policy["protect_unreviewed"]); ui_text(form.addRow, self.protect)
        self.archive_path=PathField(policy["archive_directory"],folder=True); ui_text(form.addRow, tr('보관 폴더'),self.archive_path); layout.addLayout(form)
        row=QHBoxLayout(); row.addWidget(button(tr('보존 설정 저장'),self.save_policy)); row.addWidget(button(tr('정리 미리보기'),self.make_preview))
        self.apply=button(tr('검증 후 보관 이동'),self.archive); self.apply.setEnabled(False); row.addWidget(self.apply); layout.addLayout(row)
        self.candidates=table([tr('구분'),tr('검사 ID / 폴더'),tr('원본 용량')]); self.candidates.setMaximumHeight(140); layout.addWidget(self.candidates)
        note=ui_text(QLabel, tr('보관 이동 후에도 이력과 원본을 열 수 있습니다. 같은 드라이브로 옮기면 디스크의 남은 용량은 늘지 않습니다.'))
        note.setWordWrap(True); layout.addWidget(note)
        row=QHBoxLayout()
        for label,slot in ((tr('백업 만들기'),self.backup),(tr('백업 검증'),self.verify),(tr('새 폴더에 복원'),self.restore)):
            row.addWidget(button(label,slot))
        self.open_restored=button(tr('복원본 실행'),self.launch_restored); self.open_restored.setEnabled(False); row.addWidget(self.open_restored); layout.addLayout(row)
        self.output=QPlainTextEdit(); self.output.setReadOnly(True); self.output.setMaximumHeight(130); layout.addWidget(self.output)
        self.progress=QProgressBar(); self.progress.setRange(0,0); self.progress.hide(); outer.addWidget(self.progress)
        self.message=ui_text(QLabel, tr('백업에는 운영 기록과 등록된 모델·기준 파일이 포함됩니다. 프로그램과 공통 기본 모델은 별도 설치가 필요합니다.'))
        self.message.setWordWrap(True); outer.addWidget(self.message); outer.addWidget(button(tr('닫기'),self.reject))

    def run(self,fn,done=None):
        if self.busy: return
        self.busy=True; self.body.setEnabled(False); self.progress.show(); ui_text(self.message.setText, tr('처리 중입니다. 파일 크기에 따라 시간이 걸릴 수 있습니다.'))
        self.job=Task(fn); self.job.succeeded.connect(done or self.report); self.job.failed.connect(self.failed)
        self.job.finished.connect(self.task_finished); self.job.start()

    def task_finished(self):
        self.busy=False; self.body.setEnabled(True); self.progress.hide(); self.job=None

    def failed(self,message):
        ui_text(self.message.setText, tr('작업을 완료하지 못했습니다. ')+str(message)); ui_text(self.output.setPlainText, str(message))

    def report(self,result):
        ui_text(self.message.setText, tr('처리가 완료됐습니다.'))
        labels={"path":tr('저장 위치'),"runtime":tr('복원 운영 폴더'),"files":tr('파일 수'),"bytes":tr('용량'),"archived":tr('보관한 원본 수'),"indexed":tr('색인 갱신 수'),"captures":tr('복원한 원본 수'),"operations":tr('보관 작업 기록')}
        if isinstance(result,list):
            lines=[f"{r['id']}: "+(tr('복구 완료') if r['state']=='COMPLETED' else tr('확인 필요 · ')+r.get('error','')) for r in result]
            ui_text(self.output.setPlainText, text_join('\n', lines) or tr('중단된 보관 작업이 없습니다.')); return
        if isinstance(result,dict):
            lines=[]
            for key,value in result.items():
                if key=="errors": lines.extend(tr('확인 필요: ')+r.get('error','') for r in value)
                elif key!="operations": lines.append(f"{labels.get(key,key)}: {gib(value) if key=='bytes' else value}")
            ui_text(self.output.setPlainText, text_join('\n', lines)); return
        ui_text(self.output.setPlainText, tr('완료'))

    def show_usage(self,result):
        ui_text(self.usage_label.setText, trf('운영 폴더 {v0} / 검사 원본 {v1} / 디스크 여유 {v2}', v0=gib(result['runtime_bytes']), v1=gib(result['active_bytes']), v2=gib(result['free_bytes']))
            + (tr(' · 원본 용량 기준 초과') if result['active_limit_exceeded'] else "") + (tr(' · 남은 공간 부족') if result['free_space_low'] else ""))
        ui_text(self.message.setText, tr('용량을 확인했습니다.'))

    def save_policy(self):
        value={key:widget.value() for key,widget in self.days.items()}
        value.update(protect_unreviewed=self.protect.isChecked(),max_active_gib=self.capacity.value(),min_free_gib=self.free.value(),archive_directory=self.archive_path.value())
        self.preview=None; self.apply.setEnabled(False)
        self.run(lambda:self.service.save_policy(value),lambda _:ui_text(self.message.setText, tr('보존 설정을 저장했습니다. 자동 정리는 실행되지 않습니다.')))

    def make_preview(self):
        self.run(self.service.preview_archive,self.show_preview)

    def show_preview(self,result):
        self.preview=result; self.candidates.setRowCount(len(result["candidates"]))
        for i,item in enumerate(result["candidates"]):
            for j,value in enumerate((tr('검사 원본') if item["run_id"] else tr('미등록 저장물'),item["run_id"] or Path(item["source"]).name,gib(item["bytes"]))):
                self.candidates.setItem(i,j,ui_text(QTableWidgetItem, value))
        self.apply.setEnabled(bool(result["candidates"]))
        ui_text(self.message.setText, trf('이동 대상 {v0}개 · 보호 {v1}개 · 확인 필요 {v2}개', v0=len(result['candidates']), v1=len(result['protected']), v2=len(result['errors'])))
        lines=[trf('보호 · {v0}: {v1}', v0=r['run_id'], v1=tr(r['reason'])) for r in result['protected']]
        lines += [trf('확인 필요 · {v0}: {v1}', v0=r['path'], v1=r['error']) for r in result['errors']]
        ui_text(self.output.setPlainText, text_join('\n', lines) or tr('보호 또는 오류 항목이 없습니다.'))

    def archive(self):
        if not self.preview or self.busy: return
        items=self.preview["candidates"]
        answer=QMessageBox.question(self,tr('보관 이동'),trf('원본 폴더 {v0}개 ({v1})를\n{v2}\n로 복사하고 검증한 뒤 운영 폴더의 중복 원본을 정리합니다. 진행할까요?', v0=len(items), v1=gib(sum((i['bytes'] for i in items))), v2=self.preview['policy']['archive_directory']),QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
        if answer!=QMessageBox.Yes: return
        preview=self.preview; self.preview=None; self.apply.setEnabled(False)
        self.run(lambda:self.service.archive(preview))

    def new_folder(self,title,prefix):
        # A name dialog is used to select a non-existent output directory; services enforce no overwrite.
        return QFileDialog.getSaveFileName(self,title,str(self.service.root.parent/(prefix+datetime.now().strftime("-%Y%m%d-%H%M%S"))),tr('폴더 이름 (*)'),options=QFileDialog.DontConfirmOverwrite)[0]

    def backup(self):
        path=self.new_folder(tr('새 백업 폴더 이름'),"MES-backup")
        if path: self.run(lambda:self.service.backup(path))

    def verify(self):
        path=QFileDialog.getExistingDirectory(self,tr('검증할 백업 폴더'))
        if path: self.run(lambda:{tr('검증'):tr('통과'),tr('파일 수'):len(self.service.validate_backup(path)["files"]),tr('백업'):path})

    def restore(self):
        backup=QFileDialog.getExistingDirectory(self,tr('복원할 백업 폴더'))
        if not backup: return
        destination=self.new_folder(tr('새 복원 폴더 이름'),"MES-restored")
        if destination: self.run(lambda:self.service.restore(backup,destination),self.restore_done)

    def restore_done(self,result):
        self.restored=result; self.open_restored.setEnabled(True); self.report(result)
        ui_text(self.message.setText, tr('새 폴더에 복원했습니다. VLM은 OFF이며 로봇 동작은 재개되지 않습니다. 현재 운영 데이터는 그대로입니다.'))

    def launch_restored(self):
        if self.restored:
            subprocess.Popen([sys.executable,str(self.application_root/"scripts/desktop.py"),"--runtime",self.restored["runtime"]],cwd=self.application_root,
                creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
            ui_text(self.message.setText, tr('복원본을 별도 창으로 실행했습니다. 장비 연결과 운전 기준을 확인한 뒤 사용하세요.'))

    def reject(self):
        if self.busy: ui_text(self.message.setText, tr('저장 작업이 완료된 뒤 닫을 수 있습니다.')); return
        super().reject()

    def closeEvent(self,event):
        if self.busy: event.ignore(); ui_text(self.message.setText, tr('저장 작업이 완료된 뒤 닫을 수 있습니다.'))
        else: event.accept()

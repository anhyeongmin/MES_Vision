"""Saved image model inspection; inference stays in a cancellable child process."""
from pathlib import Path
import os, sys, time
from uuid import uuid4
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QDialog,QVBoxLayout,QHBoxLayout,QLabel,QComboBox,QFileDialog,QSplitter,QWidget,QPlainTextEdit,QTableWidgetItem,QMessageBox,QHeaderView
from mes_vision.i18n import tr,trf,text_join
from mes_vision.qt_i18n import ui_text
from mes_vision.training.data import read_json,write_json,sha256,require
from mes_vision.vlm.snapshots import load_snapshot
from mes_vision.operation.widgets import button,table
from mes_vision.operation.image_view import ImagePanel
from mes_vision.operation.responsive import FlowLayout
from mes_vision.operation.finding_display import object_overlays,finding_caption
from .photo_inspection import open_results


class PhotoInspectionDialog(QDialog):
    def __init__(self,root,runtime,parent=None):
        super().__init__(parent); self.root=Path(root); self.runtime=Path(runtime)
        self.process=None; self.pending_close=False; self.cancelled=False; self.report=None; self.record=None; self.inspection=None
        self.jobs=[]; self.selected_id=None; self.sample_mode=False
        ui_text(self.setWindowTitle,tr('저장사진 검사')); self.resize(1320,820); self.setMinimumSize(950,600)
        layout=QVBoxLayout(self); row=FlowLayout()
        self.role=QComboBox(); ui_text(self.role.addItem,tr('상세사진 · 불량 검사'),'detail'); ui_text(self.role.addItem,tr('전체사진 · 위치 검출'),'overview')
        self.kind=QComboBox(); ui_text(self.kind.addItem,tr('촬영 사진'),'real'); ui_text(self.kind.addItem,tr('합성 사진'),'synthetic')
        self.choose=button(tr('사진 선택'),self.choose_photos); self.samples=button(tr('학습 자료 불러오기'),self.choose_samples)
        self.run_button=button(tr('사진 검사 시작'),lambda:self.guard(self.start)); self.stop_button=button(tr('검사 중단'),self.stop)
        self.open_button=button(tr('저장 결과 열기'),self.choose_results)
        for w in (self.role,self.kind,self.choose,self.samples,self.run_button,self.stop_button,self.open_button): row.addWidget(w)
        layout.addLayout(row)
        note=ui_text(QLabel,tr('전체사진은 위치만 검출합니다. 상세사진은 불량 후보를 표시하며, 실물 판정 기준이 등록되기 전에는 보류로 남깁니다.'))
        note.setWordWrap(True); layout.addWidget(note)
        self.status=ui_text(QLabel,tr('사진을 선택하세요.')); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.files=table([tr('사진'),tr('촬영 구분'),tr('물체 수')]); self.files.setMaximumHeight(130)
        self.files.horizontalHeader().setStretchLastSection(False)
        self.files.horizontalHeader().setSectionResizeMode(0,QHeaderView.Stretch)
        self.files.horizontalHeader().setSectionResizeMode(1,QHeaderView.ResizeToContents)
        self.files.horizontalHeader().setSectionResizeMode(2,QHeaderView.ResizeToContents)
        self.files.itemSelectionChanged.connect(self.select_photo); layout.addWidget(self.files)
        panes=QSplitter(); layout.addWidget(panes,1)
        left=QWidget(); l=QVBoxLayout(left); l.setContentsMargins(0,0,0,0)
        l.addWidget(ui_text(QLabel,tr('저장 원본'))); self.original=ImagePanel(); l.addWidget(self.original,1)
        self.original.canvas.selected.connect(self.select_object)
        self.objects=table([tr('물체 ID'),tr('판정'),tr('불량 후보')]); self.objects.setMaximumHeight(120)
        self.objects.itemSelectionChanged.connect(self.select_object_row); l.addWidget(self.objects); panes.addWidget(left)
        right=QWidget(); r=QVBoxLayout(right); r.setContentsMargins(0,0,0,0)
        r.addWidget(ui_text(QLabel,tr('선택 물체 확대'))); self.detail=ImagePanel(); r.addWidget(self.detail,1)
        self.reasons=QPlainTextEdit(); self.reasons.setReadOnly(True); self.reasons.setMaximumHeight(190); r.addWidget(self.reasons)
        r.addWidget(ui_text(QLabel,tr('VLM 설명이 없어도 기본 판정과 이유를 확인할 수 있습니다.')))
        panes.addWidget(right); panes.setSizes([720,520])
        self.timer=QTimer(self); self.timer.setInterval(200); self.timer.timeout.connect(self.poll)
        self.set_busy(False)

    def guard(self,fn):
        try: return fn()
        except Exception as exc:
            ui_text(self.status.setText,str(exc)); QMessageBox.warning(self,tr('설정 확인'),str(exc))

    def set_busy(self,value):
        for widget in (self.role,self.kind,self.choose,self.samples,self.run_button,self.open_button): widget.setEnabled(not value)
        self.kind.setEnabled(not value and not self.sample_mode)
        self.stop_button.setEnabled(value)

    def choose_photos(self):
        paths,_=QFileDialog.getOpenFileNames(self,tr('사진 선택'),str(self.root),'Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)')
        if paths:
            self.sample_mode=False; self.kind.setEnabled(True)
            self.jobs=[{'path':str(Path(p).resolve()),'role':self.role.currentData()} for p in paths]
            self.image_kind=self.kind.currentData(); self.show_jobs()

    def choose_samples(self):
        def load():
            spec=read_json(self.root/'configs/inspection/ash-photo-samples.json')
            self.jobs=[dict(row,path=str(self.root/row['path'])) for row in spec['images']]
            self.sample_mode=True; self.kind.setEnabled(False)
            self.kind.setCurrentIndex(self.kind.findData('synthetic')); self.image_kind='synthetic'; self.show_jobs()
        self.guard(load)

    def show_jobs(self):
        self.report=None; self.clear_selection(); self.files.blockSignals(True); self.files.setRowCount(len(self.jobs))
        for i,job in enumerate(self.jobs):
            for j,value in enumerate((Path(job['path']).name,tr('전체사진') if job['role']=='overview' else tr('상세사진'),'—')):
                self.files.setItem(i,j,ui_text(QTableWidgetItem,value))
        self.files.blockSignals(False)
        ui_text(self.status.setText,trf('선택한 사진 {count}장 · 검사 시작을 누르세요.',count=len(self.jobs)))

    def clear_selection(self):
        self.record=None; self.inspection=None; self.selected_id=None
        for panel in (self.original,self.detail):
            panel.canvas.pixmap=QPixmap(); panel.canvas.tracks=[]; panel.canvas.selected_id=None; panel.canvas.fit()
        self.objects.setRowCount(0); self.reasons.clear()

    def start(self):
        require(self.process is None and self.jobs,tr('사진을 선택하세요.'))
        jobs=[]
        for job in self.jobs:
            row=dict(job); actual=sha256(Path(row['path']))
            require('sha256' not in row or row['sha256']==actual,'Sample photograph changed')
            row['sha256']=actual; jobs.append(row)
        directory=self.runtime/'photo-inspections'; directory.mkdir(parents=True,exist_ok=True)
        identity=uuid4().hex; self.output=directory/identity
        request=directory/(identity+'-request.json')
        write_json(request,{'images':jobs,'image_kind':'synthetic' if self.sample_mode else self.kind.currentData(),'bundle':str(self.root/'configs/inspection/ash-trained-v3.json'),
                            'runtime':str(self.runtime),'output':str(self.output)})
        self.clear_selection(); self.report=None; self.cancelled=False; self.pending_close=False
        process=QProcess(self); self.process=process; process.setWorkingDirectory(str(self.root))
        environment=QProcessEnvironment.systemEnvironment(); environment.insert('PYTHONIOENCODING','utf-8')
        process.setProcessEnvironment(environment)
        process.setStandardOutputFile(str(directory/(identity+'-stdout.log')))
        process.setStandardErrorFile(str(directory/(identity+'-stderr.log')))
        process.finished.connect(self.process_finished); process.errorOccurred.connect(self.process_error)
        self.last_progress=None; self.progress_at=time.monotonic(); self.set_busy(True); self.timer.start()
        ui_text(self.status.setText,tr('모델 준비 및 사진 검사 중…'))
        process.start(sys.executable,[str(self.root/'scripts/inspect_photos.py'),'--request',str(request),'--watch-parent',str(os.getpid())])

    def poll(self):
        if not self.process: return
        try:
            state=read_json(self.output/'status.json'); key=(state['state'],state['completed'])
            if key!=self.last_progress: self.progress_at=time.monotonic(); self.last_progress=key
            ui_text(self.status.setText,trf('사진 검사 {done} / {total}',done=state['completed'],total=state['total']))
        except FileNotFoundError: pass
        except Exception as exc:
            self.stop(); ui_text(self.status.setText,str(exc)); return
        if time.monotonic()-self.progress_at>180:
            self.stop(); ui_text(self.status.setText,tr('사진 검사 응답 시간이 초과됐습니다.'))

    def stop(self):
        if self.process:
            self.cancelled=True; self.process.kill(); self.stop_button.setEnabled(False)

    def process_error(self,error):
        if error==QProcess.FailedToStart: self.process_finished(-1,QProcess.CrashExit)

    def process_finished(self,code,status):
        if self.process is None: return
        self.timer.stop(); process=self.process; self.process=None; process.deleteLater(); self.set_busy(False)
        if self.cancelled: ui_text(self.status.setText,tr('사진 검사를 중단했습니다.'))
        elif code==0 and status==QProcess.NormalExit:
            self.guard(lambda:self.load_results(self.output/'results.json'))
        else:
            message=tr('사진 검사에 실패했습니다. 결과 폴더의 로그를 확인하세요.')
            try: message=message+'\n'+read_json(self.output/'status.json').get('error','')
            except (OSError,ValueError): pass
            ui_text(self.status.setText,message)
        if self.pending_close: super().reject()

    def reject(self):
        if self.process:
            self.pending_close=True; self.stop(); return
        super().reject()

    def closeEvent(self,event):
        if self.process:
            event.ignore(); self.pending_close=True; self.stop()
        else: super().closeEvent(event)

    def choose_results(self):
        path,_=QFileDialog.getOpenFileName(self,tr('저장 결과 열기'),str(self.runtime/'photo-inspections'),'Results (results.json)')
        if path: self.guard(lambda:self.load_results(path))

    def load_results(self,path):
        self.clear_selection(); self.report=None
        directory,report=open_results(path); self.results_root=directory; self.report=report
        self.kind.setCurrentIndex(self.kind.findData(report['image_kind']))
        self.sample_mode=report['image_kind']=='synthetic'; self.kind.setEnabled(not self.sample_mode)
        self.jobs=[{'role':r['role'],'path':r['source_path'],'sha256':r['source_sha256']} for r in report['records']]
        self.files.blockSignals(True); self.files.setRowCount(len(report['records']))
        for i,row in enumerate(report['records']):
            for j,value in enumerate((row['name'],tr('전체사진') if row['role']=='overview' else tr('상세사진'),str(row['objects']))):
                self.files.setItem(i,j,ui_text(QTableWidgetItem,value))
        self.files.blockSignals(False)
        ui_text(self.status.setText,trf('사진 검사 완료 · {count}장',count=len(report['records'])))
        if report['records']:
            self.files.blockSignals(True); self.files.selectRow(0); self.files.blockSignals(False); self.select_photo()

    def select_photo(self):
        if not self.report: return
        self.guard(self._select_photo)

    def _select_photo(self):
        self.clear_selection(); index=self.files.currentRow()
        if not 0<=index<len(self.report['records']): return
        record=self.report['records'][index]; path=self.results_root/record['snapshot']
        manifest,result,_=load_snapshot(path,expected_digest=record['snapshot_digest'])
        self.record=record; self.inspection=result; self.manifest=manifest
        self.role.setCurrentIndex(self.role.findData(record['role']))
        self.original.canvas.set_file(path/'frame.png')
        self.objects.blockSignals(True); self.objects.setRowCount(len(result['objects'])); tracks=[]
        for i,obj in enumerate(result['objects']):
            overlays=object_overlays(obj)
            if record['role']=='overview':
                for overlay in overlays: overlay.update(status='INSPECTING',label=obj['object_id'].split(':')[-1])
            tracks.extend(overlays)
            candidates=sorted({f['defect_code'] for c in obj['checks'] for f in c['findings'] if f.get('defect_code')})
            values=(obj['object_id'].split(':')[-1],tr('위치 검출') if record['role']=='overview' else tr('판정 보류'),', '.join(candidates) or '—')
            for j,value in enumerate(values): self.objects.setItem(i,j,ui_text(QTableWidgetItem,value))
        self.objects.blockSignals(False); self.original.canvas.tracks=tracks; self.original.canvas.update()
        if result['objects']: self.select_object(result['objects'][0]['object_id'])
        else: ui_text(self.reasons.setPlainText,tr('물체 미검출 · 빈 작업대로 확정할 수 없음'))

    def select_object_row(self):
        if self.inspection and 0<=self.objects.currentRow()<len(self.inspection['objects']):
            self.select_object(self.inspection['objects'][self.objects.currentRow()]['object_id'])

    def select_object(self,identity):
        if not self.inspection: return
        obj=next((o for o in self.inspection['objects'] if o['object_id']==identity),None)
        if obj is None: return
        self.selected_id=identity; self.original.canvas.selected_id=identity; self.original.canvas.update()
        self.objects.blockSignals(True); self.objects.selectRow(self.inspection['objects'].index(obj)); self.objects.blockSignals(False)
        item=next(o for o in self.manifest['objects'] if o['object_id']==identity)
        self.detail.canvas.set_file(self.results_root/self.record['snapshot']/item['file'])
        self.detail.canvas.tracks=object_overlays(obj,crop=True); self.detail.canvas.selected_id=identity; self.detail.canvas.update()
        if self.record['role']=='overview':
            for overlay in self.detail.canvas.tracks: overlay.update(status='INSPECTING',label=identity.split(':')[-1])
        if self.record['role']=='overview': lines=[tr('위치 검출 · 전체사진으로 품질을 판정하지 않습니다.')]
        else:
            lines=[tr('최종 판정: 보류 · 실물 판정 기준 미등록')]
            if not self.record['association_confirmed']:
                lines.append(tr('상세사진에서 선택한 물체 하나를 확실하게 식별하지 못했습니다.'))
            else:
                findings=[f for c in obj['checks'] for f in c['findings']]
                lines.extend([tr('불량 후보:')]+[finding_caption(f) for f in findings] if findings else [tr('알려진 불량 후보 없음 · 정상 확정 아님')])
            for check in obj['checks']:
                if check['status']=='ERROR': lines.append(tr('검사 오류')+' · '+check['check_id'])
        lines.append(tr('표시 영역은 AI 예측입니다. 정답 영역이 아닙니다.'))
        ui_text(self.reasons.setPlainText,text_join('\n',lines))

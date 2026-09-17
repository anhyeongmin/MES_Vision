"""Hand-moved camera testing: arrival buttons replace robot events only."""
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from uuid import uuid4
from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QWidget, QPlainTextEdit, QTextEdit, QTableWidgetItem, QFileDialog, QCheckBox, QDoubleSpinBox)
from mes_vision.operation.widgets import button, table
from mes_vision.operation.image_view import ImagePanel
from mes_vision.operation.finding_display import object_overlays, finding_summary, finding_details_html
from mes_vision.training.data import read_json, write_json, sha256, require
from mes_vision.vlm.snapshots import load_snapshot
from .photo_inspection import open_results
from .manual_capture import Rectifier, fresh_after, save_capture


class ManualInspectionDialog(QDialog):
    def __init__(self, root, runtime, settings, parent=None, *, camera=None):
        super().__init__(parent)
        self.root, self.runtime = Path(root), Path(runtime)
        self.settings = deepcopy(settings)
        self.camera, self.owns_camera = camera, camera is None
        self.process = self.pending = self.selected = self.cycle = None
        self.server=None; self.server_bundle=None; self.request_id=None
        self.objects, self.details = [], {}
        self.overview_capture = None
        self.closing = self.cancelled = False
        self.close_ready = False
        self.vlm_worker = None
        self.preload_owner = uuid4().hex
        self.prepare_vlm_after_models = False
        self.vlm_preload_wanted = False
        self.vlm_model_loaded = False
        self.vlm_jobs = set()
        self.vlm_next_poll = 0
        self.server_allows_vlm = False
        self.vlm_owner = parent if parent is not None and hasattr(parent, 'toggle_vlm') else None
        from mes_vision.vlm.queue import AnalysisQueue
        self.vlm_queue = self.vlm_owner.queue if self.vlm_owner is not None else AnalysisQueue(self.runtime/'vlm')
        self.preview_key = None
        acquisition=self.settings.get('inspection_acquisition')
        self.rectifier = Rectifier(acquisition['calibration_file'] if acquisition else self.root/'artifacts/camera-calibration/video-20260914/exploratory-calibration.json')
        if acquisition: require(self.rectifier.digest==acquisition['calibration_sha256'],'Lens calibration file changed')
        self.bundle_path = self.root/'configs/inspection/ash-wall-reviewed-v2.json'
        self.bundle = read_json(self.bundle_path)
        self.setWindowTitle('MES Vision · 수동 도착 검사')
        self.resize(1240, 900)
        layout = QVBoxLayout(self)
        notice = QLabel('로봇 연결·이동 없음 | U20CAM 1280×720 임시 왜곡보정 적용 | 좌표는 보정 영상의 픽셀입니다.\n'
                        '풀뷰에서 촬영 → 물체 번호 선택 → 카메라를 해당 물체 위로 옮겨 고정 → 상세 촬영')
        notice.setWordWrap(True); layout.addWidget(notice)
        row = QHBoxLayout(); layout.addLayout(row)
        self.connect_button = button('카메라 연결', lambda: self.guard(self.connect_camera))
        self.full_button = button('① 풀뷰 도착 · 촬영', lambda: self.guard(lambda: self.arrive('overview')))
        self.detail_button = button('② 선택 물체 도착 · 상세검사', lambda: self.guard(lambda: self.arrive('detail')))
        self.stop_button = button('취소', self.stop)
        self.reset_button = button('새 회차', self.reset_cycle)
        for b in (self.connect_button, self.full_button, self.detail_button, self.stop_button, self.reset_button): row.addWidget(b)
        self.settle = QDoubleSpinBox(); self.settle.setRange(.5, 5); self.settle.setValue(1.5); self.settle.setSuffix('초 고정 대기')
        row.addWidget(self.settle)
        self.square = QCheckBox('중앙 1:1 크롭'); self.square.setChecked(True); row.addWidget(self.square)
        comparison_row=QHBoxLayout(); layout.addLayout(comparison_row)
        self.overview_inspect_button=button('③ 풀뷰로 전체 검사',lambda:self.guard(self.capture_and_inspect_overview))
        comparison_row.addWidget(self.overview_inspect_button)
        comparison_hint=QLabel('현재 풀뷰 촬영부터 모든 물체 검사까지 한 번에 진행합니다')
        comparison_hint.setWordWrap(True);comparison_row.addWidget(comparison_hint,1)
        settings_row = QHBoxLayout(); layout.addLayout(settings_row)
        self.bundle_button = button('모델 묶음 선택', lambda: self.guard(self.choose_bundle))
        self.defect_button = button('불량 모델 교체(.pth)', lambda: self.guard(self.choose_defects))
        settings_row.addWidget(self.bundle_button); settings_row.addWidget(self.defect_button)
        self.prepare_button=button('모델 미리 로드',lambda:self.guard(self.prepare_models))
        self.release_button=button('모델 해제',self.release_models)
        settings_row.addWidget(self.prepare_button); settings_row.addWidget(self.release_button)
        self.model_label = QLabel(); self.model_label.setWordWrap(True); settings_row.addWidget(self.model_label, 1)
        self.refresh_model_label()
        self.status = QLabel('카메라를 연결하고 풀뷰 위치로 옮겨주세요.'); self.status.setWordWrap(True); layout.addWidget(self.status)
        panes = QSplitter(); layout.addWidget(panes, 1)
        left = QWidget(); ll = QVBoxLayout(left); panes.addWidget(left)
        ll.addWidget(QLabel('저장된 전체사진 · 물체를 누르면 상세 결과 표시'))
        self.overview = ImagePanel(); ll.addWidget(self.overview, 1)
        self.overview.canvas.placeholder = '풀뷰 도착 버튼으로 전체사진을 촬영하세요.'
        self.overview.canvas.selected.connect(self.select_object)
        self.table = table(['물체', '중심 X / Y (px)', '상세 결과']); self.table.setMaximumHeight(150)
        self.table.itemSelectionChanged.connect(self.select_row); ll.addWidget(self.table)
        right = QSplitter(Qt.Vertical); panes.addWidget(right)
        self.preview = ImagePanel(square=False); self.preview.setMinimumHeight(240); self.preview.canvas.placeholder = '왜곡보정 실시간 영상'; right.addWidget(self.preview)
        detail_box = QWidget(); dl = QVBoxLayout(detail_box); right.addWidget(detail_box)
        self.detail_title=QLabel('선택 물체의 저장된 상세사진');dl.addWidget(self.detail_title)
        self.detail = ImagePanel(square=False); self.detail.setMinimumHeight(260); dl.addWidget(self.detail, 1)
        self.detail.canvas.placeholder = '물체 번호를 선택하고 상세사진을 촬영하세요.'
        self.reasons = QTextEdit(); self.reasons.setReadOnly(True); self.reasons.setMaximumHeight(85); dl.addWidget(self.reasons)
        vlm_row = QHBoxLayout(); dl.addLayout(vlm_row)
        self.vlm_enabled = QCheckBox('VLM 보조 설명'); self.vlm_enabled.setChecked(self.vlm_queue.enabled())
        self.vlm_enabled.toggled.connect(lambda enabled: self.vlm_guard(lambda: self.toggle_manual_vlm(enabled)))
        vlm_row.addWidget(self.vlm_enabled)
        self.vlm_model_status = QLabel('VLM 모델: 미로드'); vlm_row.addWidget(self.vlm_model_status)
        self.vlm_request = button('선택 사진 분석 / 재시도', lambda: self.vlm_guard(lambda: self.request_vlm(self.selected, retry=True)))
        vlm_row.addWidget(self.vlm_request)
        self.vlm_text = QPlainTextEdit(); self.vlm_text.setReadOnly(True); self.vlm_text.setMaximumHeight(85)
        self.vlm_text.setPlainText('VLM 보조 정보 · 기본 판정을 변경하지 않습니다.')
        dl.addWidget(self.vlm_text)
        panes.setSizes([620, 620]); right.setChildrenCollapsible(False)
        right.setStretchFactor(0, 1); right.setStretchFactor(1, 2)
        right.setSizes([280, 520])
        self.timer = QTimer(self); self.timer.timeout.connect(lambda: self.guard(self.tick)); self.timer.start(60)
        self.update_controls()

    def guard(self, fn):
        try: fn()
        except Exception as exc:
            self.pending = None; self.status.setText(str(exc)); self.update_controls()

    def refresh_model_label(self):
        self.model_label.setText('전체: '+Path(self.bundle['models']['overview']['weights']).parent.name+
            ' / 불량: '+str(Path(self.bundle['models']['defects']['weights']))+'\n미검증 후보 모델 · 최종 판정은 보류')

    def choose_bundle(self):
        path, _ = QFileDialog.getOpenFileName(self, '검사 모델 묶음', str(self.root/'configs/inspection'), '*.json')
        if path:
            data = read_json(path)
            require(data.get('purpose') == 'saved_photo_inspection' and set(data['models']) == {'overview','objects','defects'}, '검사 모델 묶음이 아닙니다.')
            self.bundle_path, self.bundle = Path(path), data; self.refresh_model_label()

    def choose_defects(self):
        path, _ = QFileDialog.getOpenFileName(self, '완료된 불량 추론 모델 선택 (학습 중 파일 제외)', str(self.root/'artifacts/training'), '*.pth')
        if path:
            spec = self.bundle['models']['defects']
            spec.update(weights=str(Path(path).resolve()), sha256=sha256(Path(path)))
            self.refresh_model_label()

    def connect_camera(self):
        from mes_vision.operation.camera import CameraProcess
        require(self.camera is None or self.camera.disposed, '카메라가 이미 연결되어 있습니다.')
        require(self.owns_camera and self.settings.get('serial'), '본 프로그램의 장비 설정에서 U20CAM을 먼저 등록하세요.')
        require(self.settings.get('driver') == 'uvc', '이 보정 데이터는 U20CAM용입니다.')
        self.camera = CameraProcess(self.settings)
        self.camera.failed.connect(lambda message: self.guard(lambda: self.camera_error(message)))
        self.camera.start(); self.status.setText('카메라 연결 중…')

    def camera_error(self, message):
        self.stop(); self.status.setText(message)

    def update_controls(self):
        busy = self.pending is not None or self.process is not None or self.closing
        live = self.camera is not None and not self.camera.disposed and not self.camera.stopping.is_set()
        self.connect_button.setEnabled(self.owns_camera and not live and not busy)
        self.full_button.setEnabled(live and not busy)
        self.detail_button.setEnabled(live and not busy and self.selected is not None)
        self.overview_inspect_button.setEnabled(live and not busy)
        self.stop_button.setEnabled((busy or self.vlm_preload_wanted) and not self.closing)
        self.reset_button.setEnabled(not busy)
        self.table.setEnabled(not busy)
        for widget in (self.bundle_button, self.defect_button, self.square): widget.setEnabled(not busy and not self.objects)
        self.settle.setEnabled(not busy)
        self.prepare_button.setEnabled(not busy)
        self.release_button.setEnabled(not busy and (self.server is not None or self.vlm_preload_wanted or self.vlm_model_loaded))

    def release_models(self, *, release_vlm=True):
        if self.process is not None: return
        if release_vlm: self.release_vlm_preload(force=True)
        if self.server is not None:
            self.cancelled=True; self.process=self.server; self.server.kill()
            self.status.setText('모델 메모리 해제 중…'); self.update_controls()

    def prepare_models(self):
        require(self.process is None and self.pending is None,'현재 작업이 끝난 뒤 로드하세요.')
        if self.server is not None:
            if self.server_bundle==self.bundle:
                self.vlm_guard(self.prepare_vlm_model); return
            self.release_models()
            require(self.server.waitForFinished(3000) if self.server else True,'이전 모델 종료 대기 중입니다.')
        self.cancelled=False; self.request_id=None
        self.prepare_vlm_after_models=self.vlm_queue.enabled()
        self.start_server(); self.process=self.server; self.started=time.monotonic(); self.update_controls()

    def reset_cycle(self):
        if self.pending is not None or self.process is not None: return
        self.vlm_guard(self.cancel_vlm_jobs)
        self.objects, self.details, self.selected, self.cycle = [], {}, None, None
        self.overview_capture=None
        self.table.setRowCount(0); self.reasons.clear()
        for panel in (self.overview, self.detail):
            panel.canvas.pixmap=QPixmap(); panel.canvas.tracks=[]; panel.canvas.selected_id=None; panel.canvas.update()
        self.status.setText('새 회차 · 기존 사진과 결과는 저장 폴더에 남습니다.'); self.update_controls()

    def arrive(self, role):
        require(self.pending is None and self.process is None, '현재 검사가 끝난 뒤 촬영하세요.')
        require(self.camera is not None and not self.camera.stopping.is_set(), '카메라를 연결하세요.')
        require(role == 'overview' or self.selected is not None, '물체 번호를 선택하세요.')
        now = time.monotonic()
        token = self.camera.command('profile', {'name': role})
        self.pending = dict(role=role, target=self.selected if role=='detail' else None,
                            token=token, deadline=now+15, not_before=None, settle=self.settle.value())
        self.status.setText('카메라를 고정하세요. 촬영 조건 적용 후 안정 대기합니다.'); self.update_controls()

    def tick(self):
        if self.closing:
            if self.process is None and (self.vlm_worker is None or not self.vlm_worker.isRunning()) and (not self.owns_camera or self.camera is None or self.camera.disposed):
                self.close_ready = True
                self.timer.stop(); super().reject()
            return
        self.update_controls()
        self.poll_server()
        now=time.monotonic()
        if now >= self.vlm_next_poll:
            self.vlm_next_poll = now + .5
            self.vlm_guard(self.refresh_vlm)
        if self.process is not None and now-self.started > 240:
            self.stop(); self.status.setText('검사 응답 시간 초과 · 학습 중 GPU 사용 여부와 로그를 확인하세요.')
        if self.camera is None: return
        raw_reader = getattr(type(self.camera), 'get_latest_raw', None)
        frame, received = raw_reader(self.camera) if raw_reader else self.camera.get_latest(); now = time.monotonic()
        if frame is not None and received != self.preview_key:
            require(not frame.transformations, '입력 영상이 이미 보정되어 있습니다.')
            require(frame.acquisition_identity is None, '입력 영상이 이미 보정되어 있습니다.')
            _, rgb, _ = self.rectifier.apply(frame.rgb, self.square.isChecked())
            self.preview.canvas.set_rgb(rgb); self.preview_key = received
        if not self.pending: return
        p = self.pending
        require(now < p['deadline'], '새 카메라 영상 대기 시간이 초과됐습니다.')
        if p['not_before'] is None:
            reply = self.camera.replies.pop(p['token'], None)
            if reply is None: return
            require('error' not in reply, reply.get('error', '촬영 조건 오류'))
            p['not_before'] = max(now, reply['result']['applied_at'])+p['settle']
        if not fresh_after(frame, received, p['not_before'], now): return
        self.pending = None
        if p['role'] == 'overview':
            self.vlm_guard(self.cancel_vlm_jobs)
            self.cycle = self.runtime/'manual-inspections'/uuid4().hex
            self.cycle.mkdir(parents=True)
            write_json(self.cycle/'bundle.json', self.bundle)
            self.objects, self.details, self.selected = [], {}, None
            self.overview_capture=None
            self.table.setRowCount(0); self.overview.canvas.tracks=[]
            self.detail.canvas.pixmap=QPixmap(); self.detail.canvas.tracks=[]; self.detail.canvas.update(); self.reasons.clear()
        folder = self.cycle/('capture-'+uuid4().hex)
        image = save_capture(folder, frame, self.rectifier, self.square.isChecked(),
            dict(role=p['role'], target=p['target'], cycle=self.cycle.name, settings=self.settings))
        if p['role']=='overview': self.overview.canvas.set_file(image)
        self.run_capture(image, folder, p)

    def run_capture(self, image, folder, capture):
        self.capture = capture; self.output = folder/'result'; self.cancelled = False
        request = folder/'request.json'
        write_json(request, dict(images=[dict(path=str(image), sha256=sha256(image), role=capture['role'])],
            image_kind='real', bundle=str(self.cycle/'bundle.json'), runtime=str(self.runtime), output=str(self.output)))
        self.submit_request(request)

    def capture_and_inspect_overview(self):
        require(not self.closing, '창을 닫는 중입니다.')
        self.arrive('overview')
        self.pending['inspect_all'] = True

    def inspect_overview(self):
        require(self.pending is None and self.process is None and not self.closing,'현재 검사가 끝난 뒤 실행하세요.')
        require(self.cycle is not None and self.overview_capture is not None and self.objects,'먼저 풀뷰 도착 · 촬영을 완료하세요.')
        origin=deepcopy(self.overview_capture)
        manifest,source,_=load_snapshot(origin['snapshot'],expected_digest=origin['snapshot_digest'])
        require({o['object_id'] for o in source['objects']}=={o['object_id'] for o in self.objects},'풀뷰 물체 목록이 변경됐습니다. 다시 촬영하세요.')
        folder=self.cycle/('overview-inspection-'+uuid4().hex);folder.mkdir()
        image=Path(origin['snapshot'])/'frame.png'
        self.capture=dict(role='overview_inspection',target=self.selected,overview_origin=origin)
        self.output=folder/'result';self.cancelled=False
        request=folder/'request.json'
        write_json(request,dict(images=[dict(path=str(image),sha256=sha256(image),role='overview_inspection',overview_origin=origin)],
            image_kind=manifest['kind'],bundle=str(self.cycle/'bundle.json'),runtime=str(self.runtime),output=str(self.output)))
        self.submit_request(request)
        self.status.setText(f'저장된 풀뷰로 {len(self.objects)}개 물체 검사 중 · 카메라 이동·재촬영 없음')

    def submit_request(self,request):
        if self.server is not None and self.server_bundle != self.bundle:
            # Reconfiguration is uncommon; reap the old worker before acquiring its GPU lock again.
            self.server.kill()
            require(self.server.waitForFinished(3000), '이전 모델 종료 대기 중입니다. 다시 촬영하세요.')
        if self.server is None: self.start_server()
        self.process=self.server
        self.request_id=uuid4().hex
        from .photo_inspection import atomic_json
        atomic_json(self.server_directory/'command.json',dict(id=self.request_id,request=str(request)))
        self.started=time.monotonic(); self.update_controls()
        self.status.setText('검사 요청 · 모델은 최초 한 번 로드하고 이후 촬영에 재사용합니다.')

    def start_server(self):
        self.server_allows_vlm = False
        self.server_directory=self.runtime/'manual-model-workers'/uuid4().hex
        self.server_directory.mkdir(parents=True)
        write_json(self.server_directory/'bundle.json',self.bundle)
        self.server_bundle=deepcopy(self.bundle)
        process = QProcess(self); self.server = process
        process.setWorkingDirectory(str(self.root))
        env = QProcessEnvironment.systemEnvironment(); env.insert('PYTHONIOENCODING','utf-8'); process.setProcessEnvironment(env)
        process.setStandardOutputFile(str(self.server_directory/'stdout.log')); process.setStandardErrorFile(str(self.server_directory/'stderr.log'))
        process.finished.connect(self.model_finished)
        process.errorOccurred.connect(lambda error: self.model_finished(-1, QProcess.CrashExit) if error==QProcess.FailedToStart else None)
        process.start(sys.executable, [str(self.root/'scripts/manual_model_server.py'), '--directory',str(self.server_directory),
            '--runtime',str(self.runtime), '--watch-parent', str(os.getpid())])

    def finished(self, code, status):
        # Compatibility for callers of the old handler; do not connect this name:
        # QDialog also exposes a Qt signal named finished.
        self.model_finished(code, status)

    def model_finished(self, code, status):
        worker=self.server or self.process
        if worker is None: return
        was_busy=self.process is not None
        worker.deleteLater(); self.process = self.server = None
        if was_busy and not self.cancelled and not self.closing:
            error='모델 작업이 종료됐습니다. 다시 촬영하면 모델을 재로드합니다.'
            if hasattr(self,'server_directory'):
                error+='\n'+str(self.server_directory/'stderr.log')
                state=self.server_directory/'state.json'
                if state.exists(): error+='\n'+read_json(state).get('error','')
            self.status.setText(error)
        self.update_controls()

    def poll_server(self):
        if self.process is None or self.server is None: return
        state_path=self.server_directory/'state.json'
        if not state_path.exists(): return
        state=read_json(state_path)
        self.server_allows_vlm = bool(state.get('allow_vlm', False))
        if state['state']=='LOADING': self.status.setText('전체·상세·불량 모델 최초 로드 중… 이후 촬영에서는 재사용합니다.')
        if state['state']=='READY' and self.request_id is None and not self.cancelled:
            self.process=None; self.status.setText('비전 모델 준비 완료 · 메모리에 유지합니다.')
            if self.prepare_vlm_after_models:
                self.prepare_vlm_after_models=False; self.vlm_guard(self.prepare_vlm_model)
            self.update_controls(); return
        if state.get('id') != self.request_id: return
        if state['state']=='INSPECTING': self.status.setText('로드된 모델로 검사 중…')
        elif state['state']=='COMPLETED' and not self.cancelled:
            self.process=None
            self.guard(self.accept_result)
            if self.prepare_vlm_after_models and self.process is None:
                self.prepare_vlm_after_models=False; self.vlm_guard(self.prepare_vlm_model)
            self.status.setText(self.status.text()+f" · 검사 처리 {state['elapsed_ms']/1000:.2f}초 (모델 로드 제외)")
            self.update_controls()

    def accept_result(self):
        directory, report = open_results(self.output/'results.json')
        record = report['records'][0]; snapshot = directory/record['snapshot']
        _, result, _ = load_snapshot(snapshot, expected_digest=record['snapshot_digest'])
        if self.capture['role']=='overview_inspection':
            self.accept_overview_inspection(snapshot,record,result)
            return
        if self.capture['role']=='overview':
            self.overview_capture=dict(snapshot=str(snapshot),snapshot_digest=record['snapshot_digest'])
            from .display_order import reading_order
            self.objects = reading_order(result['objects'])
            for i, obj in enumerate(self.objects): obj['manual_id'] = str(i+1)
            self.overview.canvas.set_file(snapshot/'frame.png')
        else:
            old = self.details.get(self.capture['target'], {})
            if old.get('vlm_job'): self.vlm_guard(lambda: self.vlm_queue.cancel(old['vlm_job']))
            self.details[self.capture['target']] = dict(snapshot=str(snapshot), snapshot_digest=record['snapshot_digest'], result=result, associated=record['association_confirmed'],capture_method='detail')
        self.save_session()
        self.refresh_objects()
        if self.objects: self.select_object(self.capture['target'] or self.objects[0]['manual_id'])
        self.status.setText('검사 완료 · 다음 물체로 카메라를 옮겨주세요. 저장 위치: '+str(self.cycle) if self.objects else '물체 미검출 · 풀뷰 위치/조명/모델을 확인하고 다시 촬영하세요.')
        if self.capture['role']=='overview' and self.capture.get('inspect_all') and self.objects:
            self.inspect_overview()
            return
        if self.capture['role']=='detail':
            try: self.request_vlm(self.capture['target'])
            except Exception as exc: self.vlm_text.setPlainText('VLM 확인 불가: '+str(exc))

    def accept_overview_inspection(self,snapshot,record,result):
        require(record['role']=='overview_inspection' and record['overview_origin']==self.overview_capture
                and self.capture['overview_origin']==self.overview_capture,'풀뷰 회차가 변경됐습니다.')
        mapping={p['source_object_id']:p['object_id'] for p in record['object_mapping']}
        require(set(mapping)=={o['object_id'] for o in self.objects},'풀뷰 검사 결과의 물체 목록이 다릅니다.')
        manifest,_,_=load_snapshot(snapshot,expected_digest=record['snapshot_digest'])
        records={o['object_id']:o for o in result['objects']}
        files={o['object_id']:o['file'] for o in manifest['objects']}
        replacements={}
        for source in self.objects:
            target=mapping[source['object_id']]
            require(target in records and target in files,'물체별 풀뷰 검사 결과가 없습니다.')
            view=deepcopy(result);view['objects']=[deepcopy(records[target])]
            replacements[source['manual_id']]=dict(snapshot=str(snapshot),snapshot_digest=record['snapshot_digest'],
                result=view,associated=True,capture_method='overview_inspection',snapshot_object_id=target,
                display_file=files[target],overview_origin=deepcopy(self.overview_capture))
        for identity,detail in replacements.items():
            previous=self.details.get(identity,{})
            if previous.get('vlm_job'):self.vlm_guard(lambda job=previous['vlm_job']:self.vlm_queue.cancel(job))
        self.details.update(replacements);self.save_session();self.refresh_objects()
        if self.objects:self.select_object(self.capture['target'] or self.objects[0]['manual_id'])
        self.status.setText(f'풀뷰 전체 검사 완료 · {len(replacements)}개 · 근접 상세검사로 다시 확인해 비교할 수 있습니다.')
        for identity in replacements:self.vlm_guard(lambda key=identity:self.request_vlm(key))

    def refresh_objects(self):
        self.table.blockSignals(True); self.table.setRowCount(len(self.objects)); tracks=[]
        for i, obj in enumerate(self.objects):
            identity = obj['manual_id']; b=obj['effective_box']; detail=self.details.get(identity)
            text = '미촬영' if detail is None else ('결과 있음 · 보류' if detail['associated'] else '재촬영 필요')
            if detail and detail['associated']:
                text = finding_summary(detail['result']['objects']) or '표시할 불량 없음 · 보류'
                text=('풀뷰 · ' if detail.get('capture_method')=='overview_inspection' else '상세 · ')+text
            for j, value in enumerate((identity, f"{(b['x1']+b['x2'])/2:.1f} / {(b['y1']+b['y2'])/2:.1f}", text)):
                self.table.setItem(i,j,QTableWidgetItem(value))
            overlay=object_overlays(obj,identity=identity)[0]; overlay.update(label=identity+' · '+text, status='REVIEW' if detail else 'INSPECTING'); tracks.append(overlay)
        self.table.blockSignals(False); self.overview.canvas.tracks=tracks; self.overview.canvas.update()

    def select_row(self):
        row=self.table.currentRow()
        if 0<=row<len(self.objects): self.select_object(self.objects[row]['manual_id'])

    def select_object(self, identity):
        if self.pending is not None or self.process is not None: return
        if not any(o['manual_id']==identity for o in self.objects): return
        self.selected=identity; self.overview.canvas.selected_id=identity; self.overview.canvas.update()
        self.table.blockSignals(True); self.table.selectRow(next(i for i,o in enumerate(self.objects) if o['manual_id']==identity)); self.table.blockSignals(False)
        self.detail_button.setText('② 물체 '+identity+'번 도착 · 상세검사')
        d=self.details.get(identity); self.detail.canvas.tracks=[]
        if d:
            is_overview=d.get('capture_method')=='overview_inspection'
            self.detail_title.setText('선택 물체 · 풀뷰에서 자른 검사 영역' if is_overview else '선택 물체의 저장된 상세사진')
            self.detail.canvas.set_file(Path(d['snapshot'])/(d['display_file'] if is_overview else 'frame.png'))
            lines=['물체 '+identity+'번 · '+('풀뷰 검사 · 실제 근접 촬영 아님' if is_overview else '사용자가 연결한 상세사진'), '최종 판정: 보류 (판정 기준 미검증)']
            if not d['associated']: lines.append('중앙에 물체 하나가 검출되지 않았습니다. 다시 촬영하세요.')
            for obj in d['result']['objects']:
                self.detail.canvas.tracks.extend(object_overlays(obj,crop=is_overview))
                lines.extend('검사 오류: '+c['check_id'] for c in obj['checks'] if c['status']=='ERROR')
            self.reasons.setHtml(finding_details_html(lines,d['result']['objects']))
        else:
            self.detail.canvas.pixmap=QPixmap(); self.reasons.setPlainText('물체 '+identity+'번 위로 카메라를 옮긴 뒤 상세검사 버튼을 누르세요.')
        self.detail.canvas.update(); self.update_controls()
        self.vlm_guard(self.refresh_vlm)

    def vlm_guard(self, fn):
        # An optional explanation must never reset a pending camera capture.
        try: fn()
        except Exception as exc: self.vlm_text.setPlainText('VLM 확인 불가: '+str(exc))

    def toggle_manual_vlm(self, enabled):
        if self.vlm_owner is not None: self.vlm_owner.toggle_vlm(enabled)
        else: self.vlm_queue.set_enabled(enabled)
        if enabled:
            if self.server is not None and self.process is None: self.prepare_vlm_model()
            elif self.server is not None: self.prepare_vlm_after_models=True
            self.request_vlm(self.selected)
        else: self.release_vlm_preload()
        self.refresh_vlm()

    def request_vlm(self, identity, *, retry=False):
        if self.closing or not self.vlm_queue.enabled(): return
        detail = self.details.get(identity)
        if not detail or not detail['associated'] or len(detail['result']['objects']) != 1: return
        # Bind to the immutable snapshot, not the current row or camera image.
        _, result, digest = load_snapshot(detail['snapshot'], expected_digest=detail['snapshot_digest'])
        target=detail.get('snapshot_object_id')
        if target is None:
            require(len(result['objects'])==1,'상세사진에 여러 물체가 연결됐습니다.')
            obj=result['objects'][0]
        else:
            matches=[o for o in result['objects'] if o['object_id']==target]
            require(len(matches)==1,'풀뷰 VLM 대상 물체가 없습니다.')
            obj=matches[0]
        from mes_vision.vlm.region_backend import region_plan
        plan=region_plan(dict(snapshot_path=detail['snapshot'],snapshot_digest=detail['snapshot_digest'],object_id=obj['object_id']),self.vlm_capabilities())
        if not plan['regions']:
            detail['vlm_no_regions']=True
            self.refresh_vlm()
            return
        detail.pop('vlm_no_regions',None)
        job_id = detail.get('vlm_job')
        if job_id and retry:
            job = self.vlm_queue.get(job_id)
            if job['state'] in {'FAILED','CANCELLED','DEFERRED','SKIPPED_DISABLED'}: self.vlm_queue.retry(job_id)
        if not job_id:
            job_id = self.vlm_queue.enqueue(detail['snapshot'], obj['object_id'], dict(max_new_tokens=64, language='ko'))
            detail.update(vlm_job=job_id, vlm_object_id=obj['object_id'])
        self.vlm_jobs.add(job_id)
        self.ensure_vlm_worker()
        # Small GPUs cannot retain both models: release only the idle manual models.
        if self.server is not None and self.process is None and not self.server_allows_vlm:
            self.release_models(release_vlm=False)
        self.save_session()
        self.refresh_vlm()

    def ensure_vlm_worker(self):
        if self.vlm_owner is not None:
            self.vlm_owner.toggle_vlm(True)
            return self.vlm_owner.worker
        elif self.vlm_worker is None or not self.vlm_worker.isRunning():
            from mes_vision.vlm.viewer import WorkerThread
            if self.vlm_worker is not None: self.vlm_worker.deleteLater()
            self.vlm_worker = WorkerThread(self.vlm_queue, self.root, backend='qwen', allow_synthetic=False)
            self.vlm_worker.failed.connect(self.vlm_failed)
            self.vlm_worker.start()
        return self.vlm_worker

    def prepare_vlm_model(self):
        if self.closing or not self.vlm_queue.enabled(): return
        worker=self.ensure_vlm_worker()
        require(worker is not None,'VLM 작업자를 시작할 수 없습니다.')
        worker.worker.request_preload(self.preload_owner)
        self.vlm_preload_wanted=True
        self.vlm_model_status.setText('VLM 모델: 준비 요청 중…')

    def release_vlm_preload(self, *, force=False):
        self.prepare_vlm_after_models=False
        worker=self.vlm_owner.worker if self.vlm_owner is not None else self.vlm_worker
        if (self.vlm_preload_wanted or force) and worker is not None:
            worker.worker.release_preload(self.preload_owner)
        self.vlm_preload_wanted=False

    def vlm_failed(self, message):
        self.vlm_queue.set_enabled(False)
        self.vlm_text.setPlainText('VLM 확인 불가: '+message+'\n기본 검사 결과는 유지됩니다.')

    def vlm_capabilities(self):
        from mes_vision.vlm.region_backend import supported_codes
        return supported_codes(read_json(self.root/'configs/vlm/backend.json'))

    def save_session(self):
        if self.cycle is not None:
            write_json(self.cycle/'session.json', dict(objects=self.objects, details=self.details,overview_capture=self.overview_capture,
                coordinate_space='rectified_crop_pixels', robot_commands_enabled=False, production_ready=False))

    def refresh_vlm(self):
        enabled = self.vlm_queue.enabled()
        state=self.vlm_queue.runtime() or {}
        # Ignore heartbeat remnants from a worker which is no longer running.
        fresh=isinstance(state,dict) and -5<=time.time()-state.get('updated',0)<5
        self.vlm_model_loaded=bool(fresh and state.get('model_ready'))
        if not enabled: model_text='OFF'
        elif fresh and state.get('preload_error'):
            model_text='준비 실패 · '+state['preload_error']; self.vlm_preload_wanted=False
        elif fresh and state.get('model_ready'): model_text='준비 완료' if state.get('phase')!='WAITING_FOREGROUND' else '준비 완료 · 비전 검사 대기'
        elif fresh and state.get('phase')=='MODEL_LOADING': model_text='로드 중…'
        elif self.vlm_preload_wanted: model_text='준비 대기 · GPU 사용 상태 확인 중'
        else: model_text='미로드'
        self.vlm_model_status.setText('VLM 모델: '+model_text)
        self.vlm_enabled.blockSignals(True); self.vlm_enabled.setChecked(enabled); self.vlm_enabled.blockSignals(False)
        detail = self.details.get(self.selected)
        self.vlm_request.setEnabled(enabled and not self.closing and self.pending is None and self.process is None and bool(detail and detail['associated']))
        text = 'VLM 보조 정보 · 기본 판정을 변경하지 않습니다.\n'
        if not enabled: text += 'OFF · 기본 판정과 불량 종류는 계속 확인할 수 있습니다.'
        elif not detail or not detail['associated']: text += '물체 하나가 연결된 상세사진을 촬영하세요.'
        elif detail.get('vlm_no_regions'): text += '표시 기준을 통과한 지원 영역 없음 · 추가 분석하지 않음 · 정상 확정 아님'
        elif not detail.get('vlm_job'): text += '분석 요청 없음 · 지원 범위: '+', '.join(sorted(self.vlm_capabilities()))
        else:
            job = self.vlm_queue.get(detail['vlm_job'])
            require(job['snapshot_digest'] == detail['snapshot_digest'] and job['object_id'] == detail['vlm_object_id'], 'VLM result identity mismatch')
            states = dict(PENDING='분석 대기',RUNNING='분석 중',FAILED='확인 불가 · 분석 실패',CANCELLED='분석 취소',DEFERRED='대기열 가득 참',SKIPPED_DISABLED='VLM OFF')
            if job['state'] == 'COMPLETED':
                text += job['result']['analysis']['observation']
            else: text += states.get(job['state'], job['state'])
        if self.vlm_text.toPlainText() != text: self.vlm_text.setPlainText(text)

    def cancel_vlm_jobs(self):
        for job_id in self.vlm_jobs: self.vlm_queue.cancel(job_id)
        self.vlm_jobs.clear()

    def stop(self):
        self.release_vlm_preload()
        self.pending=None; self.cancelled=True
        if self.process: self.process.kill()
        self.status.setText('촬영/검사를 취소했습니다. 카메라 영상은 유지됩니다.'); self.update_controls()

    def reject(self):
        if self.close_ready:
            super().reject(); return
        if self.closing: return
        self.closing=True; self.pending=None; self.cancelled=True
        self.release_vlm_preload()
        worker = self.server or self.process
        if worker is not None:
            self.process=worker; worker.kill()
        if self.vlm_worker is not None: self.vlm_worker.stop_event.set()
        if self.owns_camera and self.camera: self.camera.stopping.set()
        try: self.cancel_vlm_jobs()
        except Exception as exc: self.status.setText('VLM 요청 정리 오류: '+str(exc))
        self.update_controls()
        self.timer.start(60)
        self.tick()

    def closeEvent(self, event):
        if self.close_ready: event.accept()
        else: event.ignore(); self.reject()

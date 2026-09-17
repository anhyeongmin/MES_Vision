"""Correlated live phases, truthful elapsed time and cached evidence rendering."""
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
from copy import deepcopy
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from PySide6.QtWidgets import QApplication
from mes_vision.i18n import set_language, validate_catalog
from mes_vision.vlm.progress import overall, job_label, selected_text
from mes_vision.vlm.worker import AnalysisWorker
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.viewer import AnalysisViewer

APP = QApplication.instance() or QApplication([])
ROOT = Path(__file__).resolve().parents[1]


class ProgressTests(unittest.TestCase):
    def setUp(self):
        set_language('ko')
        self.row = dict(id='request', token='attempt-2', state='RUNNING', cancel_requested=False,
                        created=900., updated=990., result=None)
        self.runtime = dict(phase='MODEL_LOADING', job_id='request', job_token='attempt-2',
                            updated=1000., phase_started=982., model_pid=123)

    def tearDown(self): set_language('ko')

    def test_loading_and_analysis_follow_runtime_without_job_row_update(self):
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '모델 준비 중')
        self.runtime['phase'] = 'ANALYZING'
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '추가 분석 중')

    def test_other_attempt_and_stale_heartbeat_are_not_shown_as_live(self):
        self.runtime['job_token'] = 'attempt-1'
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '처리 상태 확인 중')
        self.runtime['job_token'] = 'attempt-2'
        self.assertEqual(job_label(self.row, self.runtime, True, 1008), '처리 상태 확인 중')
        self.assertEqual(overall({'enabled': True}, self.runtime, 1008), '작업자 상태 확인 필요')

    def test_cancel_and_off_take_priority_over_old_running_phase(self):
        self.row['cancel_requested'] = True
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '중단 요청 중')
        self.assertEqual(overall({'enabled': False}, self.runtime, 1000), '중단 요청 중')
        self.assertEqual(overall({'enabled': False}, self.runtime, 1008), '중단 상태 확인 필요')

    def test_pending_uses_current_priority_reason_not_historical_error(self):
        self.row.update(state='PENDING', error='FOREGROUND_PRIORITY')
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '분석 대기')
        self.runtime['phase'] = 'WAITING_FOREGROUND'
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), '검사 우선으로 대기')
        self.runtime['phase'] = 'WAITING_GPU'
        self.assertEqual(job_label(self.row, self.runtime, True, 1000), 'GPU 사용 순서 대기')

    def test_finished_elapsed_freezes_and_completed_is_not_overridden_by_other_work(self):
        self.row.update(state='COMPLETED', result={'metrics': {'model_reused': True}})
        first = selected_text(self.row, self.runtime, True, 1000)
        self.assertEqual(first, selected_text(self.row, self.runtime, False, 2000))
        self.assertIn('00:01:30', first); self.assertIn('모델 재사용', first)

    def test_elapsed_stage_does_not_claim_completion_eta(self):
        self.assertIn('현재 단계 경과 00:00:18', overall({'enabled': True}, self.runtime, 1000))
        self.runtime.update(phase='WARM_IDLE', warm_expires_at=1025)
        self.assertIn('유지 종료까지 00:00:25', overall({'enabled': True}, self.runtime, 1000))

    def test_deferred_is_not_presented_as_automatically_queued(self):
        self.row['state'] = 'DEFERRED'
        self.assertIn('다시 요청하세요', selected_text(self.row, None, True, 1000))

    def test_four_languages_and_placeholders(self):
        validate_catalog()
        for locale, expected in [('ko','모델 준비 중'),('en','Preparing model'),('zh-CN','正在准备模型'),('th','กำลังเตรียมโมเดล')]:
            set_language(locale)
            self.assertIn(expected, overall({'enabled': True}, self.runtime, 1000))
            self.assertIn('00:00:18', overall({'enabled': True}, self.runtime, 1000))


class ProgressUiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.f = make_vlm_fixture(Path(self.temp.name)/'fixture', enabled=True)
        self.queue = self.f['queue']
        self.job = self.queue.claim('progress-test')
        self.worker = AnalysisWorker(self.queue, ROOT, backend='mock', allow_synthetic=True)
        self.worker.state('MODEL_LOADING', self.job)
        self.w = AnalysisViewer(self.queue, ROOT, run_worker=False); self.w.timer.stop()

    def tearDown(self):
        self.w.stop_refresh(); self.w.close(); APP.processEvents(); self.temp.cleanup()

    def test_phase_updates_without_rereading_images_or_altering_evidence(self):
        baseline = self.w.reasons.toPlainText()
        self.assertIn('모델 준비 중', self.w.progress.text())
        with patch('mes_vision.vlm.viewer.load_snapshot', side_effect=AssertionError('redundant read')) as read:
            self.worker.state('ANALYZING', self.job); self.w.refresh()
            self.assertIn('추가 분석 중', self.w.table.item(0,2).text())
            self.assertIn('추가 분석 중', self.w.selected_progress.text())
            self.assertIn('추가 분석 중', self.w.analysis.toPlainText())
            self.assertEqual(read.call_count, 0)
        self.assertEqual(baseline, self.w.reasons.toPlainText())

    def test_read_failure_does_not_leave_live_progress_and_recovers(self):
        with patch.object(self.queue, 'runtime', side_effect=OSError('unavailable')): self.w.refresh()
        self.assertIn('마지막 조회 기준', self.w.progress.text())
        self.w.refresh(); self.assertIn('모델 준비 중', self.w.progress.text())

    def test_heartbeat_preserves_stage_start_and_throttles_database_writes(self):
        initial = self.queue.runtime()
        with patch.object(self.queue, 'runtime', wraps=self.queue.runtime) as writes:
            for _ in range(20): self.worker.state('MODEL_LOADING', self.job)
            self.assertEqual(writes.call_count, 0)
            self.worker.state_written_at -= 2
            self.worker.state('MODEL_LOADING', self.job)
            self.assertEqual(writes.call_count, 1)
        latest = self.queue.runtime()
        self.assertEqual(latest['phase_started'], initial['phase_started'])
        self.assertEqual(latest['job_token'], self.job['token'])


class WorkerProgressTests(unittest.TestCase):
    def test_long_analysis_heartbeats_then_reports_foreground_wait(self):
        with tempfile.TemporaryDirectory() as root:
            fixture = make_vlm_fixture(Path(root)/'fixture', enabled=True)
            queue = fixture['queue']
            worker = AnalysisWorker(queue, ROOT, backend='mock', allow_synthetic=True,
                                    mock_behavior='slow', idle_seconds=0)
            errors = []
            def run():
                try: worker.run()
                except Exception as exc: errors.append(repr(exc))
            def wait(predicate):
                end = time.monotonic()+8
                while time.monotonic()<end:
                    if errors: self.fail(str(errors))
                    if predicate(): return
                    time.sleep(.02)
                self.fail('worker state did not arrive')
            thread = threading.Thread(target=run); thread.start()
            try:
                wait(lambda: (queue.runtime() or {}).get('phase') == 'ANALYZING')
                initial = queue.runtime()
                wait(lambda: queue.runtime()['updated'] > initial['updated']+.8)
                self.assertEqual(queue.runtime()['phase_started'], initial['phase_started'])
                with worker.gpu.foreground(timeout=5):
                    wait(lambda: queue.runtime()['phase'] == 'WAITING_FOREGROUND')
                    self.assertEqual(queue.get(fixture['job_id'])['state'], 'PENDING')
                    worker.stop_event.set()
            finally:
                worker.stop_event.set(); thread.join(8)
            self.assertFalse(thread.is_alive()); self.assertFalse(errors)


if __name__ == '__main__': unittest.main()

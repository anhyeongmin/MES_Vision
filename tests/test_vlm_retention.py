"""Idle model reuse without weakening foreground ownership or cancellation."""
from dataclasses import asdict
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from filelock import FileLock, Timeout
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.backend import GenerationConfig
from mes_vision.vlm.worker import AnalysisWorker
from mes_vision.training.data import write_json, sha256

ROOT = Path(__file__).resolve().parents[1]


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.fixture = make_vlm_fixture(Path(self.temp.name)/"fixture", enabled=True)
        self.queue = self.fixture["queue"]
        self.worker = None
        self.thread = None
        self.errors = []

    def tearDown(self):
        if self.worker: self.worker.stop_event.set()
        if self.thread:
            self.thread.join(8)
            self.assertFalse(self.thread.is_alive())
        self.assertFalse(self.errors, self.errors)
        self.temp.cleanup()

    def wait(self, predicate, timeout=8):
        end = time.monotonic()+timeout
        while time.monotonic() < end:
            if self.errors: self.fail(str(self.errors))
            if predicate(): return
            time.sleep(.02)
        self.fail("retention condition timed out")

    def phase(self, expected):
        state = self.queue.runtime()
        return state is not None and state["phase"] == expected

    def start(self, **options):
        self.worker = AnalysisWorker(self.queue, ROOT, backend="mock", allow_synthetic=True,
                                     idle_seconds=0, **options)
        def run():
            try: self.worker.run()
            except Exception as exc: self.errors.append(repr(exc))
        self.thread = threading.Thread(target=run)
        self.thread.start()

    def enqueue_next(self):
        old = self.queue.get(self.fixture["job_id"])
        return self.queue.enqueue(self.fixture["snapshot"], old["object_id"], asdict(GenerationConfig(max_new_tokens=129)))

    def test_empty_queue_reuses_same_process_without_touching_original(self):
        original = sha256(self.fixture["snapshot"]/"inspection.json")
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        first = self.queue.runtime()["model_pid"]
        with self.assertRaises(Timeout):
            with self.worker.gpu.background_lock(): pass
        time.sleep(.15)
        second = self.enqueue_next()
        self.wait(lambda: self.queue.get(second)["state"] == "COMPLETED" and self.phase("WARM_IDLE"))
        state = self.queue.runtime()
        self.assertEqual(state["model_pid"], first); self.assertEqual(state["model_load_count"], 1)
        result = self.queue.get(second)["result"]
        self.assertTrue(result["metrics"]["model_reused"])
        self.assertEqual(result["metrics"]["model_load_seconds"], 0)
        self.assertTrue(result["advisory_only"])
        self.assertEqual(sha256(self.fixture["snapshot"]/"inspection.json"), original)

    def test_deadline_expires_then_next_request_reloads(self):
        self.start(keep_alive_seconds=.3)
        self.wait(lambda: self.phase("WARM_IDLE"))
        self.wait(lambda: self.phase("IDLE") and self.queue.runtime()["last_release_reason"] == "IDLE_EXPIRED")
        self.assertIsNone(self.queue.runtime()["model_pid"])
        second = self.enqueue_next()
        self.wait(lambda: self.queue.get(second)["state"] == "COMPLETED")
        self.assertFalse(self.queue.get(second)["result"]["metrics"]["model_reused"])

    def test_zero_retention_releases_after_completion(self):
        self.start(keep_alive_seconds=0)
        self.wait(lambda: self.queue.get(self.fixture["job_id"])["state"] == "COMPLETED" and self.phase("IDLE"))
        self.assertIsNone(self.queue.runtime()["model_pid"])

    def test_off_releases_idle_model_and_on_does_not_preload(self):
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        self.queue.set_enabled(False)
        self.wait(lambda: self.phase("DISABLED") and self.queue.runtime()["model_pid"] is None)
        self.assertEqual(self.queue.runtime()["last_release_reason"], "VLM_DISABLED")
        self.queue.set_enabled(True); time.sleep(.3)
        self.assertIsNone(self.queue.runtime()["model_pid"])
        self.assertEqual(self.queue.get(self.fixture["job_id"])["state"], "COMPLETED")

    def test_foreground_can_acquire_gpu_while_model_is_warm(self):
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        with self.worker.gpu.foreground(timeout=5):
            self.wait(lambda: self.queue.runtime()["model_pid"] is None)
            self.assertEqual(self.queue.runtime()["last_release_reason"], "FOREGROUND_PRIORITY")
        self.assertEqual(self.queue.get(self.fixture["job_id"])["state"], "COMPLETED")

    def test_nonconcurrent_resident_lease_releases_idle_model(self):
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        root = self.worker.gpu.root
        write_json(root/"resident.json", {"allow_vlm": False})
        with FileLock(str(root/"resident.lock")):
            self.wait(lambda: self.queue.runtime()["model_pid"] is None)
            self.assertEqual(self.queue.runtime()["last_release_reason"], "FOREGROUND_PRIORITY")

    def test_low_memory_releases_completed_job_without_failing_it(self):
        self.start(min_free_mib=20000)
        self.wait(lambda: self.phase("IDLE") and self.queue.runtime()["last_release_reason"] == "LOW_FREE_MEMORY")
        self.assertIsNone(self.queue.runtime()["model_pid"])
        self.assertEqual(self.queue.get(self.fixture["job_id"])["state"], "COMPLETED")

    def test_memory_probe_error_releases_without_losing_result(self):
        original = AnalysisWorker.wait_message
        def probe(worker, job, timeout):
            if job is None: raise TimeoutError("memory probe stalled")
            return original(worker, job, timeout)
        with patch.object(AnalysisWorker, "wait_message", probe):
            self.start()
            self.wait(lambda: self.phase("IDLE") and self.queue.runtime()["last_release_reason"] == "MEMORY_STATUS_UNAVAILABLE")
        self.assertEqual(self.queue.get(self.fixture["job_id"])["state"], "COMPLETED")

    def test_memory_pressure_after_warmup_is_detected_on_next_check(self):
        pressure = threading.Event()
        original = AnalysisWorker.wait_message
        def probe(worker, job, timeout):
            message = original(worker, job, timeout)
            if job is None and pressure.is_set(): message["free_bytes"] = 512 * 1024**2
            return message
        with patch.object(AnalysisWorker, "wait_message", probe):
            self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
            pressure.set()
            self.wait(lambda: self.phase("IDLE") and self.queue.runtime()["last_release_reason"] == "LOW_FREE_MEMORY")
        self.assertIsNone(self.queue.runtime()["model_pid"])

    def test_idle_crash_is_cleaned_and_next_request_can_run(self):
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        self.worker.child.process.terminate()
        self.wait(lambda: self.phase("IDLE") and self.queue.runtime()["model_pid"] is None)
        second = self.enqueue_next()
        self.wait(lambda: self.queue.get(second)["state"] == "COMPLETED")
        self.assertFalse(self.queue.get(second)["result"]["metrics"]["model_reused"])

    def test_stop_releases_warm_child_before_worker_exits(self):
        self.start(); self.wait(lambda: self.phase("WARM_IDLE"))
        self.worker.stop_event.set(); self.thread.join(5)
        self.assertFalse(self.thread.is_alive()); self.assertTrue(self.phase("STOPPED"))
        self.assertIsNone(self.queue.runtime()["model_pid"])
        with self.worker.gpu.background_lock(): pass

    def test_invalid_policy_is_rejected(self):
        for value in (-1, 301, float("nan"), float("inf"), True):
            with self.assertRaises(ValueError): AnalysisWorker(self.queue, ROOT, keep_alive_seconds=value)
        for value in (-1, 0, 32769, True, 2048.5):
            with self.assertRaises(ValueError): AnalysisWorker(self.queue, ROOT, min_free_mib=value)


if __name__ == "__main__": unittest.main()

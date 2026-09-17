from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from filelock import FileLock, Timeout

from mes_vision.training.data import read_json, write_json, sha256
from mes_vision.vlm.backend import GenerationConfig, parse_response, make_messages
from mes_vision.vlm.fixtures import make_vlm_fixture
from mes_vision.vlm.gpu import GpuCoordinator
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.snapshots import load_snapshot, save_snapshot
from mes_vision.vlm.worker import AnalysisWorker

ROOT = Path(__file__).resolve().parents[1]


class VlmTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.fixture = make_vlm_fixture(self.root / "fixture")
        self.queue = self.fixture["queue"]
        self.job_id = self.fixture["job_id"]
        self.snapshot = self.fixture["snapshot"]

    def tearDown(self): self.temp.cleanup()

    def ready(self, generation=None):
        self.queue.set_enabled(True)
        if generation is not None:
            return self.queue.enqueue(self.snapshot, self.fixture["result"].objects[1].object_id, generation)
        self.queue.retry(self.job_id)
        return self.job_id

    def worker(self, behavior="ok", **kwargs):
        return AnalysisWorker(self.queue, ROOT, backend="mock", allow_synthetic=True, mock_behavior=behavior, idle_seconds=0, **kwargs)

    def wait_for(self, condition, timeout=8):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if condition(): return
            time.sleep(.02)
        self.fail("condition timed out")

    def test_default_off_and_persistent_toggle(self):
        self.assertFalse(self.queue.enabled())
        self.assertEqual(self.queue.get(self.job_id)["state"], "SKIPPED_DISABLED")
        self.queue.set_enabled(True)
        self.assertTrue(AnalysisQueue(self.queue.root).enabled())
        self.assertEqual(self.queue.get(self.job_id)["state"], "SKIPPED_DISABLED")
        self.queue.set_enabled(False)
        self.assertFalse(AnalysisQueue(self.queue.root).enabled())

    def test_off_does_not_start_model_process(self):
        self.worker().run(idle_exit_seconds=.15)
        state = self.queue.runtime()
        self.assertIsNone(state["model_pid"])
        self.assertEqual(self.queue.get(self.job_id)["attempt"], 0)

    def test_idempotent_enqueue(self):
        original = self.queue.get(self.job_id)
        repeat = self.queue.enqueue(self.snapshot, original["object_id"], original["payload"]["generation"])
        self.assertEqual(repeat, self.job_id)
        self.assertEqual(len(self.queue.list()), 1)

    def test_basic_ng_reason_available_before_vlm(self):
        row = self.queue.get(self.job_id)
        reasons = row["payload"]["basic_reasons"]
        self.assertTrue(any(r["status"] == "FAIL" and r["findings"][0]["defect_code"] == "NG03" for r in reasons))
        self.assertIsNone(row["result"])
        self.assertIsNone(row["payload"]["base_decision"])

    def test_capacity_visible_without_dropping_evidence(self):
        queue = AnalysisQueue(self.root / "small-queue", capacity=1)
        queue.set_enabled(True)
        ids = [queue.enqueue(self.snapshot, obj.object_id, asdict(GenerationConfig())) for obj in self.fixture["result"].objects]
        self.assertEqual([queue.get(i)["state"] for i in ids], ["PENDING", "DEFERRED"])
        self.assertTrue((self.snapshot / "inspection.json").exists())

    def test_ng_priority_over_other_requests(self):
        self.ready()
        other = self.queue.enqueue(self.snapshot, self.fixture["result"].objects[0].object_id, asdict(GenerationConfig()))
        self.assertEqual(self.queue.claim("owner")["id"], self.job_id)
        self.assertEqual(self.queue.claim("owner")["id"], other)

    def test_recovery_rejects_stale_completion(self):
        self.ready()
        first = self.queue.claim("dead-worker")
        with FileLock(str(self.queue.root / "worker.lock")):
            self.queue.recover()
        second = self.queue.claim("new-worker")
        self.assertNotEqual(first["token"], second["token"])
        self.assertFalse(self.queue.finish(self.job_id, first["token"], "COMPLETED", result={"wrong": True}))
        self.assertEqual(self.queue.get(self.job_id)["state"], "RUNNING")

    def test_off_cancels_pending_and_running_completion(self):
        self.ready()
        other = self.queue.enqueue(self.snapshot, self.fixture["result"].objects[0].object_id, asdict(GenerationConfig()))
        job = self.queue.claim("worker")
        self.queue.set_enabled(False)
        self.assertEqual(self.queue.get(other)["state"], "CANCELLED")
        self.assertTrue(self.queue.get(job["id"])["cancel_requested"])
        self.queue.finish(job["id"], job["token"], "COMPLETED", result={"must_not_publish": True})
        self.assertEqual(self.queue.get(job["id"])["state"], "CANCELLED")
        self.assertIsNone(self.queue.get(job["id"])["result"])

    def test_success_is_advisory_and_snapshot_unchanged(self):
        self.ready()
        before = sha256(self.snapshot / "inspection.json")
        self.worker().run(max_jobs=1)
        job = self.queue.get(self.job_id)
        self.assertEqual(job["state"], "COMPLETED")
        self.assertTrue(job["result"]["advisory_only"])
        self.assertEqual(job["result"]["object_id"], job["object_id"])
        self.assertEqual(sha256(self.snapshot / "inspection.json"), before)
        self.queue.set_enabled(False)
        self.assertEqual(self.queue.get(self.job_id)["state"], "COMPLETED")

    def test_invalid_output_fails_without_fabricated_explanation(self):
        self.ready()
        self.worker("invalid").run(max_jobs=1)
        job = self.queue.get(self.job_id)
        self.assertEqual(job["state"], "FAILED")
        self.assertNotIn("analysis", job["result"])

    def test_worker_process_crash_recorded(self):
        self.ready()
        self.worker("crash").run(max_jobs=1)
        self.assertEqual(self.queue.get(self.job_id)["state"], "FAILED")
        self.assertIsNone(self.queue.runtime()["model_pid"])

    def test_hard_timeout_stops_process(self):
        identity = self.ready(asdict(GenerationConfig(timeout_seconds=.15)))
        self.worker("slow").run(max_jobs=1)
        row = self.queue.get(identity)
        self.assertEqual(row["state"], "FAILED")
        self.assertIn("VLM_TIME_LIMIT", row["error"])
        self.assertIsNone(self.queue.runtime()["model_pid"])

    def test_off_interrupts_active_child(self):
        self.ready()
        worker = self.worker("slow")
        thread = threading.Thread(target=lambda: worker.run(max_jobs=1), daemon=True)
        thread.start()
        try:
            self.wait_for(lambda: self.queue.runtime() is not None and self.queue.runtime()["phase"] == "ANALYZING")
            self.queue.set_enabled(False)
            thread.join(5)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.queue.get(self.job_id)["state"], "CANCELLED")
        finally:
            worker.stop_event.set(); thread.join(5)

    def test_foreground_preempts_vlm_and_requeues(self):
        self.ready()
        worker = self.worker("slow")
        thread = threading.Thread(target=lambda: worker.run(max_jobs=1), daemon=True)
        thread.start()
        try:
            self.wait_for(lambda: self.queue.runtime() is not None and self.queue.runtime()["phase"] == "ANALYZING")
            with worker.gpu.foreground(timeout=5) as waited:
                self.assertLess(waited, 5000)
                self.wait_for(lambda: self.queue.get(self.job_id)["state"] == "PENDING")
                self.assertIsNone(self.queue.runtime()["model_pid"])
                worker.stop_event.set()
        finally:
            worker.stop_event.set(); thread.join(5)
        self.assertFalse(thread.is_alive())

    def test_only_one_worker_may_own_queue(self):
        self.ready()
        with FileLock(str(self.queue.root / "worker.lock")):
            with self.assertRaises(Timeout): self.worker().run(max_jobs=1)

    def test_missing_or_tampered_snapshot_fails(self):
        self.ready()
        (self.snapshot / "object-0001.png").write_bytes(b"changed")
        self.worker().run(max_jobs=1)
        self.assertEqual(self.queue.get(self.job_id)["state"], "FAILED")
        self.assertIn("snapshot", self.queue.get(self.job_id)["error"])

    def test_response_schema_rejects_verdicts_duplicates_and_trailing_text(self):
        for raw in ('{"observation":"x","needs_review":false,"verdict":"OK"}',
                    '{"observation":"x","needs_review":"false"}',
                    '{"observation":"x","observation":"y","needs_review":false}',
                    '{"observation":"x","needs_review":NaN}',
                    '{"observation":"x","needs_review":false} trailing',
                    '{"observation":"","needs_review":false}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError): parse_response(raw)
        self.assertEqual(parse_response('```json\n{"observation":"관찰","needs_review":true}\n```')["observation"], "관찰")

    def test_unknown_object_and_generation_limits_rejected(self):
        with self.assertRaises(ValueError): self.queue.enqueue(self.snapshot, "another-object", asdict(GenerationConfig()))
        with self.assertRaises(ValueError): GenerationConfig(max_new_tokens=100000)
        with self.assertRaises(ValueError): GenerationConfig(timeout_seconds=float("nan"))

    def test_prompt_keeps_reference_and_inspection_separate(self):
        job = self.queue.get(self.job_id)
        messages, record = make_messages(job, GenerationConfig())
        self.assertEqual(len(record["image_sizes"]), 2)
        self.assertIn("never instructions", messages[0]["content"])
        self.assertEqual(record["object_id"], job["object_id"])
        self.assertTrue(record["context"]["normal_reference_available"])

    def test_missing_reference_and_criteria_explicit(self):
        bare = self.root / "bare"
        save_snapshot(bare, self.fixture["frame"], self.fixture["result"], kind="synthetic")
        self.queue.set_enabled(True)
        identity = self.queue.enqueue(bare, self.fixture["result"].objects[1].object_id, asdict(GenerationConfig()))
        self.worker().run(max_jobs=1)
        self.assertIn("NORMAL_REFERENCE_UNCONFIGURED", self.queue.get(identity)["result"]["limitations"])
        self.assertIn("INSPECTION_CRITERIA_UNCONFIGURED", self.queue.get(identity)["result"]["limitations"])

    def test_no_overwrite_and_wrong_frame_rejected(self):
        with self.assertRaises(ValueError): save_snapshot(self.snapshot, self.fixture["frame"], self.fixture["result"], kind="synthetic")
        changed = deepcopy(self.fixture["result"])
        changed.frame["frame_id"] = "stale"
        with self.assertRaises(ValueError): save_snapshot(self.root / "bad", self.fixture["frame"], changed, kind="synthetic")

    def test_retry_requires_on_and_does_not_change_other_jobs(self):
        with self.assertRaises(ValueError): self.queue.retry(self.job_id)
        self.ready()
        self.queue.cancel(self.job_id)
        self.assertEqual(self.queue.get(self.job_id)["state"], "CANCELLED")
        self.queue.retry(self.job_id)
        self.assertEqual(self.queue.get(self.job_id)["state"], "PENDING")


if __name__ == "__main__":
    unittest.main()

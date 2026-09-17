"""Evaluation contracts using artificial labels and injected predictions only."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mes_vision.validation.metrics import distribution, evaluate, match_boxes
from mes_vision.validation.replay import ReplayRunner, select_collection, session_configuration
from mes_vision.data_management.fixtures import make_fixture
from mes_vision.operation.catalog import new_product, default_equipment
from mes_vision.training.data import read_json, write_json
from mes_vision.vlm.snapshots import load_snapshot
from test_operation import InjectedModels, detection, frame


def truth(condition="NORMAL", codes=None):
    return [{"capture_id": "scene", "objects": [{"id": "t", "bbox": [0, 0, 10, 10], "condition": condition, "codes": codes or []}]}]


def prediction(decision="OK", codes=None):
    return [{"capture_id": "scene", "status": "COMPLETED", "elapsed_ms": 12.,
        "objects": [{"id": "p", "bbox": [0, 0, 10, 10], "decision": decision, "codes": codes or []}]}]


class MetricsTests(unittest.TestCase):
    def test_perfect_match(self):
        r = evaluate(truth(), prediction()); self.assertEqual(r["matched"], 1); self.assertEqual(r["normal_rejected_as_ng"]["rate"], 0)
        self.assertIsNone(r["ng_passed_as_ok"]["rate"])
    def test_unmatched_prediction_and_missing_truth(self):
        p = prediction(); p[0]["objects"][0]["bbox"] = [20, 0, 30, 10]
        r = evaluate(truth(), p); self.assertEqual((r["missed"], r["false_positives"]), (1, 1))
    def test_duplicates_are_false_positives(self):
        p = prediction(); p[0]["objects"].append(dict(p[0]["objects"][0], id="p2"))
        r = evaluate(truth(), p); self.assertEqual(r["detection_precision"]["rate"], .5)
    def test_cardinality_precedes_iou(self):
        # Greedy best-IoU would consume the only viable prediction for the second object.
        pairs = match_boxes([[0, 0, 10, 10], [4, 0, 14, 10]], [[1, 0, 11, 10], [-4, 0, 6, 10]], .4)
        self.assertEqual({(a, b) for a, b, _ in pairs}, {(0, 1), (1, 0)})
    def test_threshold_is_inclusive(self):
        self.assertEqual(len(match_boxes([[0, 0, 10, 10]], [[0, 0, 5, 10]], .5)), 1)
    def test_error_is_not_an_empty_success(self):
        p = [{"capture_id": "scene", "status": "ERROR", "error": "GPU failed", "objects": [], "elapsed_ms": 5}]
        r = evaluate(truth("UNKNOWN_NG"), p)
        self.assertEqual((r["failed_frames"], r["unassessed_due_to_error"], r["missed"]), (1, 1, 0))
        self.assertEqual(r["detection_recall_all_selected"]["rate"], 0)
    def test_empty_scene_and_null_denominators(self):
        t = [{"capture_id": "scene", "objects": []}]; p = prediction(); p[0]["objects"] = []
        r = evaluate(t, p); self.assertIsNone(r["detection_recall_all_selected"]["rate"])
    def test_ng_escape_and_review_remain_separate(self):
        r = evaluate(truth("UNKNOWN_NG"), prediction("REVIEW")); self.assertEqual(r["truth_review_rate"]["rate"], 1)
        r = evaluate(truth("KNOWN_NG", ["NG03"]), prediction()); self.assertEqual(r["ng_passed_as_ok"]["rate"], 1)
    def test_multidefect_requires_each_final_code(self):
        r = evaluate(truth("KNOWN_NG", ["NG03", "NG06"]), prediction("NG", ["NG03"]))
        self.assertEqual(r["defect_code_recall"]["NG03"]["rate"], 1); self.assertEqual(r["defect_code_recall"]["NG06"]["rate"], 0)
    def test_uncertain_not_normal(self):
        r = evaluate(truth("UNCERTAIN"), prediction()); self.assertIsNone(r["normal_rejected_as_ng"]["rate"])
    def test_missing_extra_or_duplicate_frames_rejected(self):
        for p in ([], prediction()+prediction(), [dict(prediction()[0], capture_id="other")]):
            with self.subTest(p=p), self.assertRaises(ValueError): evaluate(truth(), p)
    def test_invalid_prediction_and_codes_rejected(self):
        for p in (prediction(None), prediction("OK", ["NG03"]), prediction("NG", ["fake"]), prediction("NG", ["NG03", "NG03"])):
            with self.subTest(p=p), self.assertRaises(ValueError): evaluate(truth(), p)
    def test_invalid_duration_and_threshold(self):
        for value in (float("nan"), -1, True):
            with self.subTest(value=value), self.assertRaises(ValueError): distribution([value])
        for value in (0, 1.1, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError): evaluate(truth(), prediction(), threshold=value)
    def test_distribution_retains_tail_and_count(self):
        r = distribution([1., 1., 100.]); self.assertEqual(r["n"], 3); self.assertEqual(r["max"], 100)
        self.assertGreater(r["p95"], r["median"]); self.assertIsNone(distribution([])["median"])


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.collection = make_fixture(self.root / "fixture")
        # Injectable unit-test product contract, never registered in an operating catalog.
        self.product = new_product(); self.product.update(id="fixture-part", version=1, workspace_id="main")
        self.product["quality"] = {"version": "test-quality", "validation_reference": "UNIT-TEST-ONLY", "blur_min": 0, "brightness_min": 0, "brightness_max": 255}
        self.equipment = default_equipment(); self.equipment["camera"].update(width=320, height=200)
        self.equipment["workspace"].update(roi=[[0, 0], [320, 0], [320, 200], [0, 200]], validation_reference="UNIT-TEST-ONLY")
        self.models = InjectedModels(); self.models.policy = replace(self.models.policy, product_id="fixture-part")
        self.models.detections = (detection(20, 25, 100, 130), detection(175, 30, 115, 130))
        self.runner = ReplayRunner(self.root, self.product, self.equipment, models=self.models)
    def tearDown(self): self.temp.cleanup()
    def run_replay(self):
        return self.runner.run(self.collection.root, self.root / "report", allow_synthetic=True)
    def test_select_only_heldout_full_scenes(self):
        _, records = select_collection(self.collection.root)
        self.assertEqual({r["split"] for r in records}, {"test", "challenge"}); self.assertEqual(len(records), 2)
        with self.assertRaises(ValueError): select_collection(self.collection.root, ("train",))
    def test_tampered_image_rejected_before_model_load(self):
        record = next(r for r in self.collection.data["records"] if r["split"] == "test")
        self.collection.image_path(record).write_bytes(b"changed")
        with self.assertRaises(ValueError): self.run_replay()
        self.assertEqual(self.models.loads, 0)
    def test_unreviewed_selected_record_rejected(self):
        data = deepcopy(self.collection.data); data["records"][2]["reviewed"] = False
        write_json(self.collection.root / "collection.json", data)
        with self.assertRaises(ValueError): self.run_replay()
    def test_resident_load_once_snapshot_not_live_no_operator_database(self):
        r = self.run_replay(); self.assertEqual(r["frames"], 2); self.assertEqual(r["matched"], 4)
        self.assertEqual(self.models.loads, 1); self.assertTrue(self.models.closed)
        _, result, _ = load_snapshot(self.root / "report/snapshots/000000")
        self.assertEqual(result["mode"], "model_file"); self.assertFalse(result["robot_commands_enabled"])
        self.assertFalse(any(self.root.rglob("operation.sqlite3"))); self.assertFalse(r["provenance"]["product_accuracy_validated"])
    def test_inference_failure_counted_for_every_frame(self):
        with patch.object(self.models.detector, "detect", side_effect=RuntimeError("injected GPU failure")):
            r = self.run_replay()
        self.assertEqual(r["failed_frames"], 2); self.assertEqual(r["unassessed_due_to_error"], 4); self.assertTrue(self.models.closed)
    def test_model_load_failure_writes_failed_manifest(self):
        with patch.object(self.models, "load", side_effect=RuntimeError("load failed")), self.assertRaises(RuntimeError): self.run_replay()
        self.assertEqual(read_json(self.root / "report/run.json")["status"], "FAILED"); self.assertTrue(self.models.closed)
    def test_roi_does_not_remove_ground_truth_from_denominator(self):
        self.runner.equipment["workspace"]["roi"] = [[0, 0], [160, 0], [160, 200], [0, 200]]
        r = self.run_replay(); self.assertEqual(r["truth_objects"], 4); self.assertEqual(r["missed"], 2)
    def test_resolution_mismatch_is_error_not_rescaled_success(self):
        self.runner.equipment["camera"]["width"] = 640
        r = self.run_replay(); self.assertEqual(r["failed_frames"], 2)
    def test_live_frame_rejected(self):
        with self.assertRaises(ValueError): self.runner.inspect(frame())
    def test_unvalidated_workspace_skips_checks_and_cannot_pass(self):
        self.runner.equipment["workspace"]["validation_reference"] = ""
        self.models.defect = True
        r = self.run_replay(); self.assertEqual(self.models.calls, 0)
        self.assertEqual(r["truth_review_rate"]["rate"], 1)
    def test_bad_quality_skips_defect_calls_and_reports_review(self):
        self.runner.product["quality"]["brightness_min"] = 250
        self.models.defect = True
        r = self.run_replay(); self.assertEqual(self.models.calls, 0)
        self.assertEqual(r["truth_review_rate"]["rate"], 1)
    def test_invalid_threshold_does_not_load_models(self):
        with self.assertRaises(ValueError): self.runner.run(self.collection.root, self.root / "report", threshold=0, allow_synthetic=True)
        self.assertEqual(self.models.loads, 0); self.assertFalse((self.root / "report").exists())
    def test_missing_session_database_is_not_created(self):
        import sqlite3
        with self.assertRaises(sqlite3.OperationalError): session_configuration(self.root, "missing")
        self.assertFalse((self.root / "operation.sqlite3").exists())
    def test_frozen_session_config_read_only(self):
        from mes_vision.operation.catalog import OperationStore
        store = OperationStore(self.root / "runtime")
        session = store.start_session(self.product, self.equipment)
        p, e = session_configuration(store.root, session)
        self.assertEqual(p, self.product); self.assertEqual(e, self.equipment)
        with self.assertRaises(ValueError): session_configuration(store.root, "missing")
    def test_explicit_synthetic_opt_in_and_no_overwrite(self):
        with self.assertRaises(ValueError): self.runner.run(self.collection.root, self.root / "report")
        self.run_replay()
        with self.assertRaises(FileExistsError): self.run_replay()


if __name__ == "__main__": unittest.main()

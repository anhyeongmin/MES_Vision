from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

from mes_vision.anomaly.features import fingerprint
from mes_vision.decision import apply_policy, load_policy, CheckRule
from mes_vision.decision.fixtures import make_case
from mes_vision.decision.io import run_from_dict
from mes_vision.decision.policy import load_policy_data
from mes_vision.inspection import Box, CheckStatus, Finding, Mode, ModelRef
from mes_vision.training.data import write_json
from mes_vision.vlm.snapshots import save_snapshot, load_snapshot
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.backend import GenerationConfig, make_messages

ROOT = Path(__file__).resolve().parents[1]


class DecisionTests(unittest.TestCase):
    def setUp(self):
        self.data = make_case()
        self.raw, self.policy, self.evidence = (self.data[k] for k in ("raw", "policy", "evidence"))

    def change_check(self, name, **changes):
        checks = self.raw.objects[0].checks
        index = next(i for i, c in enumerate(checks) if c.check_id == name)
        checks[index] = replace(checks[index], **changes)

    def evaluate(self, **kwargs):
        return apply_policy(self.raw, kwargs.get("policy", self.policy), kwargs.get("evidence", self.evidence))

    def codes(self, result=None):
        return {r["code"] for r in (result or self.evaluate()).objects[0].decision_details["reasons"]}

    def test_all_required_pass_is_ok_but_never_robot_authority(self):
        self.raw.robot_commands_enabled = True
        result = self.evaluate()
        self.assertEqual(result.final_decision, "OK")
        self.assertEqual(result.objects[0].final_decision, "OK")
        self.assertEqual(result.decision_status, "SIMULATED")
        self.assertTrue(result.decision_details["required_checks_complete"])
        self.assertFalse(result.robot_commands_enabled)

    def test_mixed_objects_keep_ng_and_error_and_review_independently(self):
        data = make_case(mixed=True)
        r = data["result"]
        self.assertEqual([o.final_decision for o in r.objects], ["OK", "NG", "REVIEW"])
        self.assertEqual(r.objects[1].decision_details["defect_codes"], ["NG03", "NG06"])
        self.assertFalse(r.objects[1].decision_details["required_checks_complete"])
        self.assertEqual(r.final_decision, "NG")
        self.assertIn("CHECK_ERROR", {x["code"] for x in r.objects[1].decision_details["reasons"]})

    def test_source_is_preserved_and_output_has_independent_storage(self):
        before = deepcopy(self.raw.to_dict())
        r = self.evaluate()
        self.assertEqual(before, self.raw.to_dict())
        r.objects[0].checks[0].details["changed"] = True
        r.config["product_id"] = "other"
        self.assertEqual(before, self.raw.to_dict())

    def test_missing_error_not_run_and_uncertain_required_never_ok(self):
        for state in (CheckStatus.ERROR, CheckStatus.NOT_RUN, CheckStatus.UNCERTAIN):
            with self.subTest(state=state):
                self.change_check("geometry", status=state)
                self.assertEqual(self.evaluate().final_decision, "REVIEW")
        self.raw.objects[0].checks = [c for c in self.raw.objects[0].checks if c.check_id != "geometry"]
        self.assertIn("REQUIRED_CHECK_MISSING", self.codes())

    def test_optional_geometry_must_be_explicit_and_does_not_fake_completion(self):
        self.change_check("geometry", status=CheckStatus.NOT_RUN)
        rule = replace(self.policy.rules[2], required=False, optional_reason="SYNTHETIC symmetric test part")
        r = self.evaluate(policy=replace(self.policy, rules=(*self.policy.rules[:2], rule)))
        self.assertEqual(r.final_decision, "OK")
        self.assertEqual(r.objects[0].decision_details["checks"][2]["status"], "REVIEW")
        with self.assertRaises(ValueError): replace(rule, optional_reason=None)

    def test_optional_valid_failure_is_still_ng(self):
        self.change_check("geometry", status=CheckStatus.FAIL, findings=(Finding("hole missing", "NG05"),))
        rule = replace(self.policy.rules[2], required=False, optional_reason="test optional")
        self.assertEqual(self.evaluate(policy=replace(self.policy, rules=(*self.policy.rules[:2], rule))).final_decision, "NG")

    def test_known_and_anomaly_cannot_be_optional_or_omitted(self):
        with self.assertRaises(ValueError): replace(self.policy, rules=self.policy.rules[1:])
        with self.assertRaises(ValueError): replace(self.policy, rules=(replace(self.policy.rules[0], required=False, optional_reason="skip"), *self.policy.rules[1:]))

    def test_vlm_is_ignored_for_all_statuses_and_cannot_be_required(self):
        for state in CheckStatus:
            with self.subTest(state=state):
                self.change_check("vlm", status=state, criteria_version="anything", findings=(Finding("VLM claim", "NG01"),) if state == CheckStatus.FAIL else ())
                self.assertEqual(self.evaluate().final_decision, "OK")
        with self.assertRaises(ValueError): CheckRule("vlm", True)

    def test_policy_unvalidated_and_rule_unvalidated_never_establish_ng(self):
        self.change_check("known_defects", findings=(Finding("crack", "NG03", .95),))
        self.assertEqual(self.evaluate(policy=replace(self.policy, validated=False)).final_decision, "REVIEW")
        p = replace(self.policy, rules=(replace(self.policy.rules[0], validated=False), *self.policy.rules[1:]))
        self.assertEqual(self.evaluate(policy=p).final_decision, "REVIEW")

    def test_unconfigured_template_stays_review(self):
        p = load_policy(ROOT / "configs/decision/unconfigured.json")
        r = self.evaluate(policy=p, evidence=None)
        self.assertEqual(r.final_decision, "REVIEW")
        self.assertFalse(r.robot_commands_enabled)

    def test_defect_threshold_boundary_and_review_band(self):
        for score, expected in ((.4, "REVIEW"), (.799999, "REVIEW"), (.8, "NG"), (1., "NG")):
            with self.subTest(score=score):
                self.change_check("known_defects", findings=(Finding("crack", "NG03", score),))
                self.assertEqual(self.evaluate().final_decision, expected)

    def test_candidate_invalid_code_score_and_collection_threshold(self):
        for finding in (Finding("unknown", None, .9), Finding("missing", "NG01", .9), Finding("crack", "NG03", None), Finding("crack", "NG03", .39999)):
            with self.subTest(finding=finding):
                self.change_check("known_defects", findings=(finding,))
                self.assertIn("INVALID_DEFECT_CANDIDATE", self.codes())
                self.assertEqual(self.evaluate().final_decision, "REVIEW")
        self.change_check("known_defects", findings=(), candidate_threshold=.5)
        self.assertIn("CANDIDATE_THRESHOLD_MISMATCH", self.codes())

    def test_class_mapping_change_is_rejected_even_when_weights_match(self):
        self.change_check("known_defects", details={"class_codes": {"0": "NG06", "1": "NG03"}})
        self.assertIn("DEFECT_CLASS_MAPPING_MISMATCH", self.codes())
        self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_confirmed_candidate_survives_additional_borderline_candidate(self):
        self.change_check("known_defects", findings=(Finding("crack", "NG03", .8), Finding("surface", "NG06", .7)))
        r = self.evaluate()
        self.assertEqual(r.final_decision, "NG")
        self.assertEqual(r.objects[0].decision_details["defect_codes"], ["NG03"])
        self.assertIn("DEFECT_CANDIDATE_IN_REVIEW_BAND", self.codes(r))

    def test_anomaly_boundaries_and_unknown_code(self):
        for score, status, verdict in ((.2, CheckStatus.PASS, "OK"), (.200001, CheckStatus.UNCERTAIN, "REVIEW"),
                                       (.799999, CheckStatus.UNCERTAIN, "REVIEW"), (.8, CheckStatus.FAIL, "NG"), (2., CheckStatus.FAIL, "NG")):
            with self.subTest(score=score):
                self.change_check("anomaly", raw_score=score, status=status, findings=(Finding("different region", "NG_UNKNOWN", score),) if status == CheckStatus.FAIL else ())
                r = self.evaluate()
                self.assertEqual(r.final_decision, verdict)
                if verdict == "NG": self.assertEqual(r.objects[0].decision_details["defect_codes"], ["NG_UNKNOWN"])

    def test_anomaly_cannot_name_crack_from_distance(self):
        self.change_check("anomaly", status=CheckStatus.FAIL, raw_score=.9, findings=(Finding("crack", "NG03"),))
        self.assertIn("ANOMALY_CANNOT_NAME_DEFECT", self.codes())
        self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_anomaly_same_version_modified_thresholds_and_status_conflict_rejected(self):
        details = deepcopy(self.raw.objects[0].checks[1].details)
        details["criteria"]["fail_min"] = .7
        self.change_check("anomaly", details=details)
        self.assertIn("ANOMALY_CRITERIA_MISMATCH", self.codes())
        self.setUp()
        self.change_check("anomaly", raw_score=.9)
        self.assertIn("ANOMALY_SCORE_STATUS_CONFLICT", self.codes())

    def test_model_criteria_and_product_mismatch_never_pass(self):
        for changes in ({"model": ModelRef("different", "2", "mock")}, {"criteria_version": "old"}, {"details": {"product_id": "other"}}):
            with self.subTest(changes=changes):
                self.setUp(); self.change_check("geometry", **changes)
                self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_fail_without_confirmed_code_is_review(self):
        for findings in ((), (Finding("suspicion"),)):
            self.change_check("geometry", status=CheckStatus.FAIL, findings=findings)
            self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_stale_check_duplicate_and_crop_binding_override_ng(self):
        for mutation in ("stale", "duplicate", "crop", "finding"):
            with self.subTest(mutation=mutation):
                self.setUp()
                self.change_check("known_defects", findings=(Finding("crack", "NG03", .99),))
                obj = self.raw.objects[0]
                if mutation == "stale": self.change_check("geometry", frame_id="previous-frame")
                if mutation == "duplicate": obj.checks.append(obj.checks[0])
                if mutation == "crop": obj.crop["crop_to_original"][0][2] += 1
                if mutation == "finding": self.change_check("known_defects", findings=(Finding("crack", "NG03", .99, Box(1, 1, 2, 2), Box(1, 1, 2, 2)),))
                self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_wrong_frame_evidence_product_and_detector_override_ng(self):
        self.change_check("known_defects", findings=(Finding("crack", "NG03", .99),))
        self.assertEqual(self.evaluate(evidence=replace(self.evidence, run_id="previous")).final_decision, "REVIEW")
        self.raw.detector = ModelRef("wrong detector", "1", "mock")
        self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_no_frame_evidence_and_bad_quality_block_ok(self):
        self.assertEqual(self.evaluate(evidence=None).final_decision, "REVIEW")
        for state in (CheckStatus.FAIL, CheckStatus.ERROR, CheckStatus.NOT_RUN, CheckStatus.UNCERTAIN):
            self.assertEqual(self.evaluate(evidence=replace(self.evidence, quality_status=state)).final_decision, "REVIEW")
        self.assertEqual(self.evaluate(evidence=replace(self.evidence, workspace_status=CheckStatus.FAIL)).final_decision, "REVIEW")

    def test_count_missing_mismatch_zero_and_unknown_issues_block_ok(self):
        for value in (None, 0, 2, True):
            self.raw.config["expected_count"] = value
            self.assertEqual(self.evaluate().final_decision, "REVIEW")
        self.raw.config["expected_count"] = 1
        self.raw.issues.append("NEW_UNHANDLED_PROBLEM")
        self.assertEqual(self.evaluate().final_decision, "REVIEW")
        self.raw.issues.clear(); self.raw.objects.clear(); self.raw.config["expected_count"] = 0
        self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_duplicate_object_and_execution_failure_never_ok(self):
        self.raw.objects.append(deepcopy(self.raw.objects[0])); self.raw.config["expected_count"] = 2
        self.assertEqual(self.evaluate().final_decision, "REVIEW")
        self.setUp(); self.raw.execution_status = "ERROR"
        self.assertEqual(self.evaluate().final_decision, "REVIEW")

    def test_edge_overlap_and_unknown_object_issue_block_ok(self):
        self.raw.objects[0].issues.append("UNRECOGNIZED_OBJECT_PROBLEM")
        self.assertEqual(self.evaluate().final_decision, "REVIEW")
        data = make_case(mixed=True)
        data["raw"].objects[0].effective_box = data["raw"].objects[1].effective_box
        r = apply_policy(data["raw"], data["policy"], data["evidence"])
        self.assertEqual(r.objects[0].final_decision, "REVIEW")

    def test_policy_schema_rejects_mock_as_real_nan_duplicates_and_unknown_fields(self):
        with self.assertRaises(ValueError): replace(self.policy, kind="real")
        with self.assertRaises(ValueError): replace(self.policy, rules=(*self.policy.rules, self.policy.rules[0]))
        with self.assertRaises(ValueError): replace(self.policy.rules[0], candidate_threshold=float("nan"))
        data = asdict(self.policy); data["allow_vlm_override"] = True
        with self.assertRaises(TypeError): load_policy_data(data)

    def test_mutated_frozen_policy_dictionary_is_revalidated(self):
        self.policy.rules[0].fail_thresholds["NG03"] = float("nan")
        with self.assertRaises(ValueError): self.evaluate()

    def test_serialized_roundtrip_and_policy_fingerprint_track_full_settings(self):
        r = self.evaluate()
        loaded = run_from_dict(json.loads(json.dumps(r.to_dict())))
        self.assertEqual(json.loads(json.dumps(loaded.to_dict())), json.loads(json.dumps(r.to_dict())))
        p = replace(self.policy, version="synthetic-v2")
        self.assertNotEqual(r.decision_details["policy_digest"], self.evaluate(policy=p).decision_details["policy_digest"])
        bad = self.raw.to_dict(); bad["objects"][0]["checks"][0]["status"] = "SUCCESS"
        with self.assertRaises(ValueError): run_from_dict(bad)

    def test_pipeline_opt_in_and_frame_provider_failure(self):
        pipeline = self.data["pipeline"]
        pipeline.decision_policy = self.policy
        def evidence(frame, run_id): return replace(self.evidence, run_id=run_id)
        self.assertEqual(pipeline.run(self.data["frame"], frame_evidence_provider=evidence).final_decision, "OK")
        def fail(frame, run_id): raise RuntimeError("camera quality unavailable")
        self.assertEqual(pipeline.run(self.data["frame"], frame_evidence_provider=fail).final_decision, "REVIEW")

    def test_decision_survives_vlm_off_snapshot_and_completed_advice(self):
        r = self.evaluate()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save_snapshot(root / "snapshot", self.data["frame"], r, kind="synthetic")
            queue = AnalysisQueue(root / "queue")
            identity = queue.enqueue(root / "snapshot", r.objects[0].object_id, asdict(GenerationConfig()))
            self.assertEqual(queue.get(identity)["state"], "SKIPPED_DISABLED")
            _, prompt = make_messages(queue.get(identity), GenerationConfig())
            self.assertTrue(prompt["context"]["decision_evidence"]["required_checks_complete"])
            self.assertEqual(prompt["context"]["decision_evidence"]["policy_digest"], r.decision_details["policy_digest"])
            queue.set_enabled(True); queue.retry(identity)
            job = queue.claim("test")
            queue.finish(identity, job["token"], "COMPLETED", result={"analysis": {"observation": "different opinion", "needs_review": True}})
            queue.set_enabled(False)
            _, saved, _ = load_snapshot(root / "snapshot")
            self.assertEqual(saved["objects"][0]["final_decision"], "OK")
            self.assertEqual(queue.get(identity)["payload"]["base_decision"], "OK")


if __name__ == "__main__": unittest.main()

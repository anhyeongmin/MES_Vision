from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

from filelock import Timeout
from mes_vision.inspection import Mode
from mes_vision.robot import RobotController, UnconfiguredDobotMagician, Pose, load_profile, build_plan
from mes_vision.robot.fixtures import make_fixture, prepare, drive
from mes_vision.robot.planning import STAGES

ROOT = Path(__file__).resolve().parents[1]


class RobotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.data = make_fixture()

    def tearDown(self): self.temp.cleanup()

    def plan(self, **kwargs):
        d = dict(self.data, **kwargs)
        return prepare(d)

    def controller(self, data=None): return RobotController(self.root / "journal", (data or self.data)["adapter"])

    def test_verified_cycle_has_ordered_commands_and_requires_recapture(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=self.data["now"])
            self.assertEqual(drive(c, self.data), "RECAPTURE")
            self.assertEqual(tuple(x.stage for x in self.data["adapter"].commands), STAGES)
            self.assertFalse(self.data["adapter"].holding)
            self.assertEqual(c.journal.events()[-1]["event"], "CYCLE_VERIFIED")

    def test_acceptance_is_not_completion_or_pick_verification(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000)
            c.tick(self.data["scene"], now=1000)
            self.assertEqual(c.index, 0)
            self.assertFalse(c.journal.events()[-1]["payload"]["success_confirmed"])
            c.tick(self.data["scene"], now=1000.1)
            self.assertEqual(c.index, 0)

    def test_ng_uses_ng_destination_not_frame_aggregate(self):
        ok = self.plan()
        ng = prepare(make_fixture(decision="NG"))
        self.assertEqual(ok.commands[6].target.y, 50)
        self.assertEqual(ng.commands[6].target.y, -50)

    def test_review_requires_explicit_quarantine_configuration(self):
        with self.assertRaises(ValueError): prepare(make_fixture(decision="REVIEW"))
        self.assertEqual(prepare(make_fixture(decision="REVIEW", allow_review=True)).commands[6].target.x, 200)

    def test_missing_target_calibration_and_unvalidated_profile_are_blocked(self):
        for changes in ({"target": None}, {"profile": replace(self.data["profile"], validated=False)},
                        {"target": replace(self.data["target"], calibration_version="other")},
                        {"target": replace(self.data["target"], validated=False)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.plan(**changes)

    def test_real_mode_and_real_adapter_have_no_transport(self):
        data = deepcopy(self.data["inspection"]); data.mode = Mode.MODEL_FILE
        with self.assertRaises(ValueError): self.plan(inspection=data)
        with RobotController(self.root / "real", UnconfiguredDobotMagician()) as c:
            with self.assertRaises(ValueError): c.start(self.plan(), self.data["scene"], now=1000)
        with self.assertRaises(RuntimeError): UnconfiguredDobotMagician().submit(self.plan().commands[0])
        self.assertFalse(load_profile(ROOT / "configs/robot/unconfigured.json").validated)

    def test_stale_future_frame_and_wrong_object_are_blocked(self):
        for changes in ({"now": 1002.01}, {"now": 999.99}, {"object_id": "previous"},
                        {"target": replace(self.data["target"], frame_id="previous")}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.plan(**changes)

    def test_workspace_pixel_units_and_grasp_point_bounds(self):
        with self.assertRaises(ValueError): Pose(10, 20, 30, 0, "input_rgb_pixels")
        with self.assertRaises(ValueError): self.plan(target=replace(self.data["target"], x_mm=10000))
        with self.assertRaises(ValueError): self.plan(target=replace(self.data["target"], source_pixel=(0, 0)))
        with self.assertRaises(ValueError): Pose(float("nan"), 0, 0, 0)

    def test_inspection_and_profile_not_mutated_and_plan_independent(self):
        before = deepcopy(self.data["inspection"].to_dict())
        plan = self.plan()
        self.data["profile"].destinations["NG"] = Pose(999, 999, 10, 0)
        self.assertNotEqual(plan.profile.destinations["NG"].x, 999)
        self.assertEqual(self.data["inspection"].to_dict(), before)

    def test_changed_decision_without_matching_evidence_is_blocked(self):
        self.data["inspection"].objects[0].final_decision = "NG"
        with self.assertRaises(ValueError): self.plan()

    def test_single_controller_lock(self):
        with self.controller():
            with self.assertRaises(Timeout): self.controller()

    def test_duplicate_plan_and_same_frame_other_object_cannot_move(self):
        with self.controller() as c:
            plan = self.plan(); c.start(plan, self.data["scene"], now=1000)
            drive(c, self.data)
            with self.assertRaises(ValueError): c.start(plan, self.data["scene"], now=1001)
            other = next(o for o in self.data["inspection"].objects if o.final_decision == "NG")
            b = other.effective_box
            target = replace(self.data["target"], object_id=other.object_id, source_pixel=((b.x1+b.x2)/2, (b.y1+b.y2)/2))
            plan2 = self.plan(object_id=other.object_id, target=target)
            with self.assertRaises(ValueError): c.start(plan2, self.data["scene"], now=1001)

    def test_second_start_while_busy_is_rejected(self):
        with self.controller() as c:
            plan = self.plan(); c.start(plan, self.data["scene"], now=1000)
            with self.assertRaises(ValueError): c.start(plan, self.data["scene"], now=1000)

    def test_scene_calibration_and_connection_changes_stop_motion(self):
        for mutation in ("scene", "calibration", "connection"):
            with self.subTest(mutation=mutation):
                data = make_fixture()
                with RobotController(self.root / mutation, data["adapter"]) as c:
                    c.start(prepare(data), data["scene"], now=1000)
                    c.tick(data["scene"], now=1000)
                    scene = replace(data["scene"], revision=2) if mutation == "scene" else replace(data["scene"], calibration_version="new") if mutation == "calibration" else data["scene"]
                    if mutation == "connection": data["adapter"].epoch = "reconnected"
                    self.assertEqual(c.tick(scene, now=1000.1), "FAULT")
                    self.assertEqual(len(data["adapter"].commands), 1)

    def test_reject_timeout_disconnect_failure_unknown_stop_without_retry(self):
        for fault in ("reject", "timeout", "disconnect", "failure", "unknown", "wrong_id", "wrong_pose"):
            with self.subTest(fault=fault):
                data = make_fixture(faults={"APPROACH_PICK": fault})
                with RobotController(self.root / fault, data["adapter"]) as c:
                    c.start(prepare(data), data["scene"], now=1000)
                    self.assertEqual(drive(c, data), "FAULT")
                    c.tick(data["scene"], now=1010)
                    self.assertEqual(len(data["adapter"].commands), 1)
                    self.assertGreaterEqual(data["adapter"].stop_calls, 1)

    def test_pick_and_place_must_be_independently_verified(self):
        for stage in ("VERIFY_PICK", "VERIFY_PLACE"):
            data = make_fixture(faults={stage: "unverified"})
            with RobotController(self.root / stage, data["adapter"]) as c:
                c.start(prepare(data), data["scene"], now=1000)
                self.assertEqual(drive(c, data), "FAULT")
                self.assertEqual(data["adapter"].commands[-1].stage, stage)

    def test_user_stop_preserves_held_part_and_requires_reconciliation(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000)
            self.assertEqual(drive(c, self.data, stop_stage="LIFT_PICK"), "STOPPED")
            self.assertTrue(self.data["adapter"].holding)
            self.assertNotIn("release", [x.action for x in self.data["adapter"].commands])
            with self.assertRaises(ValueError): c.recover(confirmation_reference="test")
            self.data["adapter"].reconcile_empty()
            with self.assertRaises(ValueError): c.recover(confirmation_reference="")
            c.recover(confirmation_reference="SYNTHETIC_OPERATOR_CONFIRMATION")
            self.assertEqual(c.state, "RECAPTURE")

    def test_restart_never_resumes_interrupted_motion(self):
        c = self.controller(); c.start(self.plan(), self.data["scene"], now=1000)
        c.tick(self.data["scene"], now=1000)
        # Simulate process death: release ownership without normal close/stop.
        c.owner.release(); c.closed = True
        count = len(self.data["adapter"].commands)
        with self.controller() as restarted:
            self.assertEqual(restarted.state, "RECOVERY")
            restarted.tick(self.data["scene"], now=1000.2)
            self.assertEqual(len(self.data["adapter"].commands), count)
            with self.assertRaises(ValueError): restarted.start(self.plan(), self.data["scene"], now=1000.2)

    def test_completed_frame_remains_consumed_after_restart(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000); drive(c, self.data)
        with self.controller() as c:
            with self.assertRaises(ValueError): c.start(self.plan(), self.data["scene"], now=1001)

    def test_plan_mutation_is_rejected(self):
        plan = self.plan()
        plan.profile.destinations["NG"] = Pose(0, 0, 10, 0)
        with self.controller() as c:
            with self.assertRaises(ValueError): c.start(plan, self.data["scene"], now=1000)

    def test_frame_expiry_before_first_dispatch(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000)
            self.assertEqual(c.tick(self.data["scene"], now=1003), "FAULT")
            self.assertEqual(self.data["adapter"].commands, [])

    def test_journal_failure_before_dispatch_sends_no_move(self):
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000)
            original = c.journal.record
            def fail(*args, **kwargs): raise OSError("SIMULATED_DISK_FULL")
            c.journal.record = fail
            with self.assertRaises(OSError): c.tick(self.data["scene"], now=1000)
            self.assertEqual(self.data["adapter"].commands, [])
            self.assertGreaterEqual(self.data["adapter"].stop_calls, 1)
            c.journal.record = original

    def test_stop_at_every_stage_prevents_later_commands(self):
        for stage in STAGES:
            with self.subTest(stage=stage):
                data = make_fixture()
                with RobotController(self.root / stage, data["adapter"]) as c:
                    c.start(prepare(data), data["scene"], now=1000)
                    self.assertEqual(drive(c, data, stop_stage=stage), "STOPPED")
                    count = len(data["adapter"].commands)
                    c.tick(data["scene"], now=1020)
                    self.assertEqual(len(data["adapter"].commands), count)
                    self.assertFalse(any(e["event"] == "CYCLE_VERIFIED" for e in c.journal.events()))

    def test_next_cycle_requires_and_accepts_a_new_frame(self):
        from mes_vision.decision.policy import apply_policy, load_policy_data
        from mes_vision.decision.io import frame_evidence_from_dict
        with self.controller() as c:
            c.start(self.plan(), self.data["scene"], now=1000); drive(c, self.data)
            fresh = make_fixture(now=1010)
            fresh["adapter"] = self.data["adapter"]
            frame_id = "synthetic-policy:1"
            result = fresh["inspection"]
            result.frame["frame_id"] = frame_id
            for obj in result.objects: obj.checks = [replace(check, frame_id=frame_id) for check in obj.checks]
            ev = dict(result.decision_details["frame_evidence"], frame_id=frame_id)
            result = apply_policy(result, load_policy_data(result.decision_details["policy"]), frame_evidence_from_dict(ev))
            fresh["inspection"] = result
            fresh["target"] = replace(fresh["target"], frame_id=frame_id)
            fresh["scene"] = replace(fresh["scene"], frame_id=frame_id, revision=2, connection_epoch=fresh["adapter"].epoch)
            c.start(prepare(fresh), fresh["scene"], now=1010)
            self.assertEqual(drive(c, fresh), "RECAPTURE")
            self.assertEqual(len(fresh["adapter"].commands), 20)


if __name__ == "__main__": unittest.main()

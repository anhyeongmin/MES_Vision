from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from uuid import uuid4

from mes_vision.desktop.service import AppStore, DEFAULTS, run_inspection, validate_settings, load_recipe
from mes_vision.training.data import write_json


class DesktopServiceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.store = AppStore(self.root / "runtime")

    def inspect(self):
        output = self.root / uuid4().hex
        run_inspection(dict(mode="synthetic", settings=dict(DEFAULTS), output=str(output)), self.root)
        return output

    def test_simulation_runs_have_unique_frame_ids_and_complete_evidence(self):
        a = self.store.register(self.inspect()); b = self.store.register(self.inspect())
        self.assertNotEqual(a[0]["frame_id"], b[0]["frame_id"])
        self.assertEqual([o["final_decision"] for o in a[1]["objects"]], ["OK", "NG", "REVIEW"])
        self.assertEqual(a[0]["kind"], "synthetic")
        self.assertFalse(a[1]["robot_commands_enabled"])

    def test_history_persists_and_checks_original_integrity(self):
        path = self.inspect(); manifest, result = self.store.register(path)
        reopened = AppStore(self.store.root)
        self.assertEqual(len(reopened.list()), 1)
        self.assertEqual(reopened.open(manifest["run_id"])[2], result)
        (path / "frame.png").write_bytes(b"changed")
        with self.assertRaises(ValueError): reopened.open(manifest["run_id"])

    def test_duplicate_registration_is_idempotent_but_changed_result_rejected(self):
        path = self.inspect(); self.store.register(path); self.store.register(path)
        self.assertEqual(len(self.store.list()), 1)
        (path / "inspection.json").write_text("{}", encoding="utf-8")
        with self.assertRaises(ValueError): self.store.register(path)

    def test_settings_persist_without_treating_workspace_size_as_calibration(self):
        value = dict(DEFAULTS, product_id="sample", workspace_width_mm=300., workspace_height_mm=200.)
        self.store.save_settings(value)
        self.assertEqual(AppStore(self.store.root).settings(), value)
        self.assertNotIn("calibration", value)

    def test_nonfinite_invalid_or_mismatched_settings_rejected(self):
        for change in ({"threshold": float("nan")}, {"workspace_width_mm": -1}, {"expected_count": True},
                       {"product_id": " "}, {"recipe": {"product_id": "different"}}, {"threshold": 2}):
            with self.subTest(change=change), self.assertRaises((ValueError, KeyError)):
                validate_settings(dict(DEFAULTS, **change))
        self.assertEqual(self.store.settings(), DEFAULTS)

    def test_hardware_mode_cannot_write_a_result(self):
        with self.assertRaises(ValueError): run_inspection(dict(mode="hardware", settings=dict(DEFAULTS), output=str(self.root / "bad")), self.root)
        self.assertFalse((self.root / "bad").exists())

    def test_missing_recipe_asset_is_not_silently_replaced(self):
        path = self.root / "recipe.json"
        write_json(path, {"schema_version": 1, "product_id": "a", "objects": {
            "weights": "missing.pth", "sha256": "a"*64, "class_names": ["part"], "threshold": .4}})
        with self.assertRaises(ValueError): load_recipe(path)

    def test_saved_simulation_remains_eligible_for_policy_replay(self):
        from dataclasses import replace
        from mes_vision.calibration import fit, map_target
        from mes_vision.calibration.fixtures import make_spec
        from mes_vision.decision.io import run_from_dict
        from mes_vision.robot import SceneStamp, build_plan
        from mes_vision.robot.fixtures import make_fixture
        _, data = self.store.register(self.inspect()); result = run_from_dict(data)
        spec = make_spec(); calibration = fit(spec).accept("SYNTHETIC-TEST")
        fixture = make_fixture(); profile = replace(fixture["profile"], calibration_version=calibration.identity)
        obj = result.objects[0]
        target = map_target(calibration, result, obj.object_id, (75., 90.), spec.context,
            plane_z_mm=spec.plane_z_mm, r_deg=0., grasp_policy_version=profile.grasp_policy_version)
        scene = SceneStamp(result.run_id, result.frame["frame_id"], 1, 1000., calibration.identity, fixture["adapter"].epoch)
        plan = build_plan(result, obj.object_id, target, profile, scene, now=1000.)
        self.assertEqual(plan.object_id, obj.object_id)


if __name__ == "__main__": unittest.main()

from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from mes_vision.calibration import fit, load, spec_from_dict, map_target, Lens
from mes_vision.calibration.core import project, undistort
from mes_vision.calibration.fixtures import make_spec
from mes_vision.anomaly.features import fingerprint
from mes_vision.robot import RobotController, build_plan
from mes_vision.robot.fixtures import make_fixture, drive


class CalibrationTests(unittest.TestCase):
    def setUp(self): self.spec = make_spec()
    def accepted(self, spec=None): return fit(spec or self.spec).accept("SYNTHETIC-VALIDATION")
    def map(self, calibration, pixel=(75., 90.), **kwargs):
        return calibration.map_xy(pixel, kwargs.get("context", self.spec.context), plane_z_mm=kwargs.get("plane", 20.), kind=kwargs.get("kind", "synthetic"))

    def test_projective_fit_and_axis_direction_match_known_mapping(self):
        c = self.accepted()
        expected = project([[.5, .04, -50], [-.02, -.6, 70], [.0003, -.0001, 1]], [(75, 90)])[0]
        np.testing.assert_allclose(self.map(c), expected, atol=1e-9)
        self.assertLess(c.data["metrics"]["check_max_mm"], 1e-9)
        self.assertLess(self.map(c, (75., 120.))[1], self.map(c)[1])

    def test_acceptance_is_explicit_and_missing_reference_rejected(self):
        c = fit(self.spec)
        self.assertFalse(c.ready)
        with self.assertRaises(ValueError): self.map(c)
        with self.assertRaises(ValueError): c.accept("")
        self.assertTrue(c.accept("synthetic").ready)

    def test_independent_check_failure_blocks_acceptance(self):
        pairs = list(self.spec.pairs)
        pairs[-1] = replace(pairs[-1], robot_xy_mm=(pairs[-1].robot_xy_mm[0]+3, pairs[-1].robot_xy_mm[1]))
        c = fit(replace(self.spec, pairs=tuple(pairs)))
        self.assertIn("CHECK_MAX_MM_EXCEEDED", c.data["failures"])
        with self.assertRaises(ValueError): c.accept("cannot approve failed checks")

    def test_outlier_not_silently_removed(self):
        pairs = list(self.spec.pairs)
        pairs[2] = replace(pairs[2], robot_xy_mm=(pairs[2].robot_xy_mm[0]+10, pairs[2].robot_xy_mm[1]))
        c = fit(replace(self.spec, pairs=tuple(pairs)))
        self.assertTrue(c.data["failures"])
        self.assertEqual(len(c.data["residuals"]), len(pairs))

    def test_duplicate_fit_check_coordinates_and_ids_rejected(self):
        for changes in ({"point_id": self.spec.pairs[0].point_id}, {"pixel": self.spec.pairs[0].pixel}, {"robot_xy_mm": self.spec.pairs[0].robot_xy_mm}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(self.spec, pairs=(*self.spec.pairs[:-1], replace(self.spec.pairs[-1], **changes)))

    def test_check_sessions_must_be_independent(self):
        pairs = tuple(replace(p, measurement_session="same-session") for p in self.spec.pairs)
        with self.assertRaises(ValueError): replace(self.spec, pairs=pairs)

    def test_minimum_fit_and_holdout_counts(self):
        with self.assertRaises(ValueError): replace(self.spec, pairs=self.spec.pairs[:4]+self.spec.pairs[8:])
        with self.assertRaises(ValueError): replace(self.spec, pairs=self.spec.pairs[:10])

    def test_collinear_robot_points_are_degenerate(self):
        pairs = tuple(replace(p, robot_xy_mm=(float(i), float(i*2))) if p.role == "fit" else p for i, p in enumerate(self.spec.pairs))
        with self.assertRaises(ValueError): fit(replace(self.spec, pairs=pairs))

    def test_application_polygon_must_be_supported_and_convex(self):
        with self.assertRaises(ValueError): fit(replace(self.spec, application_polygon=((0., 0.), (479., 0.), (479., 179.), (0., 179.))))
        with self.assertRaises(ValueError): replace(self.spec, application_polygon=((15., 15.), (465., 165.), (465., 15.), (15., 165.)))

    def test_holdout_spatial_coverage_cannot_be_only_center(self):
        pairs = list(self.spec.pairs[:8])
        matrix = [[.5, .04, -50], [-.02, -.6, 70], [.0003, -.0001, 1]]
        from mes_vision.calibration import Pair
        for i, px in enumerate(((230., 80.), (250., 80.), (250., 100.), (230., 100.))):
            pairs.append(Pair(f"check-new-{i}", "check", "check-session", px, tuple(float(v) for v in project(matrix, [px])[0])))
        c = fit(replace(self.spec, pairs=tuple(pairs)))
        self.assertIn("CHECK_COVERAGE_INSUFFICIENT", c.data["failures"])

    def test_extrapolation_and_boundary(self):
        c = self.accepted()
        self.map(c, (15., 15.))
        for px in ((14., 15.), (480., 30.), (30., 170.)):
            with self.subTest(pixel=px), self.assertRaises(ValueError): self.map(c, px)

    def test_camera_mount_acquisition_base_tool_resolution_and_transforms_bound(self):
        c = self.accepted()
        for changes in ({"camera_id": "other"}, {"mount_revision": "moved"}, {"acquisition_revision": "zoomed"},
                        {"robot_base_id": "rehomed-differently"}, {"tool_frame_id": "changed"}, {"image_size": (960, 360)}, {"transformations": ("flipped",)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError): self.map(c, context=replace(self.spec.context, **changes))

    def test_unknown_wrong_height_and_kind_rejected(self):
        c = self.accepted()
        for height in (None, 20.1, float("nan"), True):
            with self.subTest(height=height), self.assertRaises(ValueError): self.map(c, plane=height)
        with self.assertRaises(ValueError): self.map(c, kind="real")

    def test_height_tolerance_is_explicit(self):
        c = self.accepted(replace(self.spec, plane_tolerance_mm=.5))
        self.map(c, plane=20.5)
        with self.assertRaises(ValueError): self.map(c, plane=20.50001)

    def test_lens_brown_correction_matches_generated_geometry(self):
        lens = Lens("opencv_brown5", "SYNTHETIC-INTRINSICS", ((400., 0., 240.), (0., 400., 90.), (0., 0., 1.)), (.2, -.05, .001, -.001, .01))
        matrix = [[.5, .04, -50], [-.02, -.6, 70], [.0003, -.0001, 1]]
        pairs = tuple(replace(p, robot_xy_mm=tuple(float(v) for v in project(matrix, undistort([p.pixel], lens))[0])) for p in self.spec.pairs)
        spec = replace(self.spec, lens=lens, pairs=pairs)
        actual = self.map(self.accepted(spec))
        np.testing.assert_allclose(actual, project(matrix, undistort([(75., 90.)], lens))[0], atol=1e-8)

    def test_unknown_distortion_model_and_invalid_matrix_rejected(self):
        with self.assertRaises(ValueError): Lens("inverse_brown_conrady", "SDK enum is not an OpenCV model")
        with self.assertRaises(ValueError): Lens("none_verified", "")
        with self.assertRaises(ValueError): Lens("opencv_brown5", "test", ((0, 0, 1), (0, 1, 1), (0, 0, 1)), (0, 0, 0, 0, 0))

    def test_saved_reload_and_no_overwrite(self):
        c = self.accepted()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"
            c.save(path); loaded = load(path)
            self.assertEqual(loaded.identity, c.identity)
            self.assertEqual(self.map(loaded), self.map(c))
            with self.assertRaises(FileExistsError): c.save(path)

    def test_tampered_matrix_even_rehashed_is_rejected_by_measurement_replay(self):
        c = self.accepted()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "calibration.json"; c.save(path)
            data = json.loads(path.read_text(encoding="utf-8")); data["data"]["matrix_ideal_pixel_to_robot_xy"][0][2] += 1
            path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError): load(path)
            data["sha256"] = fingerprint(data["data"]); path.write_text(json.dumps(data), encoding="utf-8")
            with self.assertRaises(ValueError): load(path)

    def test_same_version_changed_measurements_or_acceptance_changes_identity(self):
        a = fit(self.spec).accept("test-A"); b = fit(self.spec).accept("test-B")
        self.assertNotEqual(a.identity, b.identity)
        data = a.data; data["matrix_ideal_pixel_to_robot_xy"][0][0] = 999
        self.assertNotEqual(a.data["matrix_ideal_pixel_to_robot_xy"][0][0], 999)

    def test_spec_unknown_fields_and_unconfigured_template_rejected(self):
        data = asdict(self.spec); data["automatic_robot_motion"] = True
        with self.assertRaises(TypeError): spec_from_dict(data)
        with self.assertRaises((KeyError, TypeError, ValueError)): spec_from_dict({"state": "UNCONFIGURED", "workspace_width": 100})

    def test_calibrated_target_runs_through_robot_simulation(self):
        data = make_fixture()
        c = self.accepted()
        target = map_target(c, data["inspection"], data["object_id"], (75., 90.), self.spec.context,
                            plane_z_mm=20., r_deg=0, grasp_policy_version=data["profile"].grasp_policy_version)
        self.assertEqual(target.calibration_version, c.identity)
        data["target"] = target
        data["profile"] = replace(data["profile"], calibration_version=c.identity)
        data["scene"] = replace(data["scene"], calibration_version=c.identity)
        plan = build_plan(data["inspection"], data["object_id"], target, data["profile"], data["scene"], now=data["now"])
        with tempfile.TemporaryDirectory() as temporary, RobotController(temporary, data["adapter"]) as controller:
            controller.start(plan, data["scene"], now=data["now"])
            self.assertEqual(drive(controller, data), "RECAPTURE")

    def test_bridge_rejects_wrong_object_pixel_product_and_resized_frame(self):
        c = self.accepted(); data = make_fixture()
        def attempt(pixel=(75., 90.)):
            return map_target(c, data["inspection"], data["object_id"], pixel, self.spec.context, plane_z_mm=20., r_deg=0, grasp_policy_version="test-grasp")
        with self.assertRaises(ValueError): attempt((250., 90.))
        data["inspection"].frame["width"] = 960
        with self.assertRaises(ValueError): attempt()
        data["inspection"].frame["width"] = 480; data["inspection"].config["product_id"] = "different"
        with self.assertRaises(ValueError): attempt()


if __name__ == "__main__": unittest.main()

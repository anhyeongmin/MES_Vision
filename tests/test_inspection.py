from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.inputs import Frame, SourceKind
from mes_vision.inspection import *
from mes_vision.inspection.adapters import MockDetector, MockInspector
from mes_vision.inspection.geometry import ResizeMap, extract_crop
from mes_vision.inspection.rfdetr_adapter import RFDETRBackend, RFDETRDetector, RFDETRDefectInspector


def frame(identity="session:0"):
    rgb = np.zeros((80, 160, 3), dtype=np.uint8)
    rgb[:, :, 0] = np.arange(160)
    rgb[:, :, 1] = np.arange(80)[:, None]
    rgb[23, 17] = [255, 254, 253]
    rgb.setflags(write=False)
    return Frame(identity, "session", 0, SourceKind.IMAGE, "file:///synthetic.png", datetime.now(timezone.utc), rgb)


def detections():
    return (Detection(Box(10.2, 20.2, 39.2, 50.2), .9, 0, "part"),
            Detection(Box(70, 10, 110, 60), .8, 0, "part"))


def result(crop, model, name="known_defects", status=CheckStatus.PASS, findings=()):
    return CheckResult(name, crop.frame_id, crop.object_id, status, model, findings,
                       criteria_version="synthetic-test-v1")


class InspectionTests(unittest.TestCase):
    def pipeline(self, inspectors=(), boxes=None, **kwargs):
        return InspectionPipeline(MockDetector(detections() if boxes is None else boxes), inspectors,
                                  mode=Mode.SIMULATION, **kwargs)

    def test_crops_preserve_original_single_pixel_detail_and_fractional_edges(self):
        original = frame()
        crop = extract_crop(original, "obj", detections()[0].box, margin_px=2)
        self.assertEqual(crop.bounds, Box(8, 18, 42, 53))
        np.testing.assert_array_equal(crop.rgb, original.rgb[18:53, 8:42])
        np.testing.assert_array_equal(crop.rgb[5, 9], [255, 254, 253])
        self.assertFalse(crop.rgb.flags.writeable)
        self.assertFalse(np.shares_memory(original.rgb, crop.rgb))

    def test_resize_and_letterbox_inverse_uses_both_actual_scales(self):
        mapped = ResizeMap(.5, .25, 4, 10).to_original(Box(14, 15, 54, 25))
        self.assertEqual(mapped, Box(20, 20, 100, 60))

    def test_boundary_crop_never_wraps_negative_numpy_indices(self):
        crop = extract_crop(frame(), "obj", Box(-10, -5, 12.2, 20), 5)
        self.assertEqual(crop.bounds, Box(0, 0, 18, 25))
        self.assertEqual(crop.to_original(Box(1, 2, 3, 4)), Box(1, 2, 3, 4))

    def test_multiple_defects_map_to_original_coordinates(self):
        findings = (Finding("crack", "NG03", .9, Box(1, 2, 5, 7)),
                    Finding("surface", "NG06", .8, Box(6, 7, 10, 12)))
        inspector = MockInspector("known_defects", lambda c, m: result(c, m, status=CheckStatus.FAIL, findings=findings))
        report = self.pipeline((inspector,)).run(frame())
        first = report.objects[0].checks[0]
        self.assertEqual([x.defect_code for x in first.findings], ["NG03", "NG06"])
        self.assertEqual(first.findings[0].original_box, Box(11, 22, 15, 27))
        self.assertIsNone(report.objects[0].final_decision)
        self.assertIsNone(report.objects[0].grasp_point)

    def test_anomaly_is_called_for_every_object_even_after_known_fail_or_error(self):
        calls = []
        def known(crop, model):
            if crop.object_id.endswith("0001"):
                raise RuntimeError("known model failed")
            return result(crop, model, status=CheckStatus.FAIL, findings=(Finding("crack", "NG03"),))
        def anomaly(crop, model):
            calls.append(crop.object_id)
            return result(crop, model, "anomaly", CheckStatus.UNCERTAIN)
        report = self.pipeline((MockInspector("known_defects", known), MockInspector("anomaly", anomaly))).run(frame())
        self.assertEqual(len(calls), 2)
        self.assertEqual(report.objects[0].checks[0].status, CheckStatus.ERROR)
        self.assertEqual(report.objects[1].checks[0].status, CheckStatus.FAIL)
        self.assertEqual(report.objects[1].checks[1].status, CheckStatus.UNCERTAIN)

    def test_missing_inspectors_are_not_run_never_pass(self):
        report = self.pipeline().run(frame())
        for item in report.objects:
            self.assertEqual({c.check_id for c in item.checks}, {"known_defects", "anomaly", "geometry", "vlm"})
            self.assertTrue(all(c.status == CheckStatus.NOT_RUN for c in item.checks))

    def test_stale_inspector_identity_becomes_error(self):
        inspector = MockInspector("known_defects", lambda c, m: replace(result(c, m), frame_id="old-frame"))
        report = self.pipeline((inspector,)).run(frame())
        self.assertEqual(report.objects[0].checks[0].status, CheckStatus.ERROR)
        self.assertIn("mismatched", report.objects[0].checks[0].messages[0])

    def test_evidence_outside_crop_is_error_not_clipped_as_valid(self):
        inspector = MockInspector("known_defects", lambda c, m: result(c, m, status=CheckStatus.FAIL,
                                  findings=(Finding("bad", "NG03", .8, Box(-1, 0, 2, 3)),)))
        report = self.pipeline((inspector,)).run(frame())
        self.assertEqual(report.objects[0].checks[0].status, CheckStatus.ERROR)
        self.assertFalse(report.objects[0].checks[0].findings)

    def test_wrong_original_evidence_mapping_is_error(self):
        inspector = MockInspector("known_defects", lambda c, m: result(c, m, status=CheckStatus.FAIL,
                                  findings=(Finding("bad", "NG03", .8, Box(1, 2, 3, 4), Box(1, 2, 3, 4)),)))
        self.assertEqual(self.pipeline((inspector,)).run(frame()).objects[0].checks[0].status, CheckStatus.ERROR)

    def test_stale_detector_batch_is_rejected(self):
        detector = MockDetector(detections())
        class Stale:
            model = detector.model
            def detect(self, current):
                return detector.detect(frame("old-frame"))
        report = InspectionPipeline(Stale(), mode=Mode.SIMULATION).run(frame())
        self.assertEqual(report.execution_status, "ERROR")
        self.assertFalse(report.objects)

    def test_wrong_detector_dimensions_or_coordinate_space_is_rejected(self):
        for change in ({"image_size": (512, 512)}, {"coordinate_space": "normalized_xyxy"}):
            detector = MockDetector(detections())
            class WrongSpace:
                model = detector.model
                def detect(self, current):
                    return replace(detector.detect(current), **change)
            report = InspectionPipeline(WrongSpace(), mode=Mode.SIMULATION).run(frame())
            self.assertEqual(report.execution_status, "ERROR")

    def test_detector_exception_does_not_reuse_previous_results(self):
        detector = MockDetector(detections())
        class Flaky:
            model = detector.model
            count = 0
            def detect(self, current):
                self.count += 1
                if self.count == 2:
                    raise RuntimeError("disconnected")
                return detector.detect(current)
        pipe = InspectionPipeline(Flaky(), mode=Mode.SIMULATION)
        self.assertEqual(len(pipe.run(frame()).objects), 2)
        failed = pipe.run(frame("session:1"))
        self.assertFalse(failed.objects)
        self.assertEqual(failed.execution_status, "ERROR")

    def test_zero_detections_does_not_confirm_empty_or_ok(self):
        report = self.pipeline(boxes=(), expected_count=0).run(frame())
        self.assertIn("ZERO_DETECTIONS_NOT_CONFIRMED_EMPTY", report.issues)
        self.assertIsNone(report.final_decision)
        self.assertFalse(report.robot_commands_enabled)

    def test_outside_detection_and_count_mismatch_are_visible(self):
        outside = Detection(Box(200, 100, 220, 130), .9, 0, "part")
        report = self.pipeline(boxes=(outside,), expected_count=1).run(frame())
        self.assertTrue(any("DETECTION_OUTSIDE_IMAGE" in issue for issue in report.issues))
        self.assertTrue(any("OBJECT_COUNT_MISMATCH" in issue for issue in report.issues))

    def test_overlap_and_margin_contamination_are_visible(self):
        boxes = (Detection(Box(10, 10, 30, 30), .9, 0, "part"), Detection(Box(32, 10, 50, 30), .9, 0, "part"))
        report = self.pipeline(boxes=boxes, crop_margin_px=5).run(frame())
        self.assertIn("CROP_INTERSECTS_OTHER_DETECTION", report.objects[0].issues)
        report = self.pipeline(boxes=(boxes[0], replace(boxes[1], box=Box(20, 20, 50, 50)))).run(frame())
        self.assertIn("OVERLAPPING_DETECTION_BOXES", report.objects[0].issues)

    def test_object_limit_does_not_silently_drop_candidates(self):
        report = self.pipeline(max_objects=1).run(frame())
        self.assertEqual(report.execution_status, "ERROR")
        self.assertFalse(report.objects)
        self.assertTrue(any("OBJECT_LIMIT_EXCEEDED" in issue for issue in report.issues))

    def test_mock_cannot_be_used_in_model_file_mode(self):
        with self.assertRaises(ValueError):
            InspectionPipeline(MockDetector(detections()))

    def test_new_run_object_ids_are_never_reused(self):
        pipe = self.pipeline()
        a, b = pipe.run(frame()), pipe.run(frame())
        self.assertNotEqual(a.run_id, b.run_id)
        self.assertNotEqual(a.objects[0].object_id, b.objects[0].object_id)

    def test_json_has_no_rgb_arrays_nan_or_automatic_verdict(self):
        value = json.loads(json.dumps(self.pipeline().run(frame()).to_dict(), allow_nan=False))
        self.assertEqual(value["schema_version"], 1)
        self.assertEqual(value["mode"], "simulation")
        self.assertIsNone(value["final_decision"])
        self.assertNotIn("rgb", value["objects"][0]["crop"])

    def test_invalid_numeric_and_defect_contracts_are_rejected(self):
        with self.assertRaises(ValueError):
            Box(float("nan"), 0, 4, 4)
        with self.assertRaises(ValueError):
            Detection(Box(0, 0, 4, 4), 1.1, 0, "part")
        with self.assertRaises(ValueError):
            Finding("unknown", "NG99")
        with self.assertRaises(ValueError):
            CheckResult("x", "f", "o", CheckStatus.PASS, ModelRef("a", "1", "mock"))

    def test_coco_checkpoint_cannot_be_relabeled_as_defect_model(self):
        backend = RFDETRBackend(ROOT / "models/rf-detr-small.pth", "a" * 64, threshold=.5, training_scope="coco_general")
        with self.assertRaises(ValueError):
            RFDETRDefectInspector(backend, {0: "NG03"})

    def test_defect_bridge_keeps_multiple_candidates_and_no_detection_uncertain(self):
        class Backend:
            model = ModelRef("test-defect-model", "1", "mock", training_scope="product_defects")
            threshold = .4
            output = (Detection(Box(1, 1, 3, 3), .9, 0, "crack"), Detection(Box(4, 4, 6, 6), .8, 1, "surface"))
            def predict_rgb(self, rgb):
                return self.output
        backend = Backend()
        inspector = RFDETRDefectInspector(backend, {0: "NG03", 1: "NG06"})
        crop = extract_crop(frame(), "obj", detections()[0].box)
        check = inspector.inspect(crop)
        self.assertEqual(len(check.findings), 2)
        self.assertEqual(check.status, CheckStatus.UNCERTAIN)
        self.assertEqual(check.candidate_threshold, .4)
        backend.output = ()
        self.assertEqual(inspector.inspect(crop).status, CheckStatus.UNCERTAIN)

    def test_rfdetr_predict_bridge_does_not_double_scale_boxes(self):
        backend = RFDETRBackend("unused", "a" * 64, threshold=.5, training_scope="coco_general")
        class Network:
            def predict(self, image, **kwargs):
                self.size = image.size
                self.kwargs = kwargs
                return SimpleNamespace(xyxy=np.array([[70, 10, 110, 60]]), confidence=np.array([.9]), class_id=np.array([1]), data={"class_name": ["test"]})
        network = Network()
        backend._network = network
        batch = RFDETRDetector(backend).detect(frame())
        self.assertEqual(batch.detections[0].box, Box(70, 10, 110, 60))
        self.assertEqual(batch.candidate_threshold, .5)
        self.assertEqual(network.size, (160, 80))
        self.assertFalse(network.kwargs["include_source_image"])

    def test_bad_weight_hash_blocks_loading(self):
        import tempfile
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad.pth"
            path.write_bytes(b"bad checkpoint")
            backend = RFDETRBackend(path, "a" * 64, threshold=.5, training_scope="coco_general")
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                backend.load()
            self.assertIsNone(backend._network)

    def test_custom_model_excludes_only_registered_reserved_slot(self):
        with self.assertRaisesRegex(ValueError, "class_names"):
            RFDETRBackend("unused", "a"*64, threshold=0, training_scope="product_objects")
        class Network:
            def predict(self, *args, **kwargs):
                return SimpleNamespace(xyxy=np.array([[1, 1, 5, 5], [10, 10, 20, 20]]),
                                       confidence=np.array([.8, .9]), class_id=np.array([0, 1]), data={})
        backend = RFDETRBackend("unused", "a"*64, threshold=0, training_scope="product_objects", class_names=("part",))
        backend._network = Network()
        batch = RFDETRDetector(backend).detect(frame())
        self.assertEqual(len(batch.detections), 1)
        self.assertEqual(batch.detections[0].label, "part")
        self.assertEqual(batch.excluded_reserved_slots, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Guard against coordinate and matched-comparison mistakes without loading models."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from analyze_ash_reinforcement import translated_targets, assert_matched


class ReinforcedReviewTests(unittest.TestCase):
    def test_translation_uses_crop_origin_not_object_center(self):
        coco = {"categories": [{"id": 3, "name": "NG03"}], "annotations": [
            {"image_id": 8, "category_id": 3, "bbox": [5, 7, 11, 13]},
            {"image_id": 9, "category_id": 3, "bbox": [0, 0, 1, 1]}]}
        self.assertEqual(translated_targets(coco, {"id": 8, "source_crop_xyxy": [101, 202, 200, 300]}),
                         [{"label": "NG03", "box": [106, 209, 117, 222]}])

    def test_same_filename_different_pixels_rejected(self):
        row = {"file_name": "one.png", "image_sha256": "a", "targets": []}
        with self.assertRaisesRegex(ValueError, "image_sha256"):
            assert_matched([row], [{**row, "image_sha256": "b"}])

    def test_same_pixels_different_targets_rejected(self):
        row = {"file_name": "one.png", "image_sha256": "a", "targets": []}
        with self.assertRaisesRegex(ValueError, "targets"):
            assert_matched([row], [{**row, "targets": [{"label": "NG01", "box": [1, 2, 3, 4]}]}])

    def test_predictions_may_differ_but_duplicate_inputs_rejected(self):
        row = {"file_name": "one.png", "image_sha256": "a", "targets": []}
        assert_matched([{**row, "predictions": []}], [{**row, "predictions": ["different"]}])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            assert_matched([row, row], [row, row])

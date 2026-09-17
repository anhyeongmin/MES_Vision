import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mes_vision.synthetic.detection_review import match_image, summarize, select_threshold


def box(label="NG01", score=None):
    item = {"label": label, "box": [0, 0, 10, 10]}
    if score is not None:
        item["score"] = score
    return item


class DetectionReviewTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_reuse_one_target(self):
        result = match_image({"targets": [box()], "predictions": [box(score=.9), box(score=.8)]}, .5)
        self.assertEqual((len(result["tp"]), len(result["fp"]), len(result["fn"])), (1, 1, 0))

    def test_wrong_class_is_false_positive_and_miss(self):
        result = match_image({"targets": [box()], "predictions": [box("NG02", .9)]}, .5)
        self.assertEqual((len(result["tp"]), len(result["fp"]), len(result["fn"])), (0, 1, 1))

    def test_negative_false_alarm_and_empty_detection(self):
        rows = [{"file_name": "normal", "targets": [], "predictions": [box(score=.8)]},
                {"file_name": "defect", "targets": [box()], "predictions": []}]
        report = summarize(rows, .5, ["NG01"])
        self.assertEqual(report["false_alarm_images"], 1)
        self.assertEqual(report["positive_images_with_no_detection"], 1)
        self.assertEqual(report["total"]["tp"], 0)

    def test_nonoverlapping_box_is_not_a_match(self):
        pred = box(score=.9)
        pred["box"] = [20, 20, 30, 30]
        self.assertEqual(len(match_image({"targets": [box()], "predictions": [pred]}, .5)["tp"]), 0)

    def test_validation_threshold_excludes_low_confidence_false_alarm(self):
        rows = [{"file_name": "positive", "targets": [box()], "predictions": [box(score=.8)]},
                {"file_name": "negative", "targets": [], "predictions": [box(score=.2)]}]
        threshold, sweep = select_threshold(rows, ["NG01"])
        self.assertEqual(threshold, .2)
        self.assertEqual(len(sweep), 19)
        self.assertEqual(summarize(rows, threshold, ["NG01"])["total"]["f1"], 1)

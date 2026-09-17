import unittest
from mes_vision.validation.inference_comparison import compare_predictions


def detection(class_id=0,score=.9,x=0):
    return {'class_id':class_id,'score':score,'box':{'x1':x,'y1':0,'x2':x+10,'y2':10}}


class InferenceComparisonTests(unittest.TestCase):
    def test_reorder_and_coordinate_change(self):
        result=compare_predictions({'a':[detection(),detection(x=20)]},{'a':[detection(x=20),detection(x=.1,score=.89)]})
        self.assertEqual(result['matched'],2); self.assertAlmostEqual(result['max_coordinate_delta_px'],.1)
        self.assertAlmostEqual(result['max_score_delta'],.01)
    def test_class_change_and_duplicate_cannot_match_twice(self):
        result=compare_predictions({'a':[detection()]},{'a':[detection(),detection(),detection(class_id=1)]})
        self.assertEqual(result['matched'],1); self.assertEqual(result['unmatched_candidate'],2)
    def test_threshold_crossing_is_unmatched(self):
        result=compare_predictions({'a':[detection(score=.401)]},{'a':[detection(score=.399)]})
        self.assertEqual(result['unmatched_baseline'],1); self.assertIsNone(result['max_score_delta'])
    def test_empty_output_is_not_positive_evidence(self):
        result=compare_predictions({'a':[]},{'a':[]})
        self.assertEqual(result['matched'],0); self.assertIsNone(result['minimum_matched_iou'])
    def test_missing_image_is_rejected(self):
        with self.assertRaises(ValueError): compare_predictions({'a':[]},{})


if __name__=='__main__': unittest.main()

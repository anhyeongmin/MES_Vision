import unittest
from mes_vision.inspection.contracts import Box,Detection,DetectionBatch,ModelRef
from mes_vision.station.detail_dedup import deduplicate_detail

class DedupTests(unittest.TestCase):
    def batch(self,*detections):return DetectionBatch('f',(720,720),detections,ModelRef('test','1','mock'))
    def d(self,b,score=.9,label='ASH'):return Detection(Box(*b),score,0,label)
    def test_duplicate_keeps_high_score_and_audits_original(self):
        a=self.d((10,10,110,110),.5);b=self.d((11,11,111,111))
        original=self.batch(a,b);result,audit=deduplicate_detail(original)
        self.assertEqual(result.detections,(b,));self.assertEqual(len(original.detections),2)
        self.assertEqual(audit['removed'][0]['index'],0)
        self.assertEqual(audit['removed'][0]['kept_index'],1)
        self.assertEqual(result.frame_id,original.frame_id)
    def test_board_separate_overlapping_and_other_class_are_retained(self):
        for other in (self.d((0,0,700,700)),self.d((115,10,215,110)),
                      self.d((30,10,130,110)),self.d((10,10,110,110),label='OTHER')):
            batch=self.batch(self.d((10,10,110,110)),other)
            result,audit=deduplicate_detail(batch)
            self.assertEqual(len(result.detections),2);self.assertEqual(audit['removed'],[])
    def test_empty(self):self.assertEqual(deduplicate_detail(self.batch())[0].detections,())

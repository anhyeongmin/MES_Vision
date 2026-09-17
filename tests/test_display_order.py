import random
import unittest
from mes_vision.station.display_order import reading_order


def obj(name, x, y):
    return dict(object_id=name, effective_box=dict(x1=x-10,y1=y-10,x2=x+10,y2=y+10))


class ReadingOrderTests(unittest.TestCase):
    def test_unequal_rows_and_y_jitter_ignore_detector_order(self):
        items=[obj('1',20,22),obj('2',60,18),obj('3',100,20),
               obj('4',30,60),obj('5',90,58),obj('6',40,100)]
        original={o['object_id']:o for o in items}
        for seed in range(20):
            random.Random(seed).shuffle(items)
            result=reading_order(items)
            self.assertEqual([o['object_id'] for o in result],list('123456'))
            for o in result:
                self.assertIs(o,original[o['object_id']])
                self.assertNotIn('manual_id',o)

    def test_rows_do_not_merge_through_a_chain(self):
        items=[obj('a',90,0),obj('b',10,8),obj('c',0,16)]
        self.assertEqual([o['object_id'] for o in reading_order(items)],['b','a','c'])

    def test_empty_single_and_column(self):
        self.assertEqual(reading_order([]),[])
        a=obj('a',10,10);b=obj('b',10,50)
        self.assertEqual(reading_order([a]),[a])
        self.assertEqual(reading_order([b,a]),[a,b])

from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from monitor_reinforced_training import plateau


class PlateauTests(unittest.TestCase):
    def history(self,values):
        return [{"epoch":i+1,"map":v} for i,v in enumerate(values)]

    def test_minimum_and_patience(self):
        self.assertFalse(plateau(self.history([.9]*7)))
        self.assertTrue(plateau(self.history([.9]*8)))

    def test_meaningful_improvement_restarts_patience(self):
        self.assertFalse(plateau(self.history([.9]*6+[.91,.91,.91])))

    def test_small_fluctuations_do_not_restart_patience(self):
        self.assertTrue(plateau(self.history([.9,.901,.902,.899,.903,.900,.904,.902])))

    def test_invalid_metric_rejected(self):
        with self.assertRaises(ValueError):
            plateau(self.history([float("nan")]))

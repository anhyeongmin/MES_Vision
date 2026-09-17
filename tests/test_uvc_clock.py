"""Clock tick collisions never publish ambiguous capture timing."""
import math
import unittest
import test_uvc_camera

class ClockTests(unittest.TestCase):
    def setUp(self):
        self.fixture=test_uvc_camera.DriverTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.camera=self.fixture.camera
        self.camera.open()

    def test_same_tick_is_discarded_then_capture_recovers(self):
        self.camera.clock=iter([100.,100.,100.015625]).__next__
        first=self.camera.read()
        self.assertIsNone(self.camera.read())
        self.assertEqual(self.camera.sequence,1)
        next_frame=self.camera.read()
        self.assertEqual(next_frame.sequence,first.sequence+1)
        self.assertGreater(next_frame.host_read_completed_monotonic,first.host_read_completed_monotonic)

    def test_true_reversal_remains_blocked_and_reports_values(self):
        self.camera.clock=iter([100.,99.]).__next__
        self.camera.read()
        with self.assertRaisesRegex(ValueError, 'previous=100.0, current=99.0'):
            self.camera.read()
        self.assertEqual(self.camera.sequence,1)
        self.assertEqual(self.camera.last_read,100.)

    def test_frozen_clock_never_publishes_fresh_frames(self):
        self.camera.read()
        for _ in range(20):
            self.assertIsNone(self.camera.read())
        self.assertEqual(self.camera.sequence,1)

    def test_nonfinite_clock_is_rejected(self):
        for value in (math.nan,math.inf,-math.inf):
            self.camera.clock=lambda:value
            with self.assertRaisesRegex(ValueError,'not finite'):
                self.camera.read()
        self.assertEqual(self.camera.sequence,0)

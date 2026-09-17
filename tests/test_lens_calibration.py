import unittest
import numpy as np
import cv2
from mes_vision.operation.lens_calibration import board_image,detect,solve
from mes_vision.operation.acquisition import processing

class CalibrationTests(unittest.TestCase):
    def test_display_pattern_is_detectable(self):
        corners=detect(cv2.cvtColor(board_image(40),cv2.COLOR_GRAY2RGB))
        self.assertEqual(corners.shape,(54,1,2))
    def test_insufficient_and_stationary_views_rejected(self):
        corners=detect(cv2.cvtColor(board_image(40),cv2.COLOR_GRAY2RGB))
        for samples in ([corners]*5,[corners]*18):
            with self.assertRaises(ValueError):solve(samples,(480,360))
    def test_synthetic_calibration_and_heldout(self):
        points=np.zeros((54,3),np.float32);points[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
        k=np.array([[800.,0,640],[0,800,360],[0,0,1]])
        rng=np.random.default_rng(5);samples=[]
        for i in range(30):
            r=rng.uniform(-.4,.4,3);t=np.array([rng.uniform(-10,2),rng.uniform(-7,2),rng.uniform(17,24)])
            corners,_=cv2.projectPoints(points,r,t,k,np.array([-.12,.02,0,0,0]))
            samples.append(corners.astype(np.float32))
        result=solve(samples,(1280,720))
        self.assertTrue(result['quality_passed'])
        self.assertTrue(set(result['train_frames']).isdisjoint(result['held_out_frames']))
        self.assertAlmostEqual(result['K'][0][0],800,delta=1)
    def test_wrong_camera_profile_fails(self):
        with self.assertRaises(ValueError):
            processing(dict(driver='uvc',serial='b',width=1280,height=720,inspection_acquisition=dict(camera_serial='a',kind='undistorted_center_square_v1',sensor_size=[1280,720])))

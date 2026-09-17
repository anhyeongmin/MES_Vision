import math,unittest
from mes_vision.operation.cropped_fov import cropped_fov

class FOVTests(unittest.TestCase):
    def test_centered_square_matches_analytic_angles(self):
        k=[[800,0,639.5],[0,800,359.5],[0,0,1]]
        f=cropped_fov(k,[0]*5,(1280,720))
        self.assertEqual(f['crop_xyxy'],[280,0,1000,720])
        self.assertAlmostEqual(f['h_degrees'],math.degrees(2*math.atan(360/800)))
        self.assertAlmostEqual(f['v_degrees'],f['h_degrees'])
        self.assertAlmostEqual(f['d_degrees'],math.degrees(2*math.atan(math.sqrt(2)*360/800)))
        self.assertFalse(f['includes_invalid_border'])
    def test_offcentre_anisotropic_matrix_is_not_forced_square_fov(self):
        f=cropped_fov([[900,0,700],[0,700,330],[0,0,1]],[0]*5,(1280,720))
        self.assertNotAlmostEqual(f['h_degrees'],f['v_degrees'])
        self.assertNotAlmostEqual(*f['diagonal_degrees'])
    def test_invalid_borders_are_disclosed(self):
        f=cropped_fov([[400,0,639.5],[0,400,359.5],[0,0,1]],[.7,0,0,0,0],(1280,720))
        self.assertTrue(f['includes_invalid_border'])
        self.assertLess(f['valid_pixel_fraction'],1)

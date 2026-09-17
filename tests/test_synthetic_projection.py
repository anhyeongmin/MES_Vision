import unittest
import numpy as np
from mes_vision.synthetic.projection import raster_depth,world_triangles,difference_mask,bbox


def camera():
    return dict(width=100,height=100,fx=50,fy=50,cx=50,cy=50,position=[0,0,2],orientation='vertical_down_world_minus_Z')


def square(z=0):
    return np.array([[[-1,-1,z],[1,-1,z],[1,1,z]],[[-1,-1,z],[1,1,z],[-1,1,z]]],float)


class ProjectionTests(unittest.TestCase):
    def test_front_plane_has_correct_extent_and_metric_depth(self):
        depth=raster_depth(square(),camera())
        self.assertEqual(bbox(np.isfinite(depth)),[25,25,50,50])
        self.assertTrue(np.allclose(depth[np.isfinite(depth)],2))
    def test_foreground_wins_independent_of_triangle_order(self):
        near=square(1)*[.3,.3,1]
        a=raster_depth(np.concatenate([square(),near]),camera())
        b=raster_depth(np.concatenate([near,square()]),camera())
        np.testing.assert_equal(a,b); self.assertAlmostEqual(a[50,50],1)
        self.assertAlmostEqual(a[28,28],2)
    def test_sloped_triangle_depth_matches_ray_plane_intersection(self):
        t=np.array([[[-1,-1,0],[1,-1,1],[0,1,.2]]],float)
        d=raster_depth(t,camera()); n=np.cross(t[0,1]-t[0,0],t[0,2]-t[0,0])
        for y,x in zip(*np.nonzero(np.isfinite(d))):
            if (x+y)%13: continue
            direction=np.array([(x+.5-50)/50,-(y+.5-50)/50,-1])
            distance=np.dot(t[0,0]-[0,0,2],n)/np.dot(direction,n)
            self.assertAlmostEqual(d[y,x],distance,places=10)
    def test_difference_includes_missing_and_added_surface_but_not_background(self):
        reference=np.array([[np.inf,2.,2.],[np.inf,np.inf,2.]])
        actual=np.array([[np.inf,np.inf,1.],[3.,np.inf,2.+1e-8]])
        expected=np.array([[False,True,True],[True,False,False]])
        np.testing.assert_equal(difference_mask(actual,reference),expected)
        self.assertFalse(difference_mask(reference,reference).any())
    def test_transform_and_invalid_camera(self):
        m=np.eye(4); m[:3,3]=[1,2,3]
        np.testing.assert_allclose(world_triangles(square(),m),square()+[1,2,3])
        with self.assertRaises(ValueError): raster_depth(square(3),camera())
        with self.assertRaises(ValueError): raster_depth(square(),dict(camera(),fx=0))
    def test_padding_and_empty_masks(self):
        self.assertIsNone(bbox(np.zeros((2,3),bool)))
        mask=np.zeros((10,12),bool); mask[0,0]=True
        self.assertEqual(bbox(mask,padding=2),[0,0,3,3])


if __name__=='__main__': unittest.main()

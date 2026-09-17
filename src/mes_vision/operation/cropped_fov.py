"""Angular extent of the actual K-preserving rectified central-square output."""
import cv2
import numpy as np
from mes_vision.training.data import require


def cropped_fov(k,d,size):
    k=np.asarray(k,dtype=float);d=np.asarray(d,dtype=float)
    require(k.shape==(3,3) and np.isfinite(k).all() and np.isfinite(d).all()
            and k[0,0]>0 and k[1,1]>0,'Invalid camera matrix for FOV')
    w,h=map(int,size);require(w>0 and h>0,'Invalid image size')
    side=min(w,h);x,y=(w-side)//2,(h-side)//2
    # OpenCV centres are integer pixels; outer edges extend by half a pixel.
    left,right=x-.5,x+side-.5;top,bottom=y-.5,y+side-.5
    cx,cy=(left+right)/2,(top+bottom)/2
    inv=np.linalg.inv(k)
    def angle(a,b):
        a=inv@np.array([*a,1.]);b=inv@np.array([*b,1.])
        return float(np.degrees(np.arctan2(np.linalg.norm(np.cross(a,b)),np.dot(a,b))))
    diagonals=[angle((left,top),(right,bottom)),angle((right,top),(left,bottom))]
    # Match Rectifier: same K, same sensor dimensions, then central square.
    mx,my=cv2.initUndistortRectifyMap(k,d,None,k,(w,h),cv2.CV_32FC1)
    mx=mx[y:y+side,x:x+side];my=my[y:y+side,x:x+side]
    # Floating-point roundoff can map an identity border to -1e-14.
    eps=1e-4
    valid=(mx>=-eps)&(mx<=w-1+eps)&(my>=-eps)&(my<=h-1+eps)
    return dict(kind='rectified_center_square_geometric_estimate',image_size=[side,side],
                crop_xyxy=[x,y,x+side,y+side],h_degrees=angle((left,cy),(right,cy)),
                v_degrees=angle((cx,top),(cx,bottom)),d_degrees=max(diagonals),
                diagonal_degrees=diagonals,diagonal_definition='maximum of two corner-to-corner angles',
                axes_definition='rays through opposite edge midpoints; outer pixel edges',
                output_camera_matrix=k.tolist(),valid_pixel_fraction=float(valid.mean()),
                includes_invalid_border=bool(not valid.all()),estimated=True)

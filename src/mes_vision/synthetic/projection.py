"""Pixel-centre pinhole z-buffer for rigid CAD geometry, in metres.

Camera convention: vertical down world -Z, image right +X, image up +Y.
This produces geometry annotations, not photo appearance or a physical calibration.
"""
import numpy as np


def world_triangles(triangles, matrix):
    t = np.asarray(triangles, dtype=float); m = np.asarray(matrix, dtype=float)
    if t.ndim != 3 or t.shape[1:] != (3,3) or m.shape != (4,4) or not np.isfinite(t).all() or not np.isfinite(m).all():
        raise ValueError('Finite triangles and 4x4 transform required')
    if not np.allclose(m[3], [0,0,0,1]): raise ValueError('Affine transform required')
    return t @ m[:3,:3].T + m[:3,3]


def raster_depth(triangles, camera):
    if camera['orientation'] != 'vertical_down_world_minus_Z': raise ValueError('Unsupported camera orientation')
    width,height = camera['width'],camera['height']
    if type(width) is not int or type(height) is not int or not (0<width<=4096 and 0<height<=4096):
        raise ValueError('Unsupported image size')
    values=np.asarray([*camera['position'],camera['fx'],camera['fy'],camera['cx'],camera['cy']],dtype=float)
    if not np.isfinite(values).all() or min(camera['fx'],camera['fy'])<=0: raise ValueError('Invalid camera')
    xyz=np.asarray(triangles,dtype=float)-np.array(camera['position'])
    if xyz.ndim!=3 or xyz.shape[1:]!=(3,3) or not np.isfinite(xyz).all(): raise ValueError('Invalid mesh')
    distance=-xyz[:,:,2]
    if np.any(distance<=0): raise ValueError('Geometry crosses or is behind camera plane')
    p=np.empty_like(xyz)
    p[:,:,0]=camera['cx']+camera['fx']*xyz[:,:,0]/distance
    p[:,:,1]=camera['cy']-camera['fy']*xyz[:,:,1]/distance
    p[:,:,2]=distance
    depth=np.full((height,width),np.inf)
    for tri in p:
        x0,y0=np.maximum(np.floor(tri[:,:2].min(0)).astype(int),0)
        x1,y1=np.minimum(np.ceil(tri[:,:2].max(0)).astype(int),[width-1,height-1])
        if x1<x0 or y1<y0: continue
        den=(tri[1,1]-tri[2,1])*(tri[0,0]-tri[2,0])+(tri[2,0]-tri[1,0])*(tri[0,1]-tri[2,1])
        if abs(den)<1e-12: continue
        yy,xx=np.mgrid[y0:y1+1,x0:x1+1]; xx=xx+.5; yy=yy+.5
        a=((tri[1,1]-tri[2,1])*(xx-tri[2,0])+(tri[2,0]-tri[1,0])*(yy-tri[2,1]))/den
        b=((tri[2,1]-tri[0,1])*(xx-tri[2,0])+(tri[0,0]-tri[2,0])*(yy-tri[2,1]))/den
        c=1-a-b
        inside=(a>=-1e-8)&(b>=-1e-8)&(c>=-1e-8)
        # Perspective-correct inverse-depth interpolation; screen-linear depth is wrong.
        denominator=a/tri[0,2]+b/tri[1,2]+c/tri[2,2]
        z=np.full_like(denominator,np.inf)
        np.divide(1,denominator,out=z,where=inside&(denominator>0))
        target=depth[y0:y1+1,x0:x1+1]
        np.minimum(target,z,out=target)
    return depth


def difference_mask(actual, reference, tolerance=1e-6):
    if actual.shape!=reference.shape or tolerance<=0: raise ValueError('Matching depth arrays and positive tolerance required')
    if np.isnan(actual).any() or np.isnan(reference).any(): raise ValueError('NaN depth')
    a,b=np.isfinite(actual),np.isfinite(reference)
    result=a^b; both=a&b
    result[both] |= np.abs(actual[both]-reference[both])>tolerance
    return result


def bbox(mask, padding=0):
    ys,xs=np.nonzero(mask)
    if not len(xs): return None
    height,width=mask.shape
    x0=max(0,int(xs.min())-padding); y0=max(0,int(ys.min())-padding)
    x1=min(width,int(xs.max())+1+padding); y1=min(height,int(ys.max())+1+padding)
    return [x0,y0,x1-x0,y1-y0]

"""Screen-board calibration in raw sensor pixels, with separate verification views."""
import cv2
import numpy as np
from mes_vision.training.data import require

PATTERN=(9,6)

def board_image(cell=80):
    image=np.full((9*cell,12*cell),255,np.uint8)
    for y in range(7):
        for x in range(10):
            if (x+y)%2==0:image[(y+1)*cell:(y+2)*cell,(x+1)*cell:(x+2)*cell]=0
    return image

def detect(rgb):
    gray=cv2.cvtColor(rgb,cv2.COLOR_RGB2GRAY)
    ok,corners=cv2.findChessboardCornersSB(gray,PATTERN,cv2.CALIB_CB_NORMALIZE_IMAGE)
    require(ok,'체크보드 전체가 선명하게 보이도록 위치·거리·반사를 조절하세요.')
    return corners.astype(np.float32).reshape(-1,1,2)

def solve(samples,size):
    require(len(samples)>=18,'서로 다른 구도의 사진을 18장 이상 수집하세요.')
    points=np.zeros((54,3),np.float32);points[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)
    centres=np.array([c.reshape(-1,2).mean(0) for c in samples])
    require(np.ptp(centres[:,0])>size[0]*.25 and np.ptp(centres[:,1])>size[1]*.25,
            '화면 중앙뿐 아니라 왼쪽·오른쪽·위·아래로 옮겨 촬영하세요.')
    train=[i for i in range(len(samples)) if i%3!=2];held=[i for i in range(len(samples)) if i%3==2]
    rms,k,d,rotations,_=cv2.calibrateCamera([points]*len(train),[samples[i] for i in train],size,None,None)
    require(np.isfinite(k).all() and np.isfinite(d).all() and k[0,0]>0 and k[1,1]>0,'보정 계산에 실패했습니다.')
    errors=[]
    for i in held:
        ok,r,t=cv2.solvePnP(points,samples[i],k,d)
        require(ok,'검증 사진의 자세 계산 실패')
        projected,_=cv2.projectPoints(points,r,t,k,d)
        errors.append(float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-samples[i].reshape(-1,2))**2,axis=1)))))
    normals=np.array([cv2.Rodrigues(r)[0][:,2] for r in rotations])
    tilt_span=float(np.degrees(np.arccos(np.clip((normals@normals.T).min(),-1,1))))
    passed=bool(rms<=1. and max(errors)<=2. and np.mean(errors)<=1. and tilt_span>=10.)
    from .cropped_fov import cropped_fov
    return dict(status='SCREEN_BOARD_CHECK_PASSED' if passed else 'SCREEN_BOARD_CHECK_FAILED',
                image_size=list(size),K=k.tolist(),D=d.tolist(),pattern_inner_corners=list(PATTERN),
                cropped_fov=cropped_fov(k,d,size),
                object_units='square=1; no metric scale',train_frames=train,held_out_frames=held,
                train_rms_px=float(rms),held_out_errors_px=errors,tilt_span_degrees=tilt_span,quality_passed=passed,
                production_ready=False,limitations=['Same-session verification; not independent metrology',
                    'Requires flat display and physically square pixels','Manual focus must remain fixed',
                    'Robot mapping and inspection models need verification after changing image geometry'])

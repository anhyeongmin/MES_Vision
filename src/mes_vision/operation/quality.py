from pathlib import Path
import math
import cv2
import numpy as np
from mes_vision.training.data import require, read_json
from mes_vision.inspection import CheckResult, CheckStatus, Finding, Box, ModelRef
from mes_vision.anomaly.features import fingerprint


def validate_equipment(e):
    c=e["camera"]; w=e["workspace"]; t=e["tracking"]
    from mes_vision.inputs.camera_identity import camera_driver
    if camera_driver(c)=='uvc':
        from .camera_modes import UVC_PRESETS
        require(c.get('pixel_format','MJPG') in UVC_PRESETS,'카메라 전송 형식 오류')
        require(c['fps'] in UVC_PRESETS[c.get('pixel_format','MJPG')].get((c['width'],c['height']),()),'U20CAM 해상도·속도 설정 오류')
    require(type(c["width"]) is int and type(c["height"]) is int and 1<=c["width"]<=4096 and 1<=c["height"]<=2160
            and c["fps"] in {5,6,10,15,20,25,30,60,90},"카메라 해상도·속도 설정 오류")
    require(type(c['auto_exposure']) is bool and (c['auto_exposure'] or type(c['exposure']) in (int,float) and math.isfinite(c['exposure'])), '카메라 노출 설정 오류')
    profiles=c.get('capture_profiles')
    if profiles is not None:
        require(isinstance(profiles,dict) and set(profiles)=={'overview','detail'},'전체·상세 촬영 조건 오류')
        for values in profiles.values():
            require(type(values.get('auto_exposure')) is bool,'자동 노출 설정 오류')
            require(values['auto_exposure'] or type(values.get('exposure')) in (int,float) and math.isfinite(values['exposure']),'수동 노출 설정 오류')
            require(values.get('gain') is None or type(values['gain']) in (int,float) and math.isfinite(values['gain']),'게인 설정 오류')
    from .acquisition import inspection_camera, identity
    image_camera=inspection_camera(c)
    if identity(c):
        require(w.get("acquisition_identity")==identity(c),"Workspace acquisition changed")
    for polygon in [w["roi"],*w["excluded"]]:
        if not polygon: continue
        require(len(polygon)>=3 and all(len(p)==2 and all(type(v) in {int,float} and math.isfinite(v) for v in p)
               and 0<=p[0]<=image_camera["width"] and 0<=p[1]<=image_camera["height"] for p in polygon),"검사 영역 좌표가 영상 범위를 벗어났습니다.")
        require(abs(cv2.contourArea(np.asarray(polygon,np.float32)))>1,"검사 영역의 면적이 없습니다.")
    require(.1<=t["stable_seconds"]<=10 and .1<=t["max_gap"]<=10 and 0<t["motion_px"]<=100 and 0<t["appearance_delta"]<=1,"추적 조건 오류")
    require(e["robot"]["baudrate"]==115200,"Magician 통신 속도는 115200입니다.")
    if e["robot"]["input_address"] is not None:
        require(type(e["robot"]["input_address"]) is int and 1<=e["robot"]["input_address"]<=20,"집기 확인 입력 주소 오류")
    require(e["robot"]["holding_level"] in {0,1} and 0<e["robot"]["pose_tolerance_mm"]<=10
            and 0<e["robot"]["rotation_tolerance_deg"]<=10,"로봇 확인 조건 오류")


def inside(point,polygon):
    return len(polygon)>=3 and cv2.pointPolygonTest(np.asarray(polygon,np.float32),tuple(map(float,point)),False)>=0


def in_workspace(box,workspace):
    points=((box.x1,box.y1),(box.x2,box.y1),(box.x2,box.y2),(box.x1,box.y2))
    if not all(inside(p,workspace["roi"]) for p in points): return False
    rectangle=np.array(points,np.float32)
    for polygon in workspace["excluded"]:
        # Exclusions are rectangles drawn by the UI; reject any intersection.
        if cv2.intersectConvexConvex(rectangle,cv2.convexHull(np.asarray(polygon,np.float32)))[0]>0: return False
    return True


def quality(rgb,boxes,criteria):
    measurements=[]
    for b in boxes:
        crop=rgb[max(0,int(b.y1)):int(b.y2),max(0,int(b.x1)):int(b.x2)]
        if not crop.size: return False,[]
        gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY)
        measurements.append({"blur":float(cv2.Laplacian(gray,cv2.CV_64F).var()),"brightness":float(gray.mean())})
    passed=bool(measurements) and all(m["blur"]>=criteria["blur_min"]
        and criteria["brightness_min"]<=m["brightness"]<=criteria["brightness_max"] for m in measurements)
    return passed,measurements


class GeometryInspector:
    """Registered, pose-sensitive feature occupancy and silhouette criteria."""
    check_id="geometry"
    def __init__(self,path):
        self.config=read_json(Path(path)); self.product_id=self.config["product_id"]
        c=self.config
        require(c["kind"]=="real" and c["validation_reference"] and c["version"],"실물 형상 검증 기록이 필요합니다.")
        require(0<=c["binary_threshold"]<=255 and type(c["foreground_dark"]) is bool,"형상 이진화 설정 오류")
        require(c["features"] or c.get("aspect_range"),"형상 검사 항목이 없습니다.")
        for f in c["features"]:
            x1,y1,x2,y2=f["region"]
            require(0<=x1<x2<=1 and 0<=y1<y2<=1 and 0<=f["min_fraction"]<=f["max_fraction"]<=1
                    and f["code"] in {"NG01","NG02","NG04","NG05","NG06"},"형상 검사 영역·기준 오류")
        self.model=ModelRef("Registered geometry checks",c["version"],"model",fingerprint(c),"product_geometry")
    def inspect(self,crop):
        c=self.config; gray=cv2.cvtColor(crop.rgb,cv2.COLOR_RGB2GRAY)
        mask=gray<c["binary_threshold"] if c["foreground_dark"] else gray>c["binary_threshold"]
        findings=[]; measurements=[]; height,width=gray.shape
        if c.get("aspect_range"):
            ratio=width/height; measurements.append({"aspect_ratio":ratio})
            if not c["aspect_range"][0]<=ratio<=c["aspect_range"][1]: findings.append(Finding("외곽 가로세로 비율","NG04",1.))
        for f in c["features"]:
            x1,y1,x2,y2=f["region"]; bounds=Box(int(x1*width),int(y1*height),max(int(x1*width)+1,int(x2*width)),max(int(y1*height)+1,int(y2*height)))
            area=mask[int(bounds.y1):int(bounds.y2),int(bounds.x1):int(bounds.x2)]
            fraction=float(area.mean()); measurements.append({"name":f["name"],"fraction":fraction})
            if not f["min_fraction"]<=fraction<=f["max_fraction"]: findings.append(Finding(f["name"],f["code"],1.,bounds))
        return CheckResult(self.check_id,crop.frame_id,crop.object_id,CheckStatus.FAIL if findings else CheckStatus.PASS,
            self.model,tuple(findings),criteria_version=c["version"],details={"measurements":measurements,"criteria":c,"product_id":self.product_id})

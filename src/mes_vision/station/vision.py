"""Separate overview localization from close-up inspection.

The overview detector cannot issue an OK/NG decision. The detail detector and
inspectors use their own registered models and policy. VLM remains advisory.
"""
from copy import deepcopy
from dataclasses import asdict
import math
from mes_vision.inspection import InspectionPipeline, Mode, Box, CheckStatus
from mes_vision.inspection.contracts import DetectionBatch
from mes_vision.decision import apply_policy, FrameEvidence
from mes_vision.operation.quality import quality, in_workspace
from mes_vision.training.data import require


def checked_batch(detector, frame, limit):
    batch=detector.detect(frame)
    require(isinstance(batch,DetectionBatch) and batch.frame_id==frame.frame_id
            and batch.image_size==(frame.width,frame.height) and batch.model==detector.model
            and batch.coordinate_space=='input_rgb_pixels', 'Detector returned unrelated frame or coordinates')
    require(len(batch.detections)<=limit,'Object limit exceeded; no detections may be silently discarded')
    for item in batch.detections:
        require(item.box.clip(frame.width,frame.height)==item.box,'Object is outside the original image')
    return batch


class _Detection:
    def __init__(self,batch): self.batch=batch; self.model=batch.model
    def detect(self,frame):
        require(frame.frame_id==self.batch.frame_id,'Frozen detail frame changed')
        return self.batch


class StationVision:
    def __init__(self,overview_detector,detail_detector,inspectors,policy,*,product_id,
                 overview_workspace,detail_workspace,detail_quality,max_objects=100,mode=Mode.LIVE):
        require(type(max_objects) is int and 1<=max_objects<=100,'Invalid object capacity')
        require(mode in (Mode.LIVE,Mode.SIMULATION),'Explicit live or injected-test mode required')
        require(policy.product_id==product_id and policy.detector==detail_detector.model,
                'Detail detector and decision policy do not match')
        require(detail_quality['version']==policy.frame_criteria_version,'Detail image quality policy does not match')
        if mode==Mode.LIVE:
            require(policy.kind=='real' and policy.validated and policy.validation_reference,
                    'Validated close-up decision policy required')
            require(all(m.model.kind=='model' for m in (overview_detector,detail_detector)), 'Registered live models required')
        self.overview_detector=overview_detector; self.detail_detector=detail_detector
        self.inspectors=tuple(inspectors); self.policy=policy; self.product_id=product_id
        self.overview_workspace=deepcopy(overview_workspace); self.detail_workspace=deepcopy(detail_workspace)
        self.detail_quality=deepcopy(detail_quality); self.max_objects=max_objects; self.mode=mode

    def locate(self,frame):
        require(frame.is_live==(self.mode==Mode.LIVE),'Input mode mismatch')
        batch=checked_batch(self.overview_detector,frame,self.max_objects)
        result=[]
        # Freeze deterministic per-cycle ordering; this is not a cross-frame track ID.
        for index,item in enumerate(sorted(batch.detections,key=lambda d:(d.box.y1,d.box.x1,d.class_id))):
            visible=in_workspace(item.box,self.overview_workspace)
            touching=any(item.box.overlaps(other.box) for other in batch.detections if other is not item)
            result.append({'ordinal':index+1,'box':asdict(item.box),'class_id':item.class_id,
                'label':item.label,'score':item.score,'model':asdict(batch.model),
                'localization_status':'READY' if visible and not touching else 'REVIEW',
                'reason':None if visible and not touching else 'OVERVIEW_OVERLAP' if touching else 'OVERVIEW_WORKSPACE'})
        return result

    def inspect_detail(self,frame,*,cycle_id,target_id,overview_frame_id,expected_label,
                       center_tolerance_fraction=.25):
        require(cycle_id and target_id and overview_frame_id and expected_label and frame.frame_id!=overview_frame_id,
                'A separate detail frame and cycle/target link are required')
        require(type(center_tolerance_fraction) in (int,float) and math.isfinite(center_tolerance_fraction)
                and 0<center_tolerance_fraction<=.5,'Invalid detail target center tolerance')
        batch=checked_batch(self.detail_detector,frame,self.max_objects)
        # One intended object must be identifiable before its verdict can be used.
        association=len(batch.detections)==1
        if association:
            d=batch.detections[0]; b=d.box
            association=(d.label==expected_label and
                abs((b.x1+b.x2)/2/frame.width-.5)<=center_tolerance_fraction and
                abs((b.y1+b.y2)/2/frame.height-.5)<=center_tolerance_fraction)
        if not association:
            return {'cycle_id':cycle_id,'target_id':target_id,'detail_frame_id':frame.frame_id,
                    'overview_frame_id':overview_frame_id,'decision':'REVIEW',
                    'reason':'DETAIL_TARGET_UNCONFIRMED','inspection':None}
        good,measurements=quality(frame.rgb,[d.box],self.detail_quality)
        workspace=in_workspace(d.box,self.detail_workspace)
        if not good or not workspace:
            return {'cycle_id':cycle_id,'target_id':target_id,'detail_frame_id':frame.frame_id,
                    'overview_frame_id':overview_frame_id,'decision':'REVIEW',
                    'reason':'DETAIL_IMAGE_QUALITY' if not good else 'DETAIL_WORKSPACE','inspection':None}
        pipeline=InspectionPipeline(_Detection(batch),self.inspectors,mode=self.mode,
                                    expected_count=1,product_id=self.product_id)
        raw=pipeline.run(frame)
        raw.config['station_link']={'cycle_id':cycle_id,'target_id':target_id,
                                   'overview_frame_id':overview_frame_id,'detail_frame_id':frame.frame_id}
        raw.config['quality_measurements']=measurements
        q=self.detail_quality
        evidence=FrameEvidence(raw.run_id,frame.frame_id,self.product_id,
            'real' if self.mode==Mode.LIVE else 'synthetic',q['version'],CheckStatus.PASS,
            CheckStatus.PASS,True,q['validation_reference'])
        result=apply_policy(raw,self.policy,evidence)
        require(len(result.objects)==1,'Detail pipeline did not produce one linked object')
        return {'cycle_id':cycle_id,'target_id':target_id,'detail_frame_id':frame.frame_id,
                'overview_frame_id':overview_frame_id,'decision':result.objects[0].final_decision,
                'reason':None,'inspection':result}

from __future__ import annotations

from dataclasses import replace
import time
from uuid import uuid4

import numpy as np

from mes_vision.inputs import Frame
from .adapters import Detector, Inspector, UnconfiguredInspector
from .contracts import CheckResult, CheckStatus, Detection, Mode, ObjectResult, RunResult
from .geometry import extract_crop
from .batching import validate_results


class InspectionPipeline:
    """Synchronous inspection with an optional versioned policy. No robot authority."""

    def __init__(self, detector: Detector, inspectors: tuple[Inspector, ...] = (), *,
                 mode: Mode = Mode.MODEL_FILE, crop_margin_px: int = 0,
                 expected_count: int | None = None, max_objects: int = 100, product_id: str | None = None,
                 gpu_coordinator=None, decision_policy=None, inspection_batch_size=4):
        if not isinstance(mode, Mode):
            raise ValueError("explicit Mode enum required")
        if type(crop_margin_px) is not int or crop_margin_px < 0:
            raise ValueError("invalid crop margin")
        if type(max_objects) is not int or max_objects < 1:
            raise ValueError("invalid max_objects")
        if expected_count is not None and (type(expected_count) is not int or expected_count < 0):
            raise ValueError("invalid expected_count")
        if len({item.check_id for item in inspectors}) != len(inspectors):
            raise ValueError("duplicate inspector IDs")
        if product_id is not None and (not isinstance(product_id, str) or not product_id.strip()):
            raise ValueError("product_id must be a nonempty string or None")
        if type(inspection_batch_size) is not int or inspection_batch_size not in (1,2,4):
            raise ValueError("Invalid inspection batch limit")
        self.inspection_batch_size=inspection_batch_size
        self.product_id = product_id
        if decision_policy is not None:
            from mes_vision.decision import DecisionPolicy
            if not isinstance(decision_policy, DecisionPolicy):
                raise ValueError("explicit DecisionPolicy required")
        self.decision_policy = decision_policy
        self.gpu_coordinator = gpu_coordinator
        self.detector, self.mode = detector, mode
        self.crop_margin_px, self.expected_count, self.max_objects = crop_margin_px, expected_count, max_objects
        items = list(inspectors)
        for name in ("known_defects", "anomaly", "geometry", "vlm"):
            if name not in {item.check_id for item in items}:
                items.append(UnconfiguredInspector(name))
        self.inspectors = tuple(items)
        for inspector in self.inspectors:
            if getattr(inspector, "product_id", None) is not None and inspector.product_id != self.product_id:
                raise ValueError("inspector reference bank and pipeline product mismatch")
        if mode != Mode.SIMULATION and any(x.model.kind == "mock" for x in (detector, *self.inspectors)):
            raise ValueError("mock adapters require simulation mode")

    def run(self, frame: Frame, *, frame_evidence_provider=None) -> RunResult:
        if self.gpu_coordinator is None:
            result = self._run(frame)
        else:
            with self.gpu_coordinator.foreground() as waited_ms:
                result = self._run(frame)
            result.config["gpu_wait_ms"] = waited_ms
        if self.decision_policy is not None:
            from mes_vision.decision import apply_policy
            decision_started = time.perf_counter()
            evidence = None
            if frame_evidence_provider is not None:
                try: evidence = frame_evidence_provider(frame, result.run_id)
                except Exception as exc: result.issues.append(f"FRAME_EVIDENCE_ERROR: {type(exc).__name__}: {exc}")
            result = apply_policy(result, self.decision_policy, evidence)
            result.decision_details["elapsed_ms"] = (time.perf_counter()-decision_started)*1000
        return result

    def _run(self, frame: Frame) -> RunResult:
        if frame.is_live != (self.mode == Mode.LIVE):
            raise ValueError("live camera and inspection mode must match")
        if (not frame.frame_id or frame.coordinate_space != "input_rgb_pixels" or
                frame.rgb.dtype != np.uint8 or frame.rgb.ndim != 3 or frame.rgb.shape[2] != 3 or
                frame.width < 1 or frame.height < 1):
            raise ValueError("invalid original RGB frame")
        started = time.perf_counter()
        result = RunResult(uuid4().hex, frame.metadata(), self.mode, self.detector.model,
                           {"crop_margin_px": self.crop_margin_px, "expected_count": self.expected_count,
                            "max_objects": self.max_objects},
                           issues=["IMAGE_QUALITY_NOT_CONFIGURED", "ROI_NOT_CONFIGURED", "POLICY_NOT_IMPLEMENTED"])
        result.config["product_id"] = self.product_id
        result.config["detector_candidate_threshold"] = None
        if self.product_id is None:
            result.issues.append("PRODUCT_NOT_CONFIGURED")
        if self.mode == Mode.SIMULATION:
            result.issues.append("SIMULATED_RESULTS_NOT_PRODUCT_INSPECTION")
        if self.detector.model.training_scope != "product_objects":
            result.issues.append("OBJECT_MODEL_NOT_PRODUCT_TRAINED")
        try:
            batch = self.detector.detect(frame)
            if batch.frame_id != frame.frame_id or batch.image_size != (frame.width, frame.height):
                raise ValueError("detector frame identity or original dimensions mismatch")
            if batch.coordinate_space != "input_rgb_pixels" or batch.model != self.detector.model:
                raise ValueError("detector coordinate space or provenance mismatch")
            if len(batch.detections) > self.max_objects:
                raise ValueError("OBJECT_LIMIT_EXCEEDED: no candidates silently discarded")
            if not all(isinstance(item, Detection) for item in batch.detections):
                raise ValueError("detector returned invalid detection objects")
            result.config["detector_candidate_threshold"] = batch.candidate_threshold
            result.config["excluded_reserved_slots"] = batch.excluded_reserved_slots
        except Exception as exc:
            # Adapter failures are recorded; interrupt/exit signals still propagate.
            result.execution_status = "ERROR"
            result.issues.append(f"DETECTOR_ERROR: {type(exc).__name__}: {exc}")
            result.elapsed_ms = (time.perf_counter() - started) * 1000
            return result

        if not batch.detections:
            result.issues.append("ZERO_DETECTIONS_NOT_CONFIRMED_EMPTY")
        if self.expected_count is None:
            result.issues.append("EXPECTED_COUNT_NOT_CONFIGURED")

        result.config['inspection_batch_size']=self.inspection_batch_size
        result.config['inspection_batches']=[]
        for offset in range(0,len(batch.detections),self.inspection_batch_size):
            prepared=[]
            for index in range(offset,min(offset+self.inspection_batch_size,len(batch.detections))):
                detection=batch.detections[index]
                object_id = f"{result.run_id}:OBJ{index + 1:04d}"
                clipped = detection.box.clip(frame.width, frame.height)
                if clipped is None:
                    result.issues.append(f"DETECTION_OUTSIDE_IMAGE: {index}")
                    continue
                crop = extract_crop(frame, object_id, detection.box, self.crop_margin_px)
                item = ObjectResult(object_id, detection, clipped, crop.metadata())
                if clipped != detection.box:
                    item.issues.append("DETECTION_CLIPPED_TO_IMAGE")
                if clipped.x1 == 0 or clipped.y1 == 0 or clipped.x2 == frame.width or clipped.y2 == frame.height:
                    item.issues.append("OBJECT_TOUCHES_IMAGE_EDGE")
                for other_index, other in enumerate(batch.detections):
                    if other_index != index:
                        if clipped.overlaps(other.box):
                            item.issues.append("OVERLAPPING_DETECTION_BOXES")
                        elif crop.bounds.overlaps(other.box):
                            item.issues.append("CROP_INTERSECTS_OTHER_DETECTION")
                item.issues = list(dict.fromkeys(item.issues))
                prepared.append((crop,item))
            if not prepared: continue
            crops=tuple(crop for crop,_ in prepared)
            for inspector in self.inspectors:
                method=getattr(inspector,'inspect_many',None)
                if callable(method) and len(crops)>1:
                    began=time.perf_counter()
                    try:
                        checks=validate_results(crops,method(crops),inspector)
                    except Exception as exc:
                        checks=tuple(self._error_check(inspector,crop,exc) for crop in crops)
                    elapsed=(time.perf_counter()-began)*1000
                    checks=tuple(replace(check,elapsed_ms=elapsed) for check in checks)
                    result.config['inspection_batches'].append({'check_id':inspector.check_id,
                        'object_ids':[crop.object_id for crop in crops],'elapsed_ms':elapsed})
                else:
                    checks=[]
                    for crop in crops:
                        began=time.perf_counter()
                        try: check=validate_results((crop,),(inspector.inspect(crop),),inspector)[0]
                        except Exception as exc: check=self._error_check(inspector,crop,exc)
                        checks.append(replace(check,elapsed_ms=(time.perf_counter()-began)*1000))
                for (crop,item),check in zip(prepared,checks,strict=True):
                    try:
                        findings=[]
                        for finding in check.findings:
                            mapped=crop.to_original(finding.crop_box) if finding.crop_box else None
                            if finding.original_box is not None and finding.original_box!=mapped:
                                raise ValueError('inspector supplied inconsistent original coordinates')
                            findings.append(replace(finding,original_box=mapped))
                        check=replace(check,findings=tuple(findings))
                    except Exception as exc: check=replace(self._error_check(inspector,crop,exc),elapsed_ms=check.elapsed_ms)
                    item.checks.append(check)
                    if check.status in {CheckStatus.ERROR,CheckStatus.NOT_RUN,CheckStatus.UNCERTAIN}:
                        item.issues.append(f'CHECK_{check.status.value}: {check.check_id}')
            result.objects.extend(item for _,item in prepared)
            # Crop images are retained for at most one bounded group, never the entire scene.
            prepared.clear()
            crops=(); del crop
        if self.expected_count is not None and len(result.objects) != self.expected_count:
            result.issues.append(f"OBJECT_COUNT_MISMATCH: expected={self.expected_count}, actual={len(result.objects)}")
        result.execution_status = "COMPLETED_WITH_ISSUES" if result.issues or any(x.issues for x in result.objects) else "COMPLETED"
        result.elapsed_ms = (time.perf_counter() - started) * 1000
        return result

    @staticmethod
    def _error_check(inspector,crop,exc):
        return CheckResult(inspector.check_id,crop.frame_id,crop.object_id,CheckStatus.ERROR,inspector.model,
            messages=(f'{type(exc).__name__}: {exc}',))

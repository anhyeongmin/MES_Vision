"""RF-DETR 1.9.4 bridge. Local hash-verified weights; no model download API."""
from __future__ import annotations

import hashlib
from copy import copy
from importlib.metadata import version
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image

from mes_vision.inputs import Frame
from .contracts import (Box, CheckResult, CheckStatus, Crop, DEFECT_CODES, Detection,
                        DetectionBatch, Finding, ModelRef, finite)

RFDETR_VERSION = "1.9.4"
DEFAULT_CUDA_PROFILE = "fp32_jit"
INFERENCE_PROFILES = frozenset({"standard", "fp32_jit", "fp16", "fp16_jit"})


def model_reference(weights_sha256, training_scope, *, inference_profile=DEFAULT_CUDA_PROFILE, max_batch_size=None):
    if not isinstance(inference_profile, str) or inference_profile not in INFERENCE_PROFILES:
        raise ValueError("Unsupported RF-DETR inference profile")
    # Existing standard policies keep their identity. An optimized path needs its own validated policy.
    execution_version = RFDETR_VERSION if inference_profile == "standard" else RFDETR_VERSION+"+"+inference_profile
    if max_batch_size is None: max_batch_size = 4 if training_scope == "product_defects" else 1
    if type(max_batch_size) is not int or max_batch_size not in (1,2,4): raise ValueError("Invalid inference batch limit")
    if max_batch_size > 1: execution_version += f"+batch{max_batch_size}"
    return ModelRef("RF-DETR Small", execution_version, "model", weights_sha256, training_scope)


class RFDETRBackend:
    """Explicit load before use; one backend instance per independently trained role."""

    def __init__(self, weights: str | Path, sha256: str, *, threshold: float,
                 training_scope: str, device: str = "cuda", class_names: tuple[str, ...] | None = None,
                 inference_profile: str | None = None, max_batch_size: int | None = None):
        finite(threshold, "candidate threshold")
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be in [0, 1]")
        if training_scope not in {"coco_general", "product_objects", "product_defects"}:
            raise ValueError("training_scope must be explicit")
        if training_scope != "coco_general":
            if not class_names or len(set(class_names)) != len(class_names) or not all(isinstance(x, str) and x.strip() for x in class_names):
                raise ValueError("trained models require their exact ordered class_names")
        elif class_names is not None:
            raise ValueError("COCO model retains its official sparse class mapping")
        if inference_profile is None: inference_profile = DEFAULT_CUDA_PROFILE if device == "cuda" else "standard"
        if not isinstance(inference_profile, str) or inference_profile not in INFERENCE_PROFILES:
            raise ValueError("Unsupported RF-DETR inference profile")
        if inference_profile != "standard" and device != "cuda":
            raise ValueError("Optimized RF-DETR profiles require CUDA")
        self.inference_profile = inference_profile
        if max_batch_size is None: max_batch_size = 4 if training_scope == "product_defects" and device == "cuda" else 1
        if type(max_batch_size) is not int or max_batch_size not in (1,2,4): raise ValueError("Invalid inference batch limit")
        self.max_batch_size = max_batch_size
        self._batch_networks = {}
        self.excluded_reserved_slots_batch = ()
        self.actual_batch_sizes = ()
        self.class_names = tuple(class_names) if class_names is not None else None
        self.excluded_reserved_slots = 0
        self.weights = Path(weights).resolve()
        self.threshold, self.device = threshold, device
        self.model = model_reference(sha256, training_scope, inference_profile=inference_profile, max_batch_size=max_batch_size)
        self._network = None

    def load(self) -> None:
        if self._network is not None:
            return
        if version("rfdetr") != RFDETR_VERSION:
            raise RuntimeError("RF-DETR version differs from the verified adapter version")
        with self.weights.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != self.model.weights_sha256:
            raise RuntimeError("RF-DETR weights SHA256 mismatch; model not loaded")
        from rfdetr import RFDETRSmall
        options = {}
        if self.class_names is not None:
            import torch
            saved = torch.load(self.weights, map_location="cpu", weights_only=True, mmap=True)
            if saved["model"]["class_embed.weight"].shape[0] != len(self.class_names) + 1:
                raise ValueError("checkpoint head does not match registered classes; refusing to reinitialize")
            if tuple(saved.get("args", {}).get("class_names", ())) != self.class_names:
                raise ValueError("checkpoint class order differs from registered classes")
            del saved
            options["num_classes"] = len(self.class_names)
        network = RFDETRSmall(pretrain_weights=str(self.weights), device=self.device, **options)
        if self.inference_profile != "standard":
            import torch
            # The base module is retained; no trained head or weight file is changed.
            # Publish the network only after optimization succeeds. No silent precision fallback.
            network.inference(compile=self.inference_profile.endswith("_jit"), batch_size=1,
                dtype=torch.float32 if self.inference_profile == "fp32_jit" else torch.float16)
        try:
            batches = {1: network}
            for size in (2,4):
                if size > self.max_batch_size: continue
                if not self.inference_profile.endswith("_jit"):
                    batches[size] = network
                else:
                    # RF-DETR 1.9.4 owns shape-specific JIT state in its wrapper/context.
                    # Share the unchanged base weights, but keep each compiled context independent.
                    wrapper = copy(network); wrapper.model = copy(network.model)
                    wrapper.inference(compile=True, batch_size=size,
                        dtype=torch.float32 if self.inference_profile == "fp32_jit" else torch.float16)
                    batches[size] = wrapper
            self._batch_networks = batches
            self._network = network
        except Exception:
            self._network = None; self._batch_networks = {}; raise

    def close(self) -> None:
        self._network = None
        self._batch_networks = {}
        self.excluded_reserved_slots_batch = ()
        self.actual_batch_sizes = ()

    def predict_rgb(self, rgb: np.ndarray) -> tuple[Detection, ...]:
        if self._network is None:
            raise RuntimeError("RF-DETR backend has not been loaded")
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("RGB uint8 required")
        import torch
        with torch.inference_mode():
            output = self._network.predict(Image.fromarray(rgb), threshold=self.threshold, include_source_image=False)
        detections, excluded = self._decode(output)
        self.excluded_reserved_slots = excluded
        self.excluded_reserved_slots_batch = (excluded,)
        self.actual_batch_sizes = (1,)
        return detections

    def predict_many_rgb(self, images):
        images = tuple(images)
        if self._network is None: raise RuntimeError("RF-DETR backend has not been loaded")
        self.excluded_reserved_slots_batch = ()
        self.actual_batch_sizes = ()
        if not images: return ()
        if len(images) > 4: raise ValueError("At most four current images per inference request")
        for rgb in images:
            if not isinstance(rgb,np.ndarray) or rgb.dtype!=np.uint8 or rgb.ndim!=3 or rgb.shape[2]!=3 or min(rgb.shape[:2])<1:
                raise ValueError("RGB uint8 required")
        outputs=[]; exclusions=[]; sizes=[]; position=0
        import torch
        while position < len(images):
            size=max(n for n in (1,2,4) if n<=min(self.max_batch_size,len(images)-position))
            if size==1:
                outputs.append(self.predict_rgb(images[position])); exclusions.append(self.excluded_reserved_slots)
            else:
                network=self._batch_networks[size]
                with torch.inference_mode():
                    rows=network.predict([Image.fromarray(rgb) for rgb in images[position:position+size]],
                        threshold=self.threshold,include_source_image=False)
                if not isinstance(rows,(list,tuple)) or len(rows)!=size: raise ValueError("RF-DETR batch output count mismatch")
                for row in rows:
                    detections,excluded=self._decode(row); outputs.append(detections); exclusions.append(excluded)
            sizes.extend([size]*size); position+=size
        self.excluded_reserved_slots_batch=tuple(exclusions)
        self.actual_batch_sizes=tuple(sizes)
        return tuple(outputs)

    def _decode(self, output):
        boxes = np.asarray(output.xyxy)
        scores = np.asarray(output.confidence)
        classes = np.asarray(output.class_id)
        if boxes.shape != (len(scores), 4) or scores.ndim != 1 or classes.shape != scores.shape:
            raise ValueError("RF-DETR output array shapes disagree")
        if not np.isfinite(boxes).all() or not np.isfinite(scores).all() or not np.isfinite(classes).all():
            raise ValueError("non-finite RF-DETR output")
        if not np.equal(classes, np.floor(classes)).all():
            raise ValueError("RF-DETR class IDs must be integers")
        excluded = 0
        keep = np.ones(len(scores), dtype=bool)
        if self.class_names is not None:
            # Custom training uses labels 0..N-1 with an extra unused head slot N.
            # Public predict can expose that slot at low thresholds. COCO's slot
            # 90 is a REAL class and must not be filtered using this custom rule.
            if np.any(classes < 0) or np.any(classes > len(self.class_names)):
                raise ValueError("unknown class outside registered foreground/reserved slots")
            keep = classes < len(self.class_names)
            excluded = int((~keep).sum())
        names = output.data.get("class_name")
        if names is not None and len(names) != len(scores):
            raise ValueError("RF-DETR label count mismatch")
        # predict postprocess uses original input dimensions. Do not rescale again.
        return tuple(Detection(Box(*(float(v) for v in box)), float(scores[i]), int(classes[i]),
                               self.class_names[int(classes[i])] if self.class_names is not None else
                               str(names[i]) if names is not None else f"class_{classes[i]}")
                     for i, box in enumerate(boxes) if keep[i]), excluded


class RFDETRDetector:
    def __init__(self, backend: RFDETRBackend):
        if backend.model.training_scope == "product_defects":
            raise ValueError("defect checkpoint cannot serve as the product object detector")
        self.backend, self.model = backend, backend.model

    def detect(self, frame: Frame) -> DetectionBatch:
        return DetectionBatch(frame.frame_id, (frame.width, frame.height), self.backend.predict_rgb(frame.rgb), self.model,
                              candidate_threshold=self.backend.threshold, excluded_reserved_slots=self.backend.excluded_reserved_slots)


class RFDETRDefectInspector:
    """Collect candidates only; no detections never becomes PASS automatically.

    Site-validated decision thresholds/policy will be connected in later steps.
    The general COCO checkpoint cannot be relabeled as an NG model.
    """
    check_id = "known_defects"

    def __init__(self, backend: RFDETRBackend, class_codes: Mapping[int, str]):
        if backend.model.training_scope != "product_defects":
            raise ValueError("a separately trained product_defects checkpoint is required")
        if not class_codes or any(type(k) is not int or k < 0 or v not in DEFECT_CODES - {"NG_UNKNOWN"} for k, v in class_codes.items()):
            raise ValueError("explicit class ID to registered defect code mapping required")
        self.backend, self.model = backend, backend.model
        self.class_codes = dict(class_codes)

    def inspect(self, crop: Crop) -> CheckResult:
        detections = self.backend.predict_rgb(crop.rgb)
        return self._result(crop,detections,getattr(self.backend,'excluded_reserved_slots',0),1)

    def inspect_many(self, crops):
        from .batching import validate_crops
        crops=validate_crops(crops)
        if not crops: return ()
        outputs=self.backend.predict_many_rgb([crop.rgb for crop in crops])
        if len(outputs)!=len(crops): raise ValueError("Defect output count mismatch")
        return tuple(self._result(crop,rows,excluded,size) for crop,rows,excluded,size in
            zip(crops,outputs,self.backend.excluded_reserved_slots_batch,self.backend.actual_batch_sizes,strict=True))

    def _result(self,crop,detections,excluded,batch_size):
        findings = []
        for detection in detections:
            if detection.class_id not in self.class_codes:
                raise ValueError(f"unmapped defect class: {detection.class_id}")
            findings.append(Finding(detection.label, self.class_codes[detection.class_id], detection.score, detection.box))
        return CheckResult(self.check_id, crop.frame_id, crop.object_id, CheckStatus.UNCERTAIN, self.model,
                           tuple(findings), messages=("CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED",
                                                      f"EXCLUDED_RESERVED_SLOTS: {excluded}"),
                           candidate_threshold=self.backend.threshold,
                           details={"class_codes": {str(k): v for k, v in sorted(self.class_codes.items())},"inference_batch_size":batch_size})

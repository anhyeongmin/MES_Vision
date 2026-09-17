from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4
import time

import cv2
import numpy as np
from PIL import Image

from mes_vision.inspection import Box, CheckResult, CheckStatus, Finding, ModelRef
from mes_vision.training.data import read_json, require, sha256, write_json
from .bank import Bank, checked_features
from .features import check_rgb


@dataclass(frozen=True)
class Criteria:
    version: str
    bank_digest: str
    product_id: str
    pass_max: float
    fail_min: float
    pixel_threshold: float
    validated: bool = False
    validation_reference: str | None = None
    kind: str = "real"

    def __post_init__(self):
        require(isinstance(self.version, str) and self.version.strip() and self.version != "UNCONFIGURED", "criteria version required")
        require(isinstance(self.product_id, str) and self.product_id.strip(), "criteria product required")
        require(isinstance(self.bank_digest, str) and len(self.bank_digest) == 64 and all(c in "0123456789abcdef" for c in self.bank_digest), "invalid criteria bank digest")
        for value in (self.pass_max, self.fail_min, self.pixel_threshold):
            require(type(value) in {int, float} and np.isfinite(value), "criteria thresholds must be finite numbers")
        require(0 <= self.pass_max < self.fail_min <= 2 and self.pass_max < self.pixel_threshold <= 2,
                "require 0 <= pass_max < fail_min <= 2 and pass_max < pixel_threshold <= 2")
        require(type(self.validated) is bool and self.kind in {"real", "synthetic"}, "invalid criteria validation scope")
        if self.validated:
            require(isinstance(self.validation_reference, str) and self.validation_reference.strip(), "validated criteria require a validation record reference")


def nearest_neighbors(query, memory, *, device="cpu", query_chunk=256, bank_chunk=2048):
    import torch
    checked_features(query)
    checked_features(memory)
    require(query.ndim == memory.ndim == 2 and query.shape[1] == memory.shape[1], "distance feature dimensions mismatch")
    require(type(query_chunk) is int and 1 <= query_chunk <= 1024 and type(bank_chunk) is int and 1 <= bank_chunk <= 8192,
            "invalid distance chunk size")
    require(device in {"cpu", "cuda"} and (device != "cuda" or torch.cuda.is_available()), "requested distance device unavailable")
    distances, indices = [], []
    with torch.inference_mode():
        for start in range(0, len(query), query_chunk):
            q = torch.from_numpy(np.array(query[start:start+query_chunk], copy=True)).to(device)
            best = torch.full((len(q),), float("inf"), device=device)
            closest = torch.zeros(len(q), dtype=torch.long, device=device)
            for bank_start in range(0, len(memory), bank_chunk):
                m = torch.from_numpy(np.array(memory[bank_start:bank_start+bank_chunk], copy=True)).to(device)
                # Direct accumulation avoids cancellation for identical normalized vectors.
                scores = torch.cdist(q, m, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                minimum, local_index = scores.min(dim=1)
                better = minimum < best
                closest = torch.where(better, local_index + bank_start, closest)
                best = torch.minimum(best, minimum)
            distances.append(best.cpu().numpy())
            indices.append(closest.cpu().numpy())
    result = np.concatenate(distances).astype(np.float32)
    require(np.isfinite(result).all() and (result >= 0).all() and (result <= 2.001).all(), "invalid nearest-neighbor distances")
    return np.minimum(result, 2), np.concatenate(indices)


def connected_regions(grid, width, height, threshold):
    require(np.isfinite(grid).all() and grid.ndim == 2 and min(grid.shape) > 0, "invalid score grid")
    require(type(width) is int and type(height) is int and width > 0 and height > 0, "invalid original image size")
    if threshold is None: return []
    require(type(threshold) in {float, int} and 0 < threshold <= 2, "invalid region threshold")
    mask = np.ascontiguousarray(grid >= threshold, dtype=np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    regions = []
    for label in range(1, count):
        x, y, w, h, area = (int(v) for v in stats[label])
        regions.append({"box": [x*width/grid.shape[1], y*height/grid.shape[0],
                                (x+w)*width/grid.shape[1], (y+h)*height/grid.shape[0]],
                        "peak_distance": float(grid[labels == label].max()), "patch_count": area})
    return regions


class AnomalyEngine:
    def __init__(self, bank_directory, extractor, *, product_id, criteria=None, allow_synthetic=False,
                 distance_execution=None):
        self.extractor = extractor
        self.distance_execution = ('resident' if extractor.device == 'cuda' else 'legacy') if distance_execution is None else distance_execution
        require(self.distance_execution in {'resident', 'legacy'}, 'invalid distance execution')
        self._neighbors = None
        self._closed = False
        self.bank = Bank(bank_directory, extractor.signature, product_id=product_id, allow_synthetic=allow_synthetic)
        self.product_id = product_id
        self.criteria = Criteria(**read_json(Path(criteria))) if isinstance(criteria, (str, Path)) else criteria
        if self.criteria is not None:
            require(isinstance(self.criteria, Criteria), "invalid criteria")
            require(self.criteria.bank_digest == self.bank.digest and self.criteria.product_id == product_id
                    and self.criteria.kind == self.bank.meta["kind"], "criteria are for another bank, product, or data scope")

    def prepare(self):
        require(not self._closed, 'anomaly engine is closed')
        if self.distance_execution == 'resident' and self._neighbors is None:
            from .resident import ResidentNeighbors
            self._neighbors = ResidentNeighbors(self.bank.features, device=self.extractor.device)

    def close(self):
        if self._neighbors is not None: self._neighbors.close()
        self._neighbors = None
        self._closed = True

    def score(self, rgb):
        if self.distance_execution == 'resident': return self.score_many((rgb,))[0]
        self.prepare()
        check_rgb(rgb)
        started = time.perf_counter()
        features = checked_features(self.extractor.extract(rgb))
        return self._score_features(rgb,features,started)

    def score_many(self, images):
        require(not self._closed, 'anomaly engine is closed')
        images=tuple(images)
        require(len(images)<=4,'At most four current images per anomaly request')
        for rgb in images: check_rgb(rgb)
        if not images: return ()
        self.prepare()
        started=time.perf_counter()
        method = 'extract_many_device' if self.distance_execution == 'resident' else 'extract_many'
        extract_many=getattr(self.extractor,method,None)
        if not callable(extract_many): extract_many=getattr(self.extractor,'extract_many',None)
        features=extract_many(images) if callable(extract_many) else tuple(self.extractor.extract(rgb) for rgb in images)
        require(len(features)==len(images),'Anomaly feature count mismatch')
        return tuple(self._score_features(rgb,value,started) for rgb,value in zip(images,features,strict=True))

    def _score_features(self,rgb,features,started):
        require(list(features.shape) == self.bank.meta["grid_shape"], "query feature grid differs from bank")
        if self._neighbors is not None:
            distances, nearest = self._neighbors.search(features.reshape(-1, features.shape[-1]))
        else:
            checked_features(features)
            distances, nearest = nearest_neighbors(features.reshape(-1, features.shape[-1]), self.bank.features,
                                                   device=self.extractor.device)
        grid = distances.reshape(features.shape[:2])
        score = float(grid.max())
        criteria = self.criteria
        status, messages = CheckStatus.UNCERTAIN, []
        if criteria is None:
            messages.append("ANOMALY_CRITERIA_UNCONFIGURED")
        elif not criteria.validated:
            messages.append("ANOMALY_CRITERIA_NOT_VALIDATED")
        elif score <= criteria.pass_max:
            status = CheckStatus.PASS
        elif score >= criteria.fail_min:
            status = CheckStatus.FAIL
        else:
            messages.append("ANOMALY_SCORE_IN_REVIEW_BAND")
        if self.bank.meta["kind"] == "synthetic":
            messages.append("SYNTHETIC_REFERENCE_NOT_PRODUCT_VALIDATION")
        regions = connected_regions(grid, rgb.shape[1], rgb.shape[0], criteria.pixel_threshold if criteria else None)
        peak_index = int(distances.argmax())
        nearest_index = int(nearest[peak_index])
        details = {"status": status.value, "raw_score": score, "score_metric": self.bank.meta["metric"],
            "score_range": [0, 2], "score_is_probability": False, "image_score": "max_patch_distance",
            "product_id": self.product_id, "bank_id": self.bank.meta["bank_id"], "bank_digest": self.bank.digest,
            "feature_fingerprint": self.bank.meta["feature_fingerprint"], "kind": self.bank.meta["kind"],
            "criteria": asdict(criteria) if criteria else None, "messages": messages,
            "coordinate_space": "crop_rgb_pixels", "image_size": [rgb.shape[1], rgb.shape[0]],
            "grid_shape": list(grid.shape), "regions": regions, "alignment": "NOT_APPLIED",
            "peak_patch_yx": [peak_index//grid.shape[1], peak_index % grid.shape[1]],
            "peak_nearest_reference": {"bank_index": nearest_index, "source_image_index": int(self.bank.origins[nearest_index, 0]),
                                       "source_patch_index": int(self.bank.origins[nearest_index, 1])},
            "distance_execution": self.distance_execution,
            "elapsed_ms": (time.perf_counter()-started)*1000, "final_decision": None, "robot_commands_enabled": False}
        return details, grid, nearest.reshape(grid.shape)


def save_score(output, rgb, details, grid, nearest):
    """Save quantitative grid plus fixed 0..2 visual scale; no per-image normalization."""
    output = Path(output).resolve()
    require(not output.exists(), "use a new score directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.parent / ("." + output.name + "-building-" + uuid4().hex[:8])
    staging.mkdir()
    np.save(staging / "patch-distances.npy", grid, allow_pickle=False)
    np.save(staging / "nearest-bank-indices.npy", nearest, allow_pickle=False)
    Image.fromarray(rgb).save(staging / "input.png")
    dense = cv2.resize(grid, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    intensity = np.rint(np.clip(dense/2, 0, 1)*255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(intensity, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    Image.fromarray(heat).save(staging / "heatmap.png")
    overlay = (.60*rgb + .40*heat).clip(0, 255).astype(np.uint8)
    for region in details["regions"]:
        x1, y1, x2, y2 = region["box"]
        cv2.rectangle(overlay, (int(x1), int(y1)), (min(rgb.shape[1]-1, int(np.ceil(x2))-1), min(rgb.shape[0]-1, int(np.ceil(y2))-1)), (255, 70, 20), 2)
    Image.fromarray(overlay).save(staging / "overlay.png")
    report = dict(details, visual_scale={"min": 0, "max": 2, "map": "TURBO", "resampling": "nearest_patch", "quantitative_file": "patch-distances.npy"},
                  files={name: sha256(staging / name) for name in ("patch-distances.npy", "nearest-bank-indices.npy", "input.png", "heatmap.png", "overlay.png")})
    write_json(staging / "result.json", report)
    require(not output.exists(), "score output created concurrently")
    staging.rename(output)
    return report


class AnomalyInspector:
    check_id = "anomaly"

    def __init__(self, engine, *, evidence_dir=None):
        self.engine, self.product_id = engine, engine.product_id
        self.evidence_dir = Path(evidence_dir) if evidence_dir is not None else None
        self.model = ModelRef("DINOv2 normal-reference anomaly", "1", "model", engine.bank.digest,
                              "product_normal_reference" if engine.bank.meta["kind"] == "real" else "synthetic_reference")

    def inspect(self, crop):
        details, grid, nearest = self.engine.score(crop.rgb)
        return self._result(crop,details,grid,nearest)

    def inspect_many(self,crops):
        from mes_vision.inspection.batching import validate_crops
        crops=validate_crops(crops)
        rows=self.engine.score_many([crop.rgb for crop in crops])
        require(len(rows)==len(crops),'Anomaly result count mismatch')
        return tuple(self._result(crop,*row) for crop,row in zip(crops,rows,strict=True))

    def _result(self,crop,details,grid,nearest):
        details.update(frame_id=crop.frame_id, object_id=crop.object_id,
                       crop_bounds_original=asdict(crop.bounds))
        status = CheckStatus(details["status"])
        findings = tuple(Finding("정상 참조와 다른 영역", "NG_UNKNOWN" if status == CheckStatus.FAIL else None,
                                 region["peak_distance"], Box(*region["box"])) for region in details["regions"])
        if status == CheckStatus.FAIL and not findings:
            findings = (Finding("이상 점수가 검증된 불합격 기준 이상", "NG_UNKNOWN", details["raw_score"]),)
        if self.evidence_dir is not None:
            directory = self.evidence_dir / uuid4().hex
            save_score(directory, crop.rgb, details, grid, nearest)
            details["evidence_directory"] = str(directory.resolve())
        criteria = self.engine.criteria
        return CheckResult(self.check_id, crop.frame_id, crop.object_id, status, self.model, findings,
                           tuple(details["messages"]), raw_score=details["raw_score"],
                           criteria_version=criteria.version if criteria else None, details=details)

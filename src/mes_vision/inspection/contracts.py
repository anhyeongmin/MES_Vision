from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import math
import json
from typing import Any

import numpy as np


class Mode(StrEnum):
    SIMULATION = "simulation"
    MODEL_FILE = "model_file"
    LIVE = "live"


class CheckStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"
    ERROR = "ERROR"
    NOT_RUN = "NOT_RUN"


DEFECT_CODES = frozenset({"NG01", "NG02", "NG03", "NG04", "NG05", "NG06", "NG_UNKNOWN"})


def finite(value: float, name: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class Box:
    """Continuous pixel-edge xyxy, exclusive right/bottom; never normalized or mm."""
    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self):
        for value in (self.x1, self.y1, self.x2, self.y2):
            finite(value, "box coordinate")
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("box must have positive width and height")

    def clip(self, width: int, height: int) -> Box | None:
        x1, y1, x2, y2 = max(0, self.x1), max(0, self.y1), min(width, self.x2), min(height, self.y2)
        return Box(x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None

    def translated(self, x: float, y: float) -> Box:
        return Box(self.x1 + x, self.y1 + y, self.x2 + x, self.y2 + y)

    def overlaps(self, other: Box) -> bool:
        return min(self.x2, other.x2) > max(self.x1, other.x1) and min(self.y2, other.y2) > max(self.y1, other.y1)


@dataclass(frozen=True, slots=True)
class ModelRef:
    name: str
    version: str
    kind: str  # model, mock, unconfigured
    weights_sha256: str | None = None
    training_scope: str | None = None

    def __post_init__(self):
        if not self.name or not self.version or self.kind not in {"model", "mock", "unconfigured"}:
            raise ValueError("invalid model provenance")
        if self.weights_sha256 is not None:
            if len(self.weights_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.weights_sha256):
                raise ValueError("weights SHA256 must be 64 lower-case hex characters")


@dataclass(frozen=True, slots=True)
class Detection:
    box: Box
    score: float
    class_id: int
    label: str

    def __post_init__(self):
        finite(self.score, "detection score")
        if not isinstance(self.box, Box) or not 0 <= self.score <= 1 or type(self.class_id) is not int or self.class_id < 0 or not self.label:
            raise ValueError("invalid detection class or score")


@dataclass(frozen=True, slots=True)
class DetectionBatch:
    frame_id: str
    image_size: tuple[int, int]
    detections: tuple[Detection, ...]
    model: ModelRef
    coordinate_space: str = "input_rgb_pixels"
    candidate_threshold: float | None = None
    excluded_reserved_slots: int = 0

    def __post_init__(self):
        if type(self.excluded_reserved_slots) is not int or self.excluded_reserved_slots < 0:
            raise ValueError("invalid reserved slot count")
        if self.candidate_threshold is not None:
            finite(self.candidate_threshold, "candidate threshold")
            if not 0 <= self.candidate_threshold <= 1:
                raise ValueError("invalid candidate threshold")


@dataclass(frozen=True, slots=True)
class Crop:
    frame_id: str
    object_id: str
    bounds: Box
    object_box: Box
    rgb: np.ndarray

    @property
    def width(self) -> int:
        return self.rgb.shape[1]

    @property
    def height(self) -> int:
        return self.rgb.shape[0]

    def to_original(self, box: Box) -> Box:
        if box.clip(self.width, self.height) != box:
            raise ValueError("finding lies outside its crop")
        return box.translated(self.bounds.x1, self.bounds.y1)

    def metadata(self) -> dict[str, Any]:
        return {"bounds_original_xyxy": asdict(self.bounds), "width": self.width, "height": self.height,
                "coordinate_space": "crop_rgb_pixels", "resized": False, "alignment": "NOT_APPLIED",
                "crop_to_original": [[1, 0, self.bounds.x1], [0, 1, self.bounds.y1], [0, 0, 1]]}


@dataclass(frozen=True, slots=True)
class Finding:
    label: str
    defect_code: str | None = None
    score: float | None = None
    crop_box: Box | None = None
    original_box: Box | None = None

    def __post_init__(self):
        if not self.label or (self.defect_code is not None and self.defect_code not in DEFECT_CODES):
            raise ValueError("unknown defect code or empty finding label")
        if self.score is not None:
            finite(self.score, "finding score")


@dataclass(frozen=True, slots=True)
class CheckResult:
    check_id: str
    frame_id: str
    object_id: str
    status: CheckStatus
    model: ModelRef
    findings: tuple[Finding, ...] = ()
    messages: tuple[str, ...] = ()
    raw_score: float | None = None
    criteria_version: str | None = None
    elapsed_ms: float | None = None
    candidate_threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.details, dict):
            raise ValueError("check details must be a JSON object")
        json.dumps(self.details, allow_nan=False)
        if not isinstance(self.status, CheckStatus) or not self.check_id or not self.frame_id or not self.object_id:
            raise ValueError("invalid check identity/status")
        for name in ("raw_score", "elapsed_ms", "candidate_threshold"):
            value = getattr(self, name)
            if value is not None:
                finite(value, name)
        if self.elapsed_ms is not None and self.elapsed_ms < 0:
            raise ValueError("negative duration")
        if self.candidate_threshold is not None and not 0 <= self.candidate_threshold <= 1:
            raise ValueError("invalid candidate threshold")
        if self.status == CheckStatus.PASS and self.findings:
            raise ValueError("PASS cannot carry unresolved defect findings")
        if self.status in {CheckStatus.PASS, CheckStatus.FAIL} and not self.criteria_version:
            raise ValueError("PASS/FAIL requires an explicit criteria version")


@dataclass(slots=True)
class ObjectResult:
    object_id: str
    detection: Detection
    effective_box: Box
    crop: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    final_decision: str | None = None
    grasp_point: None = None
    decision_details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RunResult:
    run_id: str
    frame: dict[str, Any]
    mode: Mode
    detector: ModelRef
    config: dict[str, Any]
    objects: list[ObjectResult] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    execution_status: str = "COMPLETED"
    elapsed_ms: float = 0.0
    schema_version: int = 1
    final_decision: str | None = None
    decision_status: str = "NOT_IMPLEMENTED"
    robot_commands_enabled: bool = False
    calibration_version: None = None
    decision_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

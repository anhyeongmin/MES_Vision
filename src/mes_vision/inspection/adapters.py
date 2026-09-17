"""Interfaces, explicit placeholders and deterministic simulations."""
from dataclasses import dataclass
from typing import Callable, Protocol

from mes_vision.inputs import Frame
from .contracts import CheckResult, CheckStatus, Crop, Detection, DetectionBatch, ModelRef


class Detector(Protocol):
    model: ModelRef
    def detect(self, frame: Frame) -> DetectionBatch: ...


class Inspector(Protocol):
    check_id: str
    model: ModelRef
    def inspect(self, crop: Crop) -> CheckResult: ...


@dataclass(frozen=True)
class UnconfiguredInspector:
    check_id: str
    model: ModelRef = ModelRef("unconfigured", "1", "unconfigured")

    def inspect(self, crop: Crop) -> CheckResult:
        return CheckResult(self.check_id, crop.frame_id, crop.object_id, CheckStatus.NOT_RUN,
                           self.model, messages=("INSPECTOR_NOT_CONFIGURED",))


@dataclass(frozen=True)
class MockDetector:
    detections: tuple[Detection, ...]
    model: ModelRef = ModelRef("simulated-object-detector", "1", "mock", training_scope="synthetic_fixture")

    def detect(self, frame: Frame) -> DetectionBatch:
        return DetectionBatch(frame.frame_id, (frame.width, frame.height), self.detections, self.model)


@dataclass(frozen=True)
class MockInspector:
    check_id: str
    callback: Callable[[Crop, ModelRef], CheckResult]
    model: ModelRef = ModelRef("simulated-inspector", "1", "mock", training_scope="synthetic_fixture")

    def inspect(self, crop: Crop) -> CheckResult:
        return self.callback(crop, self.model)

"""Detection, original-resolution crops and inspection orchestration."""

from .contracts import (
    Box, CheckResult, CheckStatus, Crop, Detection, DetectionBatch, Finding,
    Mode, ModelRef, ObjectResult, RunResult,
)
from .pipeline import InspectionPipeline

__all__ = ["Box", "CheckResult", "CheckStatus", "Crop", "Detection", "DetectionBatch",
           "Finding", "Mode", "ModelRef", "ObjectResult", "RunResult", "InspectionPipeline"]

"""Strict JSON reconstruction; unknown fields and invalid enums are rejected."""
from copy import deepcopy
from mes_vision.inspection.contracts import Box, CheckResult, CheckStatus, Detection, Finding, ModelRef, Mode, ObjectResult, RunResult
from mes_vision.training.data import require
from .policy import FrameEvidence


def run_from_dict(data):
    data = deepcopy(data)
    require(data["schema_version"] == 1, "unsupported inspection result schema")
    data["mode"] = Mode(data["mode"])
    data["detector"] = ModelRef(**data["detector"])
    objects = []
    for item in data["objects"]:
        item["detection"] = Detection(**dict(item["detection"], box=Box(**item["detection"]["box"])))
        item["effective_box"] = Box(**item["effective_box"])
        checks = []
        for check in item["checks"]:
            check["model"] = ModelRef(**check["model"])
            check["status"] = CheckStatus(check["status"])
            check["messages"] = tuple(check["messages"])
            check["findings"] = tuple(Finding(**dict(f, crop_box=Box(**f["crop_box"]) if f["crop_box"] else None,
                                                     original_box=Box(**f["original_box"]) if f["original_box"] else None)) for f in check["findings"])
            checks.append(CheckResult(**check))
        item["checks"] = checks
        objects.append(ObjectResult(**item))
    data["objects"] = objects
    return RunResult(**data)


def frame_evidence_from_dict(data):
    return FrameEvidence(**dict(data, quality_status=CheckStatus(data["quality_status"]), workspace_status=CheckStatus(data["workspace_status"])))

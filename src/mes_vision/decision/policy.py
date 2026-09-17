from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import re

from mes_vision.anomaly.features import fingerprint
from mes_vision.inspection.contracts import Box, CheckStatus, DEFECT_CODES, Mode, ModelRef
from mes_vision.training.data import read_json, require

ENGINE_VERSION = "decision-v1"
BASE_CHECKS = {"known_defects", "anomaly", "geometry"}


def text_or_none(value):
    return value is None or isinstance(value, str) and bool(value.strip())


def digest_or_none(value):
    return value is None or isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


@dataclass(frozen=True)
class CheckRule:
    check_id: str
    required: bool
    method: str = "status"
    optional_reason: str | None = None
    model: ModelRef | None = None
    criteria_version: str | None = None
    validated: bool = False
    validation_reference: str | None = None
    criteria_digest: str | None = None
    candidate_threshold: float | None = None
    fail_thresholds: dict | None = None
    class_codes: dict | None = None

    def __post_init__(self):
        require(isinstance(self.check_id, str) and re.fullmatch("[a-z][a-z0-9_]{0,63}", self.check_id)
                and self.check_id != "vlm", "invalid check ID; VLM cannot enter the decision policy")
        require(type(self.required) is bool and type(self.validated) is bool, "rule flags must be boolean")
        require(self.method in {"status", "defect_candidates", "anomaly_distance"}, "unsupported rule method")
        require(all(text_or_none(v) for v in (self.optional_reason, self.criteria_version, self.validation_reference)), "invalid rule text")
        require(self.required or self.optional_reason is not None, "optional check requires an explicit reason")
        require(self.model is None or isinstance(self.model, ModelRef), "invalid rule model")
        require(digest_or_none(self.criteria_digest), "invalid criteria digest")
        if self.validated:
            require(self.model is not None and self.criteria_version is not None and self.criteria_version != "UNCONFIGURED"
                    and self.validation_reference is not None, "validated rule requires model, criteria version and validation reference")
        if self.method == "defect_candidates":
            require(self.check_id == "known_defects", "candidate method is for known defects")
            if self.candidate_threshold is not None:
                require(type(self.candidate_threshold) in {int, float} and math.isfinite(self.candidate_threshold)
                        and 0 <= self.candidate_threshold <= 1, "invalid collection threshold")
            require(self.fail_thresholds is None or isinstance(self.fail_thresholds, dict) and bool(self.fail_thresholds), "empty thresholds")
            for code, value in (self.fail_thresholds or {}).items():
                require(code in DEFECT_CODES - {"NG_UNKNOWN"} and type(value) in {int, float} and math.isfinite(value)
                        and self.candidate_threshold is not None and self.candidate_threshold <= value <= 1, "invalid defect threshold")
            require(self.class_codes is None or isinstance(self.class_codes, dict) and bool(self.class_codes)
                    and all(isinstance(k, str) and k.isascii() and k.isdecimal() and str(int(k)) == k
                            and v in (self.fail_thresholds or {}) for k, v in self.class_codes.items()), "invalid class-to-defect binding")
            if self.validated: require(self.candidate_threshold is not None and self.fail_thresholds and self.class_codes,
                                       "validated candidates need thresholds and class-to-defect binding")
        else:
            require(self.candidate_threshold is None and self.fail_thresholds is None and self.class_codes is None, "candidate settings do not apply to this method")
        if self.method == "anomaly_distance":
            require(self.check_id == "anomaly", "anomaly method is for anomaly checks")
            if self.validated: require(self.criteria_digest is not None, "anomaly rule must pin complete criteria")


@dataclass(frozen=True)
class DecisionPolicy:
    version: str
    product_id: str | None
    rules: tuple[CheckRule, ...]
    kind: str = "real"
    validated: bool = False
    validation_reference: str | None = None
    detector: ModelRef | None = None
    frame_criteria_version: str | None = None
    require_expected_count: bool = True
    schema_version: int = 1

    def __post_init__(self):
        require(isinstance(self.version, str) and bool(self.version.strip()) and self.version != "UNCONFIGURED", "policy version required")
        require(self.kind in {"real", "synthetic"} and type(self.validated) is bool and type(self.require_expected_count) is bool,
                "invalid policy scope/flags")
        require(type(self.schema_version) is int and self.schema_version == 1, "unsupported policy schema")
        require(all(text_or_none(v) for v in (self.product_id, self.validation_reference, self.frame_criteria_version)), "invalid policy metadata")
        require(self.detector is None or isinstance(self.detector, ModelRef), "invalid detector binding")
        require(isinstance(self.rules, tuple) and all(isinstance(r, CheckRule) for r in self.rules), "rules must be a tuple of CheckRule")
        ids = [r.check_id for r in self.rules]
        require(len(ids) == len(set(ids)) and BASE_CHECKS <= set(ids), "known defects, anomaly and geometry must be explicitly configured once")
        require(all(r.required for r in self.rules if r.check_id in {"known_defects", "anomaly"}), "known defects and anomaly cannot be skipped")
        require(next(r for r in self.rules if r.check_id == "anomaly").method == "anomaly_distance", "anomaly must use pinned distance criteria")
        if self.validated:
            require(self.product_id is not None and self.validation_reference is not None and self.detector is not None
                    and self.frame_criteria_version is not None, "validated policy requires product, detector, frame criteria and validation reference")
        if self.kind == "real":
            refs = [self.detector] + [r.model for r in self.rules]
            require(all(m is None or m.kind == "model" and m.weights_sha256 is not None for m in refs), "real policy requires pinned non-mock model/code artifacts")
            if self.detector: require(self.detector.training_scope == "product_objects", "real detector must be product trained")
            for rule in self.rules:
                if rule.model and rule.check_id in {"known_defects", "anomaly"}:
                    expected = "product_defects" if rule.check_id == "known_defects" else "product_normal_reference"
                    require(rule.model.training_scope == expected, "real rule has wrong training/reference scope")


@dataclass(frozen=True)
class FrameEvidence:
    run_id: str
    frame_id: str
    product_id: str
    kind: str
    criteria_version: str
    quality_status: CheckStatus
    workspace_status: CheckStatus
    validated: bool = False
    validation_reference: str | None = None

    def __post_init__(self):
        require(all(isinstance(v, str) and v.strip() for v in (self.run_id, self.frame_id, self.product_id, self.criteria_version)), "frame evidence identity required")
        require(self.kind in {"real", "synthetic"} and type(self.validated) is bool, "invalid frame evidence scope")
        require(isinstance(self.quality_status, CheckStatus) and isinstance(self.workspace_status, CheckStatus), "frame statuses must be explicit")
        require(text_or_none(self.validation_reference) and (not self.validated or self.validation_reference is not None), "frame validation reference required")


def load_policy(path):
    data = read_json(path)
    data["rules"] = tuple(CheckRule(**dict(r, model=ModelRef(**r["model"]) if r.get("model") else None)) for r in data["rules"])
    if data.get("detector") is not None: data["detector"] = ModelRef(**data["detector"])
    return DecisionPolicy(**data)


def reason(code, *, check_id=None, detail=None):
    return {"code": code, "check_id": check_id, "detail": detail}


def assess(check, rule, policy):
    """Assess a check without changing its raw evidence or status."""
    result = {"check_id": rule.check_id, "required": rule.required, "raw_status": check.status.value if check else None,
              "status": "REVIEW", "defect_codes": [], "criteria_version": rule.criteria_version, "reasons": []}
    def hold(code, detail=None):
        result["status"] = "REVIEW"
        result["reasons"].append(reason(code, check_id=rule.check_id, detail=detail))
        return result
    if check is None: return hold("REQUIRED_CHECK_MISSING" if rule.required else "OPTIONAL_CHECK_MISSING")
    if check.status in {CheckStatus.ERROR, CheckStatus.NOT_RUN}: return hold("CHECK_" + check.status.value, list(check.messages))
    if not rule.validated: return hold("CHECK_CRITERIA_NOT_VALIDATED")
    if check.model != rule.model: return hold("CHECK_MODEL_MISMATCH")
    if check.details.get("product_id", policy.product_id) != policy.product_id: return hold("CHECK_PRODUCT_MISMATCH")
    if check.details.get("kind", policy.kind) != policy.kind: return hold("CHECK_SCOPE_MISMATCH")
    if rule.method == "defect_candidates":
        if check.status != CheckStatus.UNCERTAIN or "CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED" not in check.messages:
            return hold("CANDIDATE_CONTRACT_MISMATCH")
        if check.candidate_threshold != rule.candidate_threshold: return hold("CANDIDATE_THRESHOLD_MISMATCH")
        if check.details.get("class_codes") != rule.class_codes: return hold("DEFECT_CLASS_MAPPING_MISMATCH")
        failed, borderline = [], False
        for finding in check.findings:
            if (finding.defect_code not in rule.fail_thresholds or finding.score is None
                    or not rule.candidate_threshold <= finding.score <= 1):
                return hold("INVALID_DEFECT_CANDIDATE")
            if finding.score >= rule.fail_thresholds[finding.defect_code]: failed.append(finding.defect_code)
            else: borderline = True
        result["status"] = "FAIL" if failed else "REVIEW" if borderline else "PASS"
        result["defect_codes"] = sorted(set(failed))
        if borderline: result["reasons"].append(reason("DEFECT_CANDIDATE_IN_REVIEW_BAND", check_id=rule.check_id))
    else:
        if check.criteria_version != rule.criteria_version: return hold("CHECK_CRITERIA_VERSION_MISMATCH")
        if rule.method == "anomaly_distance":
            criteria = check.details.get("criteria")
            if not isinstance(criteria, dict) or fingerprint(criteria) != rule.criteria_digest: return hold("ANOMALY_CRITERIA_MISMATCH")
            from mes_vision.anomaly.scoring import Criteria
            try: bound = Criteria(**criteria)
            except (TypeError, ValueError): return hold("ANOMALY_CRITERIA_INVALID")
            if (not bound.validated or bound.version != rule.criteria_version or bound.product_id != policy.product_id
                    or bound.kind != policy.kind or bound.bank_digest != check.model.weights_sha256):
                return hold("ANOMALY_REFERENCE_MISMATCH")
            score = check.raw_score
            if score is None or not 0 <= score <= 2: return hold("ANOMALY_SCORE_INVALID")
            expected = CheckStatus.PASS if score <= bound.pass_max else CheckStatus.FAIL if score >= bound.fail_min else CheckStatus.UNCERTAIN
            if check.status != expected: return hold("ANOMALY_SCORE_STATUS_CONFLICT")
        result["status"] = check.status.value if check.status in {CheckStatus.PASS, CheckStatus.FAIL} else "REVIEW"
        if check.status == CheckStatus.FAIL:
            codes = {f.defect_code for f in check.findings if f.defect_code is not None}
            if not codes: return hold("FAIL_WITHOUT_CONFIRMED_DEFECT_CODE")
            if rule.check_id == "anomaly" and codes != {"NG_UNKNOWN"}: return hold("ANOMALY_CANNOT_NAME_DEFECT")
            result["defect_codes"] = sorted(codes)
        if result["status"] == "REVIEW": hold("CHECK_UNCERTAIN", list(check.messages))
    if result["status"] == "FAIL": result["reasons"].append(reason("CONFIRMED_DEFECT", check_id=rule.check_id, detail=result["defect_codes"]))
    if result["status"] == "PASS": result["reasons"].append(reason("CHECK_PASSED", check_id=rule.check_id))
    return result


def apply_policy(source, policy, frame_evidence=None):
    """Return an independently owned decision result; source and VLM data remain untouched."""
    require(isinstance(policy, DecisionPolicy), "explicit DecisionPolicy required")
    require(frame_evidence is None or isinstance(frame_evidence, FrameEvidence), "invalid frame evidence")
    # Revalidate mutable dictionaries inside frozen policies before evaluating any data.
    policy = load_policy_data(asdict(policy))
    result = deepcopy(source)
    result.robot_commands_enabled = False
    global_reasons, invalid_identity = [], False
    def block(code, detail=None, *, invalid=False):
        nonlocal invalid_identity
        global_reasons.append(reason(code, detail=detail)); invalid_identity |= invalid
    expected_kind = "synthetic" if result.mode == Mode.SIMULATION else "real"
    if policy.kind != expected_kind: block("POLICY_SCOPE_MISMATCH", invalid=True)
    if not result.run_id or not result.frame.get("frame_id"): block("RUN_IDENTITY_INVALID", invalid=True)
    if result.config.get("product_id") != policy.product_id or policy.product_id is None: block("POLICY_PRODUCT_MISMATCH", invalid=True)
    if not policy.validated: block("POLICY_NOT_VALIDATED")
    if result.detector != policy.detector: block("DETECTOR_MODEL_MISMATCH", invalid=True)
    if result.execution_status == "ERROR": block("INSPECTION_EXECUTION_ERROR", invalid=True)
    elif result.execution_status not in {"COMPLETED", "COMPLETED_WITH_ISSUES"}: block("INSPECTION_INCOMPLETE", invalid=True)
    try:
        width, height = result.frame["width"], result.frame["height"]
        require(type(width) is int and type(height) is int and min(width, height) > 0
                and result.frame["coordinate_space"] == "input_rgb_pixels", "invalid frame geometry")
    except (ValueError, KeyError):
        width = height = 0
        block("FRAME_GEOMETRY_INVALID", invalid=True)
    if frame_evidence is None: block("FRAME_CHECKS_UNCONFIGURED")
    else:
        if (frame_evidence.run_id, frame_evidence.frame_id, frame_evidence.product_id, frame_evidence.kind) != (
                result.run_id, result.frame.get("frame_id"), policy.product_id, policy.kind):
            block("FRAME_EVIDENCE_IDENTITY_MISMATCH", invalid=True)
        if not frame_evidence.validated or frame_evidence.criteria_version != policy.frame_criteria_version:
            block("FRAME_CRITERIA_NOT_VALIDATED")
        if frame_evidence.quality_status != CheckStatus.PASS: block("IMAGE_QUALITY_NOT_PASSED", frame_evidence.quality_status.value)
        if frame_evidence.workspace_status != CheckStatus.PASS: block("WORKSPACE_NOT_PASSED", frame_evidence.workspace_status.value)
    count = result.config.get("expected_count")
    if count is None:
        if policy.require_expected_count: block("EXPECTED_COUNT_UNCONFIGURED")
    elif type(count) is not int or count < 0 or count != len(result.objects): block("OBJECT_COUNT_MISMATCH")
    if not result.objects: block("ZERO_OBJECTS_NOT_CONFIRMED_EMPTY")
    ids = [o.object_id for o in result.objects]
    if len(ids) != len(set(ids)): block("DUPLICATE_OBJECT_ID", invalid=True)
    acknowledged = {"POLICY_NOT_IMPLEMENTED", "POLICY_NOT_CONFIGURED", "IMAGE_QUALITY_NOT_CONFIGURED", "ROI_NOT_CONFIGURED",
                    "EXPECTED_COUNT_NOT_CONFIGURED", "SIMULATED_RESULTS_NOT_PRODUCT_INSPECTION", "OBJECT_MODEL_NOT_PRODUCT_TRAINED",
                    "ZERO_DETECTIONS_NOT_CONFIRMED_EMPTY", "PRODUCT_NOT_CONFIGURED"}
    for issue in result.issues:
        if issue not in acknowledged and not issue.startswith("OBJECT_COUNT_MISMATCH:"):
            block("RUN_ISSUE", issue)
    for obj in result.objects:
        local, identity_bad = [], invalid_identity
        bounds = None
        if not obj.object_id.startswith(result.run_id + ":OBJ"): identity_bad = True; local.append(reason("OBJECT_IDENTITY_INVALID"))
        try:
            bounds = Box(**obj.crop["bounds_original_xyxy"])
            require(bounds.clip(width, height) == bounds and obj.effective_box == obj.detection.box.clip(width, height)
                    and obj.effective_box.clip(width, height) == obj.effective_box
                    and bounds.x1 <= obj.effective_box.x1 < obj.effective_box.x2 <= bounds.x2
                    and bounds.y1 <= obj.effective_box.y1 < obj.effective_box.y2 <= bounds.y2
                    and obj.crop["width"] == bounds.x2-bounds.x1 and obj.crop["height"] == bounds.y2-bounds.y1
                    and obj.crop["coordinate_space"] == "crop_rgb_pixels" and obj.crop["resized"] is False
                    and obj.crop["crop_to_original"] == [[1, 0, bounds.x1], [0, 1, bounds.y1], [0, 0, 1]], "invalid crop binding")
        except (KeyError, ValueError, AttributeError, TypeError): identity_bad = True; local.append(reason("CROP_IDENTITY_INVALID"))
        checks = [c for c in obj.checks if c.check_id != "vlm"]
        check_ids = [c.check_id for c in checks]
        if len(check_ids) != len(set(check_ids)): identity_bad = True; local.append(reason("DUPLICATE_CHECK_ID"))
        for check in checks:
            if (check.frame_id, check.object_id) != (result.frame.get("frame_id"), obj.object_id):
                identity_bad = True; local.append(reason("CHECK_IDENTITY_MISMATCH", check_id=check.check_id))
            for finding in check.findings:
                if finding.crop_box is not None:
                    if (bounds is None or finding.crop_box.clip(obj.crop.get("width", 0), obj.crop.get("height", 0)) != finding.crop_box
                            or finding.original_box != finding.crop_box.translated(bounds.x1, bounds.y1)):
                        identity_bad = True; local.append(reason("FINDING_COORDINATES_INVALID", check_id=check.check_id))
                elif finding.original_box is not None:
                    identity_bad = True; local.append(reason("FINDING_COORDINATES_INVALID", check_id=check.check_id))
        rules = {r.check_id: r for r in policy.rules}
        if set(check_ids)-set(rules): local.append(reason("UNREGISTERED_CHECK", detail=sorted(set(check_ids)-set(rules))))
        for issue in obj.issues:
            if issue in {f"CHECK_{s.value}: {name}" for s in CheckStatus for name in {*rules, "vlm"}}: continue
            local.append(reason("OBJECT_ISSUE", detail=issue))
        if obj.effective_box.x1 <= 0 or obj.effective_box.y1 <= 0 or obj.effective_box.x2 >= width or obj.effective_box.y2 >= height:
            local.append(reason("OBJECT_TOUCHES_IMAGE_EDGE"))
        if obj.detection.box != obj.effective_box: local.append(reason("OBJECT_PARTIALLY_OUTSIDE_IMAGE"))
        for other in result.objects:
            if other is not obj and (obj.effective_box.overlaps(other.effective_box)
                    or bounds is not None and bounds.overlaps(other.effective_box)):
                local.append(reason("OBJECT_OR_CROP_OVERLAP")); break
        assessed = [assess(next((c for c in checks if c.check_id == r.check_id), None), r, policy) for r in policy.rules]
        complete = all(c["status"] in {"PASS", "FAIL"} for c in assessed if c["required"])
        confirmed = sorted({code for c in assessed if c["status"] == "FAIL" for code in c["defect_codes"]})
        holds = any(c["status"] == "REVIEW" for c in assessed if c["required"])
        # Invalid identity or unapproved policy can never establish a defect for this object.
        if identity_bad or not policy.validated: decision = "REVIEW"; confirmed = []
        elif confirmed: decision = "NG"
        elif global_reasons or local or holds: decision = "REVIEW"
        else: decision = "OK"
        obj.final_decision = decision
        obj.decision_details = {"policy_version": policy.version, "policy_digest": fingerprint(asdict(policy)), "engine_version": ENGINE_VERSION,
            "kind": policy.kind, "required_checks_complete": complete, "defect_codes": confirmed,
            "reasons": deepcopy(global_reasons) + local + [r for c in assessed for r in c["reasons"]], "checks": assessed,
            "identity_valid": not identity_bad, "production_validated": policy.kind == "real" and policy.validated}
    decisions = [o.final_decision for o in result.objects]
    result.final_decision = ("REVIEW" if invalid_identity or not policy.validated else "NG" if "NG" in decisions
                             else "REVIEW" if global_reasons or not decisions or "REVIEW" in decisions else "OK")
    result.decision_status = "SIMULATED" if policy.kind == "synthetic" and expected_kind == "synthetic" else "EVALUATED"
    result.decision_details = {"engine_version": ENGINE_VERSION, "policy_version": policy.version,
        "policy_digest": fingerprint(asdict(policy)), "policy": asdict(policy), "kind": policy.kind,
        "input_digest": fingerprint(source.to_dict()),
        "reasons": global_reasons, "counts": {v: decisions.count(v) for v in ("OK", "NG", "REVIEW")},
        "required_checks_complete": bool(decisions) and all(o.decision_details["required_checks_complete"] for o in result.objects),
        "frame_evidence": asdict(frame_evidence) if frame_evidence else None, "robot_authorization": "NOT_PROVIDED"}
    result.issues = [i for i in result.issues if i not in {"POLICY_NOT_IMPLEMENTED", "POLICY_NOT_CONFIGURED"}]
    return result


def load_policy_data(data):
    data = deepcopy(data)
    data["rules"] = tuple(CheckRule(**dict(r, model=ModelRef(**r["model"]) if r.get("model") else None)) for r in data["rules"])
    if data.get("detector") is not None: data["detector"] = ModelRef(**data["detector"])
    return DecisionPolicy(**data)

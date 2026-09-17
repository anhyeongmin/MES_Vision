"""Explicit synthetic policy fixtures; never register them as production criteria."""
from dataclasses import asdict
from datetime import datetime, timezone
import numpy as np

from mes_vision.anomaly.features import fingerprint
from mes_vision.anomaly.scoring import Criteria
from mes_vision.inputs import Frame, SourceKind
from mes_vision.inspection import Box, CheckResult, CheckStatus, Detection, Finding, InspectionPipeline, Mode, ModelRef
from mes_vision.inspection.adapters import MockDetector, MockInspector
from .policy import CheckRule, DecisionPolicy, FrameEvidence, apply_policy


def make_case(*, mixed=False):
    rgb = np.full((180, 480, 3), 235, dtype=np.uint8)
    boxes = (Box(20, 30, 130, 150), Box(180, 30, 290, 150), Box(340, 30, 450, 150)) if mixed else (Box(20, 30, 130, 150),)
    for box in boxes: rgb[int(box.y1):int(box.y2), int(box.x1):int(box.x2)] = [65, 125, 180]
    if mixed: rgb[60:110, 210:214] = [15, 15, 15]
    frame = Frame("synthetic-policy:0", "synthetic-policy", 0, SourceKind.IMAGE,
                  "synthetic://decision-fixture", datetime.now(timezone.utc), rgb)
    known_model = ModelRef("synthetic-defect-candidates", "1", "mock", "a"*64, "synthetic_fixture")
    anomaly_model = ModelRef("synthetic-normal-memory", "1", "mock", "b"*64, "synthetic_reference")
    geometry_model = ModelRef("synthetic-geometry", "1", "mock", "c"*64, "synthetic_fixture")
    criteria = Criteria("SYNTHETIC-distance-v1", "b"*64, "fixture-part", .2, .8, .5, True, "SYNTHETIC-TEST-ONLY", "synthetic")
    def known(crop, model):
        findings = (Finding("균열", "NG03", .9, Box(30, 30, 34, 80)), Finding("표면 결함", "NG06", .95)) if mixed and crop.object_id.endswith("0002") else ()
        return CheckResult("known_defects", crop.frame_id, crop.object_id, CheckStatus.UNCERTAIN, model, findings,
                           messages=("CANDIDATES_ONLY_CRITERIA_NOT_VALIDATED",), candidate_threshold=.4,
                           details={"class_codes": {"0": "NG03", "1": "NG06"}})
    def anomaly(crop, model):
        error = mixed and crop.object_id.endswith("0002")
        score = .5 if mixed and crop.object_id.endswith("0003") else .1
        return CheckResult("anomaly", crop.frame_id, crop.object_id, CheckStatus.ERROR if error else CheckStatus.UNCERTAIN if score == .5 else CheckStatus.PASS,
                           model, raw_score=score, criteria_version=criteria.version,
                           messages=("SYNTHETIC_TEST_FAILURE",) if error else (),
                           details={"criteria": asdict(criteria), "product_id": "fixture-part", "kind": "synthetic"})
    def geometry(crop, model):
        return CheckResult("geometry", crop.frame_id, crop.object_id, CheckStatus.PASS, model, criteria_version="SYNTHETIC-geometry-v1")
    detector = MockDetector(tuple(Detection(b, .99, 0, "fixture-part") for b in boxes))
    inspectors = (MockInspector("known_defects", known, known_model), MockInspector("anomaly", anomaly, anomaly_model),
                  MockInspector("geometry", geometry, geometry_model))
    policy = DecisionPolicy("SYNTHETIC-policy-v1", "fixture-part", (
        CheckRule("known_defects", True, "defect_candidates", model=known_model, criteria_version="SYNTHETIC-defects-v1", validated=True,
                  validation_reference="SYNTHETIC-TEST-ONLY", candidate_threshold=.4, fail_thresholds={"NG03": .8, "NG06": .9},
                  class_codes={"0": "NG03", "1": "NG06"}),
        CheckRule("anomaly", True, "anomaly_distance", model=anomaly_model, criteria_version=criteria.version, validated=True,
                  validation_reference="SYNTHETIC-TEST-ONLY", criteria_digest=fingerprint(asdict(criteria))),
        CheckRule("geometry", True, model=geometry_model, criteria_version="SYNTHETIC-geometry-v1", validated=True,
                  validation_reference="SYNTHETIC-TEST-ONLY")), kind="synthetic", validated=True,
        validation_reference="SYNTHETIC-TEST-ONLY", detector=detector.model, frame_criteria_version="SYNTHETIC-frame-v1")
    pipeline = InspectionPipeline(detector, inspectors, mode=Mode.SIMULATION, product_id="fixture-part", expected_count=len(boxes))
    raw = pipeline.run(frame)
    evidence = FrameEvidence(raw.run_id, frame.frame_id, "fixture-part", "synthetic", "SYNTHETIC-frame-v1",
                             CheckStatus.PASS, CheckStatus.PASS, True, "SYNTHETIC-TEST-ONLY")
    return {"frame": frame, "raw": raw, "policy": policy, "evidence": evidence, "pipeline": pipeline,
            "result": apply_policy(raw, policy, evidence)}

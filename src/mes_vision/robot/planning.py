from copy import deepcopy
from dataclasses import asdict, dataclass
from uuid import uuid4

from mes_vision.anomaly.features import fingerprint
from mes_vision.inspection import Mode
from mes_vision.training.data import require, read_json
from .contracts import Pose, Workspace, RobotProfile, SceneStamp, MappedTarget, Command

STAGES = ("APPROACH_PICK", "DESCEND_PICK", "PICK", "VERIFY_PICK", "LIFT_PICK", "MOVE_PLACE",
          "DESCEND_PLACE", "PLACE", "VERIFY_PLACE", "RETRACT")


@dataclass(frozen=True)
class PickPlan:
    plan_id: str
    object_id: str
    decision: str
    scene: SceneStamp
    profile: RobotProfile
    commands: tuple[Command, ...]
    inspection_digest: str
    target_digest: str
    digest: str


def profile_from_dict(data):
    data = deepcopy(data)
    if data.get("workspace") is not None: data["workspace"] = Workspace(**data["workspace"])
    if data.get("destinations") is not None: data["destinations"] = {k: Pose(**v) for k, v in data["destinations"].items()}
    return RobotProfile(**data)


def load_profile(path): return profile_from_dict(read_json(path))


def plan_digest(plan):
    data = asdict(plan); data.pop("digest")
    return fingerprint(data)


def build_plan(inspection, object_id, target, profile, scene, *, now):
    """Build a bound live or synthetic plan from verified evidence and mapped coordinates."""
    require(isinstance(profile, RobotProfile), "ROBOT_PROFILE_REQUIRED")
    profile = profile_from_dict(asdict(profile))
    require((profile.kind == "synthetic" and inspection.mode == Mode.SIMULATION)
            or (profile.kind == "real" and inspection.mode == Mode.LIVE and inspection.frame.get("is_live") is True),
            "LIVE_VERIFIED_INSPECTION_REQUIRED")
    require(profile.validated, "ROBOT_PROFILE_NOT_VALIDATED")
    require(isinstance(scene, SceneStamp) and isinstance(target, MappedTarget), "CALIBRATED_TARGET_AND_SCENE_REQUIRED")
    require(0 <= now-scene.observed_monotonic <= profile.frame_max_age_seconds, "STALE_OR_FUTURE_FRAME")
    require(inspection.decision_status == ("SIMULATED" if profile.kind == "synthetic" else "EVALUATED") and inspection.config.get("product_id") == profile.product_id,
            "DECISION_SCOPE_OR_PRODUCT_MISMATCH")
    require((scene.run_id, scene.frame_id) == (inspection.run_id, inspection.frame.get("frame_id")), "SCENE_IDENTITY_MISMATCH")
    require(target.kind == profile.kind and target.validated and (target.run_id, target.frame_id, target.object_id) ==
            (scene.run_id, scene.frame_id, object_id), "TARGET_IDENTITY_OR_VALIDATION_MISMATCH")
    require(scene.calibration_version == target.calibration_version == profile.calibration_version
            and target.grasp_policy_version == profile.grasp_policy_version, "CALIBRATION_OR_GRASP_POLICY_MISMATCH")
    objects = [o for o in inspection.objects if o.object_id == object_id]
    require(len(objects) == 1, "OBJECT_NOT_UNIQUE")
    obj = objects[0]
    from mes_vision.decision.policy import apply_policy, load_policy_data
    from mes_vision.decision.io import frame_evidence_from_dict
    details = inspection.decision_details
    require(isinstance(details.get("policy"), dict) and isinstance(details.get("frame_evidence"), dict), "VERIFIED_DECISION_EVIDENCE_REQUIRED")
    replayed = apply_policy(inspection, load_policy_data(details["policy"]), frame_evidence_from_dict(details["frame_evidence"]))
    replayed_obj = next(o for o in replayed.objects if o.object_id == object_id)
    require(replayed.final_decision == inspection.final_decision and replayed_obj.final_decision == obj.final_decision
            and replayed_obj.decision_details == obj.decision_details, "SAVED_DECISION_DOES_NOT_MATCH_EVIDENCE")
    require(obj.final_decision in {"OK", "NG", "REVIEW"} and obj.decision_details.get("identity_valid") is True, "NO_VALID_OBJECT_DECISION")
    require(not inspection.decision_details.get("reasons"), "FRAME_OR_RUN_NOT_READY_FOR_MOTION")
    forbidden = {"OBJECT_ISSUE", "OBJECT_TOUCHES_IMAGE_EDGE", "OBJECT_PARTIALLY_OUTSIDE_IMAGE", "OBJECT_OR_CROP_OVERLAP", "UNREGISTERED_CHECK"}
    require(not any(r["code"] in forbidden for r in obj.decision_details.get("reasons", [])), "OBJECT_PLACEMENT_NOT_READY")
    require(obj.final_decision != "REVIEW" or profile.allow_review_move, "REVIEW_MOVE_NOT_CONFIGURED")
    b = obj.effective_box
    require(b.x1 <= target.source_pixel[0] < b.x2 and b.y1 <= target.source_pixel[1] < b.y2, "GRASP_PIXEL_OUTSIDE_OBJECT")
    destination = profile.destinations.get(obj.final_decision)
    require(destination is not None, "DESTINATION_NOT_CONFIGURED")
    pick = Pose(target.x_mm, target.y_mm, profile.pick_z_mm, target.r_deg)
    above_pick = Pose(pick.x, pick.y, profile.travel_z_mm, pick.r)
    above_place = Pose(destination.x, destination.y, profile.travel_z_mm, destination.r)
    require(all(profile.workspace.contains(p) for p in (pick, above_pick, destination, above_place)), "POSE_OUTSIDE_CONFIGURED_WORKSPACE")
    identity = uuid4().hex
    actions = (("move", above_pick), ("move", pick), ("engage", None), ("verify_pick", None), ("move", above_pick),
               ("move", above_place), ("move", destination), ("release", None), ("verify_place", None), ("move", above_place))
    commands = tuple(Command(uuid4().hex, identity, stage, action, pose) for stage, (action, pose) in zip(STAGES, actions))
    plan = PickPlan(identity, object_id, obj.final_decision, deepcopy(scene), profile, commands,
                    fingerprint(inspection.to_dict()), fingerprint(asdict(target)), "")
    from dataclasses import replace
    return replace(plan, digest=plan_digest(plan))

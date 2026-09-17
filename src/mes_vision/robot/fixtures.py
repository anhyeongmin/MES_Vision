"""Invented coordinates for software tests only; not Magician limits or calibration."""
from mes_vision.decision.fixtures import make_case
from .contracts import Pose, Workspace, RobotProfile, MappedTarget, SceneStamp
from .simulator import SimulatedMagician
from .planning import build_plan


def make_fixture(*, decision="OK", faults=None, now=1000., allow_review=False):
    data = make_case(mixed=True)
    result = data["result"]
    obj = next(o for o in result.objects if o.final_decision == decision)
    profile = RobotProfile("SYNTHETIC-robot-v1", "fixture-part", "synthetic", True, "SYNTHETIC-TEST-ONLY",
        "SYNTHETIC-calibration-v1", "SYNTHETIC-grasp-v1", Workspace(-200, 400, -200, 200, 0, 200, -180, 180),
        10, 100, {"OK": Pose(150, 50, 10, 0), "NG": Pose(150, -50, 10, 0), "REVIEW": Pose(200, 0, 10, 0)},
        "suction", "synthetic_independent_checks", 2., 1., allow_review)
    adapter = SimulatedMagician(Pose(0, 0, 100, 0), faults=faults)
    scene = SceneStamp(result.run_id, result.frame["frame_id"], 1, now, profile.calibration_version, adapter.epoch)
    b = obj.effective_box
    target = MappedTarget(result.run_id, scene.frame_id, obj.object_id, profile.calibration_version,
        profile.grasp_policy_version, "synthetic", ((b.x1+b.x2)/2, (b.y1+b.y2)/2), 40, 30, 0, True, "SYNTHETIC-TEST-ONLY")
    return {"inspection": result, "object_id": obj.object_id, "profile": profile, "adapter": adapter,
            "scene": scene, "target": target, "now": now}


def prepare(data):
    return build_plan(data["inspection"], data["object_id"], data["target"], data["profile"], data["scene"], now=data["now"])


def drive(controller, data, *, stop_stage=None, max_ticks=200):
    for tick in range(max_ticks):
        if controller.state in {"RECAPTURE", "FAULT", "STOPPED", "RECOVERY"}: break
        if stop_stage and controller.state == stop_stage:
            controller.stop(); break
        controller.tick(data["scene"], now=data["now"]+tick*.1)
    return controller.state

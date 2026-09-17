from __future__ import annotations
from dataclasses import dataclass, asdict
import math
from typing import Protocol

from mes_vision.training.data import require


def number(value):
    return type(value) in {int, float} and math.isfinite(value)


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    z: float
    r: float
    coordinate_space: str = "robot_base_mm_deg"

    def __post_init__(self):
        require(all(number(v) for v in (self.x, self.y, self.z, self.r)), "pose must be finite mm/degrees")
        require(self.coordinate_space == "robot_base_mm_deg", "pixel coordinates cannot be robot poses")


@dataclass(frozen=True)
class Workspace:
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float
    r_min: float
    r_max: float

    def __post_init__(self):
        require(all(number(v) for v in asdict(self).values()), "workspace must be finite")
        require(self.x_min < self.x_max and self.y_min < self.y_max and self.z_min < self.z_max and self.r_min < self.r_max, "invalid workspace bounds")

    def contains(self, pose):
        return all(low <= value <= high for low, value, high in (
            (self.x_min, pose.x, self.x_max), (self.y_min, pose.y, self.y_max),
            (self.z_min, pose.z, self.z_max), (self.r_min, pose.r, self.r_max)))


@dataclass(frozen=True)
class SceneStamp:
    run_id: str
    frame_id: str
    revision: int
    observed_monotonic: float
    calibration_version: str
    connection_epoch: str

    def __post_init__(self):
        require(all(isinstance(v, str) and v.strip() for v in (self.run_id, self.frame_id, self.calibration_version, self.connection_epoch)), "scene identity required")
        require(type(self.revision) is int and self.revision >= 0 and number(self.observed_monotonic), "invalid scene revision/time")


@dataclass(frozen=True)
class MappedTarget:
    run_id: str
    frame_id: str
    object_id: str
    calibration_version: str
    grasp_policy_version: str
    kind: str
    source_pixel: tuple[float, float]
    x_mm: float
    y_mm: float
    r_deg: float
    validated: bool
    validation_reference: str | None

    def __post_init__(self):
        require(self.kind in {"real", "synthetic"} and type(self.validated) is bool, "invalid target scope")
        require(isinstance(self.source_pixel, tuple) and len(self.source_pixel) == 2
                and all(number(v) for v in (*self.source_pixel, self.x_mm, self.y_mm, self.r_deg)), "invalid mapped target")
        require(all(isinstance(v, str) and v.strip() for v in (self.run_id, self.frame_id, self.object_id,
                self.calibration_version, self.grasp_policy_version)), "mapped target identity required")
        require(not self.validated or isinstance(self.validation_reference, str) and self.validation_reference.strip(), "target validation record required")


@dataclass(frozen=True)
class RobotProfile:
    version: str
    product_id: str | None = None
    kind: str = "real"
    validated: bool = False
    validation_reference: str | None = None
    calibration_version: str | None = None
    grasp_policy_version: str | None = None
    workspace: Workspace | None = None
    pick_z_mm: float | None = None
    travel_z_mm: float | None = None
    destinations: dict[str, Pose] | None = None
    end_effector: str | None = None
    verification_method: str | None = None
    frame_max_age_seconds: float | None = None
    command_timeout_seconds: float | None = None
    allow_review_move: bool = False

    def __post_init__(self):
        require(isinstance(self.version, str) and self.version.strip(), "profile version required")
        require(self.kind in {"real", "synthetic"} and type(self.validated) is bool and type(self.allow_review_move) is bool, "invalid profile flags")
        for v in (self.product_id, self.validation_reference, self.calibration_version, self.grasp_policy_version):
            require(v is None or isinstance(v, str) and bool(v.strip()), "invalid profile metadata")
        require(self.workspace is None or isinstance(self.workspace, Workspace), "invalid workspace")
        require(self.destinations is None or isinstance(self.destinations, dict)
                and set(self.destinations) <= {"OK", "NG", "REVIEW"}
                and all(isinstance(v, Pose) for v in self.destinations.values()), "invalid destinations")
        for v in (self.pick_z_mm, self.travel_z_mm): require(v is None or number(v), "invalid height")
        for v in (self.frame_max_age_seconds, self.command_timeout_seconds): require(v is None or number(v) and 0 < v <= 300, "invalid deadline")
        require(self.end_effector in {None, "suction", "gripper"}, "unconfigured/unsupported end effector")
        require(self.verification_method is None or isinstance(self.verification_method, str) and self.verification_method.strip(), "invalid verification method")
        if self.validated:
            require(all(v is not None for v in (self.product_id, self.validation_reference, self.calibration_version,
                    self.grasp_policy_version, self.workspace, self.pick_z_mm, self.travel_z_mm, self.destinations,
                    self.end_effector, self.verification_method, self.frame_max_age_seconds, self.command_timeout_seconds)), "validated profile is incomplete")
            require({"OK", "NG"} <= set(self.destinations), "OK/NG destinations required")
            require(self.travel_z_mm > self.pick_z_mm and all(self.travel_z_mm > p.z for p in self.destinations.values()), "travel height must clear all pick/place heights")
            require(not self.allow_review_move or "REVIEW" in self.destinations, "review movement requires a destination")


@dataclass(frozen=True)
class Command:
    command_id: str
    plan_id: str
    stage: str
    action: str
    target: Pose | None = None


@dataclass(frozen=True)
class Completion:
    command_id: str
    state: str  # PENDING, DONE, FAILED, UNKNOWN
    pose: Pose | None = None
    verified: bool | None = None
    detail: str | None = None


@dataclass(frozen=True)
class RobotStatus:
    connection_epoch: str
    connected: bool
    motion_state: str
    holding: bool | None
    pose: Pose | None


class RobotAdapter(Protocol):
    kind: str
    def status(self) -> RobotStatus: ...
    def submit(self, command: Command) -> bool: ...
    def poll(self, command_id: str) -> Completion: ...
    def request_stop(self) -> bool: ...


class UnconfiguredDobotMagician:
    """No serial, SDK or automatic device discovery until hardware integration."""
    kind = "real_unconfigured"
    def status(self): return RobotStatus("unconfigured", False, "UNKNOWN", None, None)
    def submit(self, command): raise RuntimeError("DOBOT_HARDWARE_ADAPTER_NOT_IMPLEMENTED")
    def poll(self, command_id): raise RuntimeError("DOBOT_HARDWARE_ADAPTER_NOT_IMPLEMENTED")
    def request_stop(self): return False

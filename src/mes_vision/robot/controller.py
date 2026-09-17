from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from filelock import FileLock

from mes_vision.training.data import require
from .contracts import Completion, Pose
from .journal import RobotJournal
from .planning import PickPlan, STAGES, plan_digest, profile_from_dict


class RobotController:
    """Tick-driven, single-plan controller. Backend calls must be bounded/nonblocking."""
    def __init__(self, root, adapter, *, owner=None):
        self.root = Path(root).resolve(); self.root.mkdir(parents=True, exist_ok=True)
        require(owner is None or Path(owner.lock_file).resolve()==self.root/'controller.lock' and owner.is_locked,
                'Pre-acquired robot ownership required')
        self.owner = owner if owner is not None else FileLock(str(self.root / "controller.lock"), timeout=0)
        self.owner.acquire()
        try:
            self.journal = RobotJournal(root)
            self.adapter = adapter
            self.plan = None
            self.index = 0
            self.pending = None
            self.deadline = None
            self.last_time = None
            self.closed = False
            self.state = self.journal.state()
            if self.state not in {"IDLE", "RECAPTURE"}:
                self.state = "RECOVERY"
                self.journal.record(self.state, "RESTART_REQUIRES_RECONCILIATION")
        except BaseException:
            self.owner.release(); raise

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def transition(self, state, event, payload=None):
        # Write before returning success or dispatching the next physical action.
        try:
            self.journal.record(state, event, plan_id=self.plan.plan_id if self.plan else None, payload=payload)
        except Exception:
            self.state = "FAULT"
            try: self.adapter.request_stop()
            except Exception: pass
            raise
        self.state = state

    def start(self, plan, scene, *, now):
        require(not self.closed and self.state in {"IDLE", "RECAPTURE"}, "CONTROLLER_NOT_READY")
        require(self.adapter.kind in {"synthetic", "real"}, "ROBOT_ADAPTER_NOT_READY")
        require(isinstance(plan, PickPlan) and plan.digest == plan_digest(plan), "PLAN_CHANGED_OR_INVALID")
        profile = profile_from_dict(asdict(plan.profile))
        require(profile.kind == self.adapter.kind and profile.validated, "VALIDATED_MATCHING_PROFILE_REQUIRED")
        require(scene == plan.scene and 0 <= now-scene.observed_monotonic <= profile.frame_max_age_seconds, "STALE_OR_CHANGED_SCENE")
        require(tuple(c.stage for c in plan.commands) == STAGES and all(c.plan_id == plan.plan_id for c in plan.commands)
                and len({c.command_id for c in plan.commands}) == len(STAGES), "INVALID_PLAN_SEQUENCE")
        require(tuple(c.action for c in plan.commands) == ("move", "move", "engage", "verify_pick", "move", "move", "move", "release", "verify_place", "move"), "INVALID_PLAN_ACTIONS")
        require(all(c.target is None if c.action != "move" else isinstance(c.target, Pose) and profile.workspace.contains(c.target)
                    for c in plan.commands), "INVALID_PLAN_POSES")
        status = self.adapter.status()
        require(status.connected and status.connection_epoch == scene.connection_epoch and status.motion_state == "READY"
                and status.holding is False and status.pose is not None and profile.workspace.contains(status.pose)
                and abs(status.pose.z-profile.travel_z_mm) <= getattr(self.adapter,"pose_tolerance",0.), "ROBOT_STATE_UNCONFIRMED_OR_NOT_READY")
        self.journal.reserve(plan)  # Unique frame reservation persists even if the first command is rejected.
        self.plan = deepcopy(plan)
        self.state = "PREPARE_PICK"
        self.index = 0; self.pending = None; self.deadline = None; self.last_time = now

    def stop(self, reason="USER_STOP"):
        require(not self.closed, "CONTROLLER_CLOSED")
        # An intent is retained if the process dies during the stop request.
        self.transition("RECOVERY", "STOP_REQUESTED", {"reason": reason})
        try: acknowledged = self.adapter.request_stop()
        except Exception: acknowledged = False
        self.pending = None
        self.transition("STOPPED" if acknowledged else "FAULT", "STOP_RESULT", {"acknowledged": acknowledged,
                        "reason": reason, "physical_safety_confirmed": False})

    def fault(self, reason):
        self.stop(reason)
        self.transition("FAULT", "MOTION_OUTCOME_UNCONFIRMED", {"reason": reason, "retry_automatically": False})

    def tick(self, scene, *, now):
        require(not self.closed, "CONTROLLER_CLOSED")
        if self.state in {"IDLE", "RECAPTURE", "FAULT", "RECOVERY", "STOPPED"}: return self.state
        try:
            require(now >= self.last_time, "MONOTONIC_CLOCK_MOVED_BACKWARD")
            self.last_time = now
            require(scene == self.plan.scene, "SCENE_OR_CALIBRATION_CHANGED")
            status = self.adapter.status()
            require(status.connected and status.connection_epoch == scene.connection_epoch, "ROBOT_CONNECTION_CHANGED")
            if self.pending is None:
                if self.index == 0: require(now-scene.observed_monotonic <= self.plan.profile.frame_max_age_seconds, "FRAME_EXPIRED_BEFORE_MOTION")
                command = self.plan.commands[self.index]
                self.transition(command.stage, "COMMAND_INTENT", asdict(command))
                require(self.adapter.submit(command) is True, "COMMAND_REJECTED")
                self.pending = command
                self.deadline = now+self.plan.profile.command_timeout_seconds
                self.transition(command.stage, "COMMAND_ACCEPTED", {"command_id": command.command_id, "success_confirmed": False})
                return self.state
            require(now < self.deadline, "COMMAND_TIMEOUT")
            response = self.adapter.poll(self.pending.command_id)
            require(isinstance(response, Completion) and response.command_id == self.pending.command_id, "STALE_COMPLETION")
            require(response.state in {"PENDING", "DONE", "FAILED", "UNKNOWN"}, "INVALID_COMPLETION")
            if response.state == "PENDING": return self.state
            require(response.state == "DONE", "COMMAND_" + response.state)
            if self.pending.action == "move": require(response.pose == self.pending.target, "MOTION_POSITION_UNCONFIRMED")
            if self.pending.action.startswith("verify_"): require(response.verified is True, "PICK_OR_PLACE_NOT_VERIFIED")
            self.transition(self.pending.stage, "COMMAND_COMPLETED", asdict(response))
            self.pending = None; self.index += 1
            if self.index == len(self.plan.commands):
                self.transition("RECAPTURE", "CYCLE_VERIFIED", {"object_id": self.plan.object_id, "decision": self.plan.decision,
                    "new_frame_required": True, "synthetic": self.adapter.kind == "synthetic", "physical_execution_tested": self.adapter.kind == "real"})
            return self.state
        except Exception as exc:
            self.fault(f"{type(exc).__name__}: {exc}")
            return self.state

    def recover(self, *, confirmation_reference):
        require(not self.closed and self.state in {"FAULT", "RECOVERY", "STOPPED"}, "RECOVERY_NOT_REQUIRED")
        require(self.adapter.kind in {"synthetic", "real"}, "ROBOT_ADAPTER_NOT_READY")
        require(isinstance(confirmation_reference, str) and confirmation_reference.strip(), "RECOVERY_CONFIRMATION_REQUIRED")
        status = self.adapter.status()
        require(status.connected and status.motion_state == "STOPPED" and status.holding is False, "ROBOT_STATE_MUST_BE_RECONCILED_FIRST")
        # Explicit state reconciliation, never retry the previous motion.
        self.adapter.ready_after_recovery()
        self.transition("RECAPTURE", "RECOVERY_CONFIRMED", {"reference": confirmation_reference, "old_plan_discarded": True, "new_frame_required": True})
        self.plan = None; self.pending = None

    def close(self):
        if self.closed: return
        try:
            if self.state not in {"IDLE", "RECAPTURE", "STOPPED", "FAULT", "RECOVERY"}: self.stop("CONTROLLER_CLOSED")
        finally:
            self.closed = True
            self.owner.release()

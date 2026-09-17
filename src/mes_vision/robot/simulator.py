from uuid import uuid4
from .contracts import Completion, RobotStatus
from mes_vision.training.data import require


class SimulatedMagician:
    """Deterministic virtual robot. No hardware imports or transport calls."""
    kind = "synthetic"
    def __init__(self, initial_pose, *, faults=None, pending_polls=1):
        self.epoch = uuid4().hex
        self.pose = initial_pose
        self.holding = False
        self.connected = True
        self.motion_state = "READY"
        self.faults = dict(faults or {})
        require(set(self.faults.values()) <= {"reject", "timeout", "disconnect", "failure", "unknown", "unverified", "wrong_id", "wrong_pose"}, "invalid simulated fault")
        require(type(pending_polls) is int and pending_polls >= 0, "invalid simulated delay")
        self.pending_polls = pending_polls
        self.commands = []
        self.pending = None
        self.polls = 0
        self.stop_calls = 0

    def status(self): return RobotStatus(self.epoch, self.connected, self.motion_state, self.holding, self.pose)

    def submit(self, command):
        require(self.connected and self.pending is None and self.motion_state != "STOPPED", "simulator is not ready")
        require(command.command_id not in {c.command_id for c in self.commands}, "duplicate command cannot be resent")
        self.commands.append(command)
        if self.faults.get(command.stage) == "reject": return False
        self.pending, self.polls = command, 0
        if command.action in {"engage", "release"}: self.holding = None
        self.motion_state = "BUSY"
        return True

    def poll(self, command_id):
        require(self.pending is not None and self.pending.command_id == command_id, "no matching simulated command")
        command = self.pending
        fault = self.faults.get(command.stage)
        if fault == "disconnect":
            self.connected = False
            raise ConnectionError("SIMULATED_CONNECTION_LOST")
        self.polls += 1
        if fault == "timeout" or self.polls <= self.pending_polls: return Completion(command_id, "PENDING")
        self.pending = None
        self.motion_state = "READY"
        if fault in {"failure", "unknown"}:
            self.holding = None
            return Completion(command_id, "FAILED" if fault == "failure" else "UNKNOWN", detail="SIMULATED_" + fault.upper())
        if command.action == "move": self.pose = command.target
        if command.action == "engage": self.holding = True
        if command.action == "release": self.holding = False
        verified = None
        if command.action.startswith("verify_"): verified = False if fault == "unverified" else self.holding if command.action == "verify_pick" else not self.holding
        return Completion("stale-command" if fault == "wrong_id" else command_id, "DONE",
                          None if fault == "wrong_pose" else self.pose, verified, "SYNTHETIC_COMPLETION")

    def request_stop(self):
        self.stop_calls += 1
        self.pending = None
        self.motion_state = "STOPPED" if self.connected else "UNKNOWN"
        # Never release a potentially held part as a side effect of stop.
        return self.connected

    def reconcile_empty(self):
        """Explicit test-only stand-in for human/independent physical state verification."""
        self.pending = None
        self.connected = True
        self.holding = False
        self.motion_state = "STOPPED"

    def ready_after_recovery(self):
        require(self.connected and self.motion_state == "STOPPED" and self.holding is False, "state not reconciled")
        self.epoch = uuid4().hex
        self.motion_state = "READY"

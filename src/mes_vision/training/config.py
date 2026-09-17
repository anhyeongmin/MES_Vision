from dataclasses import asdict, dataclass
import math
from pathlib import Path

from .data import read_json, require, ROLES


@dataclass(frozen=True)
class JobConfig:
    role: str
    dataset_dir: str
    initial_weights: str
    initial_sha256: str
    epochs: int = 100
    batch_size: int = 2
    grad_accum_steps: int = 4
    lr: float = 0.0001
    lr_encoder: float = 0.00015
    num_workers: int = 0
    seed: int = 42
    precision: str = "bf16-mixed"
    synthetic: bool = False

    def __post_init__(self):
        require(self.role in ROLES, "invalid training role")
        for name in ("epochs", "batch_size", "grad_accum_steps"):
            require(type(getattr(self, name)) is int and getattr(self, name) > 0, f"invalid {name}")
        require(type(self.num_workers) is int and self.num_workers >= 0, "invalid num_workers")
        require(type(self.seed) is int and 0 <= self.seed < 2**32, "invalid seed")
        for value in (self.lr, self.lr_encoder):
            require(type(value) in (float, int) and math.isfinite(value) and value > 0, "invalid learning rate")
        require(self.precision in {"32-true", "bf16-mixed", "16-mixed"}, "invalid precision")
        require(type(self.synthetic) is bool, "synthetic must be boolean")
        require(len(self.initial_sha256) == 64 and all(c in "0123456789abcdef" for c in self.initial_sha256), "invalid initial weights SHA256")

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path).resolve()
        value = read_json(path)
        for name in ("dataset_dir", "initial_weights"):
            value[name] = str((path.parent / value[name]).resolve())
        return cls(**value)

    def snapshot(self):
        return asdict(self)

    def compatibility(self):
        data = self.snapshot()
        for key in ("epochs", "dataset_dir", "initial_weights"):
            data.pop(key)
        return data

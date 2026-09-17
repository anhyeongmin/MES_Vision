from contextlib import contextmanager
from pathlib import Path
import time

from filelock import FileLock, Timeout

from mes_vision.training.data import read_json, write_json


class GpuCoordinator:
    """Cooperating foreground inspections announce demand before waiting for the GPU."""
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.gpu_path = str(self.root / "gpu.lock")
        self.intent_path = str(self.root / "foreground.lock")

    def foreground_requested(self):
        try:
            with FileLock(self.intent_path, timeout=0): return False
        except Timeout: return True

    def recently_foreground(self, idle_seconds):
        path = self.root / "last-foreground.json"
        with FileLock(str(self.root / "timing.lock"), timeout=2):
            try: return time.time()-read_json(path)["time"] < idle_seconds
            except FileNotFoundError: return False

    def mark_foreground(self):
        with FileLock(str(self.root / "timing.lock"), timeout=2):
            write_json(self.root / "last-foreground.json", {"time": time.time()})

    @contextmanager
    def foreground(self, *, timeout=10):
        started = time.perf_counter()
        with FileLock(self.intent_path, timeout=timeout):
            self.mark_foreground()
            with FileLock(self.gpu_path, timeout=timeout):
                try: yield (time.perf_counter()-started)*1000
                finally: self.mark_foreground()

    def background_lock(self):
        return FileLock(self.gpu_path, timeout=0)

    def resident_blocks_vlm(self):
        # The OS-held lease disappears on a crash; a stale metadata file cannot block forever.
        try:
            with FileLock(str(self.root / "resident.lock"), timeout=0): return False
        except Timeout:
            try: return not read_json(self.root / "resident.json").get("allow_vlm", False)
            except (FileNotFoundError, ValueError): return True

    def resident_allows_retention(self):
        """Only an active, explicit memory-sharing lease permits retained weights."""
        try:
            with FileLock(str(self.root / 'resident.lock'), timeout=0): return False
        except Timeout:
            try:
                state=read_json(self.root/'resident.json')
                return state.get('active') is True and state.get('allow_vlm') is True
            except (FileNotFoundError, ValueError): return False

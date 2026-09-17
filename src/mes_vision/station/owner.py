"""Acquire the existing controller lock before a serial adapter can connect."""
from pathlib import Path
from filelock import FileLock
from mes_vision.robot.controller import RobotController


class OwnedAdapter:
    def __init__(self,root,factory):
        self.root=Path(root); self.factory=factory; self.adapter=None; self.controller=None
        self.owner=FileLock(str(self.root/'controller.lock'),timeout=0)

    def __enter__(self):
        self.root.mkdir(parents=True,exist_ok=True); self.owner.acquire()
        try:
            self.adapter=self.factory()
            # Pass the acquired owner through the narrow factory below, so the
            # controller retains its normal journal/recovery/close semantics.
            self.controller=RobotController(self.root,self.adapter,owner=self.owner)
            return self
        except BaseException:
            if self.adapter:
                try: self.adapter.close()
                except Exception: pass
            self.owner.release(); raise

    def __exit__(self,*args):
        try:
            if self.controller: self.controller.close()
        finally:
            try:
                if self.adapter: self.adapter.close()
            finally: self.owner.release()

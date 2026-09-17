"""Isolated SAM annotation child process."""
from pathlib import Path
import json,os,sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
if __name__=='__main__':
    from mes_vision.vlm.worker import watch_parent
    watch_parent(os.getppid())
    from mes_vision.data_management.sam_assist import propose
    try:
        request=json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
        print(propose(request,ROOT))
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)

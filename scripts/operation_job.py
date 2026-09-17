"""Supervised, exclusive GPU job used by the production training dialog."""
import argparse
import os
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--runtime",type=Path,required=True)
    parser.add_argument("--script",choices=["training.py","anomaly.py"],required=True); parser.add_argument("arguments",nargs=argparse.REMAINDER)
    args=parser.parse_args()
    from mes_vision.vlm.worker import watch_parent
    from mes_vision.vlm.gpu import GpuCoordinator
    from filelock import FileLock
    watch_parent(os.getppid())
    gpu=GpuCoordinator(args.runtime/"vlm/gpu-coordination")
    arguments=args.arguments[1:] if args.arguments[:1]==["--"] else args.arguments
    sys.argv=[str(ROOT/"scripts"/args.script),*arguments]
    with FileLock(str(gpu.root/"resident.lock"),timeout=0),gpu.foreground(timeout=60):
        runpy.run_path(sys.argv[0],run_name="__main__")

if __name__=="__main__": main()

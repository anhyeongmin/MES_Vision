"""Isolated saved-photograph worker. Never connects a camera or robot."""
import argparse, os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request',required=True,type=Path)
    parser.add_argument('--watch-parent',type=int)
    args=parser.parse_args()
    if args.watch_parent:
        from mes_vision.vlm.worker import watch_parent
        watch_parent(args.watch_parent)
    from mes_vision.training.data import read_json
    from mes_vision.station.photo_inspection import run_photos
    run_photos(read_json(args.request),ROOT)


if __name__=='__main__': main()

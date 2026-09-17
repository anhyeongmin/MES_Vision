"""Persistent, single-request photo worker. No camera or robot connection."""
import argparse,time,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--runtime',type=Path,required=True); p.add_argument('--watch-parent',type=int,required=True)
    args=p.parse_args()
    from mes_vision.vlm.worker import watch_parent
    watch_parent(args.watch_parent)
    from mes_vision.station.photo_inspection import ResidentPhotos,run_photos,atomic_json
    from mes_vision.training.data import read_json
    state=args.directory/'state.json'
    try:
        atomic_json(state,dict(state='LOADING'))
        with ResidentPhotos(args.directory/'bundle.json',ROOT,args.runtime,allow_background=True) as resident:
            atomic_json(state,dict(state='READY',allow_vlm=resident.allow_vlm)); previous=None
            while True:
                command_path=args.directory/'command.json'
                if not command_path.exists(): time.sleep(.05); continue
                command=read_json(command_path)
                if command['id']==previous: time.sleep(.05); continue
                previous=command['id']; atomic_json(state,dict(state='INSPECTING',id=previous))
                try:
                    report=run_photos(read_json(Path(command['request'])),ROOT,resident=resident)
                    atomic_json(state,dict(state='COMPLETED',id=previous,elapsed_ms=report['total_ms'],allow_vlm=resident.allow_vlm))
                except Exception as exc:
                    # A failed CUDA operation may poison a context; terminate instead of reusing it.
                    atomic_json(state,dict(state='FAILED',id=previous,error=str(exc))); return 1
    except Exception as exc:
        atomic_json(state,dict(state='FAILED',error=str(exc))); return 1


if __name__=='__main__': raise SystemExit(main())

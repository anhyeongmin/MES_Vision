"""Check completed render receipts while the current render batch is running."""
from pathlib import Path
import argparse
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from mes_vision.synthetic.dataset_export import read, sha, quality


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    plan = read(root/"plan.json")
    ph = sha(root/"plan.json")
    geometry = {k:np.asarray(v) for k,v in read(root/"geometry.json").items()}
    for index,task in enumerate(plan["tasks"]):
        deadline = time.monotonic()+300
        while not (root/"receipts"/(task["id"]+".json")).exists():
            if time.monotonic() > deadline:
                raise TimeoutError("No completed render receipt in five minutes")
            time.sleep(1)
        result = quality(root,task,geometry,ph)
        if not result["accepted"]:
            print(f"QUALITY_REJECT {task['id']}: {result['reason']}",flush=True)
        if (index+1)%28 == 0:
            print(f"QUALITY_PROGRESS {index+1}/{len(plan['tasks'])}",flush=True)
    from mes_vision.synthetic.reinforcement_export import export
    report = export(root,ROOT/"datasets/ash-synthetic-v2")
    print(f"EXPORT_COMPLETE crops={report['crop_images']} sources={report['original_source_images']}",flush=True)

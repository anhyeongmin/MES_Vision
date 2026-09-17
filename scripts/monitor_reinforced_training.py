"""Validation-only early stopping for this local training run."""
from pathlib import Path
import argparse
import csv
import io
import json
import math
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def plateau(history, minimum=8, patience=5, delta=.005):
    best, best_epoch = -math.inf, 0
    for row in history:
        if not math.isfinite(row["map"]):
            raise ValueError("Non-finite validation metric")
        if row["map"] >= best+delta:
            best, best_epoch = row["map"], row["epoch"]
    return bool(history and history[-1]["epoch"] >= minimum and history[-1]["epoch"]-best_epoch >= patience)


def main(run, protocol):
    policy = json.loads(protocol.read_text())["early_stopping"]
    while True:
        manifest = json.loads((run/"run.json").read_text())
        if manifest["status"] not in ("RUNNING", "VALIDATING"):
            print("TRAINING_FINAL "+manifest["status"],flush=True)
            return
        path = run/"engine/metrics.csv"
        if path.exists():
            text = path.read_text(encoding="utf-8")
            text = text[:text.rfind("\n")+1]
            history = [{"epoch":int(float(r["epoch"]))+1,"map":float(r["val/mAP_50_95"])}
                       for r in csv.DictReader(io.StringIO(text)) if r.get("val/mAP_50_95")]
            if plateau(history,policy["minimum_epochs"],policy["patience"],policy["min_delta"]):
                reason = {"reason":"Validation improvement plateau","uses_test":False,"policy":policy,"history":history}
                (run/"early-stop-reason.json").write_text(json.dumps(reason,indent=2),encoding="utf-8")
                subprocess.run([sys.executable,str(ROOT/"scripts/training.py"),"stop","--run",str(run)],check=True,cwd=ROOT)
                print("VALIDATION_PLATEAU_STOP_REQUESTED",flush=True)
                return
        time.sleep(10)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",required=True,type=Path)
    parser.add_argument("--protocol",required=True,type=Path)
    args=parser.parse_args()
    main(args.run.resolve(),args.protocol.resolve())

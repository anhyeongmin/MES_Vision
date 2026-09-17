"""Run the three prepared ASH synthetic jobs sequentially on one GPU."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
TASKS = ("overview-objects", "detail-objects", "detail-defects")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    state = {"status": "RUNNING", "synthetic": True, "production_ready": False,
             "started_at_utc": datetime.now(timezone.utc).isoformat(), "jobs": []}

    def save():
        temp = output / "batch.pending.json"
        temp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        temp.replace(output / "batch.json")

    save()
    try:
        for task in TASKS:
            for phase in ("train", "evaluate"):
                entry = {"task": task, "phase": phase, "status": "RUNNING"}
                state["jobs"].append(entry)
                save()
                command = [sys.executable, "-u", str(ROOT / "scripts/training.py"), phase]
                if phase == "train":
                    command += ["--config", str(ROOT / "configs/training" / f"ash-synthetic-v2-{task}.json"),
                                "--output", str(output / task)]
                else:
                    command += ["--run", str(output / task), "--output", str(output / f"{task}-test")]
                print(f"START {task} {phase}", flush=True)
                with (output / f"{task}-{phase}.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                entry.update(returncode=result.returncode, status="COMPLETED" if result.returncode == 0 else "FAILED")
                save()
                if result.returncode:
                    raise RuntimeError(f"{task} {phase} failed; see its log")
                print(f"DONE {task} {phase}", flush=True)
        state["status"] = "COMPLETED"
    except BaseException as exc:
        state.update(status="FAILED", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()

"""Prepare/render/export stronger synthetic defect data, without training models."""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mes_vision.synthetic.reinforcement import make_plan
from mes_vision.synthetic.dataset_export import read, write, sha


def prepare(root):
    source = ROOT / "datasets/ash-synthetic-v2"
    old_contract = read(source / "contract.json")
    if sha(source / "geometry.json") != old_contract["geometry_sha256"]:
        raise ValueError("Parent geometry changed")
    old = read(source / "plan.json")
    if sha(source / "plan.json") != old_contract["plan_sha256"]:
        raise ValueError("Parent plan changed")
    for item in old["cad_lineage"].values():
        path = Path(item["source_registry"]).parent / item["source_file"]
        if sha(path) != item["source_sha256"]:
            raise ValueError("CAD source changed")
    plan = make_plan()
    plan["cad_lineage"] = old["cad_lineage"]
    root.mkdir(parents=True, exist_ok=False)
    for folder in ("raw", "receipts", "quality", "masks"):
        (root / folder).mkdir()
    (root / "geometry.json").write_bytes((source / "geometry.json").read_bytes())
    write(root / "plan.json", plan)
    files = ["scripts/build_ash_reinforcement.py", "scripts/render_ash_samples.py", "scripts/render_ash_dataset.py",
             "src/mes_vision/synthetic/reinforcement.py", "src/mes_vision/synthetic/planning.py", "src/mes_vision/synthetic/projection.py"]
    write(root / "contract.json", dict(plan_sha256=sha(root / "plan.json"), geometry_sha256=sha(root / "geometry.json"),
        source_hashes={name:sha(ROOT/name) for name in files}, samples=32, parent_plan_sha256=old_contract["plan_sha256"]))
    write(root / "status.json", dict(status="PREPARED", planned_new_rgb=len(plan["tasks"]), model_trained=False))
    print(f"Prepared {len(plan['tasks'])} new RGB scenes", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "render", "export"))
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == "prepare":
        prepare(root)
    elif args.action == "render":
        raise SystemExit(subprocess.call([str(ROOT/".tools/blender-4.5.7-windows-x64/blender.exe"), "--background",
            "--factory-startup", "--python-exit-code", "1", "--python", str(ROOT/"scripts/render_ash_dataset.py"),
            "--", "--root", str(root)], cwd=ROOT))
    else:
        from mes_vision.synthetic.reinforcement_export import export
        print(export(root, ROOT/"datasets/ash-synthetic-v2"))

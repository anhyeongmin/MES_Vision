"""Manage reviewed source collections and export data for the training program."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from mes_vision.data_management.collection import Collection, SPLITS
from mes_vision.data_management.export import export_collection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("workspace", type=Path)
    create.add_argument("--product-id", required=True)
    create.add_argument("--kind", choices=["real", "synthetic"], default="real")
    add = commands.add_parser("import")
    add.add_argument("workspace", type=Path)
    add.add_argument("images", type=Path, nargs="+")
    add.add_argument("--session", required=True)
    video = commands.add_parser("import-video")
    video.add_argument("workspace", type=Path)
    video.add_argument("video", type=Path)
    video.add_argument("--session", required=True)
    video.add_argument("--interval", type=float, default=1.0)
    video.add_argument("--max-frames", type=int, default=120)
    split = commands.add_parser("assign-session")
    split.add_argument("workspace", type=Path)
    split.add_argument("--session", required=True)
    split.add_argument("--split", choices=SPLITS, required=True)
    inventory = commands.add_parser("inventory")
    inventory.add_argument("workspace", type=Path)
    export = commands.add_parser("export")
    export.add_argument("workspace", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--role", choices=["object_detector", "known_defect_detector", "normal_bank", "challenge"], required=True)
    export.add_argument("--defect-codes", nargs="+", default=[])
    args = parser.parse_args()
    try:
        if args.command == "create":
            collection = Collection.create(args.workspace, args.product_id, kind=args.kind)
        else:
            collection = Collection(args.workspace)
        if args.command == "import":
            for path in args.images:
                collection.import_image(path, args.session)
        elif args.command == "import-video":
            from mes_vision.data_management.video import import_video
            result = import_video(collection, args.video, args.session,
                                  interval_seconds=args.interval, max_frames=args.max_frames)
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "assign-session":
            collection.assign_session(args.session.strip(), args.split)
        elif args.command == "export":
            report = export_collection(collection, args.output, args.role, defect_codes=args.defect_codes)
            print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "images": len(report["items"]), "excluded": report["excluded"]}, ensure_ascii=False, indent=2))
            return 0
        print(json.dumps(collection.inventory(), ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

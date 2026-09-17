"""Build normal memory and score original product crops with pinned local DINOv2."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
from mes_vision.anomaly.features import DinoFeatures
from mes_vision.anomaly.bank import Bank, BankConfig, build_bank
from mes_vision.anomaly.scoring import AnomalyEngine, save_score
from mes_vision.inputs import ImageSource, InputStatus
from mes_vision.training.data import read_json, require, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    build = sub.add_parser("build")
    build.add_argument("--normal-export", type=Path, required=True)
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--config", type=Path, default=ROOT / "configs/anomaly/build.json")
    for name in ("inspect", "validate-bank", "criteria-template"):
        command = sub.add_parser(name)
        command.add_argument("--bank", type=Path, required=True)
        command.add_argument("--product-id", required=True)
        if name == "inspect":
            command.add_argument("--image", type=Path, required=True, help="one original-resolution product crop")
            command.add_argument("--criteria", type=Path)
        if name in {"inspect", "criteria-template"}:
            command.add_argument("--output", type=Path, required=True)
    for command in sub.choices.values():
        command.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
        command.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()
    extractor = engine = None
    try:
        if args.command == "build":
            config = read_json(args.config)
            require(set(config) == {"image_size", "bank"}, "unknown or missing build config fields")
            extractor = DinoFeatures(ROOT, device=args.device, image_size=config["image_size"])
            result = build_bank(args.normal_export, args.output, extractor, config=BankConfig(**config["bank"]), allow_synthetic=args.synthetic)
            print(json.dumps({"status": result["status"], "bank_patches": result["bank_patches"], "source_images": result["source_images"], "thresholds": None, "output": str(args.output.resolve())}, ensure_ascii=False))
        else:
            meta = read_json(args.bank / "bank.json")
            extractor = DinoFeatures(ROOT, device=args.device, image_size=meta["feature_signature"]["preprocessing"]["size"],max_batch_size=meta['feature_signature'].get('batch_limit',1))
            bank = Bank(args.bank, extractor.signature, product_id=args.product_id, allow_synthetic=args.synthetic)
            if args.command == "validate-bank":
                print(json.dumps({"status": "VERIFIED", "bank_digest": bank.digest, "product_id": args.product_id, "kind": bank.meta["kind"], "inference_performed": False}))
            elif args.command == "criteria-template":
                require(not args.output.exists(), "criteria output exists")
                args.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(args.output, {"version": "UNCONFIGURED", "bank_digest": bank.digest, "product_id": args.product_id,
                    "pass_max": None, "fail_min": None, "pixel_threshold": None, "validated": False,
                    "validation_reference": None, "kind": bank.meta["kind"]})
                print("Unconfigured criteria template written; use inspect without --criteria until actual thresholds are ready.")
            else:
                engine = AnomalyEngine(args.bank, extractor, product_id=args.product_id, criteria=args.criteria, allow_synthetic=args.synthetic)
                with ImageSource(args.image) as source:
                    event = source.read()
                    require(event.status == InputStatus.FRAME, "could not read crop image")
                    frame = event.frame
                details, grid, nearest = engine.score(frame.rgb)
                details["input_frame"] = frame.metadata()
                save_score(args.output, frame.rgb, details, grid, nearest)
                print(json.dumps({"status": details["status"], "raw_score": details["raw_score"], "messages": details["messages"], "output": str(args.output.resolve()), "final_decision": None}, ensure_ascii=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}", "final_decision": None}, ensure_ascii=False), file=sys.stderr)
        return 1
    finally:
        if engine is not None: engine.close()
        if extractor is not None: extractor.close()


if __name__ == "__main__":
    raise SystemExit(main())

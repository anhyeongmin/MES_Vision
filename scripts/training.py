"""MES Vision local RF-DETR train, resume, evaluate, preflight and graceful stop."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for key, folder in {"HF_HOME": "huggingface", "TORCH_HOME": "torch", "MPLCONFIGDIR": "matplotlib"}.items():
    os.environ[key] = str(ROOT / ".cache" / folder)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["DO_NOT_TRACK"] = "1"


def main():
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate", help="데이터 및 초기 가중치 확인")
    check.add_argument("--config", required=True, type=Path)
    train = commands.add_parser("train", help="학습 또는 전체 상태 복원 후 재개")
    train.add_argument("--config", required=True, type=Path)
    train.add_argument("--output", required=True, type=Path)
    train.add_argument("--resume-from", type=Path)
    train.add_argument("--stop-after-epoch", type=int)
    evaluate = commands.add_parser("evaluate", help="저장된 최선 가중치의 별도 평가")
    evaluate.add_argument("--run", required=True, type=Path)
    evaluate.add_argument("--output", required=True, type=Path)
    evaluate.add_argument("--split", choices=["valid", "test"], default="test")
    stop = commands.add_parser("stop", help="현재 epoch 완료·저장 후 정지 요청")
    stop.add_argument("--run", required=True, type=Path)
    fixture = commands.add_parser("make-fixture", help="실물과 구분된 시험 데이터 생성")
    fixture.add_argument("--output", required=True, type=Path)
    fixture.add_argument("--role", choices=["object_detector", "known_defect_detector"], required=True)
    args = parser.parse_args()
    from mes_vision.training.config import JobConfig
    from mes_vision.training.data import read_json, require
    from mes_vision.training import jobs
    try:
        if args.command == "validate":
            data = jobs.preflight(JobConfig.load(args.config))
            result = {"status": "VALID", "fingerprint": data["fingerprint"], "metadata": data["metadata"], "categories": data["categories"], "splits": data["splits"]}
        elif args.command == "train":
            result = jobs.train(JobConfig.load(args.config), args.output, resume_from=args.resume_from, stop_after_epoch=args.stop_after_epoch)
        elif args.command == "evaluate":
            result = jobs.evaluate(args.run, args.output, split=args.split)
        elif args.command == "stop":
            manifest = read_json(args.run / "run.json")
            require(manifest["status"] == "RUNNING", "학습 실행 중인 폴더가 아닙니다")
            (args.run / "STOP").touch(exist_ok=True)
            result = {"status": "STOP_REQUESTED", "message": "현재 epoch 완료 및 저장 후 멈춥니다"}
        else:
            from mes_vision.training.fixtures import make_fixture
            result = {"status": "CREATED", "kind": "synthetic", "path": str(make_fixture(args.output, args.role))}
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

"""Inspect local inputs; emit status/metadata and optionally save the first RGB frame."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image
from mes_vision.inputs import D405Source, FolderSource, ImageSource, InputStatus, VideoSource


def main() -> int:
    # Keep Korean status text intact both in Windows terminals and redirected logs.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="MES Vision 사진·폴더·영상 입력 확인")
    parser.add_argument("kind", choices=["image", "folder", "video", "d405"])
    parser.add_argument("path", nargs="?", help="로컬 파일 또는 폴더")
    parser.add_argument("--recursive", action="store_true", help="하위 폴더 포함")
    parser.add_argument("--limit", type=int, default=10, help="최대 읽기 프레임 수 (기본 10)")
    parser.add_argument("--output", type=Path, help="새 결과 폴더: events.jsonl, first-frame.png, summary.json")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("--limit는 1 이상이어야 합니다")
    if args.kind != "d405" and not args.path:
        parser.error("파일 또는 폴더 경로가 필요합니다")
    if args.recursive and args.kind != "folder":
        parser.error("--recursive는 folder 입력 전용입니다")
    if args.output:
        try:
            args.output.mkdir(parents=True, exist_ok=False)
        except OSError as exc:
            parser.error(f"새 결과 폴더를 만들 수 없습니다: {exc}")
    source = (D405Source() if args.kind == "d405" else
              FolderSource(args.path, recursive=args.recursive) if args.kind == "folder" else
              ImageSource(args.path) if args.kind == "image" else VideoSource(args.path))
    journal = (args.output / "events.jsonl").open("w", encoding="utf-8") if args.output else None
    count = 0
    reason = "limit_reached"
    code = None

    def emit(event):
        row = {"status": event.status.value, "code": event.code, "message": event.message,
               "frame": event.frame.metadata() if event.frame is not None else None}
        line = json.dumps(row, ensure_ascii=False)
        print(line)
        if journal:
            journal.write(line + "\n")
            journal.flush()

    try:
        emit(source.open())
        while count < args.limit:
            event = source.read()
            emit(event)
            if event.status != InputStatus.FRAME:
                reason, code = event.status.value, event.code
                break
            count += 1
            if count == 1 and args.output:
                Image.fromarray(event.frame.rgb).save(args.output / "first-frame.png")
    finally:
        emit(source.close())
        if journal:
            journal.close()
    summary = {"kind": args.kind, "frames_read": count, "stop_reason": reason,
               "code": code, "is_live": False, "inspection_performed": False}
    if args.output:
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    return 1 if reason in ("error", "not_configured") else 0


if __name__ == "__main__":
    raise SystemExit(main())

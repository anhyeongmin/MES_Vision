"""Create a readable report from completed, recorded synthetic training results."""
from pathlib import Path
import argparse
import csv
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ["MPLCONFIGDIR"] = str(ROOT / ".cache/matplotlib")
from mes_vision.training.data import read_json, require, write_json, sha256
from mes_vision.synthetic.detection_review import summarize


def percent(value):
    return "—" if value is None else f"{value*100:.1f}%"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", required=True, type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    review = read_json(batch / "review/summary.json")
    require(review.get("status") == "COMPLETED", "inference review not complete")
    tasks = ("overview-objects", "detail-objects", "detail-defects")
    names = {"overview-objects": "전체사진 물체 위치", "detail-objects": "상세사진 물체 위치", "detail-defects": "불량 검출 — 정답 물체 영역"}
    lines = ["# ASH 합성 데이터 1차 학습 결과", "", "2026-09-11. RF-DETR Small 세 모델 학습과 별도 시험 평가를 완료했다. 운영 모델 등록은 변경하지 않았다.", "",
             "## 학습 및 위치·영역 평가", "", "| 모델 | 완료 epoch | 시험 mAP50:95 | 시험 mAP50 |", "|---|---:|---:|---:|"]
    records = {}
    for task in tasks:
        run = read_json(batch / task / "run.json")
        evaluation = read_json(batch / f"{task}-test/evaluation.json")
        require(evaluation["status"] == "COMPLETED", "COCO evaluation not complete")
        metric = evaluation["metrics"][0]
        records[task] = {"epochs": run["completed_epochs"], "status": run["status"], "selection": run["selection"],
                         "weights": run["checkpoints"]["inference"], "metrics": metric}
        lines.append(f'| {names[task]} | {run["completed_epochs"]} | {percent(metric["test/mAP_50_95"])} | {percent(metric["test/mAP_50"])} |')
    lines += ["", "mAP는 불량 판정 정확도와 다른 검출 지표다. 상세 물체 모델은 검증 mAP가 5회 연속 상한값에 도달해 저장 후 종료를 요청했고 7epoch에서 마쳤다. 가중치 선택에는 검증 자료만 사용했다.", "",
              "## 앱 추론 어댑터로 시험 이미지 확인", "", "아래 임계값은 검증 자료에서 선택한 뒤 고정했다. 운영 판정 기준으로 승인한 값은 아니다. IoU 0.5와 클래스 일치를 동시에 만족해야 정답 검출로 센다.", "",
              "| 평가 | 찾은 정답 / 전체 정답 | 잘못된 검출 수 | 정상·빈 사진 오검출 |", "|---|---:|---:|---:|"]
    for task in (*tasks, "detail-predicted-crop-chain"):
        result = review["models"][task]
        total = result["total"]
        name = names.get(task, "물체 위치 검출 → 잘라내기 → 불량 검출")
        negative_result = f'{result["false_alarm_images"]} / {result["negative_images"]}장' if result["negative_images"] else "해당 없음"
        lines.append(f'| {name} | {total["tp"]} / {total["tp"]+total["fn"]} | {total["fp"]} | {negative_result} |')
    chain = review["models"]["detail-predicted-crop-chain"]
    lines += ["", "## 실제 물체 검출 영역을 사용한 불량별 결과", "", "| 불량 | 찾은 정답 / 전체 | 잘못된 검출 | 정밀도 | 재현율 |", "|---|---:|---:|---:|---:|"]
    descriptions = {"NG01": "구조 누락", "NG02": "추가 돌출", "NG03": "균열", "NG04": "벽 변형", "NG05": "구멍 불량", "NG06": "표면 함몰"}
    for code, result in chain["classes"].items():
        lines.append(f'| {code} {descriptions[code]} | {result["tp"]} / {result["tp"]+result["fn"]} | {result["fp"]} | {percent(result["precision"])} | {percent(result["recall"])} |')
    lines += ["", f'물체 영역을 확정하지 못한 상세사진: {chain["unconfirmed_crop_images"]}장. 불량 검출이 없다는 사실을 OK 판정으로 간주하지 않았다.', "",
              "## 잘라내기와 임계값 영향 구분", "", "정답 영역과 예측 영역의 검증 자료에서 선택된 임계값이 다르므로, 기본 결과의 차이를 전부 잘라내기 오차 때문이라고 해석할 수 없다. 아래는 두 검증 단계에서 이미 선택한 임계값을 공통으로 적용한 진단 비교다. 시험 자료를 보고 임계값을 다시 선택하지 않았다.", "",
              "| 입력 | 공통 임계값 | 찾은 정답 / 전체 | 잘못된 검출 |", "|---|---:|---:|---:|"]
    comparison = []
    thresholds = sorted({review["models"]["detail-defects"]["threshold"], chain["threshold"]})
    for task, name in (("detail-defects", "정답 물체 영역"), ("detail-predicted-crop-chain", "예측 물체 영역")):
        rows = read_json(batch / "review" / task / "test-predictions.json")
        for threshold in thresholds:
            result = summarize(rows, threshold, list(descriptions))
            counts = result["total"]
            comparison.append({"input": task, "threshold": threshold, "counts": counts})
            lines.append(f'| {name} | {threshold:.2f} | {counts["tp"]} / {counts["tp"]+counts["fn"]} | {counts["fp"]} |')
    write_json(batch / "review/crop-threshold-comparison.json", {"threshold_sources": "validation only", "results": comparison})
    lines += ["", "남은 약점은 특히 NG01의 누락 위치 혼동과 낮은 확신도다. 다음 개선 후보는 물체 잘라내기 위치·배율의 작은 오차를 포함한 학습, 누락/정상 구조의 어려운 사례 보강, 검증 자료 확충 후 종류별 임계값 및 보류 정책 검토다. 같은 시험 자료로 반복 조정하면 이후 결과는 개발 진단으로 취급하고 새로운 최종 시험 자료를 확보해야 한다.", "",
              "## 확인 자료", "", "- [학습 곡선](learning-curves.png)", "- [전체사진 검출 예시](review/overview-objects/test-review.png)",
              "- [상세 불량 검출 예시 — 예측 물체 영역 사용](review/detail-predicted-crop-chain/test-review.png)",
              "- `review/*/test-predictions.json`: 전체 시험 이미지의 예측과 정답.", "- `review/*/test-counts.json`: 누락·오검출 사례 목록.",
              "- 각 모델 폴더 `inference.pth`: 해시가 기록된 학습 결과 가중치.", "",
              "## 결과의 한계와 다음 단계", "", "모든 분할에서 같은 CAD 7종을 공유한다. 이번 결과는 알려진 형상을 새로운 합성 촬영 조건에서 검사한 결과이며 실물 성능을 보장하지 않는다. 원본 540장 가운데 상세 420장은 프로그램과 같은 물체 영역 입력으로 변환해 불량 학습에 사용했다. 정답 영역 평가와 예측 영역 평가를 따로 기록했다.", "",
              "다음은 발견한 누락·오검출 유형을 기준으로 학습 자료를 보완하고, U20CAM 실물 사진을 확보하면 같은 입력 처리로 보완 학습과 독립 실물 평가를 진행하는 것이다. 품질 검사, 미지 불량, 보류 판정 기준, 로봇 좌표·동작, VLM 설명 및 최적화 추론 프로필은 이번 검출 평가에 포함되지 않는다.", ""]
    (batch / "README.md").write_text("\n".join(lines), encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    for task in tasks:
        with (batch / task / "engine/metrics.csv").open() as stream:
            rows = [r for r in csv.DictReader(stream) if r.get("val/mAP_50_95")]
        axes[0].plot([int(r["epoch"])+1 for r in rows], [float(r["val/mAP_50_95"]) for r in rows], label=task)
        if task == "detail-defects":
            for code in descriptions:
                axes[1].plot([int(r["epoch"])+1 for r in rows], [float(r[f"val/AP/{code}"]) for r in rows], label=code)
    for ax, title in zip(axes, ("Validation detection mAP", "Validation defect AP by class")):
        ax.set(title=title, xlabel="Epoch", ylabel="AP @ IoU 0.50:0.95", ylim=(0, 1.03))
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="lower right")
    fig.suptitle("Synthetic CAD only - not physical inspection accuracy")
    fig.savefig(batch / "learning-curves.png", dpi=160)
    plt.close(fig)
    write_json(batch / "result-summary.json", {"status": "COMPLETED", "synthetic": True, "production_ready": False,
               "training": records, "adapter_review": review,
               "review_sources_sha256": {f: sha256(ROOT / f) for f in ("scripts/analyze_ash_training.py", "scripts/report_ash_training.py", "src/mes_vision/synthetic/detection_review.py")}})
    print(batch / "README.md")


if __name__ == "__main__":
    main()

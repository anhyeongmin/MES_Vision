"""Write the Korean results report from completed, matched synthetic evaluations."""
import csv
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_ash_reinforcement import BATCH, LABELS
from mes_vision.training.data import read_json, require


def fraction(report):
    c = report["total"]
    return f"{c['tp']}/{c['tp']+c['fn']}"


def main():
    review = BATCH / "review"
    summary = read_json(review / "summary.json")
    require(summary["status"] == "COMPLETED" and summary["matched_inputs_verified"], "comparison incomplete")
    old_ap = read_json(BATCH / "baseline-on-v3-test/evaluation.json")
    new_ap = read_json(BATCH / "detail-defects-test/evaluation.json")
    require(old_ap["status"] == new_ap["status"] == "COMPLETED", "native evaluations incomplete")
    require(old_ap["dataset_fingerprint"] == new_ap["dataset_fingerprint"] == summary["dataset_fingerprint"], "native datasets differ")
    run = read_json(BATCH / "detail-defects/run.json")
    with (BATCH / "detail-defects/engine/metrics.csv").open(encoding="utf-8") as stream:
        history = [(int(float(r["epoch"]))+1, float(r["val/mAP_50_95"]))
                   for r in csv.DictReader(stream) if r.get("val/mAP_50_95")]
    best_epoch, best_map = max(history, key=lambda r: r[1])
    old, new = summary["models"]["baseline"], summary["models"]["reinforced"]
    chain_old, chain_new = old["predicted_crop_chain"], new["predicted_crop_chain"]
    routes = {"ground_truth_crop": "정답 물체 영역", "predicted_crop_chain": "물체 검출→자르기→불량 검출", "crop_stress": "자르기 오차 변형"}
    text = ["# ASH 보강 재학습 및 같은 새 시험 자료 비교", "",
        f"RTX 3090에서 보강 학습 {run['completed_epochs']}회차를 완료했다. 검증 성능이 가장 높은 {best_epoch}회차 가중치를 선택했다(검증 mAP50:95 {best_map:.4f}).",
        f"실제 물체 검출과 자르기를 거친 합성 시험에서 불량 위치·종류가 맞은 수는 **{fraction(chain_old)} → {fraction(chain_new)}**다. 추가 오검출은 **{chain_old['total']['fp']} → {chain_new['total']['fp']}건**이다.",
        "", "이 결과는 **같은 CAD 7종의 새로운 합성 촬영 조건**에서 얻었다. 실제 인쇄물·U20CAM 촬영·새 불량 형상의 정확도나 최종 제품 판정률이 아니다. 운영 모델 등록과 장비 구동은 수행하지 않았다.",
        "", "## 비교 방법", "",
        "기존 모델과 새 모델 모두 동일한 새 검증/시험 자료에 적용했다. 이미지 파일 이름·바이트 해시·정답 좌표가 동일한지 검사했다. 이전 v2 시험 수치와 새 시험 수치를 직접 비교하지 않았다.",
        "", "- 학습: 2,430장. 기본 검증/시험: 각각 126장(정상 18장, NG 6종 각18장). 각 분할은 촬영 조건 18개×7종이다.",
        "- 자르기 오차 검증/시험: 각각 378장. 기본 사진을 각각 3번 변형했으므로 독립 표본 수에 합산하지 않는다.",
        "- 각 모델의 검증 자료에서 micro-F1이 가장 높은 검출 기준을 선택한 뒤 고정했다. 오차 시험은 정답 영역 검증에서 정한 기준을 그대로 썼다.",
        "- 모든 기본 검출 개수는 종류가 일치하고 정답과의 영역 겹침(IoU)이 0.5 이상일 때만 정검출로 센다. 중복 검출·위치가 맞지 않는 검출은 오검출이다.",
        "- 상세 물체 모델은 기존 가중치와 기준 0.2를 고정했다. 실제 프로그램의 영역 자르기·좌표 복원 함수를 사용했다.",
        "", "## 검증 자료로 정한 기준에서의 결과", "",
        "| 검사 경로 | 검출 기준 기존→보강 | 정검출 기존→보강 | 오검출 기존→보강 | 누락 기존→보강 |", "|---|---:|---:|---:|---:|"]
    for route, title in routes.items():
        a, b = old[route], new[route]
        text.append(f"| {title} | {a['threshold']:.2f}→{b['threshold']:.2f} | {fraction(a)}→{fraction(b)} | {a['total']['fp']}→{b['total']['fp']} | {a['total']['fn']}→{b['total']['fn']} |")
    text += ["", f"예측 영역 경로에서 정상 사진의 오검출은 {chain_old['false_alarm_images']}/{chain_old['negative_images']} → {chain_new['false_alarm_images']}/{chain_new['negative_images']}장이다. 물체 연결 실패는 {chain_old['unconfirmed_crop_images']} → {chain_new['unconfirmed_crop_images']}장이다.",
        "", "### 불량 종류별 — 물체 검출·자르기를 거친 결과", "",
        "| 종류 | 정검출 기존→보강 | 오검출 기존→보강 | 누락 기존→보강 |", "|---|---:|---:|---:|"]
    names = ["구조 누락", "추가 돌출", "균열 홈", "벽 변형", "구멍 막힘", "표면 함몰"]
    for code, name in zip(LABELS, names):
        a, b = chain_old["classes"][code], chain_new["classes"][code]
        text.append(f"| {code} {name} | {a['tp']}/{a['tp']+a['fn']}→{b['tp']}/{b['tp']+b['fn']} | {a['fp']}→{b['fp']} | {a['fn']}→{b['fn']} |")
    text += ["", "### 남은 오류", "",
        "보강 모델의 검증 선택 기준에서 남은 오검출을 아래에 모두 기록했다. 추가 오검출은 이미 다른 불량을 맞힌 사진에서 잘못된 불량 종류를 더 표시하는 경우도 포함한다.", ""]
    errors = read_json(review / "reinforced/predicted_crop_chain-test-counts.json")["errors"]
    for row in errors:
        details = ", ".join(f"{p['label']} 점수 {p['score']:.3f}" for p in row["fp"])
        text.append(f"- `{row['file_name']}`: 추가 오검출 {details or '없음'}, 누락 {len(row['fn'])}개.")
    if not errors:
        text.append("- 이 시험에서는 오검출·누락이 없었다.")
    text.append("")
    text.append("실물 보완 학습과 별도 검증에서 오검출·누락 비용에 맞는 운영 기준을 정해야 한다. 아래 공통 기준 결과를 시험 후 운영 기준 재선택의 근거로 사용하지 않았다.")
    text += ["", "## 두 모델에 같은 검출 기준을 적용한 비교", "",
        "기준 변경에 따른 효과를 구분하기 위해 실행 전에 고정한 기준으로도 비교했다. 시험 결과를 보고 고른 기준이 아니다.", "",
        "| 검사 경로 | 공통 기준 | 정검출 기존→보강 | 오검출 기존→보강 | 누락 기존→보강 |", "|---|---:|---:|---:|---:|"]
    for route, title in routes.items():
        a = read_json(review / "baseline" / f"{route}-fixed-threshold-counts.json")
        b = read_json(review / "reinforced" / f"{route}-fixed-threshold-counts.json")
        require(a["threshold"] == b["threshold"], "fixed thresholds differ")
        text.append(f"| {title} | {a['threshold']:.2f} | {fraction(a)}→{fraction(b)} | {a['total']['fp']}→{b['total']['fp']} | {a['total']['fn']}→{b['total']['fn']} |")
    text += ["", "## 같은 새 시험의 COCO 지표", "",
        "정답 물체 영역에서 두 모델을 같은 RF-DETR 평가기·BF16 설정으로 평가했다. 위 프로그램 경로 검출 개수는 standard 추론 설정을 사용하므로 서로 다른 지표로 구분한다.", "",
        "| 지표 | 기존 | 보강 |", "|---|---:|---:|"]
    metrics = [("mAP50:95", "test/mAP_50_95"), ("mAP50", "test/mAP_50")]
    metrics += [(f"{code} AP50:95", f"test/AP/{code}") for code in LABELS]
    for label, key in metrics:
        text.append(f"| {label} | {old_ap['metrics'][0][key]:.4f} | {new_ap['metrics'][0][key]:.4f} |")
    text += ["", "## 학습·평가 기록", "",
        f"- 학습 시간: {run['seconds']/60:.1f}분. 기존 불량 모델을 초기값으로 별도 학습. 배치8, BF16, 학습률0.00005, 최대30회차.",
        "- 조기 종료: 최소8회차 이후 검증 mAP50:95가 유의미하게 0.005 이상 개선되지 않는 상태가 5회 지속될 때 현재 회차를 끝내고 종료. 가중치는 전체 회차 중 실제 최고 검증 점수로 선택. 시험 자료는 학습 종료·가중치·검출 기준 선택에 사용하지 않음.",
        "- 검출이 없다는 이유만으로 OK를 승인하지 않음. 미지 불량, 보류 처리, 촬영 품질, 좌표 보정, 로봇 동작, VLM과 최종 판정 정책은 이번 수치의 평가 대상이 아님.",
        "- 좌표 변환·비교 입력 일치·조기 종료 관련 신규 단위시험 8개 통과. 학습·자료 생성 소스의 고정 해시 확인. 전체 앱 회귀검사는 이번 학습 작업에서 재실행하지 않음.",
        "", "[학습 곡선과 공통 기준 비교](review/learning-and-comparison.png) · [검증 선택 기준의 예측 사진·잔여 오류](review/paired-chain-selected.png) · [공통 기준의 예측 사진 비교](review/paired-chain-review.png)",
        "", "사진 비교는 종류별 1장씩 오류를 우선 선택한 예시이며 무작위 표본이 아니다. 전체 오류와 예측은 `review/baseline`, `review/reinforced`의 JSON에 저장했다.",
        "", "- 새 가중치: `detail-defects/inference.pth`", f"- SHA256: `{run['checkpoints']['inference']['sha256']}`",
        "- 설정·종료 규칙: `protocol.json`, `detail-defects/run.json`, `detail-defects/early-stop-reason.json`(조기 종료 시)",
        "- 검출 기준: `review/thresholds.json`", "- 동일 자료 비교 결과: `review/summary.json`",
        "- COCO 평가: `detail-defects-test/evaluation.json`, `baseline-on-v3-test/evaluation.json`", ""]
    (BATCH / "README.md").write_text("\n".join(text), encoding="utf-8")
    print("Wrote", BATCH / "README.md")


if __name__ == "__main__":
    main()

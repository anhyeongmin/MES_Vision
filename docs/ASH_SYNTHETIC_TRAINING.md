# ASH 합성 데이터 1차 학습

**완료:** 세 모델 학습과 별도 시험·앱 어댑터 검토를 끝냈다. 최종 수량, 불량별 오류, 그림은 `artifacts/training/ash-synthetic-v2-first/README.md`에 있다. 전체/상세 물체는 각각 시험 정답 53/53, 63/63개를 검출했다. 검증에서 선택한 임계값 0.85의 예측 물체 영역 기반 불량 검출은 49/54개, 잘못된 검출 1개였다. 정상 9장의 오검출은 없었다. 이는 합성 시험 결과이며 산업 현장 성능 승인이 아니다.

2026-09-11. 준비된 540장의 합성 사진으로 RF-DETR Small 모델 세 개를 각각 학습한다. 전체사진 물체 위치 → 상세사진 물체 위치 → 상세사진 불량 영역 순서이며, 각 학습은 검증용 자료의 mAP로 가중치를 선택한다. 시험 자료는 학습 업데이트나 가중치 선택에 사용하지 않는다.

결과 위치: `artifacts/training/ash-synthetic-v2-first`. `batch.json`은 세 학습과 별도 시험 평가의 상태다. 각 모델 폴더의 `run.json`에는 실제 설정, 데이터 지문, 선택 근거와 가중치 해시가 기록된다. 진행 중에는 `progress.json`에서 마지막으로 학습을 마친 epoch를 확인할 수 있다. 검증·저장이 끝나기 전에는 다음 단계로 넘어가지 않는다.

## 입력과 설정

| 모델 | 학습 입력 | 학습/검증/시험 | 최초 epoch | 물리 배치 / 누적 |
|---|---|---|---:|---|
| overview-objects | 전체 RGB | 84/18/18 | 30 | 2 / 4 |
| detail-objects | 상세 RGB | 294/63/63 | 30 | 8 / 1 |
| detail-defects | 상세사진의 물체 영역 | 294/63/63 | 30 | 8 / 1 |

세 설정 모두 유효 배치 크기는 명목상 8이며 BF16 혼합 정밀도를 쓴다. 상세 학습이 시작되기 전 RTX3090 메모리 여유를 활용하도록 물리 배치 수를 조정했다. 실제 설정은 각 실행의 `run.json`이 기준이다. 패키지와 소스 해시는 `provenance.json`에 기록한다.

상세 물체 검출은 검증 mAP50:95가 5회 연속 1.0에 도달해 저장 후 중단을 요청했고, 7epoch에서 정상 종료했다. `early-stop-reason.json`에 판단에 사용한 검증 이력이 있다. 최대 30회 설정을 모두 수행했다는 뜻은 아니며, 종료 상태 `STOPPED`와 완료 epoch 수를 유지한다. 불량 모델은 별도 모델로 학습한다.

불량 모델에 전체 상세사진을 그대로 넣으면 실제 프로그램의 물체 잘라내기와 입력 배율이 달라진다. 이를 맞추기 위해 `datasets/ash-synthetic-v2/training/detail-defects-crops-v1`을 파생 생성했다. 운영 코드의 `extract_crop`과 여백 0을 사용하며, 학습 자료에서는 정답 물체 상자로 자른다. 원본·촬영 조건·분할·클래스를 유지하고 불량 정답 좌표만 이동했다. 420장 픽셀 일치와 360개 정답 좌표의 원본 복원 일치도 확인했다. 추가로 촬영한 420장이 아니라 기존 상세사진의 학습 입력 변환이다.

## 평가 범위

- `*-test/evaluation.json`: 저장된 최선 가중치의 COCO 시험 지표.
- 추론 검토 결과는 실제 RF-DETR 앱 어댑터로 계산한다. 진단 임계값은 검증용 자료에서 IoU 0.5 기준 F1로 고르고, 고정한 뒤 시험 자료의 TP/FP/FN과 정상 사진 오검출을 집계한다.
- 물체 상자를 정답으로 주는 불량 단독 평가와, 상세 물체 모델이 예측한 영역을 잘라 불량을 검출하는 연속 평가를 구분한다. 잘못되거나 여러 개인 물체 검출을 임의로 하나 고르지 않는다.
- 검출이 없다는 사실을 최종 OK 판정으로 바꾸지 않는다. 품질 조건, 미지 불량 검사, 보류 정책과 실제 장비의 성능 승인은 이 평가 범위에 포함되지 않는다.
- 모든 분할이 같은 CAD 7종을 공유한다. 합성 시험 성능은 실제 U20CAM 촬영이나 새로운 불량 형상에 대한 정확도가 아니다.

## 실행

```powershell
& ./.venv/Scripts/python.exe -u scripts/train_ash_synthetic.py --output artifacts/training/ash-synthetic-v2-first
& ./.venv/Scripts/python.exe -u scripts/analyze_ash_training.py --batch artifacts/training/ash-synthetic-v2-first --output artifacts/training/ash-synthetic-v2-first/review
```

첫 명령의 출력 폴더는 새 폴더여야 한다. 완료된 실행에 다시 실행하지 않는다. 중단된 학습은 기존 `scripts/training.py train --resume-from` 기능으로 새 실행 폴더에 재개하며, 원래 실행 기록을 덮어쓰지 않는다. 운영 모델 등록이나 실제 카메라·로봇 동작은 이 학습 명령에 포함되지 않는다.

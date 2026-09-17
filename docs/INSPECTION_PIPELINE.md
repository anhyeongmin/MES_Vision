# 05 검출·검사 연결 코드

갱신일: 2026-09-05

입력 모듈의 원본 RGB 프레임을 받아 여러 물체를 식별하고, 물체별 원본 이미지를 잘라 검사기에 전달한다. 검출·검사 결과와 근거 영역을 JSON으로 기록할 수 있다. 모의 검사와 실제 RF-DETR 실행을 분리했다.

10 단계에서 선택형 OK/NG/REVIEW 판정 규칙을 연결했다. [DECISION.md](DECISION.md)를 따른다. 정책을 생략한 기존 원시 검사 경로의 최종 판정은 null이며 `decision_status=NOT_IMPLEMENTED`다. 명시적 정책을 전달하면 판정 및 decision_details를 별도로 남긴다. 두 경로 모두 로봇 명령은 비활성이고 미학습·미설정을 정상으로 처리하지 않는다.

## 구성

| 파일 | 역할 |
|---|---|
| [contracts.py](../src/mes_vision/inspection/contracts.py) | 좌표·모델 출처·검출·복수 결함·검사·실행 결과 자료형 |
| [geometry.py](../src/mes_vision/inspection/geometry.py) | 원본 해상도 추출, 축소/여백 좌표 복원 |
| [adapters.py](../src/mes_vision/inspection/adapters.py) | 검출기/검사기 인터페이스, 미설정 검사기, 명시적 모의 검사기 |
| [pipeline.py](../src/mes_vision/inspection/pipeline.py) | 다중 물체 순차 검사, 프레임·물체 연결 검증, 결과 통합 |
| [rfdetr_adapter.py](../src/mes_vision/inspection/rfdetr_adapter.py) | RF-DETR Small 실제 추론 연결, 별도 결함 모델 연결 준비 |
| [demo_inspection.py](../scripts/demo_inspection.py) | 모의 물체 3개의 원본·추출·영역 표시·JSON 예제 |
| [verify_inspection.py](../scripts/verify_inspection.py) | 입력 회귀 시험 + 검사 연결 시험, 선택적 GPU 확인 |

## 픽셀 좌표와 추출

- 모든 검출기는 **현재 입력 RGB 원본 크기의 xyxy 픽셀 좌표**를 반환해야 한다. 좌표 범위는 왼쪽/위 포함, 오른쪽/아래 제외다. 정규화 값이나 mm는 받지 않는다.
- RF-DETR 1.9.4의 `predict`는 내부 후처리에서 원본 크기를 반영한다. 어댑터는 그 좌표를 다시 확대하지 않는다. 설치된 공식 패키지의 `detr.py` 원본 크기 후처리 경로를 확인했다.
- 다른 모델이 축소·여백 영상 좌표를 반환하면 `ResizeMap`으로 실제 x/y 축소 비율과 왼쪽/위 여백을 역변환한다. 이 도구는 RF-DETR에 중복 적용하지 않는다.
- 물체 추출은 원본 RGB 배열에서 직접 수행한다. 소수점 상자는 시작을 내림, 끝을 올림하여 포함하고, 선택한 정수 픽셀 여백을 더한 뒤 이미지 경계에서 자른다. 모델 입력용 축소 영상에서 잘라오지 않는다.
- 추출 이미지→원본 이동 행렬을 저장한다. 결함 영역은 추출 이미지 좌표로 받고, 같은 물체의 원본 좌표로 변환하여 함께 기록한다. 추출 영역 밖의 결함 좌표나 잘못된 원본 좌표는 해당 검사 오류다.
- 이미지 밖에 있는 검출은 제외 사실을 실행 오류 목록에 남긴다. 일부만 걸치는 검출, 이미지 가장자리에 닿은 물체, 검출 상자 겹침, 추출 여백에 다른 검출이 들어오는 경우도 표시한다.
- 회전 정렬·마스크·배경 제거는 아직 하지 않는다(`alignment=NOT_APPLIED`). 추출 상자 중심을 집기점으로 사용하지 않으며 집기점·보정 버전은 null이다.

## 검사 순서와 오류

각 물체마다 등록된 검사기를 순차 호출한다. `known_defects`, `anomaly`, `geometry`, `vlm` 중 미등록 항목에는 `NOT_RUN` 검사기를 자동으로 넣는다. 실제 이상 탐지와 VLM 구현은 각각 08·09 단계에서 연결한다.

- 알려진 불량 검사가 PASS/FAIL/ERROR여도 다음 물체로 건너뛰지 않고 그 물체의 이상 탐지를 호출한다. 한 검사기의 오류가 다른 검사의 근거를 지우지 않는다.
- 검사마다 현재 프레임 ID·물체 ID·검사기 ID·모델 출처를 대조한다. 이전 결과나 다른 물체 결과가 돌아오면 ERROR로 바꾼다.
- 검출기 실패·잘못된 프레임/해상도/좌표계는 실행 ERROR다. 검출 개수 상한을 넘으면 몰래 일부를 버리지 않고 실행을 멈춘다.
- 검출 0개는 빈 작업대나 OK를 뜻하지 않는다. 예상 수량은 미설정이면 null이고, 설정했다면 유효한 검출 물체 수와 비교한다. 수량 일치만으로 누락이 없다고 보장하지 않는다.
- PASS/FAIL은 기준 버전이 있어야 한다. 모의 기준 버전은 실제 품목 기준으로 사용하지 않는다. 필수 검사 설정과 최종 판정 우선순위는 10 단계에서 완성한다.
- 촬영 품질·작업 영역 검증은 아직 미설정으로 기록한다. 제품 ID를 지정하지 않으면 null과 미설정 사유가 남는다. `COMPLETED_WITH_ISSUES`는 연결 코드 실행이 끝났다는 의미이며 검사 합격이 아니다.
- 엔진은 동기식·단일 작업용이다. UI 작업 스레드, 시간 제한·취소·복구는 후속 단계다. 현재는 잘라낸 이미지를 하나씩 처리하고 보고서에는 픽셀 배열 대신 메타데이터만 남긴다.

## 결과 형식 v1

| 묶음 | 주요 필드 |
|---|---|
| 실행 | schema_version, run_id, mode, execution_status, elapsed_ms |
| 입력 | frame ID·출처·해상도·시간 정보. 04의 입력 계약 유지 |
| 설정 | product_id, crop_margin_px, expected_count, max_objects, detector_candidate_threshold |
| 검출 출처 | 모델 이름·버전·가중치 SHA256·학습 용도, 원본 상자·점수·클래스 |
| 물체 | 실행별 고유 object_id, 경계 보정 상자, 추출 범위·이동 행렬·해상도 |
| 검사 | 검사 ID·대상 ID, PASS/FAIL/UNCERTAIN/ERROR/NOT_RUN, 모델 출처, 기준 버전, 시간·사유 |
| 근거 | 복수 findings: 불량 코드·설명·점수·추출 영역·원본 영역. 위치 없는 누락 등의 근거도 허용 |
| 이상 점수 | raw_score는 불량 확률과 구분. 검사 점수의 의미는 해당 어댑터가 정의 |
| 판정·로봇 | final_decision=null, decision_status=NOT_IMPLEMENTED, robot_commands_enabled=false |

등록 불량 코드는 NG01~NG06, NG_UNKNOWN이다. 신규 의심은 코드 null의 설명으로 남길 수 있으며 자동으로 새 코드를 생성하지 않는다. NG_UNKNOWN 확정 규칙도 후속 판정 정책에서 검증해야 한다.

결과 예제: [모의 JSON](../artifacts/inspection-check/demo-v2/result.json), [전체 영역 표시](../artifacts/inspection-check/demo-v2/overlay.png), [물체 2 원본 추출](../artifacts/inspection-check/demo-v2/object-2.png).

## RF-DETR 연결 범위

`RFDETRBackend`에 로컬 가중치 경로, 정확한 SHA256, 후보 검출 임계값, 학습 용도와 장치를 지정하고 `load()`한 뒤 사용한다. 해시 불일치나 누락 파일은 모델을 읽지 못하게 한다. 이 모듈은 모델을 다운로드하는 기능을 제공하지 않는다.

- `RFDETRDetector`: 일반 COCO 사전학습 또는 제품 위치용 모델을 전체 프레임에 적용한다. 일반 모델이면 미학습 경고가 결과에 남는다.
- `RFDETRDefectInspector`: 별도 제품 결함 학습 모델과 명시적인 클래스 ID→NG 코드 매핑을 요구한다. COCO 일반 모델을 NG 모델로 바꿔 부를 수 없다. 미등록 클래스는 검사 오류다.
- 결함 어댑터는 후보 수집만 수행하며 판정 기준 미검증으로 UNCERTAIN을 반환한다. 후보와 함께 클래스→불량 코드 매핑을 기록한다. 10 판정 엔진이 명시적으로 검증된 모델·코드 매핑·수집 및 확정 임계값을 대조한 경우에만 별도 평가 상태를 낸다. 원시 검사를 PASS로 고쳐 쓰지 않는다. 실제 결함 모델은 아직 없으므로 이 어댑터의 클래스 매핑·복수 후보 경로는 모의 백엔드로 시험했다.
- 실제 라이브 카메라 모드는 아직 받지 않는다. `model_file` 모드에서는 모의 검사기 사용을 거부한다. 모의 검사를 섞으려면 명시적으로 `simulation` 모드를 사용해야 한다.

## 실행 및 검증

프로젝트 폴더의 PowerShell에서 실행한다.

```powershell
# 입력+검사 연결의 파일/모의 검증
.\.venv\Scripts\python.exe scripts\verify_inspection.py

# 위 검증에 실제 RF-DETR CUDA 연결 확인 추가
.\.venv\Scripts\python.exe scripts\verify_inspection.py --gpu

# 새 출력 폴더에 모의 예제 생성
.\.venv\Scripts\python.exe scripts\demo_inspection.py --output artifacts\my-inspection-demo
```

확인 결과:

- 입력 26개 + 검사 연결 24개 = 50개 시험 통과. [기능 보고서](../artifacts/inspection-check/report.json)
- RTX 3090에서 공식 가중치 SHA256 확인 후 실제 RF-DETR 어댑터 실행. 원본 1280×720, 후보 임계값 0에서 300개, 시험 임계값 0.5에서 0개. 이는 합성 도형에 대한 실행 확인이며 제품 검출·불량 정확도 평가가 아니다. [GPU 보고서](../artifacts/inspection-check/gpu.json)
- 모의 예제: 물체 1은 일부 모의 검사 PASS, 물체 2는 균열+표면 결함, 물체 3은 알려진 불량 검사 ERROR+이상 의심 UNCERTAIN. 세 물체 모두 이상 검사 호출. 미연결 형상 검사·VLM은 NOT_RUN, 최종 판정은 null.
- 실제 D405·Dobot·노트북·실물 학습·불량 정확도는 검증하지 않았다.

06 학습 프로그램도 이후 완료했다. [학습 사용법](TRAINING.md)을 참고한다. 학습 가중치는 model.json의 정확한 순서로 class_names를 넘겨 읽으며, 클래스 이름·헤드 크기를 대조하고 추가 예약 슬롯을 제외한다. 제외 개수는 결과에 기록한다.

07 [데이터 관리·라벨링](DATA_MANAGEMENT.md), 08 [이상 탐지](ANOMALY.md)도 완료했다. AnomalyInspector를 등록하면 모든 물체의 원본 crop에 실제 DINOv2 기반 이상 점수·영역을 계산한다. 정상 기준 품목과 pipeline.product_id가 일치해야 한다. CheckResult.details는 추가적인 JSON 메타데이터 필드이며, 기준 ID/해시·임계값·지도 형태·선택적 증거 파일 경로를 보존한다. 판정 정책과 로봇 명령은 아직 활성화하지 않는다.

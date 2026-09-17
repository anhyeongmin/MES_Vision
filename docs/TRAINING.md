# 06 학습 프로그램

갱신일: 2026-09-05

로컬 RF-DETR Small의 데이터 사전 검사, 학습 중 검증, 가중치 선택·저장, epoch 단위 정지·재개, 별도 평가, 검사 엔진 재사용을 구현했다. 물체 위치 모델과 알려진 불량 모델은 별도 데이터·별도 가중치를 사용한다.

실제 물품 학습은 아직 하지 않았다. 이번 실행은 합성 데이터로 프로그램 동작을 확인한 것이며, 시험 가중치는 제품 모델 등록부에 활성화하지 않았다.

## 구현 파일과 설정

- [학습 실행 도구](../scripts/training.py): validate / train / stop / evaluate / make-fixture
- [학습 실행부](../src/mes_vision/training/jobs.py), [데이터 검사](../src/mes_vision/training/data.py), [설정](../src/mes_vision/training/config.py)
- [물체용 설정](../configs/training/objects.json), [불량용 설정](../configs/training/defects.json)
- [전체 검증 도구](../scripts/verify_training.py), [검증 보고서](../artifacts/training-check/report.json)

실물용 설정 파일은 준비했지만 데이터 폴더가 없으므로 지금 실물 학습을 실행하면 사전 검사에서 멈춘다. 임의의 데이터나 합격 기준을 자동 생성하지 않는다.

기본 설정은 RF-DETR Small 512×512, GPU 1개, BF16, batch 2, 누적 4회, 100 epochs, workers 0, seed 42다. 학습률 등은 설정 파일에서 바꿀 수 있다. 이는 시작 설정이며 최적 성능을 검증한 값은 아니다. 현재 EMA·다중 해상도 학습은 끄고 고정된 torchvision 증강 경로를 사용한다. 외부 학습 서비스·유료 API를 호출하지 않는다.

## 데이터 형식

각 데이터 폴더에는 아래 구조가 필요하다. JSON은 UTF-8이다.

```text
datasets/object_detector/           # 또는 known_defect_detector/
  dataset.json
  train/
    _annotations.coco.json
    image-001.png
  valid/
    _annotations.coco.json
    image-101.png
  test/
    _annotations.coco.json
    image-201.png
```

`dataset.json` 예시:

```json
{
  "schema_version": 1,
  "role": "object_detector",
  "kind": "real",
  "product_id": "사용자가 정한 품목 ID"
}
```

불량 데이터는 role을 `known_defect_detector`로 지정한다. 시험 데이터는 kind=`synthetic`이고, 학습 설정에서도 synthetic=true여야 한다. 선언이 다르면 실행하지 않는다.

COCO `images` 항목에는 원본 크기·파일명 외에 실물 식별자와 촬영 회차를 넣는다.

```json
{
  "id": 1,
  "file_name": "image-001.png",
  "width": 1280,
  "height": 720,
  "specimen_ids": ["실물-001", "실물-002"],
  "capture_session_id": "촬영회차-001"
}
```

- `specimen_ids`: 사진에 등장하는 실제 개체 ID들. 같은 실물을 회전·재촬영하거나 여러 이미지로 잘라도 같은 ID를 사용한다. 물체가 없는 배경 이미지는 빈 목록을 허용한다. 불량 검사에 쓰는 정상 물체 이미지에는 실물 ID가 필요하다.
- `capture_session_id`: 같은 촬영 회차를 나타내는 ID다. 촬영 조건과 분리 정책은 [07 라벨링 지침](LABELING_GUIDE.md)을 따른다. 검사는 이 ID가 올바르게 기입됐다는 전제에서 수행되며, 숨겨진 동일 실물을 사진만으로 식별하지는 않는다.
- 같은 실물 ID, 촬영 회차, 동일한 RGB 픽셀이 train/valid/test 사이에 겹치면 거부한다. 파일 이름만 바꾼 복사본도 픽셀 비교로 확인한다.
- 8비트 불투명 RGB PNG/JPEG를 받는다. EXIF 회전이 있는 사진은 이미지와 라벨 좌표를 함께 정규화한 뒤 넣어야 한다. 라벨을 유지한 채 사진만 자동 회전하지 않는다.

COCO `annotations`는 image_id, category_id, bbox=[x,y,width,height], area=width×height, iscrowd=0 형식이다. 원본 픽셀 좌표를 사용하며 이미지 밖 영역·0 크기·NaN·중복 ID·없는 이미지/클래스 참조를 거부한다.

`categories`의 ID와 이름은 세 분할에서 같아야 한다. 모델의 label_index는 category_id를 정렬한 0부터 시작하는 번호이며, `model.json`에 매핑을 기록한다. 각 분할에 등록 클래스별 라벨이 최소 1개는 있어야 한다. 이 조건은 실행에 필요한 최소 커버리지이며 충분한 학습·평가 표본 수가 아니다.

## 두 모델의 라벨 차이

| 역할 | 학습 이미지와 라벨 |
|---|---|
| 물체 위치 | 작업대 전체 사진. 정상·불량 모두 물체 전체를 같은 품목 클래스로 라벨링할 수 있음. 여러 물체는 여러 상자 |
| 알려진 불량 | 물체별 원본 추출 이미지. 클래스 이름은 NG01~NG06. 한 이미지에 균열과 표면 결함 등 여러 상자 허용 |

불량 모델에 OK 클래스를 만들지 않는다. 정상 이미지에는 불량 annotation이 0개다. 불량 검출 결과가 0개라는 사실이 최종 OK 판정은 아니다. 이상 탐지와 형상 검사도 필요하다.

누락·구멍 불량처럼 상자만으로 검사 기준을 충분히 표현하지 못하는 항목은 기대 위치·개수 등의 검사도 사용해야 한다. 이를 위해 임의의 불량 상자를 만들지 않는다. [07 라벨링 도구](DATA_MANAGEMENT.md)에서 영역 없는 관찰로 기록할 수 있으며, 이런 물체는 불량 상자 학습에서 제외한다. 미등록·불확실은 challenge로 보관한다.

07 도구가 내보낸 `datasets/object_detector`, `datasets/known_defect_detector`는 기본 학습 설정과 연결된다. 아래 학습 명령을 실행하기 전에 해당 폴더의 실물·분할·검토가 끝나 있어야 한다. 다른 버전 폴더로 내보내면 설정의 `dataset_dir`도 수정한다.

## 실행 방법

프로젝트 폴더의 PowerShell에서 실행한다. 데이터·초기 가중치 경로는 설정 JSON의 위치를 기준으로 해석한다. 출력 폴더는 항상 새 이름을 사용한다.

```powershell
# 실물 데이터 준비 후 형식·분할·초기 가중치 확인
.\.venv\Scripts\python.exe scripts\training.py validate --config configs\training\objects.json

# 물체 위치 학습
.\.venv\Scripts\python.exe scripts\training.py train --config configs\training\objects.json --output runs\objects-001

# 별도 불량 모델 학습
.\.venv\Scripts\python.exe scripts\training.py train --config configs\training\defects.json --output runs\defects-001

# 다른 터미널에서 현재 epoch 완료·저장 후 정지 요청
.\.venv\Scripts\python.exe scripts\training.py stop --run runs\objects-001

# 원래 실행을 보존하고 새 실행에서 재개
.\.venv\Scripts\python.exe scripts\training.py train --config configs\training\objects.json --resume-from runs\objects-001 --output runs\objects-002

# 선택된 가중치의 별도 test 평가
.\.venv\Scripts\python.exe scripts\training.py evaluate --run runs\objects-002 --split test --output runs\objects-002-test
```

`--stop-after-epoch 1`을 train에 추가하면 1번째 epoch 후 정지한다. 정지는 즉시 취소가 아닌 epoch 경계의 저장 요청이다.

재개 시 설정의 epochs는 추가 횟수가 아니라 **총 목표 epoch 수**다. 완료된 epoch보다 커야 한다. 데이터 해시·클래스·역할·seed·batch·학습률·정밀도 등이 바뀌면 재개를 거부한다. 데이터 위치와 초기 가중치 위치는 옮길 수 있지만 내용 해시는 같아야 한다. 목표 epochs 변경은 허용하며 원래 계획을 연장한 것으로 기록한다.

복원 대상은 모델, optimizer 상태·누적값, scheduler, global step과 완료 epoch다. 이번 시험에서는 누적값 일부와 scheduler 진행값까지 대조했다. RNG·데이터 로더의 중간 위치까지 동일한 비트 단위 재현을 보장하지 않으며, 중간에 끊긴 epoch의 일부 배치를 다시 처리할 수 있다.

Ctrl+C나 잡힌 예외가 발생하면 INTERRUPTED/FAILED와 마지막 완료 epoch의 체크포인트를 기록한다. 운영체제 강제 종료·전원 차단처럼 finally가 실행되지 않는 상황은 RUNNING 표시가 남을 수 있다. 이러한 비정상 종료의 상태 복구는 15 단계에서 완성한다.

## 저장 결과와 평가 기준

| 파일 | 내용 |
|---|---|
| run.json | 실행 상태·설정·부모 실행·데이터 지문·선택 기준·파일 해시·GPU 확인 |
| dataset-snapshot.json | 파일/픽셀 SHA256, 클래스 매핑, 분할별 이미지·라벨 수 |
| progress.json | 최근 완료 epoch, global step, 학습 장치 |
| engine/metrics.csv | 학습·검증 손실과 검출 지표 |
| engine/last.ckpt | 모델·optimizer·scheduler를 포함한 재개용 전체 체크포인트 |
| inference.pth | validation mAP로 선택한 검사 실행용 가중치. 재개용 optimizer 정보 없음 |
| model.json | 모델 역할·클래스 매핑·synthetic 여부·가중치 해시. 배포 승인 false, 판정 임계값 null |
| evaluation.json | 별도 평가 폴더에 저장되는 valid/test 결과, 입력·가중치 지문 |

실행 상태는 VALIDATING → RUNNING → COMPLETED/STOPPED/INTERRUPTED/FAILED다. 저장은 원래 실행을 덮어쓰지 않는다. 평가 중에는 원래 학습 파일과 가중치를 수정하지 않는다.

가중치는 validation의 `mAP 50:95`로 고른다. 재개 전 실행의 최선 점수가 더 높으면 그 가중치를 보존한다. test 분할은 학습 종료 시 자동 평가하거나 모델 선택에 사용하지 않고, evaluate 명령에서만 읽는다. 데이터 사전 검사에서는 test 파일의 무결성과 형식도 확인한다.

평가 지표는 검출 mAP·클래스별 AP 등이다. 라이브러리의 F1 값은 해당 데이터에서 임계값을 훑은 결과이므로, 고정된 제품 판정 임계값의 성능으로 해석하면 안 된다. 최종 OK/NG 오판·누락·미등록 불량·집기 성공률 검증은 별도로 진행한다.

검사 엔진에서 학습된 가중치를 읽을 때는 `model.json`의 순서대로 class_names를 넘긴다. 클래스 이름·개수와 가중치 헤드가 다르면 거부한다. RF-DETR의 추가 예약 슬롯은 제품 클래스로 취급하지 않고 제외 개수를 기록한다. COCO 일반 모델의 실제 클래스 90에는 이 규칙을 적용하지 않는다. 이 보완은 저장 가중치의 실제 재사용 시험에서 확인한 문제를 수정한 것이다.

## 장비 없이 확인한 결과

- 입력·검사 연결·데이터 검사 총 **69개 시험 통과**.
- RTX 3090에서 물체 모델 FP32 학습: 첫 실행 5 optimizer updates → 저장 → 5 step·486개 optimizer 상태 복원 → 누적 10 step까지 학습.
- optimizer 누적값 일부와 scheduler 진행값이 저장값과 일치하는지 확인.
- 별도 불량 모델 BF16 학습·저장 완료. 합성 균열·표면 결함 두 클래스와 정상 음성 이미지를 사용.
- 저장 가중치로 별도 test/valid 평가 완료. 물체 모델·불량 모델을 05의 실제 검사 어댑터에서 다시 읽어 클래스·출력 연결 확인.
- 아주 작은 데이터에는 RF-DETR의 최소 길이 sampler가 이미지를 반복하므로 이미지 2개와 학습 update 수가 같지 않다.
- 실물 검사 정확도는 확인하지 않았고, 합성 데이터의 높은 점수를 제품 정확도로 사용하지 않는다.

검증을 다시 실행하려면:

```powershell
# 입력·검사·학습 데이터 관련 회귀 시험
.\.venv\Scripts\python.exe scripts\verify_training.py

# 새 시험 폴더에서 실제 GPU 학습→정지→재개→평가→가중치 재사용까지
.\.venv\Scripts\python.exe scripts\verify_training.py --gpu
```

GPU 검증은 여러 학습 파일을 보존하므로 실행마다 수 GB의 저장 공간을 사용한다. 현재 [통합 보고서](../artifacts/training-check/report.json)의 gpu.path가 검증한 실행 폴더다. 초기 가중치나 제품 모델 등록부는 이 시험으로 교체하지 않는다.

다음은 07 데이터 관리·라벨링 준비다. 실제 데이터 수집표, 촬영 회차·실물 ID 운영, 복수 불량 라벨링 규칙과 관리 도구를 준비한다.

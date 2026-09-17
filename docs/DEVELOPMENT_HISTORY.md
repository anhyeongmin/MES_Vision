# MES Vision — 상면 검사 및 자동 분류

**새 학습 모델 연결:** `2 품목 설정 → 저장사진 검사`에서 ASH 전체/상세 물체 모델과 새 v3 불량 모델을 사용합니다. 저장사진 144장으로 실제 추론·원본 좌표 표시를 확인했습니다. 실제 검사 화면에도 불량 후보 표시선을 추가했습니다. 실물 기준이 없는 최종 판정은 보류입니다. [사용 안내](docs/SAVED_PHOTO_INSPECTION.md).

**U20CAM 입력:** USB 카메라 선택, 해상도·MJPEG/YUY2 후보, 미리보기와 전체/상세 촬영·저장 경로를 추가했습니다. 실제 카메라 연결·촬영 지연·렌즈/좌표 보정은 장비 준비 후 확인합니다. [사용법과 구현 범위](docs/U20CAM_INPUT.md).

**성능 개선 4:** Qwen 영상 패치 투영의 실행 방식을 바꿨습니다. 3090 합성 입력 비교에서 응답 완료 중앙값 11.58→3.84초. 같은 가중치를 사용하지만 생성 문장은 달라질 수 있으며 기본 검사 판정은 보존합니다. [측정·적용 범위](docs/VLM_OPTIMIZATION.md).

**성능 개선 3:** 3090 예제 최종 측정에서 4개 152→114ms, 8개 283→208ms(촬영·저장·VLM 제외), 전체 402개 시험 통과. 정상 기준을 GPU에 한 번 올려 유지하고, DINO 특징을 GPU에서 바로 비교합니다. 판정 기준과 거리 계산은 유지하며 품목 모델 해제 시 기준도 해제합니다. [적용 범위와 검증](docs/RESIDENT_ANOMALY.md).

**성능 개선 2:** 현재 물체를 최대 4개씩 검사합니다. 새 물체와 재검사 대상만 RF-DETR·DINO 묶음 처리에 전달하고, 결과의 ID/좌표를 확인합니다. 3090 최종 비교에서 4개 검사 205→141ms, 8개 398→266ms(예제 입력, 촬영·저장·VLM 제외). 전체 393개 시험을 통과했습니다. [적용 범위와 검증](docs/INSPECTION_BATCHES.md).

**성능 개선 1:** RF-DETR CUDA 추론에 FP32 JIT를 적용했습니다. 3090 예제 입력 비교에서 검출 호출 중앙값 34.68→21.90ms. 실물 불량 정확도는 별도 검증 대상이며 실행 방식이 다른 기존 판정 기준은 자동 재사용하지 않습니다. [측정과 적용 범위](docs/INFERENCE_OPTIMIZATION.md).

**환경설정:** 오른쪽 위 버튼으로 한국어/영어/중국어 간체/태국어, 글자 크기, 시작 화면, 창 배치를 설정하고 저장 위치와 진단 내보내기를 확인할 수 있습니다. 언어는 저장하면 재실행 없이 바로 적용됩니다. 검사와 장치 연결은 유지됩니다. [사용법](docs/PREFERENCES.md).

17단계 시험 도구: 전체 장면 재생 평가·누락/오판/보류 집계, 연속 추적/저장 시험, RTX 3090 실제 모델과 VLM 동시 부하 측정, 현장 시험표를 추가했습니다. [사용법](docs/VALIDATION.md) · [실측 결과](docs/VALIDATION_RESULTS.md). 실제 장비 검증 후 16단계 설치 배포를 진행합니다.

15단계 오류 처리·복구: 카메라 프로세스 격리, 검사 진행 감시, 정지 대기 명령 폐기, 오류 차단과 재시작 복구, 확인자/조치 기록, 진단 내보내기를 추가했습니다. 상단 **오류 확인 · 재개 준비**에서 사용합니다. [오류 처리 사용법](docs/RECOVERY.md).

14단계 저장·조회 확장: 조건 검색, 전체 CSV·JSON·원본 내보내기, 보존 보호와 검증된 보관 이동, 중단 복구, 자산 포함 백업·새 폴더 복원을 추가했습니다. **4 검사 이력 → 저장 · 백업 관리**에서 사용합니다. [저장·백업 사용법](docs/STORAGE.md).

2026-09-06: 14단계에 앞서 승인한 운영 수정 8가지를 반영했습니다. 실시간 D405 입력, 물체 추적·연속 검사, 버전별 품목 설정, 실제 Magician 통신, 원본·검사·로봇 이력 연결, 선택형 VLM 분석을 PySide6 운영 화면에서 사용합니다. 제품 학습 데이터·현장 보정은 아직 미등록입니다.

**화면 실행:** 프로젝트의 `MES Vision.cmd`를 더블클릭하세요. 운영 UI에는 데모·합성 예제·모의 집기 메뉴가 없습니다. [운영 화면 사용법](docs/OPERATOR_REVISION.md). 기본 저장 위치는 `artifacts/operation`이며 이전 개발 확인 자료와 구분됩니다. 설치 패키지와 현장 출고 검증은 후속 단계입니다.

- [시스템 설계](docs/SYSTEM_ARCHITECTURE.md): 모듈 역할, 검사 흐름, 판정·로봇 운전 규칙
- [개발 진행표](docs/DEVELOPMENT_PLAN.md): 장비 없이 진행할 17개 작업과 현장 의존 작업
- [모델·라이선스 선정 결과](docs/MODEL_AND_LICENSE_REVIEW.md): 확정 모델·UI 버전, 확보 파일, 배포 의무
- [자산 검증 결과](artifacts/selected-assets-check.json): 고정된 파일의 해시 대조 결과; 모델 실행 시험은 아님
- [개발 환경 결과](docs/DEVELOPMENT_ENVIRONMENT.md): GPU 추론·합성 학습·Windows UI 실행, 설치·재검증 방법
- [이미지 입력 사용법](docs/IMAGE_INPUT.md): 사진·폴더·영상 읽기, 상태 표시, D405 미설정 경계
- [학습 프로그램 사용법](docs/TRAINING.md): 데이터 검사·학습·정지·재개·평가·가중치 재사용
- [데이터 관리·라벨링](docs/DATA_MANAGEMENT.md): 로컬 데스크톱 라벨링, 검토 이력, 실물·회차 분할, 네 종류 내보내기
- [촬영·라벨링 지침](docs/LABELING_GUIDE.md): 실물 도착 후 수집·검토 기준 및 준비 양식
- [데이터 관리 검증](artifacts/data-management-check/report.json): 89개 기능·회귀 시험, 네 종류 내보내기, Qt 화면 조작 15개 점검
- [정상 등록·이상 탐지](docs/ANOMALY.md): DINOv2 정상 특징 저장·재불러오기, 거리 지도·영역, 미설정/미검증 기준 보류
- [이상 탐지 검증](artifacts/anomaly-check/report.json): 111개 기능·회귀 시험 및 실제 RTX 3090 등록·추론·두 물체 연결
- [VLM 후속 분석](docs/VLM.md): 기본 NG 근거와 별도 설명, ON/OFF·중단·복구·GPU 우선순위·실행 방법
- [VLM 검증](artifacts/vlm-check/report.json): 133개 시험, Qt 조작 12개, 실제 Qwen 응답·RF-DETR 우선 실행; 실물 정확도는 후속 검증
- [OK·NG·판정 보류](docs/DECISION.md): 필수 검사·모델/기준 연결, 복수 불량·오류 보존, 미설정 보류, 모의 실행과 저장 재평가
- [판정 검증](artifacts/decision-check/report.json): 163개 시험, 판정 Qt 조작 14개·VLM Qt 조작 12개, 원본 보존·재평가 일치
- [Dobot 제어 구조·모의 동작](docs/ROBOT.md): 단일 물체 계획, 집기/놓기 확인, 재촬영, 중단·복구·실물 차단
- [로봇 모의 검증](artifacts/robot-check/report.json): 187개 시험, CLI 시나리오 9개, 강제 종료 후 명령 재전송 차단; 실제 장비 시험은 아님
- [픽셀·로봇 좌표 보정](docs/CALIBRATION.md): 독립 확인점·오차·영역 검사, 렌즈 처리·높이/설정 연결, PySide6 편집·XY 확인
- [좌표 보정 검증](artifacts/calibration-check/report.json): 209개 시험, Qt 조작 10개, 저장 변환 일치와 모의 로봇 연결; 실물 정밀도는 미검증
- [학습 검증 결과](artifacts/training-check/report.json): 69개 기능/회귀 시험과 실제 GPU 학습 검증
- [검출·검사 연결 사용법](docs/INSPECTION_PIPELINE.md): 다중 물체 추출·복수 결함·오류·모의 결과
- [입력 검증 결과](artifacts/input-check/report.json): 실제 생성 파일을 이용한 기능·오류 처리 시험
- [환경 실행 보고서](artifacts/development-check/summary.json): 모델과 UI별 실제 확인 결과

아래는 앞서 완료한 RF-DETR 모델 준비 및 실행 확인 기록입니다.
모든 검사 기능이나 현장 성능 검증이 완료되었다는 의미는 아닙니다.

RF-DETR Small 다운로드와 Windows / RTX 3090 GPU 추론 확인을 완료했습니다.
프로젝트 위치: `C:\Users\alexi\Desktop\Workspace\MES_Vision`

## 확인한 환경

- Windows 11, Python 3.12.10
- NVIDIA GeForce RTX 3090, GPU 메모리 24GB, 드라이버 591.86
- PyTorch 2.9.1+cu128 / torchvision 0.24.1+cu128
- RF-DETR 1.9.4, Small Detection, 입력 해상도 512×512
- 패키지 버전 전체: `requirements-lock.txt`

RF-DETR은 1.10.0 공개 직후의 변경을 바로 도입하는 대신 1.9.4 패치 버전으로 고정했습니다.
CUDA 12.8 배포본은 이 PC에서 실행 확인했으며, 포함된 GPU 아키텍처 목록은
`artifacts/environment-check.json`에 기록했습니다. RTX 5070 노트북의 실제 실행은 아직 확인하지 않았습니다.
기존 시스템 Python에는 패키지를 설치하지 않고 `.venv`에 설치했습니다.

## 완료한 확인

- 공식 가중치 386,045,550바이트 다운로드
- 공식 패키지에 기재된 MD5와 일치 확인, SHA-256 별도 기록
- 모델의 실제 CUDA 배치 및 GPU 추론 확인
- 합성 도형 이미지 1280×720 입력, 모델 입력은 512×512
- 워밍업 2회 후 5회 실행, 출력 수치가 유한한지 확인
- 첫 성공 실행 중앙값 약 36.5ms (최적화 전, 단일 이미지 predict 호출)
- `pip check` 통과

이 수치는 테스트 이미지로 실행 경로를 확인한 값이며 제품 처리량 보장이 아닙니다.
현재 가중치는 COCO 일반 물체용입니다. **실제 제품의 검출·불량 검사 학습, 결함 정확도 검증, 카메라 연결,
좌표 보정, Dobot Magician 제어는 아직 진행하지 않았습니다.**
학습 의존성과 합성 데이터 실행 검증은 완료했습니다. 실제 데이터용 학습 프로그램은 06 단계에서 구성합니다.

## 다시 확인하기

프로젝트 폴더의 PowerShell에서 실행합니다. 가상환경 활성화는 필요 없습니다.

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py
```

본인 사진으로 추론 실행만 확인하려면:

```powershell
.\.venv\Scripts\python.exe scripts\verify_environment.py --image 'D:\photos\sample.jpg'
```

검증 보고서는 `artifacts/environment-check.json`에 갱신됩니다.
이 스크립트는 가중치를 변경하거나 모델을 학습하지 않습니다.

## 새 PC에서 설치하기

Python 3.12와 해당 GPU를 지원하는 NVIDIA 드라이버가 필요합니다.
`.venv` 폴더는 복사하지 말고 새 PC에서 다시 만드세요.
프로젝트 파일, src, tests, scripts, configs, requirements 파일, models와 licenses 폴더를 복사한 뒤:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\setup.ps1
```

이 명령의 실행 정책 설정은 해당 PowerShell 프로세스에만 적용됩니다.
setup은 CUDA PyTorch → 고정 의존성 → 패키지 검사 → 모든 선정 자산 검증 → RF-DETR·DINOv2·Qwen·합성 학습·Windows UI·이미지 입력·검사 연결·학습 데이터 검사 확인을 수행합니다. 학습 프로그램의 전체 GPU 시험은 별도 verify_training.py --gpu로 실행합니다.
패키지 설치에는 인터넷이 필요합니다. 모델 파일이 있으면 해시 확인 후 재사용합니다.

## 주요 파일

- `models/rf-detr-small.pth`: 공식 사전학습 모델
- `models/manifest.json`: 공식 URL, 버전, 파일 크기, MD5, SHA-256
- `scripts/prepare_model.py`: 공식 모델 및 라이선스 다운로드·검증
- `scripts/verify_environment.py`: 실제 GPU 추론 확인
- `scripts/setup.ps1`: 다른 Windows PC에 동일 환경 설치
- `scripts/snapshot_environment.py`: 버전 고정 및 라이선스 고지 수집
- `artifacts/environment-check.json`: 실제 실행 결과
- `licenses/RF-DETR-LICENSE.txt`: Apache-2.0 원문
- `licenses/RF-DETR-README.md`: 선택한 버전의 공식 안내 사본
- `licenses/dependency-inventory.json`: 설치 패키지의 라이선스 메타데이터
- `licenses/dependencies/`: 패키지가 제공한 배포 고지 사본

## 상업용 사용 범위

선택한 RF-DETR Small Detection 모델과 rfdetr 코드는 공식 자료상 Apache-2.0 대상입니다.
Plus, XL/2XL Detection 모델 또는 별도 클라우드 서비스를 도입하지 않았습니다.
사용자 제품 전체의 라이선스를 Apache-2.0으로 지정한 것은 아닙니다.

상업 배포 시 Apache-2.0의 라이선스·저작권·NOTICE 보존 및 수정 표시 조건을 적용해야 합니다.
설치 의존성에는 각각 별도 라이선스와 바이너리 구성요소가 있으므로, 수집된 목록만으로
전체 제품의 상업 배포 검토가 끝난 것으로 간주하면 안 됩니다. 실제 배포 구성에 맞춰 검토해야 합니다.

공식 출처:
- https://github.com/roboflow/rf-detr/tree/1.9.4
- https://blog.roboflow.com/rf-detr-is-free-to-use-commercially/
- https://pytorch.org/get-started/previous-versions/

## 실행 중 참고 메시지

DINOv2 기본 패치 크기와 다르다는 메시지는 공식 RF-DETR Small 전체 가중치를 불러오는
현재 구성에서는 예상된 안내입니다. 가중치 로딩 실패를 의미하지 않습니다.
OpenCV headless는 설치와 이미지 읽기 확인을 마쳤습니다. 추론 최적화 미적용 안내는 남을 수 있습니다. 현재 환경의 실행 확인과 별도로 제품 영상 처리 기능과 성능 최적화는 후속 단계에서 구성합니다.


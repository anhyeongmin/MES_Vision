# 개발 환경 구축 결과

확인일: 2026-09-05 · 대상: Windows 11 / RTX 3090 개발 PC

03 단계는 모델을 로컬 GPU에서 실행하고 학습·UI 개발에 필요한 환경을 준비하는 작업이다. 실제 제품 데이터, D405, Dobot, 노트북에 대한 검증은 포함하지 않는다.

## 확인 결과

| 항목 | 결과 | 범위 |
|---|---|---|
| 모델 파일 | 선정 파일 49개 검증 통과 | 실행 직전에 해당 모델 파일도 다시 해시 확인 |
| DINOv2 Small | CUDA 특징 추출 통과 | 448×448 합성 이미지, 출력 1×1025×384, 유한한 특징값 |
| Qwen3.5-4B | CUDA BF16, 두 이미지 비교 응답 생성 | 원본 모델, 양자화 없음, 430 입력 토큰 → 30 출력 토큰 |
| 모델 동시 적재 | RF-DETR 1개 + DINOv2 + Qwen 적재·순차 실행 통과 | 세 모델 적재 상태에서 각각 실행; 제품용 두 번째 RF-DETR·실시간 동시 요청은 미검증 |
| RF-DETR 학습 | 합성 데이터에서 optimizer 2회 및 검증 배치 실행 | 학습 중 cuda:0과 유한한 비영(非零) gradient 확인; 제품 학습 가중치는 생성하지 않음 |
| PySide6 | Windows 플랫폼의 창 표시·버튼 이벤트·화면 저장 통과 | 일시적인 환경 확인 창이며 제품 UI는 13 단계 |
| OpenCV | 이미지 디코딩·BGR/RGB 변환 확인 | PIL로 읽은 원본 RGB 값과 비교 |
| 의존성 | 패키지 78개 목록·고지 수집, pip check 통과 | 설치 메타데이터 검토; 최종 배포 바이너리 라이선스 검토는 별도 |

실행 결과는 [환경 확인 보고서](../artifacts/development-check/summary.json)와 각 구성의 JSON에 저장했다. [Windows 시험 화면](../artifacts/development-check/desktop-smoke.png)도 남겼다.

학습 종료 시 Lightning이 모델을 CPU로 옮기므로 종료 후의 device만으로 학습 장치를 판단하지 않는다. 보고서의 training_devices는 실제 학습 배치 중에 기록한 값이며, device_after_teardown은 종료 후 상태다.

## 현재 속도와 메모리의 한계

- Qwen 첫 비교 응답 생성은 약 25.19초가 걸렸다. 모델 로딩 약 5.59초와 별도이며, 최적화하지 않은 한 번의 합성 이미지 실행이다. 제품 처리량으로 보장할 수 없다.
- 세 모델을 적재한 시험에서 PyTorch가 기록한 최대 할당량은 약 9052.4MiB였다. GPU 드라이버·다른 프로그램 메모리까지 포함한 전체 VRAM 수치는 아니다.
- Qwen은 Windows에서 사용 가능한 기본 PyTorch/SDPA 실행 경로로 확인했다. 별도 가속 커널·양자화·내보내기 최적화는 적용하지 않았다. 실제 입력 해상도, 출력 길이, 처리 정책을 정한 후 09 및 성능 단계에서 조정한다.
- 노트북에서 같은 BF16 구성이 적합한지는 미확인이다. 실제 GPU 메모리를 확인하고 필요하면 양자화 등의 배포 설정을 별도로 검증한다.
- 학습·검증에 같은 합성 도형을 의도적으로 사용했다. 테스트에서 출력된 mAP 등은 데이터 흐름과 평가 도구의 실행 확인에만 쓰며 정확도 자료로 사용하지 않는다.

## 주요 버전

| 구성 | 버전 |
|---|---|
| Python | 3.12.10 |
| torch / torchvision | 2.9.1+cu128 / 0.24.1+cu128 |
| rfdetr / transformers | 1.9.4 / 5.16.1 |
| accelerate / peft | 1.14.0 / 0.20.0 |
| pytorch-lightning / torchmetrics | 2.6.5 / 1.9.0 |
| faster-coco-eval / pycocotools | 1.8.0 / 2.0.11 |
| opencv-python-headless | 5.0.0.93 |
| PySide6-Essentials / shiboken6 | 6.11.2 / 6.11.2 |

전체 버전은 [requirements-lock.txt](../requirements-lock.txt), 직접 선택한 개발 패키지는 [requirements.txt](../requirements.txt)에 기록했다. 설치 전 기존 잠금 파일은 artifacts/requirements-before-step03.txt에 보존했다. 추가 패키지 19개는 설치 계획의 버전과 SHA-256으로 고정해 설치했으며 artifacts/install-step03.json에 출처를 남겼다.

RF-DETR의 로컬 학습에 필요한 패키지를 직접 설치했다. 클라우드 데이터 다운로드·업로드 패키지가 포함된 rfdetr[train] 전체를 설치하지 않았지만, 실제 로컬 COCO 파일을 이용한 학습·검증 경로를 확인했다. 향후 다른 학습 기능에 추가 패키지가 필요하면 잠금 목록과 라이선스 고지를 함께 갱신한다.

학습의 기본 augmentation은 torchvision으로 고정해 시험했다. 이미지 입출력의 OpenCV는 headless 패키지 하나만 사용하고 창 표시는 PySide6로 처리한다. PyInstaller wheel은 02에서 확보했지만 빌드 도구 설치·제품 패키징은 16 단계에 남아 있다.

## 다시 실행하기

프로젝트 폴더에서 실행한다. 가상환경을 별도로 활성화할 필요는 없다.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_selected_assets.py --verify-only
.\.venv\Scripts\python.exe scripts\verify_development_environment.py
```

전체 확인은 구성별 별도 프로세스로 순차 실행한다. 각 구성에는 600초 제한을 두며, 개별 실패는 JSON과 로그에 기록한다. 모델을 실행하기 전에 고정 파일 해시가 다르면 로딩을 중단한다. 검증 스크립트는 HF 오프라인 모드와 로컬 파일 경로를 사용한다.

실패한 구성만 다시 확인할 수도 있다.

```powershell
.\.venv\Scripts\python.exe scripts\verify_development_environment.py --component qwen
.\.venv\Scripts\python.exe scripts\verify_development_environment.py --component training
.\.venv\Scripts\python.exe scripts\verify_development_environment.py --component ui
```

개별 실행을 시작하면 해당 결과를 running으로 바꾸고, 성공·실패 후 전체 요약을 갱신한다. 이전 성공 결과를 새 실패의 결과처럼 표시하지 않는다. UI 확인 시 시험 창이 잠시 표시된 뒤 닫힌다.

## 새 PC에서 설치하기

Python 3.12와 호환 NVIDIA 드라이버를 준비하고 프로젝트 파일을 복사한다. .venv를 복사하지 말고 새로 만든다.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\setup.ps1
```

설치 순서는 CUDA PyTorch → 전체 잠금 버전 → 의존성 검사 → 고정 자산 확보·검증 → RF-DETR 추론 → 추가 모델·학습·UI 확인 → 이미지 입력·검사 연결 검증이다. 패키지와 없는 모델을 내려받을 때는 인터넷이 필요하다. 실제 설치 파일과 오프라인 설치용 전체 wheel 묶음은 16 단계에서 만든다.

설치만 수행하고 실제 실행 검증을 명시적으로 생략하려면 -SkipRuntimeChecks를 사용할 수 있다. 이 경우 설치 성공을 모델 실행 검증 성공으로 기록하지 않는다. 현재 자동 GPU 확인은 3090 개발 PC에서 검증한 원본 모델 구성이며 노트북 성능 인증은 아니다.

이번 단계에서 설치 스크립트의 구문과 현재 환경의 일관성을 확인했다. 별도의 깨끗한 PC에 재설치하는 시험은 수행하지 않았다.

## 다음 단계

04 이미지 입력 기능은 이후 완료했다. [입력 기능과 사용법](IMAGE_INPUT.md)을 참고한다. 05 검출·검사 연결 코드도 완료했으며 06 학습 프로그램도 완료했으며 현재 다음 단계는 07 데이터 관리·라벨링 준비다. 실제 카메라 연결 확인은 사용자가 장비 준비를 알린 뒤 진행한다.

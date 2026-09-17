# MES Vision

PySide6 기반 Windows 검사 프로그램. U20CAM 촬영, RF-DETR 물체·불량 검출, 비동기 VLM 보조 설명, 학습데이터 수집 및 SAM 라벨링을 포함합니다.

## 처음 실행

대상: **Windows 10/11 x64, Python 3.12 x64, NVIDIA CUDA 지원 GPU**. 개발 확인 장비는 RTX 3090입니다. 노트북 실측은 별도입니다.

GitHub CLI와 Python이 없다면 PowerShell에서 설치한 뒤 터미널을 다시 여세요.

```powershell
winget install GitHub.cli
winget install Python.Python.3.12
```

저장소 접근 권한이 있는 계정으로 로그인하고 클론합니다.

```powershell
gh auth login
gh repo clone anhyeongmin/MES_Vision
cd MES_Vision
.\Setup.cmd
```

`Setup.cmd`는 가상환경 생성 → 고정 버전 의존성 설치 → 비공개 Release에서 모델 다운로드 → SHA-256 검사 → UI 및 CUDA 기본 확인을 수행합니다. 처음 한 번 인터넷 연결이 필요합니다. 모델 약 **5.93 GB**, Python 패키지와 다운로드 임시 공간은 별도입니다. 여유 공간 25 GB 이상을 권장합니다. 다운로드가 끊기면 같은 명령을 다시 실행하세요. 완료된 모델·분할 파일은 해시를 확인하고 재사용합니다.

설치가 끝나면:

- **MES Vision.cmd**: 전체 프로그램 실행
- **MES Vision Manual.cmd**: 로봇 없는 수동 도착 검사

새 PC에서는 전체 프로그램의 장비 설정에서 카메라를 먼저 등록하세요. 카메라 초점·장착·촬영 해상도가 달라지면 렌즈 보정과 로봇 좌표 보정을 다시 해야 합니다. 제공된 보정 파일은 기존 U20CAM의 기록이며, 장치 일치 검사와 좌표계 가드는 유지됩니다. 장치 설정·검사 이력 DB는 복사하지 않습니다. 따라서 새 환경은 빈 품목·장비 설정에서 시작합니다.

## 포함 모델

- 수동 검사 기본 모델: `configs/inspection/ash-wall-reviewed-v2.json`
- 전체/상세 물체 위치 모델 및 이전 불량 비교 모델
- VLM: Qwen3.5 2B + 전체 클래스 어댑터 + NG04/NG05 유지 어댑터
- SAM 2.1 Small: 학습 이미지 라벨링 보조
- DINOv2 Small: 정상 특징 추출

큰 가중치는 Git 이력에 넣지 않고 `runtime-assets.json`의 고정 Release와 해시로 복원합니다. 로그인 정보는 GitHub CLI가 관리하며 소스에 저장하지 않습니다. 이전 4B VLM과 전체 실험 모델은 포함하지 않습니다. 옛 개발 스크립트는 별도 학습 데이터와 실험 산출물을 요구할 수 있습니다.

## 검증 범위

설치 확인은 실행 환경과 파일 무결성 확인입니다. 실물 정확도 또는 로봇 운전 승인이 아닙니다. 현재 모델의 미검증 표시, 보류 판정, 장치/좌표계 안전장치를 보존합니다. VLM 설명은 기본 판정을 변경하지 않습니다.

추가 확인:

```powershell
./.venv/Scripts/python.exe scripts/restore_runtime.py --verify-only
./.venv/Scripts/python.exe scripts/check_clone.py --gpu
```

## 데이터와 개발

영상, 학습 사진, 검사 이력, 기존 가상환경은 제외되어 있습니다. 새로 수집한 데이터는 로컬 `collections/`, 실행 이력은 `artifacts/operation/`에 저장되며 Git에 올라가지 않습니다.

- [UI/백엔드 분담 및 좌표계 계약](docs/UI_HANDOFF.md)
- [라이선스 검토 기록](docs/MODEL_AND_LICENSE_REVIEW.md)
- [이전 개발 기록](docs/DEVELOPMENT_HISTORY.md) — 당시 결과와 제외된 로컬 산출물 링크 포함
- [오픈소스 고지](THIRD_PARTY_NOTICES.md)

이 저장소는 비공개 프로젝트 백업입니다. 자체 코드에 임의의 공개 라이선스를 부여하지 않았습니다. 외부 모델·패키지의 라이선스는 각 원본을 따릅니다.

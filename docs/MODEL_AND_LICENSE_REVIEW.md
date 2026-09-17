# 모델 및 라이선스 선정 결과

검토일: 2026-09-05 · 대상: Windows 설치형 상면 검사 및 Dobot 분류 제품

02 작업은 구현에 사용할 모델·직접 도구를 선정하고 공식 자산과 배포 조건을 확보하는 단계다. 실제 불량 성능, 추가 모델의 GPU 실행, 완성된 설치 파일의 배포 적합성 검증은 각각 후속 단계다. 본 문서는 라이선스 원문을 바탕으로 한 개발·배포 계획이며 법적 무분쟁 보증이 아니다.

## 1. 확정한 구성

| 역할 | 선택 | 코드·가중치 조건 | 준비 결과 |
|---|---|---|---|
| 전체 물체 검출 | RF-DETR Small, rfdetr 1.9.4 | 선정한 Small Detection 코드·공식 가중치 Apache-2.0 | 기존 공식 가중치 재사용; 제품 검출용으로 별도 학습 예정 |
| 알려진 불량 위치 검출 | RF-DETR Small을 별도 결함 모델로 학습 | 동일한 공식 초기 가중치 사용 | 추가 사전학습 모델 계열 없이 다중 결함 상자 출력; 최종 결함 가중치는 실제 데이터로 생성 |
| 이상 탐지 | PatchCore 방식을 참고한 로컬 구현 + DINOv2 Small ViT-S/14 | 참고 코드 Apache-2.0, 선택한 일반 DINOv2 가중치 Apache-2.0 | 정상 특징 추출 가중치 확보; 정상 메모리와 임계값은 실제 데이터로 생성 |
| VLM 설명·보조 검토 | Qwen/Qwen3.5-4B | 공식 모델 Apache-2.0 | 공식 원본 가중치·설정·토크나이저 확보 |
| 데스크톱 UI | PySide6-Essentials 6.11.2 + shiboken6 6.11.2 | LGPL-3.0 경로 선택; 해당 조건 이행 필요 | Python 3.12 / Windows x64용 wheel 확보; 미설치 |
| 실행 파일 묶기 | PyInstaller 6.22.2, onedir 방식 | GPL-2.0-or-later + Bootloader Exception | Windows x64 wheel과 예외 조항 확보; 실제 패키징은 16 단계 |

설정 파일과 모델 파일의 정확한 커밋·URL·파일 크기·기대 해시는 [선정 자산 고정 목록](../models/selection-lock.json)에 기록했다. 별도 검사 모델과 정상 메모리의 상태는 [모델 등록부](../models/model-registry.json)에 명시했다.

## 2. 이렇게 선택한 이유와 한계

### 물체 검출과 알려진 불량

현재 검증한 RF-DETR 실행 환경을 유지한다. 전체 장면에서 제품을 찾는 모델과 제품 이미지에서 결함을 찾는 모델은 학습 데이터와 최종 가중치를 분리한다. 한 제품에서 여러 결함 상자를 출력할 수 있으므로 제품마다 하나의 NG 종류만 강제하지 않는다.

초기 가중치는 COCO 일반 물체용이다. 파일 확보를 실제 제품 검출·불량 검사 학습 완료로 간주하지 않는다. 두 역할은 같은 사전학습 파일로 시작하지만 현장에서 학습한 서로 다른 모델이 된다. 필요하면 같은 GPU에서 순차 실행한다. 최종 메모리와 지연은 03 및 현장 단계에서 측정한다.

| 불량 | 기본 검사 경로 | 라벨·기준에서 지킬 점 |
|---|---|---|
| NG01 누락 | 결함 검출 + 품목별 필수 형상 확인 | 사라진 요소의 기대 위치가 정의되어야 함; '상자가 검출되지 않았다'만으로 정상 판정 금지 |
| NG02 돌출·버 | 결함 위치 검출 | 실제 돌출 영역 표기; 크기 허용 기준은 현장 설정 |
| NG03 균열 | 결함 위치 검출 | 원본에서 보이는 균열 영역 표기; 미세 결함은 촬영 해상도 확보 필요 |
| NG04 변형 | 결함 검출 + 기준 외곽 비교 | 상면에서 관찰할 수 있는 변형으로 범위 한정 |
| NG05 구멍 불량 | 결함 검출 + 구멍 개수·형상 검사 | 상면에서 확인 가능한 기준만 사용; 내부 깊이·관통 여부를 추정하지 않음 |
| NG06 표면 결함 | 결함 위치 검출 | 정상 출력 흔적과 구분할 기준 필요 |

결함의 위치가 정의되지 않는 사례를 억지로 상자 라벨로 만들지 않는다. 해당 항목은 품목별 검사 또는 이상 탐지·보류 경로에서 처리하고 데이터 지침에 명시한다. 알려진 결함 검출이 없다는 사실만으로 OK가 되지 않으며 필수 형상 검사와 이상 탐지도 통과해야 한다.

### 이상 탐지

선택한 특징 추출 모델은 facebook/dinov2-small의 일반 ViT-S/14이며 Hugging Face의 safetensors 파일을 사용한다. 저장소 안의 Cell-DINO·XRay-DINO 등 별도 비상업용 모델은 포함하지 않는다. 모델 카드뿐 아니라 Meta의 일반 DINOv2 코드·가중치 라이선스 설명도 보관했다. [DINOv2 공식 저장소](https://github.com/facebookresearch/dinov2), [선택한 모델](https://huggingface.co/facebook/dinov2-small).

PatchCore 원본의 기본 CNN 특징과 같은 설정은 아니다. DINOv2의 공간 패치 특징으로 정상 메모리·대표 샘플 선택·최근접 거리 기반 검사를 구현하는 **변형 구성**이다. 원본 논문의 정확도 수치를 이 제품에 적용하지 않는다. 선택 이유는 가중치 사용 조건의 명확성, 기존 PyTorch·Transformers와의 통합, 노트북 배포를 고려한 크기다. 미세 결함 성능은 아직 입증하지 않았다.

Amazon 원본 저장소는 고정 커밋의 참고 자료로 보관한다. 기존 requirements를 통째로 설치하지 않는다. 원본은 pretrainedmodels·FAISS 등 다른 패키지와 가중치를 도입하므로, 선택하지 않은 ImageNet 가중치를 자동으로 다운로드하는 경로를 만들지 않는다. 최근접 검색은 먼저 PyTorch에서 크기를 제한해 나누어 수행하도록 구현하며 메모리 상한과 시간은 08에서 검증한다. 참고 코드를 수정·전용하면 Apache 고지와 수정 표시를 보존한다. [PatchCore 원본](https://github.com/amazon-science/patchcore-inspection).

### VLM

Qwen3.5-4B의 공식 원본 가중치를 선택하고, 별도 제작자의 양자화 파일은 도입하지 않았다. 모든 모델 중 최신·최고라는 결론이 아니라 현재 하드웨어와 로컬 설명 역할을 위한 구현 선택이다. 공식 원본을 기준으로 정확도와 실행을 확인한 후 필요할 때 노트북용 양자화를 검토한다. [Qwen 공식 모델](https://huggingface.co/Qwen/Qwen3.5-4B).

모델 파일 크기는 GPU 실행 메모리와 다르다. 3090 동시 실행 및 노트북 메모리 적합성은 아직 확인하지 않았다. VLM은 정상 참조와 검사 근거를 받아 설명하며 검증된 NG를 OK로 취소하지 못한다. 초기에는 추가 학습 없이 입력·출력 경로를 구현한다.

## 3. 상업 배포 적용 조건

### Apache-2.0 구성

선정한 코드와 가중치는 저작권·라이선스·해당 NOTICE를 보존하고 수정한 파일에는 변경을 표시하는 조건으로 상업적 사용·수정·배포가 가능한 경로를 선택했다. 제품 전체 소스에 Apache-2.0을 적용하는 결정은 아니다. 모델 이름·상표에 대한 별도 사용 권리나 모든 데이터 관련 권리를 보장하는 것으로 해석하지 않는다. 정확한 범위는 저장한 원문을 따른다.

- RF-DETR: 기존 licenses/RF-DETR-LICENSE.txt, README, 모델 출처 파일과 models/manifest.json 유지.
- DINOv2: licenses/selection/dinov2-code-LICENSE, 같은 커밋의 README, 해당 모델 카드 유지.
- PatchCore: licenses/selection/patchcore-LICENSE 및 참고 소스의 기존 고지 유지.
- Qwen: models/qwen3.5-4b/LICENSE와 모델 카드 유지.
- Transformers·PyTorch 및 기타 실행 의존성은 별도 패키지의 고지를 유지한다. 이 문서가 전체 의존성 검토를 대신하지 않는다.

### PySide6와 Qt

초기 화면은 QtCore / QtGui / QtWidgets를 사용한다. Qt Addons·WebEngine·별도 차트 모듈 등은 현재 선정하지 않는다. 추가 모듈은 도입 전에 라이선스 범위를 확인한다. Essentials wheel 전체가 곧 최종 배포 허용 파일 목록은 아니며 실제 수집된 DLL·플러그인·제3자 라이브러리는 패키징 시 확인한다. [Qt Core](https://doc.qt.io/qt-6/qtcore-index.html), [Qt GUI](https://doc.qt.io/qt-6/qtgui-index.html), [Qt Widgets](https://doc.qt.io/qt-6/qtwidgets-index.html).

LGPL 경로에서는 제품 소스를 비공개로 유지할 수 있지만 조건 이행이 필요하다. 동적 라이브러리로 배포하고 사용자에게 고지·원문을 제공하며, 사용한 라이브러리의 대응 소스와 필요한 교체·재연결 정보를 제공하는 방식으로 구성한다. 해당 권리를 제한하는 사용 약관이나 잠금 구조를 두지 않는다. 원문 전체를 기준으로 적용한다. [Qt 공식 LGPL 안내](https://www.qt.io/development/open-source-lgpl-obligations).

확인한 6.11.2 PyPI wheel의 METADATA는 LGPL 선택지를 표시하지만 동봉 라이선스 파일은 LicenseRef-Qt-Commercial.txt뿐이었다. 이것만으로 무료 배포 의무가 충족됐다고 처리하지 않았다. 정확한 PySide·QtBase v6.11.2 소스 커밋의 LGPLv3/GPLv3 원문을 추가 확보했다. 이 보완은 라이브러리의 대응 소스·제3자 고지·교체 절차 제공 자체를 완료한 것은 아니다. 이 항목은 설치 파일이 만들어지는 16 단계에서 실제 배포 파일과 함께 완료해야 한다.

### PyInstaller

PyInstaller의 Bootloader Exception은 자체 코드로 만든 비공개 상업용 프로그램을 묶어 배포하는 경로를 허용한다. 포함된 Qt나 다른 의존성의 의무가 사라지는 것은 아니다. onedir 형태로 라이브러리와 모델을 외부 파일로 두는 구성을 선택하고 라이브러리 교체가 실제로 가능한지 검증한다. onedir 선택만으로 LGPL 준수가 자동으로 완료되지는 않는다. [PyInstaller 공식 예외 설명](https://pyinstaller.org/en/v6.22.2/license.html).

## 4. 확보한 파일과 확인 방법

- models/selection-lock.json: 정확한 모델 커밋, 기대 해시, wheel 버전·URL, 출처 원문 목록.
- models/model-registry.json: 역할별 초기 파일과 아직 생성되지 않은 학습 결과 구분.
- models/dinov2-small/: 특징 추출 모델·설정·모델 카드.
- models/qwen3.5-4b/: 공식 모델 두 개의 분할 가중치, 토크나이저, 입력 설정, 라이선스.
- vendor/wheels/: 선정한 Windows x64 UI·패키징 파일. 설치는 하지 않음.
- licenses/selection/: 커밋·라이선스·공식 문서 사본 및 wheel에서 추출한 고지.
- artifacts/selected-assets-check.json: 파일별 실제 SHA-256, 검증 상태, 오류, 고정 목록 해시.
- scripts/prepare_selected_assets.py: 고정된 자산 다운로드·무결성 검증. 모델이나 다운로드된 코드를 실행하지 않음.
- requirements-selected-ui.txt, requirements-selected-build.txt: 다음 단계용 직접 버전·wheel 해시 고정. 전체 의존성 잠금 파일은 아님.

프로젝트 폴더에서 기존 Python으로 파일만 다시 확인할 수 있다.

```powershell
.\.venv\Scripts\python.exe scripts\prepare_selected_assets.py --verify-only
```

일반 실행은 누락된 고정 파일을 다운로드하고 검증한다. 설치 환경을 변경하거나 실제 모델을 학습하지 않는다. 검증된 파일과 다운로드 중인 .partial 파일을 구분하며, 이미 존재하는 파일이 해시와 다르면 덮어써서 숨기지 않고 실패한다.

## 5. 후속 단계에서 완료할 사항

| 항목 | 완료 단계 |
|---|---|
| PySide6 설치·창 표시, DINOv2 및 Qwen GPU 실행·메모리 측정 | 03 |
| 새 패키지를 포함한 전체 의존성 해결·잠금 및 고지 수집 | 03; 이후 의존성 변경 시 갱신 |
| 알려진 불량 모델 학습 코드·복수 결함 라벨링 지침 | 06~07 |
| DINOv2 특징을 사용하는 이상 탐지 변형의 실제 구현 | 08 |
| VLM의 구조화된 설명·실패·시간 초과 처리 | 09 |
| D405/Dobot SDK·드라이버 도입 및 재배포 시 해당 공급사 조건 확인 | 관련 장비 연결·배포 단계 |
| Qt 대응 소스·고지·교체 절차 및 배포된 DLL/플러그인 확인 | 16 |
| PyTorch/CUDA 런타임·PyAV/FFmpeg 등 실제 포함 바이너리 조건 확인 | 16; 현재 메타데이터 목록만으로 승인하지 않음 |
| 모델 성능·임계값·노트북 동시 실행·연속 운전 검증 | 실제 데이터·현장 준비 후 |

02 완료는 위 선정 자산에 대한 출처·조건·파일 확보 완료다. 전체 제품을 지금 바로 판매해도 모든 배포 의무가 충족됐다는 뜻은 아니다.

## 6. 다운로드 무결성 이력

Qwen 첫 번째 분할 파일에서 재검증 해시 불일치가 관측되었다. 원인은 미확정이다. 연속 검사에서 공식 해시와 일치한 보관본으로 복원한 후 전체 49개 자산 검증과 독립 해시 확인을 통과했다. 불일치했던 재다운로드본은 artifacts/quarantine에 보관하며 실행·배포 대상에서 제외한다. [상세 이력](../artifacts/asset-recovery.json). 03 단계에서는 모델 실행 전에 무결성을 다시 확인하고 불일치 시 로딩하지 않는다.


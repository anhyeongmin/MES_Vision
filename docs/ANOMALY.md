# 08 정상 특징 등록·이상 탐지

갱신일: 2026-09-05

7번에서 검토한 정상 물체 이미지를 받아 DINOv2 특징을 저장하고, 새 물체의 이상 점수·패치별 거리 지도·의심 영역을 출력하는 로컬 프로그램을 구현했다. 저장한 정상 기준을 다시 불러올 수 있고, 기존 다중 물체 검사 연결 코드에 `AnomalyInspector`로 연결된다. 실제 제품 학습, 현장 판정 임계값, 미등록 불량 성능은 아직 미확정이다.

## 구현 방법과 범위

선정한 일반 DINOv2 Small의 마지막 층에서 CLS를 제외한 공간 패치 특징을 사용한다. 전체 RGB crop을 448×448로 조정하고, 14×14 패치에 대응하는 32×32개 특징을 만든다. 각 특징은 384차원이며 L2 정규화한다. 가운데를 잘라내거나 테두리를 버리지 않는다. 정사각형 변환은 가로·세로 비율을 바꾸며, 결과 좌표를 원래 crop의 가로·세로 크기로 각각 되돌린다. 회전 정렬·배경 마스킹은 적용하지 않는다.

정상 패치 중 최대 16,384개 후보를 일정한 난수 시드로 추출하고, 32차원 투영 공간의 가장 먼 특징을 차례로 선택해 최대 2,048개를 남긴다. 실제 저장·검색에는 투영 전 384차원 특징을 사용한다. 전체 패치 수, 후보 수, 저장 수와 각 특징의 원본 이미지·패치 번호를 기록한다. 한도가 넘는 자료가 모두 그대로 메모리에 저장되는 방식은 아니다.

새 이미지의 각 패치와 정상 특징 사이의 최소 유클리드 거리를 계산한다. 정규화된 벡터의 거리 범위는 0~2이며, 큰 값일수록 저장된 정상과 다르다. **확률이나 불량률이 아니다.** 물체 점수는 패치 거리의 최댓값이다. 거리 계산을 나누어 수행하므로 전체 이미지×전체 정상 특징의 큰 거리 행렬을 한 번에 유지하지 않는다.

이 방식은 정상 패치 메모리와 대표 특징 선택이라는 [PatchCore 원리](https://github.com/amazon-science/patchcore-inspection)를 참고한 로컬 변형이다. 원본 구현의 WideResNet 특징·재가중치·평활화 구성과 같지 않으며 논문의 성능을 그대로 주장하지 않는다. DINOv2의 공간 특징 사용은 [공식 Transformers 문서](https://huggingface.co/docs/transformers/model_doc/dinov2) 및 이미 고정한 로컬 구현을 기준으로 했다. 외부 모델 코드나 패키지를 새로 설치하지 않았다.

## 정상 기준 만들기

프로젝트 `C:\Users\alexi\Desktop\Workspace\MES_Vision`에서 실행한다. 실제 자료가 아직 없으므로 아래 실물 명령은 자료를 준비한 뒤 사용한다.

```powershell
# 7번 라벨링 도구에서 정상 기준 이미지를 내보낸 다음
.\.venv\Scripts\python.exe scripts\anomaly.py build --normal-export datasets\normal-reference-001 --output runs\anomaly-reference-001

# 이미지 특징을 계산하지 않고 파일·모델·품목 연결만 확인
.\.venv\Scripts\python.exe scripts\anomaly.py validate-bank --bank runs\anomaly-reference-001 --product-id part-001
```

`part-001`은 예시이며 라벨링 프로젝트에서 사용한 실제 품목 ID로 바꾼다. 임의의 이미지 폴더나 검증·시험 자료를 정상 기준으로 바로 받을 수 없다. `normal_bank` 내보내기, 원본 라벨 스냅샷, 검토자, train 분할, NORMAL 상태, crop 좌표와 이미지 해시를 확인한다. 정상·불량 사진의 의미가 올바른지는 사람의 라벨과 검토에 의존한다.

설정은 [configs/anomaly/build.json](../configs/anomaly/build.json)이다. 이미지 크기는 112~896의 14 배수, 후보 수는 최대 65,536, 저장 특징 수는 최대 8,192를 허용한다. 기본값은 시작 설정이며 실제 결함 가시성에 맞춰 검증해야 한다. 설정을 바꿔 정상 기준을 다시 만들면 기존 임계값을 재사용할 수 없다.

기본 장치는 CUDA다. GPU가 없으면 자동으로 속도가 다른 CPU로 바꾸지 않고 오류를 표시한다. CPU 실행은 `--device cpu`로 명시한다. 합성 자료는 등록·읽기·검사 각각에 `--synthetic`이 필요하고 결과에도 합성임을 남긴다.

```text
runs/anomaly-reference-001/
  bank.json                 # 품목·모델·전처리·메모리 구성·해시
  features.npy              # 선택된 정상 특징, float32
  origins.npy               # 각 특징의 원본 이미지/패치 번호
  source-export.json        # 원래 정상 내보내기 기록
  source-collection.json    # 검토·분할·라벨 스냅샷
```

출력은 새 폴더만 허용한다. 임시 폴더에 저장한 뒤 파일 검증을 통과하면 최종 폴더로 옮긴다. 중간 실패 때 기존 기준을 덮어쓰거나 자동 활성화하지 않는다. 남은 `.이름-building-...` 폴더는 미완료 작업일 수 있으므로 명령 오류를 확인하고, 그 폴더를 제품 기준으로 지정하지 않는다.

## 점수·영역 출력

명령의 이미지 입력은 **한 물체를 원본에서 잘라낸 RGB 사진**이다. 전체 작업대 사진의 다중 물체 검사는 검사 파이프라인에서 각 검출 물체를 추출한 다음 어댑터를 호출한다.

```powershell
.\.venv\Scripts\python.exe scripts\anomaly.py inspect --bank runs\anomaly-reference-001 --product-id part-001 --image incoming\object-crop.png --output runs\anomaly-score-001
```

이 상태에서는 기준이 없으므로 `UNCERTAIN`으로 출력한다. 점수가 낮아도 자동 PASS가 아니다. 결과 폴더에는 다음 파일을 저장한다.

| 파일 | 내용 |
|---|---|
| result.json | 원시 점수, 상태, 품목·기준 해시, 임계값 버전, 변환·영역·최대 점수 패치의 정상 참조 연결 |
| patch-distances.npy | 실제 32×32 패치 거리값; 이미지로 변환하기 전 정량 결과 |
| nearest-bank-indices.npy | 각 패치에 가장 가까운 정상 특징의 번호 |
| input.png | 검사한 원본 crop |
| heatmap.png | 패치 거리를 0~2 고정 색상 범위로 표시한 지도 |
| overlay.png | 원본 위에 지도를 겹친 미리보기. 영역 기준이 있으면 해당 상자도 표시 |

사진마다 색 범위를 최솟값·최댓값에 맞춰 바꾸지 않는다. 따라서 거의 정상인 사진도 무조건 붉게 보이도록 확대되지 않는다. 지도는 최근접 패치로 확대하며, 큰 이미지로 표시해도 공간 해상도가 늘어난 것은 아니다. 경계 상자는 연결된 이상 패치의 범위를 나타내며 실제 결함 윤곽이나 mm 계측값이 아니다.

## 합격·보류·불합격 기준

실물 기준값은 기본 설정에 넣지 않았다. 기준을 준비할 때 다음 명령으로 해당 정상 메모리에 연결된 빈 양식을 만든다.

```powershell
.\.venv\Scripts\python.exe scripts\anomaly.py criteria-template --bank runs\anomaly-reference-001 --product-id part-001 --output configs\anomaly\part-001-criteria.json
```

생성 직후에는 숫자가 null이므로 검사에 넘길 수 없다. 별도로 분리한 실물 검증 자료를 바탕으로 아래 항목을 채운다.

- `version`: 기준 버전. `UNCONFIGURED`를 실제 버전으로 변경한다.
- `pass_max`: 이 값 이하인 점수가 이상 검사 합격 구간이다.
- `fail_min`: 이 값 이상인 점수가 이상 검사 불합격 구간이다. `pass_max`보다 커야 한다.
- `pixel_threshold`: 지도에서 의심 영역을 표시할 패치 거리다. `pass_max`보다 커야 한다.
- `validated`: 실물 기준 검증 완료 선언. false이면 점수·영역만 표시하며 상태는 항상 UNCERTAIN이다.
- `validation_reference`: 검증 자료와 결과를 추적할 기록 식별자 또는 경로. 검증 완료 선언 시 필수다. 프로그램이 이 문자열만으로 실제 검증의 적절성을 보증하지는 않는다.

값을 채우고 검증한 뒤 `inspect`에 `--criteria configs\anomaly\part-001-criteria.json`을 추가한다. 정상 메모리의 해시, 품목, real/synthetic 구분이 다르면 거부한다. 알고리즘·모델·전처리·정상 등록 자료가 바뀌면 새 기준이 필요하다.

| 조건 | 이상 검사 상태 |
|---|---|
| 기준 없음 또는 검증 미완료 | UNCERTAIN |
| 검증된 기준, 점수 ≤ pass_max | PASS |
| pass_max < 점수 < fail_min | UNCERTAIN |
| 검증된 기준, 점수 ≥ fail_min | FAIL |

미검증 의심 영역에는 불량 코드를 확정하지 않는다. 검증된 불합격 근거에만 `NG_UNKNOWN`을 붙인다. 이 결과는 **이상 검사 한 항목의 상태**이며 최종 OK/NG/REVIEW는 10 단계 판정 로직의 책임이다. 현재 최종 판정은 null, 로봇 명령은 비활성 상태를 유지한다.

## 검사 코드 연결

- [DinoFeatures](../src/mes_vision/anomaly/features.py): 로컬 가중치·설정 해시 확인, FP32 특징 추출, 모델 해제.
- [build_bank / Bank](../src/mes_vision/anomaly/bank.py): 검토한 정상 내보내기 검증, 특징 선택·저장·무결성 확인.
- [AnomalyEngine / AnomalyInspector](../src/mes_vision/anomaly/scoring.py): 거리·영역·상태 계산과 기존 검사 인터페이스 연결.
- `CheckResult.details`에 정상 기준 ID/해시, 특징 버전, 임계값, 지도 형태, crop 좌표, 최대 패치의 정상 참조 연결을 추가했다. 기존 결과 v1에 선택 필드를 추가한 방식이다.
- `AnomalyInspector(engine, evidence_dir=...)`로 파일 저장을 연결할 수 있다. 저장 경로를 지정하지 않으면 파이프라인은 점수·영역·설정 메타데이터만 보존한다. 전수 이미지 보존·검색·보존 기간 관리는 14 단계에서 완성한다.
- 어댑터 품목과 `InspectionPipeline(product_id=...)`가 다르면 구성을 거부한다. 각 물체의 프레임 ID와 물체 ID를 유지하고 영역을 원본 작업대 좌표로 복원한다.
- 미등록 상태는 기존 `UnconfiguredInspector("anomaly")`로 NOT_RUN을 반환한다. 명시한 파일이 없거나 손상되면 초기화/CLI에서 오류를 내며, 실행 중 검사 실패는 파이프라인이 ERROR로 기록한다. 이전 점수를 재사용하지 않는다.

## 기능 검증과 현장 한계

```powershell
# CPU 기능·회귀 검증
.\.venv\Scripts\python.exe scripts\verify_anomaly.py

# 실제 RTX 3090 DINO 등록·재불러오기·추론·다중 물체 연결까지 확인
.\.venv\Scripts\python.exe scripts\verify_anomaly.py --gpu
```

[검증 보고서](../artifacts/anomaly-check/report.json)에 결과와 실제 실행 폴더를 남긴다. CPU 시험은 기존 89개에 이상 탐지 22개를 더한 111개다. 합성 기준과 색상 특징 시험으로 거리 정답·경계값·누락 기준·참조 오염·파일 변조·좌표 변환을 검증한다. 실제 GPU 시험은 DINOv2 448 입력, 512개 정상 특징을 사용해 CLI 등록/검사, 반복 추론, 새 모델 인스턴스에서 재불러오기, 두 물체 연결을 확인한다. 이때 검출 상자는 명시적인 모의 상자이며 이상 특징과 거리 계산은 실제 CUDA 실행이다.

실물 도착 후에는 정상 개체 간 편차, 회전, 조명·노출·배경 변동, 작은 균열의 가시성, crop 오차와 가림을 따로 평가해야 한다. 상면에서 보이지 않는 결함을 찾아내거나 모든 새로운 불량을 검출한다고 보장할 수 없다. 대표 특징 추출과 입력 축소로 세부 변화가 사라질 수 있고, 공간적 순서·구멍 개수·전반적인 형상은 별도 형상 검사가 필요하다.

합성 변형의 점수 변화와 현재 속도는 동작 확인 자료다. 실제 결함 정확도, 독립 실물 평가, RTX 5070 노트북 성능, 전체 검사 처리 시간 또는 판매용 합격 기준을 뜻하지 않는다. 특징 모델 파일은 기존 선정본을 유지하며 제품 정상 메모리와 임계값은 모델 등록부에 활성화하지 않았다. 다음은 09 VLM 연결이다.

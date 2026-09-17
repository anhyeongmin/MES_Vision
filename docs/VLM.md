# 선택형 VLM 후속 분석

09 단계 구현. 모델은 로컬 Qwen3.5-4B이며 기본 검사와 별도 프로세스에서 실행한다. PySide6 결과 확인창과 ON/OFF 제어를 포함한다. 전체 제품 화면은 13 단계에서 통합하고, WinForms는 PySide6 제품 완성 후 추가한다.

## ON/OFF 동작

| 조작·상태 | 동작 |
|---|---|
| 새 작업 저장소 | VLM 기본 OFF. 모델을 GPU에 올리지 않는다. |
| ON | 이후 요청을 대기열에 넣고 저장된 이미지로 추가 분석한다. |
| OFF | 신규 분석을 생략하고 대기 요청을 취소한다. 실행 중인 모델 프로세스에도 중단을 요청하고 종료를 확인한다. 중단 중 상태를 표시한다. |
| OFF 이후 | 기본 검사 근거, 원본 판정과 완료된 추가 설명을 유지한다. |
| 다시 ON | 취소·생략·적체 요청을 자동 재실행하지 않는다. 선택 후 재시도로 명시적으로 요청한다. |
| 다시 시작 | 저장한 ON/OFF를 복원한다. ON 상태에서 비정상 종료된 진행 요청만 복구한다. |

VLM을 끈 상태에서도 NG03/균열 같은 알려진 불량 근거, 근거 영역, 검사 점수와 적용 기준을 볼 수 있다. 이상 탐지 점수만으로 구체적인 결함 이름을 만들어 표시하지 않는다. 10 단계 판정 결과가 있는 자료에는 최종 판정·규칙·완료 여부·사유도 표시한다. 이 문서의 기존 09 예제는 정책을 연결하지 않아 `판정 연결 전`이며, 판정 포함 예제는 [DECISION.md](DECISION.md)를 따른다.

## 실행

프로젝트 폴더의 PowerShell에서 다음을 실행한다. 예제 출력 폴더는 새 경로여야 한다.

```powershell
.\.venv\Scripts\python.exe scripts\vlm.py make-fixture --output runtime\vlm-demo
.\.venv\Scripts\python.exe scripts\vlm.py ui --queue runtime\vlm-demo\queue --synthetic
```

합성 이미지와 기본 검사 근거가 표시된다. ON을 누른 후 생략된 행을 선택해 `선택 분석 다시 요청`을 누르면 실제 Qwen이 실행된다. 실제 제품 정확도를 확인하는 예제가 아니다. 모델 없이 화면만 시험하려면 UI 명령에 `--simulate`를 추가한다. 모의 분석은 합성 자료에만 허용한다.

별도 작업 실행 프로세스를 사용할 경우:

```powershell
.\.venv\Scripts\python.exe scripts\vlm.py worker --queue runtime\vlm-demo\queue --synthetic
.\.venv\Scripts\python.exe scripts\vlm.py ui --queue runtime\vlm-demo\queue --synthetic --read-only-worker
```

`--read-only-worker`는 창에서 작업 실행부를 시작하지 않는 옵션이다. ON/OFF 등의 제어는 가능하다. 같은 저장소에 실행부를 두 개 시작하면 두 번째는 거부한다. 운영 자료에는 `--synthetic`와 `--simulate`를 붙이지 않는다.

## 검사 엔진 연결

Qt에 종속되지 않은 API를 제공한다. `frame`과 `result`는 같은 검사 실행의 원본 RGB 프레임과 `InspectionPipeline` 결과다.

```python
from dataclasses import asdict
from mes_vision.vlm.snapshots import save_snapshot
from mes_vision.vlm.queue import AnalysisQueue
from mes_vision.vlm.backend import GenerationConfig

queue = AnalysisQueue("runtime/vlm/queue")  # 처음에는 OFF
snapshot = "runtime/vlm/snapshots/unique-inspection-id"
save_snapshot(
    snapshot, frame, result,
    kind="real", normal_export=reviewed_normal_export,
    criteria={"version": "approved-criteria-id", "description": approved_criteria},
)
for item in result.objects:
    queue.enqueue(snapshot, item.object_id, asdict(GenerationConfig()))
```

정상 참조는 07 단계에서 검토·내보낸 동일 품목의 학습용 정상 자료를 사용한다. `reference_index`로 참조를 명시적으로 선택할 수 있다. 정상 참조나 검사 기준이 없으면 해당 인자를 생략할 수 있지만, 분석 결과에 미설정 제한 사항을 남긴다. 임의 자료를 승인된 실물 기준으로 취급하지 않는다.

저장과 등록은 파일 입출력 작업이다. 통합 제품에서 기본 결과 표시 후 저장 작업부에서 수행해야 한다. 사진 저장과 해시 계산 시간을 검출 시간에 숨기지 않는다. 아직 통합 화면의 자동 저장·등록까지 완료한 것은 아니다.

## GPU 사용과 응답 시간

기본 검사가 같은 GPU를 사용할 때는 반드시 같은 저장소의 조정기를 연결한다.

```python
from mes_vision.vlm.gpu import GpuCoordinator
gate = GpuCoordinator(queue.root / "gpu-coordination")
pipeline = InspectionPipeline(detector, inspectors, product_id=product_id,
                              gpu_coordinator=gate)
```

기본 검사 요청이 들어오면 VLM 모델 프로세스를 종료한 뒤 GPU 사용권을 넘긴다. 중단된 VLM 요청은 대기로 돌아가고 기본 검사가 끝난 뒤 기본 1초간 새 검사가 없을 때 다시 시도한다. 연속 검사가 이어지면 VLM은 계속 대기할 수 있다. 대기 작업이 없으면 모델을 내린다. 다시 실행할 때 파일 검증·모델 적재 시간이 추가된다.

이 방식은 GPU 간섭을 관리하지만 즉시 전환이나 100ms 응답을 보장하지 않는다. 종료 대기 시간은 `gpu_wait_ms`로 검사 실행 시간과 따로 기록한다. 조정기를 사용하지 않는 별도 GPU 프로그램에는 우선순위를 강제하지 못한다. 고정된 짧은 응답 시간이 필요한 최종 운전에서는 검사 정지 구간에만 VLM을 실행하거나 별도 GPU/PC를 사용하는 정책을 현장 성능 시험으로 확정해야 한다.

기본 설정은 `configs/vlm/generation.json`의 thinking OFF, 최대 128개 생성 토큰, 이미지 긴 변 최대 448px, 생성 제한 120초, 한국어다. 모델 로딩 제한은 별도 120초다. 원본 이미지는 보존하고 VLM 입력만 크기를 줄인다. 작은 결함의 가시성은 실물로 확인해야 한다.

## 결과·오류·복구

- 허용 응답은 `observation` 문자열과 `needs_review` 불리언 두 항목뿐이다. 잘린 응답, 형식 오류, 중복 키, 추가 판정/동작 필드는 실패로 기록한다.
- VLM 의견은 참고용이다. 원본 판정, 확정 불량, 보류, 로봇 명령을 바꾸는 코드 경로가 없다. `needs_review`는 추가 검토 의견이며 최종 판정이 아니다. 올바른 JSON이어도 관찰 내용이 정확하다는 보장은 없다.
- 프레임·물체·검사 ID, 이미지/검사 파일 해시, 정상 참조·기준, 모델 파일/버전, 프롬프트·생성 설정을 연결한다. 파일 변조나 다른 물체의 지연 응답은 거부한다.
- 대기+실행은 기본 100건, 실행은 1건이다. 접수된 요청 안에서 NG/FAIL → 보류 → OK 순서로 처리한다. 꽉 차면 `적체로 보류`로 남기고 명시적 재시도를 기다린다. 이미 접수한 요청을 임의 삭제하지 않는다.
- OFF 후 늦게 도착한 결과는 완료로 반영하지 않는다. 시간 초과·자식 프로세스 실패는 실패로 남기며 기본 검사 결과는 유지한다.
- SQLite에 상태와 설정·변경 이력을 저장한다. 실행부 강제 종료 시 모델 프로세스도 종료되며 다음 실행에서 미완료 요청을 복구한다. 이전 실행의 완료 토큰은 무효화한다.
- 성공 시 `queue/attempts`에 원문 응답·프롬프트·모델 출처·시간 측정을 보존한다. 응답 검증 실패의 원문은 해당 작업의 진단 결과에 남긴다.
- 현재 창은 최근 200건을 표시한다. 전체 이력 검색·보존 기간·디스크 용량 정책은 14/15 단계에서 통합한다. 폴더 백업은 작업 실행부와 창을 모두 종료한 후 수행한다.

## 검증

```powershell
.\.venv\Scripts\python.exe scripts\verify_vlm.py --gpu
```

결과: `artifacts/vlm-check/report.json`. CPU 회귀·기능 시험, Qt offscreen 버튼 조작, 강제 종료 후 자식 종료/요청 복구, 실제 Qwen 구조화 응답과 RF-DETR 우선 실행을 확인한다. 실제 제품 정확도, 노트북 성능, 장시간 연속 운전은 장비 준비 후 검증한다.

2026-09-05 실행 `run-bcb9c079`: 133개 시험, Qt 조작 12개, 강제 종료·복구, 실제 GPU 시험 통과. RTX 3090 / Windows / BF16에서 합성 물체 두 개의 한국어 JSON 응답이 각각 13.09초·12.17초 걸렸다(각각 43토큰; 입력 준비·모델 로딩·대기 제외). 첫 모델 파일 검증+적재는 18.23초였다. 모델이 기록한 최고 할당 GPU 메모리는 약 8,870MiB다.

기본 RF-DETR 실행은 같은 시험에서 대기 없는 기준 23.18ms, VLM 중단을 기다린 시간 257.67ms, 중단 후 검사 40.34ms, 호출 전체 300.69ms였다. 전환 한 번의 관찰값이며 전체 검사 속도나 상한 보장이 아니다. 두 구조화 응답 성공은 실제 균열 식별 정확도를 증명하지 않는다.

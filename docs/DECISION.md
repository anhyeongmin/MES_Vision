# 10 · OK / NG / 판정 보류

장비 준비 전 판정 엔진과 모의 검증을 완료했다. 모델 출력과 고정된 품목 규칙을 대조해 판정한다. 실제 제품 기준·가중치·촬영 품질/작업 영역 검사는 미설정이며 출고 판정이나 로봇 운전을 활성화하지 않았다.

## 판정 우선순위

| 조건 | 물체 판정 |
|---|---|
| 프레임·물체·검사·근거 좌표 연결이 잘못됨, 품목·검출 모델·실물/모의 구분이 다름 | REVIEW · 판정 보류 |
| 판정 규칙 자체가 미검증 | REVIEW |
| 현재 대상의 검증된 검사에서 확정 불량 코드가 하나 이상 있음 | NG. 다른 검사 오류·보류 근거도 함께 보존 |
| 확정 불량은 없지만 필수 검사가 없음/실패/불확실/미설정, 촬영·작업 영역 확인 부족, 수량 불일치 등 | REVIEW |
| 필수 검사가 모두 합격하고 촬영 조건·대상·품목·수량 등 차단 사유가 없음 | OK |

프레임 전체의 식별 오류는 전체 REVIEW다. 그 외 전체 결과는 물체 중 NG가 있으면 NG, NG 없이 보류가 있거나 전체 차단 사유가 있으면 REVIEW, 모든 물체가 OK일 때만 OK다. **전체 NG는 모든 물체가 NG라는 뜻이 아니다.** 물체별 판정과 OK/NG/REVIEW 개수를 함께 저장한다. 0개 검출은 예상 수량이 0이어도 REVIEW이며 빈 작업대로 확정하지 않는다.

촬영 품질이나 다른 필수 검사가 미완료여도 별도의 유효한 확정 불량 근거는 NG로 보존할 수 있다. 이는 운전 허가가 아니다. `required_checks_complete`는 필수 물체 검사의 완료 여부이며 NG라도 false일 수 있다. 촬영 상태·배치·수량 사유는 별도로 보존한다. 로봇 권한은 어떤 판정에서도 false이며 11 단계에서 독립적으로 구현한다.

## 세 가지 검사 규칙

| 방식 | 처리 |
|---|---|
| `defect_candidates` | 알려진 불량 후보의 모델·클래스→불량 코드 연결·후보 수집 임계값을 대조한다. 승인한 불량 코드별 확정 임계값 이상이면 FAIL, 그 미만의 후보가 있으면 REVIEW, 후보가 없으면 이 검사만 PASS다. |
| `anomaly_distance` | 정상 메모리와 품목, 실물/모의 구분, 전체 기준 내용 해시를 대조한다. `점수 <= pass_max`는 PASS, `점수 >= fail_min`은 FAIL, 사이는 REVIEW다. 검사기가 보고한 상태와 재계산 상태가 다르면 REVIEW다. |
| `status` | 형상 등 별도 검사기의 PASS/FAIL/UNCERTAIN/ERROR/NOT_RUN을 검증된 모델·코드 자산 및 기준 버전에 연결해 평가한다. FAIL인데 확정 불량 코드가 없으면 REVIEW다. |

후보가 없다는 사실만으로 전체 OK가 되지 않는다. 알려진 불량 후보의 무검출을 해당 검사 PASS로 인정하려면 해당 모델·코드 매핑·수집 임계값·결함별 기준의 실물 검증이 먼저 필요하다. 후보 수집 임계값을 바꿔 낮은 점수 후보를 숨기면 연결 불일치로 REVIEW다. 후보 점수를 보정된 불량 확률로 취급하지 않는다.

이상 탐지의 확정 FAIL은 NG_UNKNOWN만 만들 수 있다. 거리 점수만으로 균열/누락 등의 이름을 부여하지 않는다. 알려진 불량과 다른 검사에서 확인한 NG01~NG06은 복수로 보존한다. 검증된 후보 하나와 애매한 후보 하나가 함께 있으면 NG이며 애매한 후보의 검토 사유도 남긴다.

known_defects와 anomaly는 필수다. geometry를 비필수로 지정하려면 품목별 사유를 명시해야 한다. 비필수 검사의 미실행은 합격으로 위장하지 않으며, 비필수 검사라도 실제로 수행해 유효한 불량이 확인되면 NG 근거에 포함한다. 등록되지 않은 검사 결과는 OK를 차단한다.

## 기본 설정과 실물 기준

기본 파일은 `configs/decision/unconfigured.json`이다. 품목·모델·기준은 null, validated는 false로 두었다. 실물용 숫자나 승인 기록을 임의로 만들지 않았다.

`DecisionPolicy`는 품목·규칙 버전·자료 종류·검출 모델·촬영 기준·검사별 모델/코드 자산, 필수 여부, 기준 버전·검증 기록을 가진다. 실물 정책은 mock 모델·일반 COCO 모델·합성 정상 참조를 거부한다. 알려진 불량은 제품 불량 학습 모델, 이상 탐지는 제품 정상 참조, 물체 검출은 제품 위치 학습 모델에 연결해야 한다.

`validated=true`와 검증 기록 경로는 프로그램의 연결 계약이다. 이 값만 편집한다고 현장 검증이나 승인 절차가 완료되는 것은 아니다. 실물 도착 후 분리된 검증 데이터로 기준을 결정하고 관련 검증 기록을 연결해야 한다. 승인 권한·변경 이력·전체 제품 설정 화면은 후속 통합 범위다.

`FrameEvidence`는 같은 run_id/frame_id/product_id에 연결된 촬영 품질과 작업 영역 검사 상태, 기준 버전, 실물/모의 구분, 검증 기록을 전달한다. 현재 실제 촬영 검사기는 없다. 이 근거를 생략하면 REVIEW다. 모의 예제만 명확하게 표시된 합성 근거를 제공한다.

## 실행 및 화면 확인

프로젝트 폴더의 PowerShell에서 새 출력 경로를 지정한다.

```powershell
.\.venv\Scripts\python.exe scripts\decision.py demo --output runtime\decision-demo
.\.venv\Scripts\python.exe scripts\vlm.py ui --queue runtime\decision-demo\queue --synthetic --read-only-worker
```

모의 물체 3개가 OK / NG / REVIEW로 표시된다. NG 물체에는 NG03·NG06과 이상 검사 실패가 함께 남는다. VLM은 OFF이며 최종 판정·판정 사유·필수 검사 완료 여부·원본 검사 근거를 확인할 수 있다. `--read-only-worker`는 VLM 실행부를 시작하지 않으므로 이 예제에서 ON만 눌러도 GPU 모델이 자동 실행되지는 않는다. 실제 후속 분석을 시험하려면 해당 옵션을 생략하고 선택 분석을 다시 요청한다. 이 합성 예제에는 검토된 정상 참조가 없으므로 VLM은 참조 미설정 제한을 표시한다.

저장된 원본 결과를 동일 정책으로 재평가하려면:

```powershell
.\.venv\Scripts\python.exe scripts\decision.py evaluate --input runtime\decision-demo\raw.json --policy runtime\decision-demo\policy.json --frame-evidence runtime\decision-demo\frame-evidence.json --simulate --output runtime\decision-demo\replayed.json
```

입력·기존 출력은 덮어쓰지 않는다. 결과에는 원본 검사 입력 해시, 적용 정책 전체와 해시, 규칙 버전, 검사별 평가·사유·불량 코드·기본 상태와 판정 개수가 남는다. 원래 검사기의 UNCERTAIN을 PASS/FAIL로 고쳐 쓰지 않고, 판정 계층에서 평가한 상태를 별도 보존한다.

## 엔진 연결

```python
from mes_vision.decision import apply_policy, load_policy

policy = load_policy("configs/decision/unconfigured.json")
decided = apply_policy(raw_result, policy, frame_evidence=None)
# 기준 미설정이므로 REVIEW. raw_result는 그대로 보존된다.
```

`InspectionPipeline(..., decision_policy=policy)`로 검사 후 자동 평가를 연결할 수도 있다. `run(frame, frame_evidence_provider=provider)`의 provider는 `(frame, run_id)`를 받아 FrameEvidence를 반환한다. provider가 실패하면 사유를 남기고 보류한다. 품질 검사의 실제 측정 구현과 GPU 사용 조정은 해당 검사기를 통합할 때 함께 검증해야 한다.

기존 원시 검사 도구와의 호환성을 위해 정책을 생략한 Pipeline은 최종 판정 null을 유지한다. 판정 기능이 연결된 제품 흐름에서는 명시적인 품목 정책을 전달해야 한다. null 결과는 정상으로 해석할 수 없다. 물체·실행 결과 스키마 v1에 `decision_details`를 추가했으며 오래된 자료도 읽을 수 있다.

VLM은 필수 검사 목록이나 판정 규칙에 등록할 수 없다. ON/OFF, 응답 성공·실패·내용이 판정을 바꾸지 않는다. 후속 분석 프롬프트 v2는 저장된 최종 판정과 적용 규칙·확정 코드·판정 사유를 참고 자료로 받는다. VLM이 이를 소급 수정하는 코드 경로는 없다. 분석 시간이 기본 판정의 완료 조건이 되지 않는다.

## 검증

```powershell
.\.venv\Scripts\python.exe scripts\verify_decision.py
```

163개 기능·회귀 시험(신규 판정 시험 30개), 판정 화면 Qt 조작 14개, 기존 VLM 화면 조작 12개를 수행한다. 저장 결과 재평가 일치, 원본 보존, 미설정 보류, 임계값 경계, 복수 불량과 검사 오류, 잘못된 모델·기준·물체·좌표·클래스 매핑, 모의/실물 분리, VLM 영향 차단을 포함한다. 결과는 `artifacts/decision-check/report.json`, 화면은 해당 실행 폴더의 `decision-viewer.png`에 저장된다.

이 단계의 결과는 소프트웨어 규칙과 연결의 검증이다. 모델을 재학습하거나 실물 성능을 확인한 것이 아니며, 실제 임계값·촬영 조건·노트북 응답 시간·로봇 동작은 후속 검증 대상이다. 다음은 11 단계 로봇 제어 구조와 모의 동작이다.

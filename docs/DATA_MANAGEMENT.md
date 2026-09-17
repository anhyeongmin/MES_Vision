# 07 데이터 관리·라벨링

갱신일: 2026-09-05

로컬 PySide6 라벨링 보조 프로그램, 사진 수집 이력, 실물·촬영 회차에 따른 분할 검사, 네 종류의 데이터 내보내기를 구현했다. 기존 설치 환경을 사용하며 외부 계정·웹 서버·유료 라벨링 서비스는 필요하지 않다. 제품의 검사·로봇 운전 화면은 13 단계다.

실물과 촬영 환경은 아직 준비되지 않았다. 합성 사진으로 기능을 확인했으며 실제 라벨 품질, 학습 성능, 현장 합격 기준은 미검증이다.

## 실행

소스 프로젝트 폴더 `C:\Users\alexi\Desktop\Workspace\MES_Vision`에서 실행한다.

```powershell
.\.venv\Scripts\python.exe scripts\label_data.py
```

이미 만든 수집 프로젝트를 바로 열려면:

```powershell
.\.venv\Scripts\python.exe scripts\label_data.py --workspace collections\part-001
```

실물 사진이 없는 지금은 빈 프로젝트를 만들 수 있다. 합성 사진을 연습에 쓰면 프로젝트 종류를 반드시 **합성 시험 사진**으로 지정한다. 실제 자료와 같은 수집 프로젝트에 섞지 않는다.

## 화면 작업 순서

1. **새 수집 프로젝트**에서 아직 없는 폴더 이름과 품목 ID를 정한다. 기존 자료는 **프로젝트 열기**에서 `collection.json`을 선택한다.
2. **사진 추가**로 같은 촬영 회차의 사진을 함께 가져온다. 다른 회차는 별도로 추가한다. 원본 파일을 복사한 다음 EXIF 방향을 반영한 RGB PNG를 만든다. 원본과 정규화 이미지의 해시, 원본 경로, 변환 내역을 기록한다. 실제 촬영 시각을 파일 읽기 시각으로 대신 기록하지 않는다.
3. **물체 상자 그리기**로 모든 대상 물체를 표시한다. 실물 ID는 사진 번호가 아니라 물체 자체의 식별자다. 다른 사진에서도 같은 물체라면 같은 ID를 쓴다.
4. 물체를 선택해 관찰 상태와 메모를 입력한다. 글자·상태 변경은 즉시 편집 중인 내용에 반영된다. 디스크 반영은 **저장**으로 한다. 다른 물체로 이동해도 편집 내용은 남으며, 사진 이동·종료 시 저장 여부를 확인한다.
5. 유형을 고르고 **불량 상자 그리기**로 해당 물체 안의 불량을 표시한다. 여러 유형·여러 위치를 추가할 수 있다. 상자로 표현할 수 없는 관찰은 **영역 없이 불량 관찰 추가**로 설명한다. 잘못 그린 불량은 삭제 후 다시 그린다. 물체 상자는 다시 그리기 모드로 수정한다.
6. **물체 정보 확인** 또는 **저장**으로 일관성을 확인한다. 알려진 불량에는 라벨이, 미등록·불확실에는 메모가 필요하다. 정상으로 바꾸려면 기존 불량 라벨을 먼저 지운다. 빈 작업대는 별도 확인란을 체크한다. 흐림·심한 가림 등 사용할 수 없는 사진은 이유를 적고 제외한다.
7. 학습 / 검증 / 시험 / 별도 보관 중 하나를 선택하고 **회차 전체에 적용**한다. 같은 실물·회차·동일 픽셀이 서로 다른 분할로 들어가면 저장을 거부한다. 상세 기준은 [라벨링 지침](LABELING_GUIDE.md)을 따른다.
8. 전체 사진의 물체 누락과 불량 표시를 점검한 뒤 검토자 이름을 넣고 **검토 완료**한다. 이후 라벨을 수정하면 완료 상태가 해제된다.
9. **학습 데이터 내보내기**에서 목적과 새 출력 폴더를 고른다. 불량 모델은 실제로 영역을 표시한 클래스만 명시한다. 모든 미제외 사진은 검토 및 분할 지정을 마쳐야 한다.

휠로 확대하고 선택·이동 모드에서 화면을 끌 수 있다. 화면 확대율과 관계없이 상자는 정규화된 원본 이미지 픽셀 좌표다. 편집 중 되돌리기는 최근 30회까지 지원하며 저장하면 편집 되돌리기 목록을 비운다. 저장 이전 버전은 `history/`에 남는다.

## 내보내는 결과

| 목적 | 이미지·라벨 | 보호 규칙 |
|---|---|---|
| 물체 위치 학습 | 전체 작업대 사진 + 모든 물체 상자, COCO | train/valid/test만 사용, 미등록·불확실 회차 제외 |
| 불량 상자 학습 | 원본에서 물체별로 자른 사진 + 위치를 변환한 불량 상자, COCO | 정상은 불량 라벨 0개. 영역 없는 불량이나 선택하지 않은 유형을 가진 물체는 전체 crop 제외 |
| 정상 기준 이미지 | train에서 검토한 NORMAL 물체 이미지 | 검증·시험·미등록·불확실을 정상 기준에 넣지 않음. 특징 벡터 생성은 08 단계 |
| 미등록·불확실 별도 보관 | challenge에 속한 물체 이미지 + 관찰 상태·메모·원본 좌표 | 지도 학습용 COCO를 만들지 않음. 동반 정상 물체도 원래 상태를 유지하며, 불확실 관찰을 확정 불량 정답으로 바꾸지 않음 |

물체를 자른 영역에 다른 물체 상자가 들어오면 내보내기를 거부한다. 인접 물체가 정상 기준이나 불량 배경에 섞이는 것을 막기 위한 보수적인 검사다. 잘못된 상자를 고치거나 실제 가림 사진을 제외한 후 다시 검토한다.

물체·불량 COCO는 06 단계의 데이터 검사기로 검증한 뒤 공개 폴더로 바꾼다. 학습·검증·시험 각각에 선택 클래스가 최소 한 번 등장해야 하며 실물·회차·픽셀 중복이 없어야 한다. 이 최소 조건은 성능을 보장하는 표본 수가 아니다.

모든 결과는 `export.json`에 수집 프로젝트 ID/버전, 포함 파일과 해시, 원본 crop 좌표, 제외 사유를 남긴다. `collection-snapshot.json`에는 당시 전체 라벨을 보존한다. 원본 파일 자체는 수집 프로젝트에 보관하므로 결과 폴더만으로 원본 수집 프로젝트를 복구할 수는 없다.

기존 출력 폴더는 덮어쓰지 않는다. 검증 실패 시 최종 폴더를 만들지 않고 `.이름-building-.../export.json`에 실패 이유를 남긴다. 실패 파일은 자동 삭제하지 않는다. 읽기 실패·동시 편집 충돌 중 복사된 원본/이미지가 기록 없이 남을 수도 있으며, `collection.json`에 없는 파일은 내보내지 않는다.

## 명령으로 관리하기

아래 명령은 UI와 같은 저장·검사 코드를 사용한다.

```powershell
.\.venv\Scripts\python.exe scripts\data_collection.py create collections\part-001 --product-id part-001
.\.venv\Scripts\python.exe scripts\data_collection.py import collections\part-001 incoming\frame-001.png --session train-session-001
.\.venv\Scripts\python.exe scripts\data_collection.py assign-session collections\part-001 --session train-session-001 --split train
.\.venv\Scripts\python.exe scripts\data_collection.py inventory collections\part-001

# 전체 사진의 검토·분할 완료 후
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\part-001 --role object_detector --output datasets\object_detector
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\part-001 --role known_defect_detector --defect-codes NG03 NG06 --output datasets\known_defect_detector
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\part-001 --role normal_bank --output datasets\normal-reference-001
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\part-001 --role challenge --output datasets\challenge-001
```

NG03/NG06은 명령 예시다. 실제 표시된 클래스에 맞춰 지정해야 한다. 기본 물체·불량 출력 경로는 기존 [학습 설정](TRAINING.md)과 연결된다. 다른 버전 폴더로 내보내면 학습 설정의 `dataset_dir`를 그 위치로 수정한다. 합성 출력은 `kind=synthetic`을 유지하므로 실물용 설정으로 학습할 수 없다.

## 이력·동시 편집과 한계

- `collection.json`을 기준으로 읽으며 매 저장 전 이전 JSON을 `history/revision-....json`으로 남기고 임시 파일을 교체한다.
- 파일 잠금과 버전 검사로 다른 창이 저장한 내용을 덮어쓰지 않는다. 충돌 시 미저장 내용을 따로 기록한 후 다시 연다. 자동 병합이나 이력 복구 화면은 이번 범위에 없다.
- 원본 사진, 수집 프로젝트 폴더 전체, export 결과를 함께 백업한다. 이력은 디스크 고장에 대비한 외부 백업을 대체하지 않는다.
- 동일 실물 여부는 사람이 입력한 ID를 신뢰한다. 이름을 다르게 입력한 동일 실물이나 근접 연속 프레임을 AI가 자동 판별하지 않는다. 검토 완료도 사람의 선언이며 실제 불량 판정의 정확성을 보증하지 않는다.
- 일반 D405 사진은 기존 파일 입력 경로를 사용한다. 실제 D405 촬영 버튼과 최종 제품 UI는 후속 작업이다.
- 새로운 모델·라벨링 서비스는 추가하지 않았다. 기존 PySide6와 이미 설치·고지된 filelock을 사용하며 최종 설치 파일의 배포 검토는 16 단계에서 진행한다.

## 검증

```powershell
.\.venv\Scripts\python.exe scripts\verify_data_management.py
```

결과: [보고서](../artifacts/data-management-check/report.json). 기존 69개 + 데이터 관리 20개, 총 89개 기능·회귀 시험 통과. 네 종류 내보내기와 실제 Qt 위젯 이벤트 15개 점검을 통과했다. 화면은 offscreen으로 렌더링하고 눈으로 확인했다. 오프스크린 미리보기에는 PC에 설치된 맑은 고딕을 읽어 사용하며 글꼴 파일을 제품에 복사하지 않는다. 실제 모니터·노트북에서 사람이 조작한 사용성 검증은 아니다. 이번 단계에서 GPU 재학습이나 제품 모델 교체는 하지 않았다.

다음은 08 정상 특징 등록·이상 점수·이상 영역 출력 프로그램이다. 임계값과 실물 성능은 데이터 확보 후 확정한다.

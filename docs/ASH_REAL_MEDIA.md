# ASH 실사진·영상 학습 준비

사진은 기존 라벨링 프로그램의 **사진 추가**, 영상은 새 **영상 추가**로 가져온다. 영상에서 추출한 사진도 사람이 물체·불량 영역을 확인하고 검토 완료해야 학습 자료로 내보낼 수 있다. 파일명이나 AI 예측을 정답으로 자동 확정하지 않는다.

## 준비된 작업 공간

프로젝트 루트에서 다음 명령으로 연다.

```powershell
.\.venv\Scripts\python.exe scripts\label_data.py --workspace collections\ash-real-overview-v1
.\.venv\Scripts\python.exe scripts\label_data.py --workspace collections\ash-real-detail-v1
```

- `overview`: 높은 위치에서 촬영한 전체사진. 모든 물체 외곽을 ASH로 표시한다.
- `detail`: 각 물체의 근접사진. 물체 외곽과 실제 관찰되는 NG01~NG06 불량 영역을 표시한다. 정상은 NORMAL로 확인하고 불량 영역은 비운다. 모르는 현상은 미등록/불확실로 기록한다.
- 두 공간은 빈 실물용 프로젝트다. 합성 사진은 넣지 않는다. 출력 재료는 잠허 PETG White 물체와 Olive Green 바닥, 카메라는 U20CAM-720p다.

## 영상 가져오기

영상 선택 → 촬영 회차 ID → 추출 간격 → 최대 장수 순서다. 기본은 약 1초마다 최대120장이다. 프레임 간격은 영상의 명목 FPS로 계산하므로 가변 프레임률 영상의 정확한 시간 간격은 보장하지 않는다. 최대 장수에 도달하면 영상 뒤쪽은 처리하지 않는다. 한 영상은 한 촬영 역할로 수집하고, 전체/근접 장면이 섞인 영상은 해당 역할과 무관한 프레임을 제외한다.

원본 영상은 한 번 보관하고 SHA256, 프레임 번호, 명목 FPS와 추정 시각을 각 프레임에 기록한다. 센서 시각이나 로봇 촬영 증거로 취급하지 않는다. 동일 바이트의 영상을 같은 프로젝트에 다시 넣는 것은 거부한다. 재인코딩한 복제 영상까지 자동 식별하지는 않는다.

같은 영상의 프레임은 모두 같은 회차에 속하며 회차 단위로 학습/검증/시험을 지정한다. 전체와 상세 프로젝트 사이에도 같은 실물은 같은 분할을 사용해야 한다. 서로 다른 수집 프로젝트 사이의 ID 일관성은 현재 자동 검사하지 않으므로 두 프로젝트에서 분할을 맞춘다. 이미 학습에 사용한 실물을 다른 영상에서 찍었다고 시험용으로 바꾸지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts\data_collection.py import-video collections\ash-real-detail-v1 incoming\detail.mp4 --session real-session-001 --interval 1 --max-frames 120
```

기존 백그라운드 파일 작업으로 실행해 화면 이벤트를 막지 않는다. 코덱 지원은 설치된 OpenCV에 따른다. 영상 작업 도중 취소 기능은 현재 없으며, 손상 영상의 조기 종료를 정상 영상 끝과 완전히 구별하지 못할 수 있으므로 추출된 프레임을 확인한다.

## 검토 후 내보내기와 추가 학습

라벨링 창에서 검토자 확인과 분할 지정을 마친 뒤 다음과 같이 내보낸다. 기존 폴더를 덮어쓰지 않는다.

```powershell
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\ash-real-overview-v1 --role object_detector --output datasets\ash-real-v1\overview-objects
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\ash-real-detail-v1 --role object_detector --output datasets\ash-real-v1\detail-objects
.\.venv\Scripts\python.exe scripts\data_collection.py export collections\ash-real-detail-v1 --role known_defect_detector --defect-codes NG01 NG02 NG03 NG04 NG05 NG06 --output datasets\ash-real-v1\detail-defects
```

`configs/training/ash-real-v1/`에 세 역할의 추가 학습 설정을 준비했다. 전체/상세 물체는 기존 v2, 불량은 v3 합성 학습 가중치에서 시작한다. 가중치 SHA를 확인했다. 실사진용 `synthetic=false`이며 아직 데이터가 없으므로 학습은 시작하지 않았다. 30 epochs, batch2/누적4, BF16, 학습률5e-5는 시작값이며 실물에서 선정한 최적값이 아니다.

```powershell
.\.venv\Scripts\python.exe scripts\training.py validate --config configs\training\ash-real-v1\detail-defects.json
.\.venv\Scripts\python.exe scripts\training.py train --config configs\training\ash-real-v1\detail-defects.json --output runs\ash-real-detail-defects-001
```

다른 두 역할도 해당 설정 파일로 별도 실행한다. 실물/회차/중복 픽셀 분리와 검토가 완료되어야 한다. 학습 후 검증 자료에서 모델과 후보 기준을 선정하고 별도 시험 자료로 평가한다. 새 모델을 운영 모델로 자동 교체하지 않는다.

## STL 추가 학습 판단

색 이름만으로 실제 카메라 RGB·반사·적층 무늬는 확정되지 않는다. 기존 색 비교137장에서는 위치 검출이 되었고 테두리 오검출이 남았다. 현재는 추가 합성 학습보다 실사진 도입을 우선한다. 실제 오류를 확인한 뒤 필요한 합성 조건을 보완하는 것은 가능하다. 이번 변경에서 재학습이나 운영 판정 기준 변경은 하지 않았다.

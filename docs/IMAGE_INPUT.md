# 04 이미지 입력 기능

**2026-09-06 변경:** 설정된 `D405Source(settings)`와 운영 카메라 작업자는 RealSense SDK로 실제 RGB 영상을 수신한다. 설정 없는 호출은 계속 미설정 상태다. [운영 사용법](OPERATOR_REVISION.md). 아래 D405 미구현 설명은 04단계 당시 기록이다.

갱신일: 2026-09-05

사진·폴더·녹화 영상을 같은 RGB 프레임 형식으로 제공한다. 한 번에 한 프레임씩 읽으므로 영상을 통째로 메모리에 적재하지 않는다. 이 단계는 입력 기능이며, AI 판정·제품 UI·실제 D405 연결은 아직 포함하지 않는다.

## 구현 파일

- [입력 모듈](../src/mes_vision/inputs/sources.py): 공통 프레임·상태, 사진·폴더·영상, D405 미설정 어댑터
- [입력 확인 도구](../scripts/check_input.py): 상태·프레임 메타데이터 표시, 첫 RGB 프레임 저장
- [검증 실행](../scripts/verify_inputs.py), [파일 기반 시험](../tests/test_inputs.py)
- [검증 결과](../artifacts/input-check/report.json)

## 입력 계약

| 항목 | 규칙 |
|---|---|
| 픽셀 | `uint8`, H×W×3, RGB 순서. 모델용 축소·정규화 없이 제공 |
| 버퍼 | 프레임마다 소유 버퍼를 복사하고 쓰기 금지 플래그 설정. 이후 영상 디코딩이 이전 프레임을 바꾸지 않음. 수정 필요 시 소비자가 복사 |
| 좌표 | `input_rgb_pixels`: 제공된 RGB 영상의 왼쪽 위 (0,0), 오른쪽 x, 아래쪽 y. mm 좌표 아님 |
| 해상도 | 프레임별 width/height. 인코딩된 크기는 별도 `encoded_size` |
| 식별 | 입력 인스턴스마다 새 session ID, 프레임마다 session ID + 증가 순번. 입력 재시작 시 이전 ID 재사용 안 함 |
| 읽은 시각 | UTC `read_at_utc`: 프로그램이 RGB 프레임을 받아 생성한 시각 |
| 실제 촬영 시각 | 파일에서는 `captured_at_utc=null`. 파일 수정 시각·EXIF 시각·영상 재생 시간을 실제 UTC 촬영 시각으로 대체하지 않음 |
| 영상 시간 | PTS×time_base를 `media_time_seconds`로 저장. 없으면 null, 0초 시작을 가정하지 않음 |
| 입력 출처 | `image` / `folder` / `video` / `d405`, 원본 로컬 파일 URI. 현재 모든 파일 입력은 `is_live=false` |
| 변환 기록 | EXIF 방향 보정, 회색조→RGB, 영상 픽셀 형식→RGB 등을 `transformations`로 기록 |

사진의 EXIF 방향은 반영한다. 예를 들어 인코딩 12×7, EXIF 90도 회전이면 RGB는 7×12다. 이후 검출·원본 추출·화면 오버레이는 **이 RGB 픽셀 공간을 함께 사용**해야 한다. 인코딩 파일의 방향 보정 전 좌표를 혼용하면 안 된다.

영상은 인코딩된 프레임 방향을 유지한다. 휴대전화 영상의 회전 메타데이터를 적용하지 않는다. 필요하면 촬영·내보내기 시 방향을 고정하고, 추후 UI에서도 동일한 RGB 영상을 표시한다. 이미지 ICC 프로파일 변환·색상 교정은 하지 않는다. 색상에 따른 합격 기준은 확정된 촬영 환경에서 따로 검증해야 한다.

## 상태와 오류

`created → ready → frame → frame … → end`, 또는 `error`로 종료한다. `close()` 후에는 `closed`이며 같은 인스턴스를 재사용하지 않는다. 새 입력은 새 인스턴스를 만든다.

- `read()`는 `InputEvent`를 반환한다. `status=frame`일 때만 `event.frame`을 소비한다.
- 종료·오류·미설정·닫힘 이벤트의 frame은 항상 null이다. 이전 프레임으로 검사를 재실행하면 안 된다.
- 폴더의 파일 목록은 열 때 고정하며 숫자 순서로 정렬한다(`part1`, `part2`, `part10`). 새 파일은 새 입력에서 읽는다. 하위 폴더는 명시적으로 선택한다.
- 폴더에서 손상·삭제된 파일을 만나면 전체 입력을 오류 상태로 멈춘다. 자동 건너뛰기나 반복 재생을 하지 않는다.
- 폴더 확장자: JPG/JPEG/PNG/BMP/TIF/TIFF/WEBP. 다른 확장자는 목록에서 제외한다. 지원 이미지가 0개면 `EMPTY_FOLDER`다.
- 8비트 RGB·회색조·팔레트 등 불투명 사진을 읽는다. 투명 픽셀, 다중 페이지/애니메이션, 16비트 PNG/TIFF, 고비트/CMYK 등의 미지원 색상은 오류로 처리한다.
- 영상은 PyAV로 첫 번째 영상 트랙을 순차 디코딩한다. 8비트 입력만 받으며 네트워크·외부 재생 목록 입력은 제공하지 않는다. 확인한 컨테이너/코덱은 MKV/FFV1과 MP4/MPEG4다. 모든 코덱을 시험한 것은 아니다.
- 디코더가 보고한 오류와 정상 반복자 종료를 구분한다. 단, 디코더가 정상 EOF로 받아들이는 잘린 영상까지 완전성 검사를 보장하지 않는다. END는 검사 합격이나 촬영 전체 분량 보증이 아니다.
- 파일 읽기는 동기식이며 단일 소비자용이다. 향후 UI는 작업 스레드에서 읽고 이벤트만 UI에 전달한다. 이 단계에는 재생 속도 제어·탐색·취소·검사 큐가 없다.

## D405 연결 경계

`D405Source`는 장비 준비 전 자리표시자다. 즉시 `not_configured / D405_NOT_CONFIGURED`를 반환하며 가짜 프레임, 장비 검색, SDK 설치 또는 연결을 수행하지 않는다.

현장 연결 단계에서 공통 이벤트 계약을 유지하면서 SDK 시작/중지, 연결 끊김·시간 초과, RGB 프레임 복사, 장치 시각/시리얼/내부 파라미터/촬영 모드를 구현한다. 실제 카메라 프레임에는 라이브 출처 정보를 넣는다. 깊이를 쓸지는 현장 조건 확인 후 결정한다. 작업대 크기 입력은 좌표 보정을 대체하지 않는다.

## 실행 방법

프로젝트 폴더의 PowerShell에서 실행한다. 새 결과 폴더 이름을 지정해야 하며 기존 결과를 덮어쓰지 않는다.

```powershell
# 사진: 상태·메타데이터·RGB 미리보기 저장
.\.venv\Scripts\python.exe scripts\check_input.py image 'D:\photos\part.jpg' --output artifacts\my-image-input

# 폴더: 기본 10프레임, 하위 폴더는 선택 사항
.\.venv\Scripts\python.exe scripts\check_input.py folder 'D:\photos' --recursive --limit 100 --output artifacts\my-folder-input

# 영상: 원래 프레임을 순서대로 최대 100장 읽기
.\.venv\Scripts\python.exe scripts\check_input.py video 'D:\videos\demo.mp4' --limit 100 --output artifacts\my-video-input

# D405 미설정 상태 확인: 예상 종료 코드 1
.\.venv\Scripts\python.exe scripts\check_input.py d405

# 입력 기능 검증 및 결과 갱신
.\.venv\Scripts\python.exe scripts\verify_inputs.py
```

출력은 `events.jsonl`, `first-frame.png`, `summary.json`이다. 콘솔에도 입력 상태가 표시된다. 프레임 수 제한으로 멈췄으면 `limit_reached`이며 실제 영상 끝으로 기록하지 않는다. `error`·`not_configured`는 종료 코드 1이다. 종료 시 파일 핸들을 해제한다.

제품 화면은 13 단계에서 이 모듈의 상태·프레임에 연결한다. 현재 도구의 파일 읽기 성공은 AI 검사 완료를 뜻하지 않으며 로봇 동작 기능이 없다.

## 검증 범위

생성한 파일로 RGB 채널·원본 크기·EXIF 방향·시각 구분, 자연 정렬·재시작 ID, 영상 전체 순서·PTS·버퍼 보존·파일 해제, 손상/삭제/빈 입력·미지원 색상·D405 미설정, 실행 도구의 종료 코드·결과 파일을 검증했다. 실제 물품, D405, Dobot 및 노트북은 사용하지 않았다.

05 검출·검사 연결 코드도 이후 완료했다. [검사 연결 사용법](INSPECTION_PIPELINE.md)에서 원본 RGB 프레임과 검출 좌표·물체별 원본 추출의 연결을 확인할 수 있다. 06 학습 프로그램도 완료했으며 현재 다음 작업은 07 데이터 관리·라벨링 준비다.

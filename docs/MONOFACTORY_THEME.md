# MonoFactory 스타일과 Dark / Light

사용자가 지정한 `C:/Workspace/MES_UI/AdvancedFactory/Launcher/public/index.html`의 색상과 로고 교체 방식을 MES Vision의 기존 PySide6 화면에 적용했다. 기존 검사 운영·품목 설정·장비/보정·검사 이력·VLM 메뉴는 유지한다. Launcher의 로그인이나 외부 서비스 연결 기능을 복사한 것은 아니다.

- 상단 **Light / Dark** 버튼을 누르면 즉시 전환한다.
- **환경설정 → 일반 → 화면 테마**에서도 선택하고 저장할 수 있다. 취소하면 변경하지 않는다.
- 다음 실행에도 마지막 테마를 유지한다. 기존 설정 파일에는 Light를 기본으로 추가하며 언어·글자 크기·창 위치 설정을 보존한다.
- 넓은 창에서는 왼쪽 메뉴 상단에 Launcher 원본 로고를 표시한다. 작은 창에서는 상단 왼쪽의 축소 로고와 메뉴 선택 상자를 사용한다.
- 원본 `monofactory-brand-white.png` / `monofactory-brand-navy.png`를 테마에 따라 교체한다. 이미지 비율을 유지하며 원본 그림을 수정하지 않았다.
- 흰색/남색 화면 배경, 하늘색 선택 상태, 둥근 입력창·버튼·표·탭을 적용했다. 실제 촬영 이미지의 픽셀과 불량 영역 좌표는 테마로 변경하지 않는다.

구현: `src/mes_vision/theme.py`, `operation/preferences.py`, `operation/preferences_dialog.py`, `operation/window.py`. 로고는 `src/mes_vision/assets/brand/`에 포함돼 있어 참고 폴더가 없어도 실행할 수 있다. 글꼴은 기존 설치 글꼴과 다국어 표시 정책을 유지한다.

장비 연결이나 검사·로봇·VLM 실행 명령은 테마 전환에 포함되지 않는다. 검증은 별도 임시 운영 폴더와 장비 미연결 상태에서 진행한다. 기존 실행 중인 사용자 앱은 강제로 종료하거나 재시작하지 않는다.

검증 화면과 결과: `artifacts/ui-theme-alignment/`. 이 작업은 프로그램 스타일 변경이며 실제 장비 성능 검증이나 설치 배포 완료를 의미하지 않는다.

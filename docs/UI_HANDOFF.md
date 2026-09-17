# MES Vision UI 개편 — 백엔드 인계 문서

작성: 2026-09-14 · 갱신: 2026-09-15 · 대상: `MES Vision.cmd`로 실행되는 검사 운영 창
UI 레이아웃·스타일과 실시간 패널 배선까지 변경했습니다. 카메라·검사 파이프라인·로봇 제어
로직은 건드리지 않았습니다.

> 이 문서에는 초판 내용을 **정정한 절이 두 곳** 있습니다 (§2.2 보정 좌표계, §9 검증 상태).
> 초판만 읽고 작업을 시작했다면 그 두 절을 먼저 확인하십시오.

---

## 0. 작업 분담 — 누가 무엇을 맡는가

분담 기준은 모듈이 아니라 **하드웨어 없이 옳음을 검증할 수 있는가**입니다.
UI 작업자(Claude)는 저장소를 읽고 쓸 수 있고, 저장소를 자기 환경으로 가져와
**오프스크린 Qt로 기존 테스트를 실행할 수 있습니다** (§9). 할 수 없는 것은 torch가 필요한
경로, 실물 `artifacts/`·`models/` 자산이 필요한 테스트, 그리고 카메라·로봇이 붙어야
확인되는 모든 것입니다. 그 선을 기준으로 나눴습니다.

| 항목 | 담당 | 근거 |
|---|---|---|
| `live_view` 피드 배선 | **UI 작업자 — 완료** | §3.1. `tick()`에서 급전, 테스트 포함 |
| `verdict`/`context` 훅 호출 | **백엔드(GPT)** | 판정·장비 상태를 아는 쪽이 불러야 함 (§4) |
| 정사각 크롭 유틸리티 | **불필요 — 이미 있음** | `manual_capture.Rectifier.apply()` 재사용 (§2.1) |
| 획득 단계 크롭 삽입 (`operation/camera.py`) | **백엔드(GPT)** | 실제 프레임 흐름을 돌려봐야 함 |
| §2.2.1 `manual_capture` 가드 충돌 해소 | **백엔드(GPT)** | 두 경로를 동시에 실행해 확인해야 함 |
| 로봇 모션 · `map_targets` · Dobot 제어 | **백엔드(GPT)** | 검증 불가 상태로 손대면 로봇이 잘못된 좌표로 이동 |
| RF-DETR · VLM 추론 경로 | **백엔드(GPT)** | 모델 실행 불가 |
| **720×720 기준 보정점 재측정** | **사람(운영자)** | 로봇 앞에서 물리적으로 해야 함 |

§9의 미검증 항목 체크리스트도 같이 보십시오.

---

## 1. 변경된 파일

| 파일 | 상태 | 내용 |
|---|---|---|
| `src/mes_vision/theme.py` | 교체 | MES 디자인 토큰, 라운드 3px, 그림자 없음, 새 objectName 세트 |
| `src/mes_vision/operation/shell.py` | 신규 | Sidebar / PageHeader / ContextBar / VerdictBanner / ViewArea / SquareHolder |
| `src/mes_vision/assets/brand/chevron.svg` | 신규 | 콤보박스 드롭다운 화살표 (네이티브 화살표만 테마를 안 탔음) |
| `src/mes_vision/operation/image_view.py` | 수정 | `ImagePanel`에 제목행·정사각 프레임 추가, 컨트롤 한 줄로 통합 |
| `src/mes_vision/operation/window.py` | 수정 | 헤더·사이드바 재구성, 전체화면 시작, `make_live()` 재배치 |
| `src/mes_vision/operation/responsive.py` | 수정 | 최대화 상태를 창 크기로 되돌리지 않도록 `fit_available_screen()` 보호 |
| `src/mes_vision/station/window.py` | 수정 | `make_live()`를 정사각 뷰포트 3분할로 재배치 |
| `src/mes_vision/locales/{en,zh-CN,th}.json` | 수정 | 신규 문자열 추가, 3개 언어 키 집합 1125개로 일치 |
| `tests/test_live_viewports.py` | 신규 | 실시간 화면 배선·정사각 기하 회귀 테스트 (§9.5) |
| `tests/test_laptop_ui.py` | 수정 | 사이드바 개편에 맞춰 단언 대상 변경 (§9.4) |

원본 백업은 전부 `docs/ui-proposal/*_original_backup.py` 에 있습니다. 이 폴더는 git 저장소가 아니므로 백업이 유일한 복구 수단입니다.

### 실행 경로 주의
`MES Vision.cmd` → `scripts/desktop.py` → **`StationWindow`** 입니다.
`StationWindow`는 `DesktopWindow`(operation/window.py)를 상속하지만 **`make_live()`를 오버라이드**합니다. 따라서 실시간 검사 화면을 고칠 때는 `station/window.py` 쪽을 봐야 합니다. `operation/window.py`의 `make_live()`는 `DesktopWindow`를 직접 쓰는 경우에만 적용됩니다 (양쪽 모두 같은 3분할 구조로 맞춰 두었습니다).

---

## 2. 반드시 정해야 하는 것 — 정사각 1:1 크롭

작업대가 정사각이라 **인식도 1:1**로 가기로 확정됐습니다. 표시만 자르는 게 아니라 **파이프라인 입력부터** 정사각입니다.

### 2.1 크롭 위치
`operation/camera.py`에서 프레임 획득 직후 중앙 정사각으로 크롭하고, 그 뒤 검출·이상탐지·VLM이 모두 이 프레임만 보도록 합니다.

- U20CAM 1280×720 → 중앙 720×720
- D405에도 같은 규칙을 적용할지 확인 필요

**새로 짜지 마십시오 — 같은 로직이 이미 있습니다.**
`station/manual_capture.py`의 `Rectifier.apply(rgb, square=True)`가 왜곡 보정 후 정확히
이 중앙 정사각 크롭을 수행하고, 크롭 박스를 `(x1, y1, x2, y2)`로 함께 반환합니다.

```python
full = cv2.remap(rgb, *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
w, h = self.size
side = min(w, h)
box = ((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2) if square else (0, 0, w, h)
```

수동 도착 검사(`MES Vision Manual.cmd` → `scripts/manual_inspection.py`, 그리고 품목 설정 화면의
`수동 도착 검사` 버튼)가 이 경로를 씁니다. 실시간 검사도 같은 `Rectifier`를 재사용해
두 경로의 크롭 결과가 어긋나지 않게 하십시오. 반환되는 `box`가 2.2의 오프셋 계산에
그대로 쓸 수 있는 값입니다.

### 2.2 보정 좌표계 — 이전 판 문서의 제안은 코드상 불가능합니다

> **정정 안내.** 이 문서 초판은 "(a) 크롭 시점에 280px 오프셋을 흡수하고 기존 보정값을
> 유지"를 선택지로 제시했습니다. **이 방법은 쓸 수 없습니다.** `station/calibration.py`를
> 읽고 확인한 결과이며, 아래가 정확한 내용입니다.

`station/calibration.py`의 `MountedCalibration`이 두 가지를 강제합니다.

```python
# __init__ — 보정 맵은 변형되지 않은 원본 overview 픽셀이어야 함
require(not self.context.transformations,
        'Moving-camera maps must use original overview pixels')

# verify_context — 보정 당시의 image_size와 정확히 일치해야 함
require((c['serial'], c['mount_revision'], c['acquisition_revision'],
         c['robot_base_id'], c['tool_frame_id'], (c['width'], c['height']))
        == (expected.camera_id, ..., expected.image_size),
        'Camera/mount/acquisition/base/tool context changed')
```

즉 **오프셋을 끼워 넣어 기존 보정값을 재활용하는 경로는 런타임에서 차단됩니다.**
실제 선택지는 두 개입니다.

| | 내용 | 대가 |
|---|---|---|
| **A (권장)** | 획득 단계에서 잘라 시스템이 1280×720을 아예 보지 않게 하고, **보정을 720×720 기준으로 재측정** | 기존 보정 번들 전부 무효. 실장비에서 보정점 재측정 필요 |
| **B** | 파이프라인은 1280×720 유지, 크롭은 **화면 표시에만** 적용 | "인식도 1:1" 결정과 어긋남 |

**A를 권장합니다.** 크롭이 `Frame` 생성 이전에 일어나면 `transformations`가 빈 상태로
유지되고, `image_size`가 (720, 720)인 새 보정 번들과 모든 `require`가 일관되게 통과합니다.
오프셋 보정 같은 특수 처리가 코드 어디에도 남지 않습니다.

기존 보정 번들을 그대로 두면 앱이 조용히 틀리는 게 아니라 **위 `require`에서 명시적으로
실패**합니다. 이건 의도된 안전장치이므로, 그 에러가 뜨면 재측정하라는 뜻으로 읽으십시오.

### 2.2.1 A 선택 시 같이 풀어야 하는 충돌

`station/manual_capture.py:34`:

```python
require(not frame.transformations, '이미 보정된 영상은 다시 보정하지 않습니다.')
```

수동 도착 검사는 **아직 보정되지 않은** 프레임을 받아 자기가 `Rectifier`를 적용하는 전제입니다.
실시간 경로가 획득 단계에서 보정·크롭한 프레임을 넘기기 시작하면 이 가드에 걸립니다.
`save_capture` 호출 지점이 원본 프레임을 받도록 분리하거나, 이미 보정된 프레임을
받아들이는 경로를 따로 만들어야 합니다. **가드를 그냥 지우지 마십시오** — 이중 보정을
막는 장치입니다.

### 2.3 ROI 기본값
크롭 프레임 전체가 기본 ROI가 되도록 조정. 별도 ROI는 그 안에서만 지정.
`operation/quality.py`의 `in_workspace()`와 `station/vision.py`의
`overview_workspace` / `detail_workspace`가 같은 좌표계를 쓰는지 함께 확인하십시오.

### 2.4 부수 효과
RF-DETR 입력 픽셀이 44% 줄고 letterbox 패딩이 사라집니다. 검사 주기가 내려갈 것으로 보이지만 **실측 전에는 수치를 문서에 쓰지 마십시오.**

---

## 3. 뷰포트 3개

`station/window.py`의 `make_live()`가 만드는 세 개의 정사각 패널입니다.

| 속성 | 제목 | 현재 상태 |
|---|---|---|
| `self.live_image` / `self.canvas` | 저장된 전체사진 | 기존과 동일. placeholder·`canvas.selected`→`select_track` 연결 유지 |
| `self.live_view` | 왜곡보정 실시간 영상 | **비어 있음 — 신규.** 프레임을 넣어야 함 |
| `self.detail_image` | 선택 물체의 상세사진 | 기존과 동일 |

### 3.1 `self.live_view` — 배선 완료, 아직 원본 프레임입니다

`StationWindow.tick()`이 카메라 프레임을 이 패널에 넣습니다. 별도 미리보기 창에
넣던 경로 바로 옆입니다.

```python
self.live_view.canvas.set_rgb(frame.rgb)   # numpy HxWx3 uint8
```

패널은 표시할 때만 중앙 정사각으로 자르며, **넘겨준 프레임 자체는 건드리지 않습니다.**

**남은 일 — 왜곡 보정.** 지금 들어가는 것은 보정되지 않은 원본이라 제목을
`실시간 영상 · 원본`으로 두었습니다. `manual_capture.Rectifier`로 감싼 뒤에
제목 문자열을 `왜곡보정 실시간 영상`으로 바꾸십시오. 이 문자열은 4개 언어에 이미
들어 있습니다. **보정 전에 제목만 바꾸지 마십시오** — 검사 담당자가 화면을 믿습니다.

프레임이 안 들어와도 크래시는 없고 placeholder 문구가 표시됩니다.
회귀 방지 테스트: `tests/test_live_viewports.py::test_live_pane_receives_camera_frames`.

### 3.2 공통
- 세 패널의 `화면 맞춤 / 1:1 / − / + / %` 버튼과 표시선 토글은 기존 `ImageCanvas` 동작에 연결돼 있어 추가 배선이 필요 없습니다.
- 패널 폭은 **높이에서 역산**됩니다: 좌측 = 높이 − 58, 우측 = (높이 − 8) ÷ 2 − 58. 창 리사이즈마다 재계산됩니다.

---

## 4. 새로 생긴 UI 훅 — 호출해야 값이 채워집니다

### 4.1 판정 배너 `self.verdict`
```python
self.verdict.set_verdict('ng', 'NG', '#1042 · NG01 표면 이물 · 2개소',
                         'ASH-TRAY-02 · v7 · 신뢰도 0.93 · 이상 점수 0.78',
                         '2026-09-14 14:07:33.412')
self.verdict.set_verdict('ok', 'OK', '#1041 · 기준 이내')
self.verdict.set_verdict('review', '보류', '#1044 · 가림 확인 필요')
self.verdict.set_idle('물체를 선택하면 판정이 표시됩니다.')
```
state는 `'ok' | 'ng' | 'review' | 'idle'`. **현재 아무 데서도 호출하지 않아 항상 idle입니다.**
호출 위치 후보: `select_track` / `select_row`(물체 선택), 판정 확정 시.

### 4.2 컨텍스트 바 `self.context`
```python
self.context.set_value('품목', 'ASH-TRAY-02 상면')
self.context.set_value('버전', 'v7 · 2026-09-12')
self.context.set_value('카메라', 'U20CAM 720×720', 'ok')   # state: '' | 'ok' | 'warn'
self.context.set_value('로봇', 'Magician 대기', 'ok')
self.context.set_value('모델', 'RF-DETR S · DINOv2')
```
키는 `품목 / 버전 / 카메라 / 로봇 / 모델` 다섯 개 고정. **현재 전부 `—` 입니다.**
호출 위치 후보: `apply_product`, 카메라·로봇 연결 상태 변경, 모델 준비 완료 시.

### 4.3 사이드바 `self.sidebar`
- `set_user(name, role)` — 미호출, 기본값 표시 중
- 하단 CTA는 `prepare_engine`에 연결돼 있음 (기존 '모델 준비'와 동일)

---

## 5. 전체화면 · 단축키

- 시작 시 **최대화**됩니다 (`operation/window.py` `__init__`)
- **F1~F5** — 사이드바의 5개 화면으로 바로 이동 (라벨에 적힌 그대로)
- **F11** — 테두리 없는 전체화면 토글, **Esc** — 전체화면 해제
- '창 배치 기억' 설정을 켜고 저장된 위치가 있으면 그 값이 우선입니다 (설정이 무의미해지지 않도록)
- `responsive.py`의 `fit_available_screen()`이 최대화·전체화면 상태에서는 `resize()`를 건너뜁니다. 이 가드를 지우면 창이 다시 작아집니다.

### F1~F5 배선
`self.page_shortcuts`가 `self.nav.setCurrentRow(index)`를 호출합니다. 사이드바 선택,
`QStackedWidget` 페이지, 페이지 헤더 제목이 모두 이 시그널 하나에 물려 있으므로
**화면 전환은 항상 `self.nav.setCurrentRow()`로 하십시오.** `self.pages.setCurrentIndex()`를
직접 부르면 사이드바와 헤더가 따라오지 않아 화면과 좌측 강조 표시가 어긋납니다.

화면 개수를 바꾸면 `PAGES` 리스트만 수정하면 됩니다. 단축키는 그 길이만큼 자동 생성됩니다
(F1부터 순서대로). 모달 다이얼로그가 떠 있는 동안에는 동작하지 않습니다 — 의도된 동작입니다.

---

## 6. 기존 코드와의 호환 — 건드리지 않은 것

의도적으로 그대로 둔 부분입니다. 리팩터링 시 주의하십시오.

- **`self.nav`는 여전히 `QListWidget`입니다.** 숨겨진 호스트(`self._nav_host`)에 들어 있고, 보이는 것은 `self.sidebar`입니다. 둘은 `currentRowChanged` ↔ `set_current`로 양방향 동기화됩니다.
  - 이유: `responsive.py`의 `ResponsiveWindow`와 `init_compact_navigation()`이 `self.nav.item(i)`, `.count()`, `.currentRow()`, `.setCurrentRow`, `.currentRowChanged`를 직접 사용합니다.
  - `self.nav`를 없애려면 `responsive.py`도 같이 고쳐야 합니다.
- 네비게이션 항목 이름을 코드에서 바꾸면 `self.refresh_nav_labels()`를 호출해야 사이드바와 페이지 제목에 반영됩니다. `StationWindow.__init__`이 그렇게 하고 있습니다.
- `self.title_label`은 `PageHeader`의 제목 라벨, `self.brand_logo`는 사이드바 심볼 라벨을 가리킵니다.
- 이름·동작이 그대로인 것: `start / pause / unload / auto / sort_enabled / live_state / selection / live_details / vlm_details / object_table / recheck_button / pick_button / counts / readiness_toggle / readiness_read / active_label / vlm / settings_button / theme_button / global_stop`

---

## 7. 다국어 — 4개 언어 모두 채웠습니다

`ko / en / zh-CN / th` 네 언어이고, 이번에 추가된 문자열 18개를 **en·zh-CN·th 세 파일에 모두 넣었습니다.** 세 파일의 키 집합이 동일한지 확인했습니다 (`validate_catalog()`가 검사하는 조건, 각 1124개).

추가한 항목: `설정`, `기록 · 분석`, `선택 물체 판정`, `선택 물체의 상세사진`, `왜곡보정 실시간 영상`, `중앙 1:1 크롭`, `카메라 영상을 연결하면 표시합니다.`, 페이지 설명 5종, 그리고 이전부터 누락돼 있던 station 문자열 6종(`수동 도착 검사`, `카메라 연결 중…` 등).

> **검수 필요**: 중국어·태국어 번역은 제가 작성한 것이라 원어민 검수를 받지 않았습니다. 특히 태국어는 확인해 주십시오. 영어는 문제없을 것으로 봅니다.

새 문자열을 추가할 때는 **세 파일에 동시에** 넣어야 합니다. 한 파일만 넣으면 `validate_catalog()`가 실패합니다. 카탈로그에 없으면 `tr()`이 원문(한국어)을 그대로 반환하므로 즉시 깨지지는 않지만 다른 언어에서 한국어가 보입니다.

---

## 8. Qt 관련 주의사항

1. **`WA_StyledBackground`** — objectName으로 배경·테두리를 주는 `QWidget` 컨테이너는 이 속성을 켜야 스타일시트가 적용됩니다. 안 켜면 아무것도 안 그려집니다. `shell.styled()` 헬퍼를 쓰십시오.
2. **`QPushButton` 높이** — QSS `padding`에 위젯 자체 프레임 마진이 더해져 `min-height`만으로는 높이가 안 맞습니다. 정확한 높이가 필요하면 `setFixedHeight()`를 쓰십시오 (nav 항목이 42px 지정에 62px로 나왔던 사례).
3. **QSS 미지원** — `box-shadow`, `letter-spacing`, `text-transform`, `::before` 모두 안 됩니다. 자간은 `QFont.setLetterSpacing()`, 대문자는 `QFont.setCapitalization(QFont.AllUppercase)`, 액센트 바는 자식 위젯으로 처리했습니다.
   - 대문자를 `str.upper()`로 하면 `DisplayText`의 번역 바인딩이 끊어집니다. 폰트로 처리한 이유입니다.
4. **정사각 유지** — `shell.SquareHolder`가 자식을 항상 1:1 중앙 정렬합니다.
5. **뷰포트 폭은 높이에서 나옵니다** — 세 패널이 전부 정사각이라 `ViewArea.resizeEvent()`가 가용 **높이**로 두 패널의 폭을 `setFixedWidth()` 합니다. 그게 규칙의 전부입니다. Qt가 이 고정 폭으로부터 `ViewArea`의 최대폭을 유도하고 그것을 감싸는 카드의 레이아웃까지 전파하므로, **카드나 영역에 별도 캡을 걸 필요가 없습니다** (실측으로 확인했습니다 — 캡을 넣으나 빼나 카드 폭이 동일).
   - **판정 칼럼에 `setMaximumWidth()`를 걸지 마십시오.** `QBoxLayout`은 늘릴 수 없는 위젯을 **가운데 정렬**하므로 좌우 양쪽에 빈칸이 생깁니다. 실제로 이것이 화면 오른쪽 빈 띠의 원인이었습니다. 최소폭만 지정하고 stretch를 주십시오. 회귀 방지: `test_judgement_column_has_no_maximum_width`.
6. 색·라운드·사이드바 폭 등은 `theme.py`의 `PALETTES`와 `METRICS`에서 한 번에 조정합니다.

---

## 9. 검증 상태

> **정정 안내.** 이 문서 초판은 "numpy·torch가 없어 전체 앱 실행을 확인하지 못했다"고
> 적었습니다. 실제로 없는 것은 **torch뿐**이고 numpy·OpenCV·Pillow·PySide6는 사용
> 가능했습니다. 이후 저장소를 작업 환경으로 가져와 **기존 테스트를 실제로 실행**했으며,
> 아래가 그 결과입니다.

### 9.1 통과한 테스트 (오프스크린 Qt, 주입 프레임)

| 테스트 모듈 | 결과 |
|---|---|
| `test_station_ui.py` | 14 통과 / 1 제외 |
| `test_operation_ui.py` · `test_theme_preferences.py` | 13 통과 |
| `test_laptop_ui.py` | 11 통과 |
| `test_live_language.py` · `test_settings_ui.py` · `test_readiness_ui.py` | 37 통과 |
| `test_live_viewports.py` (신규) | 7 통과 |

### 9.2 테스트 실행 방법

이 저장소는 패키지를 설치하지 않습니다. `scripts/*.py`는 각자 첫 줄에서
`sys.path.insert(0, ROOT/'src')`를 하지만 `tests/*.py`에는 그 줄이 없으므로,
**`PYTHONPATH`를 직접 지정해야 합니다.** 지정하지 않으면
`ModuleNotFoundError: No module named 'mes_vision'`으로 전부 실패합니다.

PowerShell (저장소 루트에서):

```powershell
$env:PYTHONPATH="$PWD\src"
.venv\Scripts\python -m unittest discover -s tests -t tests -v
```

`-t tests`가 필요합니다. `tests/`에 `__init__.py`가 없어서 `-t .`으로 주면
`Start directory is not importable`로 막히고, 테스트끼리의 상호 import
(`from test_station_devices import setup, live` 등)도 이 설정에서만 풀립니다.

pytest를 쓰려면 별도 설치가 필요합니다 (`python -m pip install pytest`).
기본 환경에는 없습니다.

### 9.3 이 환경에서 실행 불가

- `test_station_ui.py::test_resident_process_handles_saved_overview_and_detail_without_robot`
  — `anomaly/resident.py`가 torch를 요구합니다.
- `test_photo_inspection.py` · `test_manual_inspection.py` — 저장소의 `artifacts/`,
  `models/` 실물 자산이 있어야 합니다. **원본 PC에서 반드시 돌려 주십시오.**

### 9.4 발견해 고친 회귀 — `test_laptop_ui.py`

`test_compact_menu_tracks_both_navigation_directions`가 실패하고 있었습니다.
사이드바 개편으로 `self.nav`(QListWidget)가 화면에서 빠지고 숨은 호스트 안의
**선택 모델**로만 남았기 때문에 `w.nav.isVisible()`이 항상 False입니다.
`w.sidebar.isVisible()`을 보도록 테스트를 고쳤습니다. `self.nav`는 여전히 페이지 전환과
컴팩트 선택기를 구동하므로 나머지 단언은 그대로입니다. **테스트 파일을 고친 유일한
사례이며, 명세가 실제로 바뀐 지점입니다.**

### 9.5 신규 테스트 `tests/test_live_viewports.py`

실시간 화면의 배선과 정사각 기하를 고정합니다. 변이 테스트로 각 항목이 실제로 회귀를
잡는지 확인했습니다.

| 테스트 | 잡는 회귀 |
|---|---|
| `test_live_pane_receives_camera_frames` | `tick()`에서 `live_view` 급전이 빠지는 것 |
| `test_judgement_column_has_no_maximum_width` | 판정 칼럼에 최대폭이 다시 붙는 것 |
| `test_every_pane_is_square` | `SquareHolder`가 1:1을 놓치는 것 |
| `test_stacked_panes_are_half_the_tall_pane` | 세로 칼럼 폭 계산이 어긋나는 것 |

기하 검사는 `StationWindow`가 아니라 위젯을 직접 조립해 수행합니다.
`responsive.fit_available_screen()`이 창을 화면 크기로 줄이는데 오프스크린 화면이 작아
남는 폭이 생기지 않고, 그러면 단언이 엉뚱한 이유로 통과합니다.

### 9.6 여전히 사람이 봐야 하는 것

카메라·로봇이 붙어야만 드러나는 것들입니다.

- [ ] 창이 최대화로 뜨는지, F1~F5 / F11 / Esc가 동작하는지
- [ ] 환경설정에서 글자 크기·언어 변경 시 레이아웃
- [ ] 실시간 영상 패널에 실제 카메라 프레임이 들어오는지 (§3.1)
- [ ] 전체사진·상세사진 패널의 정사각 크롭 위치가 작업대와 맞는지
- [ ] 물체 목록 선택이 상세 패널과 연동되는지
- [ ] 다크 모드 및 영어·중국어·태국어 전환 시 레이아웃


> 2026-09-16 백엔드 후속 작업: [적용 결과와 실물에서 남은 확인](UI_BACKEND_HANDOFF_RESULT.md).

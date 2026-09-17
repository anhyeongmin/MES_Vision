"""Truthful display states; no estimated completion percentage or verdict changes."""
from mes_vision.i18n import tr, trf

PHASES = {
    "MODEL_LOADING": "모델 준비 중", "ANALYZING": "추가 분석 중",
    "WAITING_FOREGROUND": "검사 우선으로 대기", "WAITING_GPU": "GPU 사용 순서 대기",
    "WARM_IDLE": "모델 유지 중", "IDLE": "새 분석 요청 대기",
    "STOPPED": "분석 작업자 중단", "DISABLED": "VLM OFF",
}
TERMINAL = {"COMPLETED": "분석 완료", "FAILED": "분석 실패", "CANCELLED": "분석 취소",
            "DEFERRED": "대기열 가득 참", "SKIPPED_DISABLED": "OFF로 미요청"}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def fresh(runtime, now):
    return bool(runtime and -5 <= now-runtime.get("updated", 0) <= 6)


def overall(summary, runtime, now):
    if not summary["enabled"]:
        if runtime and runtime.get("model_pid"):
            return tr('중단 요청 중') if fresh(runtime, now) else tr('중단 상태 확인 필요')
        return "VLM OFF"
    if not runtime: return tr('분석 작업자 연결 대기')
    phase = runtime.get("phase")
    if phase == "STOPPED": return tr('분석 작업자 중단')
    if not fresh(runtime, now): return tr('작업자 상태 확인 필요')
    if phase == "DISABLED": return tr('분석 작업자 연결 대기')
    text = tr(PHASES.get(phase, '작업자 상태 확인 필요'))
    if phase == "WARM_IDLE":
        expires = runtime.get("warm_expires_at")
        if expires is not None:
            text += " · " + trf('유지 종료까지 {v0}', v0=duration(expires-now))
    elif phase in {"MODEL_LOADING", "ANALYZING", "WAITING_FOREGROUND", "WAITING_GPU"}:
        started = runtime.get("phase_started")
        if started is not None:
            text += " · " + trf('현재 단계 경과 {v0}', v0=duration(now-started))
    return text


def job_label(row, runtime, enabled, now):
    state = row["state"]
    if state in TERMINAL: return tr(TERMINAL[state])
    if state == "RUNNING" and (row["cancel_requested"] or not enabled): return tr('중단 요청 중')
    live = fresh(runtime, now)
    if state == "RUNNING":
        matches = (live and runtime.get("job_id") == row["id"] and row.get("token") is not None
                   and runtime.get("job_token") == row["token"])
        if matches and runtime.get("phase") in {"MODEL_LOADING", "ANALYZING"}:
            return tr(PHASES[runtime["phase"]])
        return tr('처리 상태 확인 중')
    if state == "PENDING":
        if live and runtime.get("phase") in {"WAITING_FOREGROUND", "WAITING_GPU"}:
            return tr(PHASES[runtime["phase"]])
        return tr('분석 대기')
    return tr('처리 상태 확인 중')


def selected_text(row, runtime, enabled, now):
    text = job_label(row, runtime, enabled, now)
    terminal = row["state"] in TERMINAL
    end = row["updated"] if terminal else now
    text += " · " + trf('요청부터 종료까지 {v0}' if terminal else '요청 후 경과 {v0}',
                          v0=duration(end-row["created"]))
    metrics = (row.get("result") or {}).get("metrics", {})
    if row["state"] == "COMPLETED" and metrics.get("model_reused"):
        text += " · " + tr('모델 재사용 · 재로딩 생략')
    if row["state"] == "DEFERRED":
        text += "\n" + tr('자동 분석 대상이 아닙니다. 대기열에 여유가 생기면 다시 요청하세요.')
    return text

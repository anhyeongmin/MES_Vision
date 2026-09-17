"""Backend-owned values for the existing UI; VLM never supplies the verdict."""
import time
from mes_vision.i18n import tr, trf
from mes_vision.qt_i18n import ui_text


def update_context(window):
    product = window.product or {}
    camera = bool(window.camera_info)
    robot = window.robot_state.get('state', '') if window.robot else ''
    manual=getattr(window,'robot_panel',None)
    if manual is not None and manual.worker is not None:
        live=manual.last_state.get('state')=='TEACHING' and time.monotonic()-manual.received<1.2
        robot=('수동 이동 중' if manual.last_state.get('busy') else '수동 제어 연결') if live else '수동 제어 응답 대기'
    values = {
        '품목': (product.get('name') or tr('미등록'), ''),
        '버전': (str(product.get('version', '—')), ''),
        '카메라': (tr('연결됨') if camera else tr('연결 안 됨'), 'ok' if camera else 'warn'),
        '로봇': (robot or tr('연결 안 됨'), 'ok' if robot in {'IDLE','RECAPTURE'} else 'warn'),
        '모델': (tr('준비 완료') if window.engine_ready else tr('미준비'), 'ok' if window.engine_ready else 'warn'),
    }
    for key, (value, state) in values.items():
        label = window.context.values[key]
        if label.text() != str(value) or label.objectName() != window.context.STATES[state]:
            window.context.set_value(key, value, state)
            ui_text(label.setText,value)


def update_verdict(window, decision, identity, reason=''):
    if decision not in {'OK','NG','REVIEW'}:
        text=tr('물체를 선택하면 판정이 표시됩니다.') if not identity else tr('검사 대기')
        window.verdict.set_idle(text); ui_text(window.verdict.main.setText,text)
        return
    names={'OK':'정상','NG':'불량','REVIEW':'보류'}
    window.verdict.set_verdict(decision.lower(), tr(names[decision]),
        trf('선택 물체: {id}', id=identity.split(':')[-1]), reason)
    ui_text(window.verdict.mark.setText,tr(names[decision]))
    ui_text(window.verdict.main.setText,trf('선택 물체: {id}',id=identity.split(':')[-1]))

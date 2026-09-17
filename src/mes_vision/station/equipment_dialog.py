from mes_vision.operation.setup_dialogs import EquipmentDialog
from mes_vision.i18n import tr


class StationEquipmentDialog(EquipmentDialog):
    def collect(self):
        value=super().collect()
        from mes_vision.operation.acquisition import configure_equipment
        parent=self.parent()
        if getattr(self,'lens_binding',None) is not None:
            from mes_vision.operation.lens_binding import bound_equipment
            return bound_equipment(parent.root,value,self.lens_binding)
        return configure_equipment(parent.root,value) if parent and hasattr(parent,'root') else value

    def request_lens_binding(self):
        from PySide6.QtWidgets import QMessageBox
        from mes_vision.qt_i18n import ui_text
        from mes_vision.operation.lens_binding import plan_binding
        try:
            candidate=EquipmentDialog.collect(self)
            plan=plan_binding(self.parent().root,candidate['camera'],confirmed=True)
            answer=QMessageBox.question(self,tr('기존 렌즈 보정 연결'),
                tr('전에 보정했던 동일한 카메라이며 렌즈·초점을 바꾸지 않았습니까?\n기존 렌즈 보정을 연결하고 작업 영역·로봇 좌표 검증은 해제합니다.\n렌즈나 초점을 바꿨다면 취소하고 재보정하세요.'),
                QMessageBox.Yes|QMessageBox.No,QMessageBox.No)
            if answer!=QMessageBox.Yes:return
            self.lens_binding=plan
            ui_text(self.message.setText,tr('동일 카메라 연결 확인됨 · 저장을 누르면 적용됩니다.'))
        except Exception as exc:ui_text(self.message.setText,str(exc))

    def persist(self,store):
        if self.lens_binding is None:return store.save_equipment(self.value)
        from mes_vision.operation.lens_binding import persist_binding
        return persist_binding(self.parent().root,self.value,self.lens_binding,store)

    def __init__(self,*args,**kwargs):
        self.lens_binding=None
        super().__init__(*args,**kwargs)
        from mes_vision.operation.widgets import button
        self.bind_lens=button(tr('동일 카메라 · 기존 렌즈 보정 연결'),self.request_lens_binding)
        self.layout().insertWidget(self.layout().count()-2,self.bind_lens)
        for index in range(self.pages.tabs.count()):
            if self.pages.tabs.tabText(index)==str(tr('추적 · 고급 설정')):
                self.pages.advanced_pages.remove(index); self.pages.tabs.setTabVisible(index,False)
        # Retain legacy values in old records, but do not present them as active
        # moving-camera controls. The station calibration popup owns both maps.
        self.calibration.parentWidget().layout().setRowVisible(self.calibration,False)
        # UVC exposure/gain are edited only against device-reported ranges.
        if self.value['camera'].get('driver')=='uvc':
            form=self.auto_exposure.parentWidget().layout()
            form.setRowVisible(self.auto_exposure,False); form.setRowVisible(self.exposure,False)
            from mes_vision.qt_i18n import ui_text
            from PySide6.QtWidgets import QLabel
            hint=ui_text(QLabel,tr('노출·게인은 환경설정 → 카메라 · 데이터 수집에서 실시간 영상과 함께 조절하세요.'))
            hint.setWordWrap(True); form.addRow(hint)

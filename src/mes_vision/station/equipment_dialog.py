from mes_vision.operation.setup_dialogs import EquipmentDialog
from mes_vision.i18n import tr


class StationEquipmentDialog(EquipmentDialog):
    def collect(self):
        value=super().collect()
        from mes_vision.operation.acquisition import configure_equipment
        parent=self.parent()
        return configure_equipment(parent.root,value) if parent and hasattr(parent,'root') else value

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
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

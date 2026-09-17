from mes_vision.qt_i18n import ui_text
from mes_vision.i18n import text_join
"""Device selectors and camera modes for the equipment editor."""
from PySide6.QtCore import QTimer, QSignalBlocker
from PySide6.QtWidgets import QComboBox, QLabel, QHBoxLayout
from mes_vision.i18n import tr, trf
from mes_vision.ui_refresh import BackgroundRead
from .camera_modes import COLOR_PRESETS,UVC_PRESETS
from functools import partial
from .device_discovery import discover_devices
from .widgets import button


class DeviceSelector(QComboBox):
    """Keep identifiers separate from labels; allow explicit offline entry."""
    def __init__(self, value):
        super().__init__(); self.setEditable(True); self.setInsertPolicy(QComboBox.NoInsert)
        self.setMinimumContentsLength(24)
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.populate([], value)

    def identifier(self):
        index=self.currentIndex()
        if index>=0 and self.currentText()==self.itemText(index): return self.itemData(index)
        return self.currentText().strip()

    def populate(self, rows, selected=None):
        selected=self.identifier() if selected is None else selected
        with QSignalBlocker(self):
            ui_text(self.clear); ui_text(self.addItem, tr('장치를 선택하세요'), '')
            known=set()
            for identifier, label in rows:
                if not identifier or identifier in known: continue
                known.add(identifier); ui_text(self.addItem, label, identifier)
            if selected and selected not in known:
                ui_text(self.addItem, selected+' · '+tr('검색에서 확인되지 않음'),selected)
            self.setCurrentIndex(max(0,self.findData(selected)))


class DeviceControls:
    def init_discovery(self, layout):
        self.discovered_cameras={}; self.mode_presets=COLOR_PRESETS
        self.discovery_read=BackgroundRead(self)
        self.discovery_read.succeeded.connect(self.devices_found)
        self.discovery_read.failed.connect(self.discovery_failed)
        self.discovery_close_result=None
        row=QHBoxLayout(); self.search_button=button(tr('장치 다시 검색'),self.search_devices)
        row.addWidget(self.search_button); self.search_status=ui_text(QLabel, tr('목록에서 선택하거나 장치 식별자를 직접 입력할 수 있습니다.'))
        self.search_status.setWordWrap(True); row.addWidget(self.search_status,1); layout.addLayout(row)

    def search_devices(self):
        if self.discovery_read.busy: return
        self.search_button.setEnabled(False); self.save_box.button(self.save_box.StandardButton.Save).setEnabled(False)
        ui_text(self.search_status.setText, tr('장치를 검색하고 있습니다…'))
        driver=self.driver.currentData()
        self.driver.setEnabled(False)
        self.discovery_read.start(partial(discover_devices,driver=driver))

    def devices_found(self, result):
        self.discovered_cameras={d['serial']:d for d in result['cameras']}
        self.serial.populate([(d['serial'],d['name']+' · '+(d['serial'][-8:] if d['serial'].startswith('uvc:') else d['serial'])) for d in result['cameras']])
        self.port.populate([(d['port'],d['port']+' · '+d['description']) for d in result['ports']])
        self.refresh_camera_modes()
        messages=[trf('카메라 {cameras}대 · 통신 포트 {ports}개',cameras=len(result['cameras']),ports=len(result['ports']))]
        for key,label in (('cameras',tr('카메라')),('ports',tr('통신 포트'))):
            if key in result['errors']: messages.append(label+' · '+tr('검색 실패 · 연결과 드라이버를 확인하고 다시 검색하세요.'))
        ui_text(self.search_status.setText, text_join('\n', messages))
        ui_text(self.search_status.setToolTip, text_join('\n', (f'{k}: {v}' for k,v in result['errors'].items())))
        self.finish_discovery()

    def discovery_failed(self, error):
        # Previous discoveries cannot be used as current capability verification.
        self.discovered_cameras={}; self.serial.populate([]); self.port.populate([]); self.refresh_camera_modes()
        ui_text(self.search_status.setText, tr('검색 실패 · 연결과 드라이버를 확인하고 다시 검색하세요.'))
        ui_text(self.search_status.setToolTip, error); self.finish_discovery()

    def finish_discovery(self):
        self.driver.setEnabled(True)
        self.search_button.setEnabled(True); self.save_box.button(self.save_box.StandardButton.Save).setEnabled(True)
        if self.discovery_close_result is not None:
            result=self.discovery_close_result; self.discovery_close_result=None; self.done(result)

    def done(self, result):
        if self.discovery_read.busy:
            self.discovery_close_result=result
            ui_text(self.search_status.setText, tr('검색을 정리하고 창을 닫습니다…'))
            return
        super().done(result)

    def closeEvent(self, event):
        if self.discovery_read.busy:
            self.discovery_close_result=0; event.ignore()
            ui_text(self.search_status.setText, tr('검색을 정리하고 창을 닫습니다…'))
        else: super().closeEvent(event)

    def refresh_camera_modes(self):
        old_size=self.selected_resolution(); old_rate=self.fps.currentData()
        camera=self.discovered_cameras.get(self.serial.identifier())
        profiles=camera.get('profiles') if camera else None
        self.mode_presets={}
        if profiles is None:
            self.mode_presets=UVC_PRESETS[self.pixel_format.currentData()] if self.driver.currentData()=='uvc' else COLOR_PRESETS
            ui_text(self.mode_hint.setText, tr('장치 지원 모드를 확인하지 못했습니다. 해상도 후보를 표시하며, 실제 연결 시 확인합니다.'))
        else:
            for profile in profiles:
                size=(profile['width'],profile['height'])
                self.mode_presets.setdefault(size,set()).add(profile['fps'])
            ui_text(self.mode_hint.setText, tr('선택한 D405의 RGB8 지원 모드입니다. USB 연결을 바꾸면 다시 검색하세요. 연결 성공 여부는 영상 시작 시 최종 확인합니다.') if profiles else tr('선택한 D405에서 RGB8 지원 모드를 찾지 못했습니다. 연결과 드라이버를 확인하세요.'))
        ui_text(self.mode_hint.setToolTip, camera.get('profile_error') or '' if camera else '')
        sizes=list(self.mode_presets)
        if old_size is not None and old_size not in sizes: sizes.append(old_size)
        with QSignalBlocker(self.resolution):
            ui_text(self.resolution.clear)
            for w,h in sizes:
                label=f'{w} × {h}'
                if (w,h) not in self.mode_presets: label+=' · '+tr('선택값 · 장치 지원 확인 필요')
                ui_text(self.resolution.addItem, label,f'{w}x{h}')
            self.resolution.setCurrentIndex(self.resolution.findData(f'{old_size[0]}x{old_size[1]}') if old_size else -1)
        self.update_frame_rates(old_rate, preserve=True)

    def update_frame_rates(self, preferred=None, *, preserve=False, reset=False):
        size=self.selected_resolution(); original=self.original['camera']
        if preferred is None and not reset: preferred=self.fps.currentData()
        rates=sorted(self.mode_presets.get(size,()))
        retained=preferred if preserve else original['fps'] if size==(original['width'],original['height']) else None
        if reset: retained=None
        if retained is not None and retained not in rates: rates.append(retained)
        ui_text(self.fps.clear)
        for rate in sorted(rates):
            label=f'{rate} fps'
            if rate not in self.mode_presets.get(size,()): label+=' · '+tr('선택값 · 장치 지원 확인 필요')
            ui_text(self.fps.addItem, label,rate)
        index=self.fps.findData(preferred)
        if index<0: index=self.fps.findData(30)
        self.fps.setCurrentIndex(index if index>=0 else 0)

    def selected_mode_supported(self):
        if self.driver.currentData()=='uvc':
            return self.fps.currentData() in UVC_PRESETS[self.pixel_format.currentData()].get(self.selected_resolution(),())
        camera=self.discovered_cameras.get(self.serial.identifier())
        if camera is None or camera.get('profiles') is None: return True  # offline configuration
        return self.fps.currentData() in self.mode_presets.get(self.selected_resolution(),())

    def change_camera_driver(self):
        self.discovered_cameras={}; self.serial.populate([], '')
        self.pixel_format.setEnabled(self.driver.currentData()=='uvc')
        self.auto_exposure.setChecked(True)
        self.refresh_camera_modes()

    def change_pixel_format(self):
        self.refresh_camera_modes()
        self.update_frame_rates(reset=True)

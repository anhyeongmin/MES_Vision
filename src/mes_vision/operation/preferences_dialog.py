from mes_vision.qt_i18n import ui_text
from copy import deepcopy
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QDialog, QVBoxLayout, QFormLayout, QLabel, QComboBox, QCheckBox, QTabWidget, QWidget, QDialogButtonBox, QLineEdit
from mes_vision.i18n import tr, LANGUAGES
from .widgets import button


class PreferencesDialog(QDialog):
    def __init__(self, preferences, runtime, parent=None):
        super().__init__(parent); self.preferences = preferences; self.value = deepcopy(preferences.value)
        self.setObjectName("preferencesDialog"); ui_text(self.setWindowTitle, tr("환경설정")); self.resize(660, 460)
        self.setWindowModality(Qt.ApplicationModal)
        layout = QVBoxLayout(self); tabs = QTabWidget(); layout.addWidget(tabs)
        general = QWidget(); form = QFormLayout(general); ui_text(tabs.addTab, general, tr("일반"))
        self.language = QComboBox(); self.language.setObjectName("displayLanguage")
        for code, name in LANGUAGES.items(): ui_text(self.language.addItem, name, code)
        self.language.setCurrentIndex(self.language.findData(self.value["language"]))
        ui_text(form.addRow, tr("표시 언어"), self.language)
        self.theme = QComboBox(); self.theme.setObjectName('displayTheme')
        self.theme.addItem('Light', 'light'); self.theme.addItem('Dark', 'dark')
        self.theme.setCurrentIndex(self.theme.findData(self.value['theme']))
        ui_text(form.addRow, tr('화면 테마'), self.theme)
        note = ui_text(QLabel, tr("언어는 저장하면 바로 적용됩니다. 진행 중인 검사와 장치 연결은 유지됩니다.")); note.setWordWrap(True); ui_text(form.addRow, note)
        self.scale = QComboBox(); self.scale.setObjectName("textScale")
        for value in (100, 115, 130): ui_text(self.scale.addItem, f"{value}%", value)
        self.scale.setCurrentIndex(self.scale.findData(self.value["text_scale"])); ui_text(form.addRow, tr("글자 크기"), self.scale)
        self.start_page = QComboBox(); self.start_page.setObjectName("startupPage")
        for index, label in enumerate(("실시간 검사", "품목 설정", "장비 · 보정", "검사 이력", "VLM 추가 분석")):
            ui_text(self.start_page.addItem, tr(label), index)
        self.start_page.setCurrentIndex(self.value["start_page"]); ui_text(form.addRow, tr("시작 화면"), self.start_page)
        self.remember = ui_text(QCheckBox, tr("창 크기와 위치 기억")); self.remember.setChecked(self.value["remember_window"]); ui_text(form.addRow, self.remember)
        note = ui_text(QLabel, tr("시작 화면 선택은 화면만 변경합니다. 검사·카메라·로봇·VLM을 자동으로 켜지 않습니다.")); note.setWordWrap(True); ui_text(form.addRow, note)
        storage = QWidget(); form = QFormLayout(storage); ui_text(tabs.addTab, storage, tr("저장 · 진단"))
        path = QLineEdit(str(runtime)); path.setReadOnly(True); ui_text(form.addRow, tr("운영 자료 저장 위치"), path)
        ui_text(form.addRow, button(tr("저장 폴더 열기"), lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(runtime)))))
        note = ui_text(QLabel, tr("검사 자료의 보관·백업은 검사 이력의 저장·백업 관리에서 설정합니다.")); note.setWordWrap(True); ui_text(form.addRow, note)
        if parent is not None:
            ui_text(form.addRow, button(tr("진단 자료 내보내기"), lambda: parent.guarded(parent.export_diagnostics)))
            capture=QWidget(); capture_form=QFormLayout(capture); ui_text(tabs.addTab,capture,tr('카메라 · 데이터 수집'))
            ui_text(capture_form.addRow,button(tr('실시간 촬영 조건 조절'),lambda:parent.guarded(parent.open_capture_studio)))
            ui_text(capture_form.addRow,button(tr('학습데이터 수집'),lambda:parent.guarded(lambda:parent.open_capture_studio(collect=True))))
            hint=ui_text(QLabel,tr('전체·상세 촬영의 노출·게인을 따로 저장하고, 품목별 영상을 자동 이름으로 수집합니다. 검사 모델을 해제한 상태에서 사용하세요.'))
            hint.setWordWrap(True); ui_text(capture_form.addRow,hint)
        self.message = ui_text(QLabel); self.message.setWordWrap(True); layout.addWidget(self.message)
        if preferences.warning: ui_text(self.message.setText, tr("환경설정 파일을 읽지 못해 기본값을 사용합니다. 저장하면 기존 파일을 별도로 보존합니다."))
        self.buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel | QDialogButtonBox.RestoreDefaults)
        self.buttons.accepted.connect(self.save); self.buttons.rejected.connect(self.reject)
        self.buttons.button(QDialogButtonBox.RestoreDefaults).clicked.connect(self.defaults); layout.addWidget(self.buttons)

    def defaults(self):
        self.theme.setCurrentIndex(0)
        self.language.setCurrentIndex(0); self.scale.setCurrentIndex(0); self.start_page.setCurrentIndex(0); self.remember.setChecked(True)

    def save(self):
        value = deepcopy(self.value)
        value.update(language=self.language.currentData(), theme=self.theme.currentData(), text_scale=self.scale.currentData(),
            start_page=self.start_page.currentData(), remember_window=self.remember.isChecked())
        if not value["remember_window"]: value["window_geometry"] = ""
        try: self.preferences.save(value)
        except Exception as exc: ui_text(self.message.setText, tr("환경설정을 저장하지 못했습니다: ")+str(exc)); return
        self.value = value; self.accept()

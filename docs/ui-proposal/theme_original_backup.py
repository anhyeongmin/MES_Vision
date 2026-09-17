"""MonoFactory Launcher palette, adapted to the native Qt operator application."""
from pathlib import Path
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

PALETTES = {
    'light': dict(bg='#ffffff', surface='#ffffff', soft='#f9fafb', text='#18212f',
                  muted='#667085', border='#d8dee8', accent='#56a9f9', selected='#dceeff',
                  selected_text='#09156a', sidebar='#ffffff', disabled='#98a2b3',
                  danger='#b42318', success='#067647', warning='#b54708'),
    'dark': dict(bg='#050d24', surface='#0b1836', soft='#11264d', text='#f7f9ff',
                 muted='#bcc9df', border='#294370', accent='#6db8ff', selected='#173d68',
                 selected_text='#f7f9ff', sidebar='#060e33', disabled='#72829a',
                 danger='#ff8c96', success='#63d7a4', warning='#ffc46b'),
}
_mode = 'light'


def color(role):
    return PALETTES[_mode][role]


def logo_path(mode):
    name = 'monofactory-brand-navy.png' if mode == 'dark' else 'monofactory-brand-white.png'
    return Path(__file__).parent / 'assets' / 'brand' / name


def stylesheet(mode, scale=1):
    p = PALETTES[mode]
    return '''
    QWidget { color: %(text)s; font-size: %(font)dpx; }
    QMainWindow,QDialog,QMessageBox,QWidget#central { background: %(bg)s; }
    QWidget#brandSidebar { background: %(sidebar)s; border-right:1px solid %(border)s; }
    QLabel { background: transparent; }
    QLabel#title { font-size: %(title)dpx; font-weight:800; padding:12px; }
    QLabel#brandLogo,QLabel#compactBrandLogo { border:0; padding:0; }
    QPushButton,QToolButton { padding:8px 12px; background:%(surface)s; border:1px solid %(border)s; border-radius:8px; font-weight:600; }
    QPushButton:hover,QToolButton:hover { background:%(soft)s; border-color:%(accent)s; }
    QPushButton:focus,QComboBox:focus,QLineEdit:focus { border:2px solid %(accent)s; }
    QPushButton:pressed,QPushButton:checked { background:%(selected)s; color:%(selected_text)s; border-color:%(accent)s; }
    QPushButton:disabled,QToolButton:disabled { color:%(disabled)s; background:%(soft)s; }
    QPushButton#globalRobotStop { color:%(danger)s; border-color:%(danger)s; font-weight:800; }
    QPushButton#inspectionStart { background:%(accent)s; color:#06104f; font-weight:800; }
    QPushButton#inspectionStart:disabled { background:%(soft)s; color:%(disabled)s; }
    QListWidget,QTreeWidget { background:%(surface)s; border:1px solid %(border)s; border-radius:8px; padding:6px; }
    QListWidget#mainNavigation { background:transparent; border:0; }
    QListWidget::item { padding:12px 6px; border-radius:6px; }
    QListWidget::item:selected,QTreeWidget::item:selected { background:%(selected)s; color:%(selected_text)s; }
    QLineEdit,QPlainTextEdit,QTextEdit,QTableWidget,QTableView,QComboBox,QSpinBox,QDoubleSpinBox,QDateEdit,QDateTimeEdit {
        background:%(surface)s; color:%(text)s; border:1px solid %(border)s; border-radius:6px;
        selection-background-color:%(selected)s; selection-color:%(selected_text)s; }
    QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QDateEdit,QDateTimeEdit { padding:6px; min-height:20px; }
    QComboBox QAbstractItemView { background:%(surface)s; color:%(text)s; selection-background-color:%(selected)s; selection-color:%(selected_text)s; }
    QHeaderView::section { background:%(soft)s; color:%(muted)s; padding:8px; border:0; border-bottom:1px solid %(border)s; font-weight:700; }
    QTableView { gridline-color:%(border)s; alternate-background-color:%(soft)s; }
    QTableCornerButton::section { background:%(soft)s; border:0; }
    QTabWidget::pane,QGroupBox { border:1px solid %(border)s; border-radius:8px; background:%(surface)s; }
    QGroupBox { margin-top:14px; padding:12px; }
    QGroupBox::title { subcontrol-origin:margin; left:12px; }
    QTabBar::tab { background:%(soft)s; color:%(muted)s; padding:10px 16px; border-bottom:2px solid transparent; }
    QTabBar::tab:selected { color:%(selected_text)s; background:%(surface)s; border-bottom-color:%(accent)s; }
    QScrollArea,QAbstractScrollArea::corner { background:%(bg)s; border:0; }
    QScrollBar:vertical { background:%(soft)s; width:12px; margin:0; }
    QScrollBar:horizontal { background:%(soft)s; height:12px; margin:0; }
    QScrollBar::handle { background:%(border)s; border-radius:5px; min-height:24px; min-width:24px; }
    QScrollBar::add-line,QScrollBar::sub-line { width:0; height:0; }
    QSplitter::handle { background:%(border)s; }
    QMenu,QToolTip { background:%(surface)s; color:%(text)s; border:1px solid %(border)s; padding:5px; }
    QMenu::item:selected { background:%(selected)s; }
    QCheckBox,QRadioButton { spacing:7px; background:transparent; }
    QCheckBox::indicator { width:16px; height:16px; border:1px solid %(border)s; border-radius:3px; background:%(surface)s; }
    QCheckBox::indicator:checked { background:%(accent)s; border-color:%(accent)s; image:url("%(check)s"); }
    QCheckBox::indicator:disabled { background:%(soft)s; border-color:%(disabled)s; }
    QProgressBar { border:1px solid %(border)s; background:%(soft)s; text-align:center; border-radius:5px; }
    QProgressBar::chunk { background:%(accent)s; }
    ''' % dict(p, font=round(13*scale), title=round(23*scale),
               check=(Path(__file__).parent/'assets/brand/check.svg').as_posix())


def apply_theme(mode, scale=1):
    global _mode
    if mode not in PALETTES: raise ValueError('Unsupported theme')
    _mode = mode
    app = QApplication.instance()
    if app is None: return
    app.setStyle('Fusion')
    palette = QPalette()
    for role, key in [(QPalette.Window,'bg'),(QPalette.WindowText,'text'),(QPalette.Base,'surface'),
                      (QPalette.AlternateBase,'soft'),(QPalette.Text,'text'),(QPalette.Button,'surface'),
                      (QPalette.ButtonText,'text'),(QPalette.Highlight,'selected'),
                      (QPalette.HighlightedText,'selected_text'),(QPalette.ToolTipBase,'surface'),
                      (QPalette.ToolTipText,'text'),(QPalette.PlaceholderText,'muted')]:
        palette.setColor(role,QColor(color(key)))
    for role in (QPalette.Text,QPalette.ButtonText,QPalette.WindowText):
        palette.setColor(QPalette.Disabled,role,QColor(color('disabled')))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(mode,scale))
    for widget in app.allWidgets(): widget.update()

"""MonoFactory MES design tokens, adapted to the native Qt operator application.

Colours, spacing and typography follow `AdvancedFactory/Common/frontend_theme.css`
and `MES/frontend/src/App.css`. Corner radii and shadows deliberately differ: the
inspection application uses 3px corners and no blur shadows so the image panes read
as instrument surfaces rather than web cards.
"""
from pathlib import Path
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

PALETTES = {
    'light': dict(bg='#ffffff', surface='#ffffff', soft='#f8fafc', text='#17233b',
                  muted='#43536c', border='#d8e1ec', accent='#56a9f9', accent_strong='#2f8fe9',
                  selected='#e4f0fe', selected_text='#09156a', sidebar='#ffffff',
                  sidebar_end='#ffffff', navy='#09156a', navy_strong='#06104f',
                  disabled='#98a2b3', track='#e8eef6', stage='#0c1424',
                  danger='#c53b4d', danger_soft='#fbeaec', success='#16845b',
                  success_soft='#e7f3ee', warning='#c77700', warning_soft='#fbf0e0'),
    'dark': dict(bg='#050d24', surface='#0b1836', soft='#11264d', text='#f7f9ff',
                 muted='#bcc9df', border='#294370', accent='#6db8ff', accent_strong='#91c9ff',
                 selected='#173d68', selected_text='#f7f9ff', sidebar='#060e33',
                 sidebar_end='#040a26', navy='#09156a', navy_strong='#06104f',
                 disabled='#72829a', track='#1b2d4f', stage='#070c18',
                 danger='#ff7180', danger_soft='#2b1520', success='#49c58d',
                 success_soft='#0e2b22', warning='#f1ad46', warning_soft='#2c2113'),
}
_mode = 'light'

# Layout constants shared with the widget code. Kept here so the whole shell can be
# retuned from one place, the way the CSS custom properties work on the web side.
METRICS = dict(radius=3, control_radius=3, sidebar_width=272, control_height=40,
               card_padding=12, layout_gap=10, nav_item_height=42)


def color(role):
    return PALETTES[_mode][role]


def metric(name, scale=1):
    return round(METRICS[name] * scale)


def logo_path(mode):
    name = 'monofactory-brand-navy.png' if mode == 'dark' else 'monofactory-brand-white.png'
    return Path(__file__).parent / 'assets' / 'brand' / name


def symbol_path(mode):
    name = 'monofactory-symbol-navy.png' if mode == 'dark' else 'monofactory-symbol.png'
    path = Path(__file__).parent / 'assets' / 'brand' / name
    return path if path.exists() else logo_path(mode)


def stylesheet(mode, scale=1):
    p = PALETTES[mode]
    sidebar = ('%(sidebar)s' % p if mode == 'light' else
               'qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 %(sidebar)s,stop:1 %(sidebar_end)s)' % p)
    return '''
    QWidget { color: %(text)s; font-size: %(font)dpx; }
    QMainWindow,QDialog,QMessageBox,QWidget#central { background: %(bg)s; }
    QLabel { background: transparent; }

    /* ---- sidebar ---- */
    QWidget#brandSidebar { background: %(sidebar_bg)s; border-right:1px solid %(border)s; }
    QLabel#brandTitle { font-size: %(title)dpx; font-weight:700; color: %(brand_title)s; }
    QLabel#brandSub { font-size: %(small)dpx; font-weight:700; color: %(brand_sub)s; }
    QLabel#brandLogo,QLabel#compactBrandLogo,QLabel#brandSymbol { border:0; padding:0; }
    QLabel#title { font-size: %(title)dpx; font-weight:800; padding:10px 12px; }
    QLabel#navGroupTitle { font-size: %(tiny)dpx; font-weight:800; color: %(nav_title)s; padding: 6px 12px 2px; }
    QPushButton#navItem,QToolButton#navItem { min-height:%(nav_h)dpx; padding:0 12px;
        border:1px solid transparent; border-radius:%(r)dpx; background:transparent; color:%(nav_item)s;
        font-size:%(body)dpx; font-weight:600; text-align:left; }
    QPushButton#navItem:hover,QToolButton#navItem:hover { border-color:%(nav_border)s; background:%(nav_bg)s;
        color:%(selected_text)s; }
    QPushButton#navItem:checked,QToolButton#navItem:checked { border-color:%(nav_border)s; background:%(nav_bg)s;
        color:%(selected_text)s; font-weight:800; }
    QFrame#navActiveBar { background:%(accent)s; border:0; }
    QLabel#navKey { color:%(muted)s; font-size:%(tiny)dpx; font-weight:800; }
    QWidget#userCard { border:1px solid %(border)s; border-radius:%(r)dpx; background:%(soft)s; }
    QLabel#userAvatar { background:%(accent)s; color:%(navy)s; font-size:%(body)dpx; font-weight:900;
        border-radius:%(r)dpx; min-width:36px; min-height:36px; qproperty-alignment:AlignCenter; }
    QLabel#userName { font-size:%(body)dpx; font-weight:800; color:%(user_name)s; }
    QLabel#userRole { font-size:%(tiny)dpx; font-weight:700; color:%(user_role)s; }
    QPushButton#ctaButton { background:%(accent)s; border:0; border-radius:%(r)dpx; padding:12px 14px;
        color:%(navy)s; font-size:%(body)dpx; font-weight:700; }
    QPushButton#ctaButton:hover { background:%(accent_strong)s; }
    QPushButton#ctaButton:disabled { background:%(soft)s; color:%(disabled)s; }

    /* ---- page header ---- */
    QLabel#pageKicker { color:%(accent_strong)s; font-size:%(tiny)dpx; font-weight:900; }
    QLabel#pageTitle { color:%(text)s; font-size:%(h1)dpx; font-weight:800; }
    QLabel#pageDesc { color:%(muted)s; font-size:%(small)dpx; font-weight:600; }
    QWidget#contextBar { border:1px solid %(border)s; border-radius:%(r)dpx; background:%(surface)s; }
    QFrame#contextDivider { background:%(border)s; border:0; }
    QLabel#ctxKey { color:%(muted)s; font-size:%(tiny)dpx; font-weight:800; }
    QLabel#ctxValue { color:%(text)s; font-size:%(body)dpx; font-weight:800; }
    QLabel#ctxValueOk { color:%(success)s; font-size:%(body)dpx; font-weight:800; }
    QLabel#ctxValueWarn { color:%(warning)s; font-size:%(body)dpx; font-weight:800; }

    /* ---- status pills in the header ---- */
    QLabel#statusPill,QPushButton#statusPill { min-height:%(ctrl_h)dpx; padding:0 12px; border:1px solid %(border)s;
        border-radius:%(r)dpx; background:%(surface)s; color:%(muted)s; font-size:%(small)dpx; font-weight:800; }
    QPushButton#statusPill:hover { border-color:%(accent)s; background:%(soft)s; }
    QPushButton#statusPillOn { min-height:%(ctrl_h)dpx; padding:0 12px; border:1px solid %(accent)s;
        border-radius:%(r)dpx; background:%(surface)s; color:%(accent_strong)s; font-size:%(small)dpx; font-weight:800; }
    QPushButton#themeToggle { min-width:92px; min-height:%(ctrl_h)dpx; padding:0 13px; border:1px solid %(border)s;
        border-radius:%(r)dpx; background:%(surface)s; color:%(theme_fg)s; font-size:%(small)dpx; font-weight:750; }
    QPushButton#themeToggle:hover { border-color:%(accent)s; background:%(soft)s; }
    QPushButton#globalRobotStop { min-height:%(ctrl_h)dpx; padding:0 13px; border:1px solid %(danger)s;
        border-radius:%(r)dpx; background:%(surface)s; color:%(danger)s; font-size:%(small)dpx; font-weight:800; }
    QPushButton#globalRobotStop:hover { background:%(danger_soft)s; }

    /* ---- panels ---- */
    QFrame#panel,QWidget#panel { border:1px solid %(border)s; border-radius:%(r)dpx; background:%(surface)s; }
    QWidget#panelHead { background:%(surface)s; border-bottom:1px solid %(border)s; }
    QLabel#panelTitle { font-size:%(body)dpx; font-weight:800; color:%(text)s; }
    QWidget#toolbar { background:%(soft)s; border-bottom:1px solid %(border)s; }
    QLabel#chip { padding:4px 8px; border:1px solid %(border)s; border-radius:%(r)dpx;
        background:%(surface)s; color:%(muted)s; font-size:%(tiny)dpx; font-weight:800; }

    QWidget#faultBanner { background:%(danger_soft)s; border:1px solid %(danger)s; border-radius:%(r)dpx; }

    /* ---- verdict banner ---- */
    QWidget#verdictOk { background:%(success)s; }
    QWidget#verdictNg { background:%(danger)s; }
    QWidget#verdictReview { background:%(warning)s; }
    QWidget#verdictIdle { background:%(soft)s; border-bottom:1px solid %(border)s; }
    QLabel#verdictMark { color:#ffffff; font-size:%(h2)dpx; font-weight:900; padding:1px 10px; }
    QLabel#verdictMain { color:#ffffff; font-size:%(body)dpx; font-weight:900; }
    QLabel#verdictSub { color:rgba(255,255,255,0.88); font-size:%(small)dpx; font-weight:700; }
    QLabel#verdictIdleText { color:%(muted)s; font-size:%(small)dpx; font-weight:800; }

    /* ---- image viewports ---- */
    QWidget#viewArea { background:%(soft)s; border-top:1px solid %(border)s; }
    QLabel#viewTitle { color:%(muted)s; font-size:%(tiny)dpx; font-weight:800; }
    QLabel#viewTitleStrong { color:%(text)s; font-size:%(tiny)dpx; font-weight:800; }
    QLabel#viewMeta { color:%(muted)s; font-size:%(tiny)dpx; font-weight:700; }
    QWidget#viewFrame { border:1px solid %(border)s; background:%(stage)s; }
    QPushButton#zoomButton { min-height:22px; padding:0 8px; border:1px solid %(border)s;
        border-radius:2px; background:%(surface)s; color:%(text)s; font-size:%(tiny)dpx; font-weight:700; }
    QPushButton#zoomButton:hover { border-color:%(accent)s; background:%(soft)s; }
    QPushButton#zoomButton:checked { border-color:%(accent)s; background:%(selected)s;
        color:%(selected_text)s; font-weight:800; }
    QLabel#zoomLevel { color:%(muted)s; font-size:%(tiny)dpx; font-weight:800; }

    /* ---- generic controls ---- */
    QPushButton,QToolButton { min-height:%(btn_h)dpx; padding:0 11px; background:%(surface)s;
        border:1px solid %(border)s; border-radius:%(r)dpx; font-weight:700; }
    QPushButton:hover,QToolButton:hover { background:%(soft)s; border-color:%(accent)s; }
    QPushButton:focus,QComboBox:focus,QLineEdit:focus { border:1px solid %(accent)s; }
    QPushButton:pressed,QPushButton:checked { background:%(selected)s; color:%(selected_text)s; border-color:%(accent)s; }
    QPushButton:disabled,QToolButton:disabled { color:%(disabled)s; background:%(soft)s; border-color:%(border)s; }
    QPushButton#primaryButton { background:%(accent)s; border-color:%(accent)s; color:%(navy)s; font-weight:800; }
    QPushButton#primaryButton:hover { background:%(accent_strong)s; border-color:%(accent_strong)s; }
    QPushButton#primaryButton:disabled { background:%(soft)s; color:%(disabled)s; border-color:%(border)s; }
    QPushButton#dangerButton { border-color:%(danger)s; color:%(danger)s; background:%(danger_soft)s; font-weight:800; }
    QPushButton#inspectionStart { background:%(accent)s; border-color:%(accent)s; color:%(navy)s; font-weight:800; }
    QPushButton#inspectionStart:disabled { background:%(soft)s; color:%(disabled)s; border-color:%(border)s; }

    QListWidget,QTreeWidget { background:%(surface)s; border:1px solid %(border)s; border-radius:%(r)dpx; padding:4px; }
    QListWidget#mainNavigation { background:transparent; border:0; }
    QListWidget::item { padding:9px 6px; border-radius:%(r)dpx; }
    QListWidget::item:selected,QTreeWidget::item:selected { background:%(selected)s; color:%(selected_text)s; }
    QLineEdit,QPlainTextEdit,QTextEdit,QTableWidget,QTableView,QComboBox,QSpinBox,QDoubleSpinBox,QDateEdit,QDateTimeEdit {
        background:%(surface)s; color:%(text)s; border:1px solid %(border)s; border-radius:%(r)dpx;
        selection-background-color:%(selected)s; selection-color:%(selected_text)s; }
    QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox,QDateEdit,QDateTimeEdit { padding:5px 8px; min-height:%(input_h)dpx; }
    QComboBox QAbstractItemView { background:%(surface)s; color:%(text)s;
        selection-background-color:%(selected)s; selection-color:%(selected_text)s; }
    QComboBox::drop-down { subcontrol-origin:padding; subcontrol-position:center right;
        width:20px; border:0; background:transparent; }
    QComboBox::down-arrow { image:url("%(chevron)s"); width:11px; height:11px; }
    QHeaderView::section { background:%(soft)s; color:%(muted)s; padding:7px 10px; border:0;
        border-bottom:1px solid %(border)s; font-size:%(tiny)dpx; font-weight:800; }
    QTableView { gridline-color:%(border)s; alternate-background-color:%(soft)s; }
    QTableView::item { padding:5px 8px; }
    QTableCornerButton::section { background:%(soft)s; border:0; }
    QTabWidget::pane,QGroupBox { border:1px solid %(border)s; border-radius:%(r)dpx; background:%(surface)s; }
    QGroupBox { margin-top:14px; padding:10px; font-weight:800; }
    QGroupBox::title { subcontrol-origin:margin; left:10px; color:%(muted)s; }
    QTabBar::tab { background:%(soft)s; color:%(muted)s; padding:8px 14px; border:0;
        border-bottom:2px solid transparent; font-weight:800; }
    QTabBar::tab:selected { color:%(selected_text)s; background:%(surface)s; border-bottom-color:%(accent)s; }
    QScrollArea,QAbstractScrollArea::corner { background:%(bg)s; border:0; }
    QScrollBar:vertical { background:%(soft)s; width:11px; margin:0; }
    QScrollBar:horizontal { background:%(soft)s; height:11px; margin:0; }
    QScrollBar::handle { background:%(border)s; border-radius:2px; min-height:24px; min-width:24px; }
    QScrollBar::handle:hover { background:%(muted)s; }
    QScrollBar::add-line,QScrollBar::sub-line { width:0; height:0; }
    QSplitter::handle { background:%(border)s; }
    QMenu,QToolTip { background:%(surface)s; color:%(text)s; border:1px solid %(border)s; padding:4px; }
    QMenu::item:selected { background:%(selected)s; }
    QCheckBox,QRadioButton { spacing:6px; background:transparent; font-weight:700; }
    QCheckBox::indicator { width:14px; height:14px; border:1px solid %(border)s; border-radius:2px; background:%(surface)s; }
    QCheckBox::indicator:checked { background:%(accent)s; border-color:%(accent)s; image:url("%(check)s"); }
    QCheckBox::indicator:disabled { background:%(soft)s; border-color:%(disabled)s; }
    QProgressBar { border:1px solid %(border)s; background:%(track)s; text-align:center; border-radius:2px; }
    QProgressBar::chunk { background:%(accent)s; }
    QStatusBar { background:%(soft)s; border-top:1px solid %(border)s; color:%(muted)s; font-size:%(tiny)dpx; font-weight:700; }
    QStatusBar::item { border:0; }
    ''' % dict(
        p,
        sidebar_bg=sidebar,
        brand_title=p['navy'] if mode == 'light' else '#ffffff',
        brand_sub=p['muted'] if mode == 'light' else 'rgba(255,255,255,0.68)',
        nav_title=p['muted'] if mode == 'light' else 'rgba(255,255,255,0.54)',
        nav_item=p['text'] if mode == 'light' else 'rgba(255,255,255,0.74)',
        nav_bg='rgba(86,169,249,0.12)' if mode == 'light' else 'rgba(86,169,249,0.20)',
        nav_border='rgba(86,169,249,0.52)' if mode == 'light' else 'rgba(86,169,249,0.72)',
        user_name=p['text'] if mode == 'light' else '#ffffff',
        user_role=p['muted'] if mode == 'light' else 'rgba(255,255,255,0.62)',
        theme_fg=p['navy'] if mode == 'light' else p['text'],
        r=METRICS['radius'],
        nav_h=round(METRICS['nav_item_height'] * scale),
        ctrl_h=round(METRICS['control_height'] * scale),
        btn_h=round(34 * scale), input_h=round(22 * scale),
        tiny=round(11 * scale), small=round(13 * scale), body=round(14 * scale),
        font=round(13 * scale), h1=round(28 * scale), h2=round(18 * scale),
        title=round(20 * scale),
        check=(Path(__file__).parent / 'assets/brand/check.svg').as_posix(),
        chevron=(Path(__file__).parent / 'assets/brand/chevron.svg').as_posix())


def apply_theme(mode, scale=1):
    global _mode
    if mode not in PALETTES: raise ValueError('Unsupported theme')
    _mode = mode
    app = QApplication.instance()
    if app is None: return
    app.setStyle('Fusion')
    palette = QPalette()
    for role, key in [(QPalette.Window, 'bg'), (QPalette.WindowText, 'text'), (QPalette.Base, 'surface'),
                      (QPalette.AlternateBase, 'soft'), (QPalette.Text, 'text'), (QPalette.Button, 'surface'),
                      (QPalette.ButtonText, 'text'), (QPalette.Highlight, 'selected'),
                      (QPalette.HighlightedText, 'selected_text'), (QPalette.ToolTipBase, 'surface'),
                      (QPalette.ToolTipText, 'text'), (QPalette.PlaceholderText, 'muted')]:
        palette.setColor(role, QColor(color(key)))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, QColor(color('disabled')))
    app.setPalette(palette)
    app.setStyleSheet(stylesheet(mode, scale))
    for widget in app.allWidgets(): widget.update()

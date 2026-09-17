"""Presentation-layer widgets for the MES Vision operator shell.

Layout and styling only. Nothing here touches the camera, the inspection pipeline
or the robot: every widget takes plain strings and emits plain signals, so the
window code keeps owning all behaviour.

The structure mirrors the MonoFactory MES web shell (sidebar, page header,
context bar) with the corner radii and density of the inspection application.
"""
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QPixmap
from PySide6.QtWidgets import (QWidget, QFrame, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
                               QSizePolicy)

from mes_vision.theme import METRICS, symbol_path

TRACKED = QFont.AbsoluteSpacing


def styled(widget):
    """Plain QWidget containers ignore stylesheet backgrounds and borders unless
    WA_StyledBackground is set. Every object-name styled container goes through here."""
    widget.setAttribute(Qt.WA_StyledBackground, True)
    return widget


def _tracked(label, px):
    """QSS has no letter-spacing; apply it on the font instead."""
    font = QFont(label.font())
    font.setLetterSpacing(TRACKED, px)
    label.setFont(font)


def divider(vertical=False):
    line = styled(QFrame()); line.setObjectName('contextDivider')
    if vertical: line.setFixedWidth(1)
    else: line.setFixedHeight(1)
    return line


def chip(text):
    label = QLabel(text); label.setObjectName('chip')
    label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    return label


def panel():
    frame = styled(QFrame()); frame.setObjectName('panel')
    layout = QVBoxLayout(frame); layout.setContentsMargins(0, 0, 0, 0); layout.setSpacing(0)
    return frame, layout


class NavItem(QPushButton):
    """Sidebar entry: checkable row with the 3px accent bar of the web sidebar."""

    def __init__(self, text, shortcut_hint=None):
        super().__init__(text)
        self.setObjectName('navItem'); self.setCheckable(True); self.setAutoExclusive(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # QPushButton adds its own frame margins on top of the QSS padding, so the
        # row height is pinned here instead of relying on min-height alone.
        self.setFixedHeight(METRICS['nav_item_height'])
        self.bar = styled(QFrame(self)); self.bar.setObjectName('navActiveBar'); self.bar.hide()
        self.hint = None
        if shortcut_hint:
            self.hint = QLabel(shortcut_hint, self); self.hint.setObjectName('navKey')
            self.hint.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.toggled.connect(self.bar.setVisible)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        height = min(22, self.height() - 8)
        self.bar.setGeometry(0, (self.height() - height) // 2, 3, height)
        if self.hint:
            self.hint.adjustSize()
            self.hint.move(self.width() - self.hint.width() - 12,
                           (self.height() - self.hint.height()) // 2)


class Sidebar(QWidget):
    """Brand header, grouped navigation, user card and the primary action button."""

    current_changed = Signal(int)
    action_clicked = Signal()

    def __init__(self, title='MES Vision', subtitle='Inspection Operations'):
        super().__init__()
        styled(self); self.setObjectName('brandSidebar')
        self.setFixedWidth(METRICS['sidebar_width'])
        self.items = []; self.group_titles = []

        root = QVBoxLayout(self); root.setContentsMargins(18, 24, 18, 20); root.setSpacing(20)

        head = QHBoxLayout(); head.setContentsMargins(4, 0, 4, 10); head.setSpacing(12)
        self.symbol = QLabel(); self.symbol.setObjectName('brandSymbol')
        self.symbol.setFixedSize(54, 54); self.symbol.setAlignment(Qt.AlignCenter)
        head.addWidget(self.symbol)
        names = QVBoxLayout(); names.setSpacing(2)
        self.title = QLabel(title); self.title.setObjectName('brandTitle')
        self.subtitle = QLabel(subtitle); self.subtitle.setObjectName('brandSub')
        names.addWidget(self.title); names.addWidget(self.subtitle)
        head.addLayout(names); head.addStretch(1)
        root.addLayout(head)

        self.nav_host = QWidget()
        self.nav_host.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        self.nav = QVBoxLayout(self.nav_host); self.nav.setContentsMargins(0, 0, 0, 0); self.nav.setSpacing(14)
        root.addWidget(self.nav_host)
        root.addStretch(1)

        self.foot = QVBoxLayout(); self.foot.setSpacing(10)
        self.user_card = styled(QWidget()); self.user_card.setObjectName('userCard')
        user = QHBoxLayout(self.user_card); user.setContentsMargins(10, 10, 10, 10); user.setSpacing(10)
        self.avatar = QLabel('O'); self.avatar.setObjectName('userAvatar'); self.avatar.setFixedSize(36, 36)
        user.addWidget(self.avatar)
        who = QVBoxLayout(); who.setSpacing(2)
        self.user_name = QLabel('운영자'); self.user_name.setObjectName('userName')
        self.user_role = QLabel('Inspection Operator'); self.user_role.setObjectName('userRole')
        who.addWidget(self.user_name); who.addWidget(self.user_role)
        user.addLayout(who, 1)
        self.foot.addWidget(self.user_card)
        self.action = QPushButton('모델 준비'); self.action.setObjectName('ctaButton')
        self.action.clicked.connect(self.action_clicked)
        self.foot.addWidget(self.action)
        root.addLayout(self.foot)

    def add_group(self, title, entries):
        """entries: list of (label, shortcut_hint). Returns (group title label, buttons).

        The caption is uppercased by the font rather than by the string so a
        translated caption keeps its binding and still reads as a section header.
        """
        group = QVBoxLayout(); group.setSpacing(5)
        label = QLabel(title); label.setObjectName('navGroupTitle')
        font = QFont(label.font()); font.setCapitalization(QFont.AllUppercase); label.setFont(font)
        _tracked(label, 1.5)
        self.group_titles.append(label)
        group.addWidget(label)
        created = []
        for text, hint in entries:
            item = NavItem(text, hint)
            index = len(self.items)
            item.clicked.connect(lambda _=False, i=index: self.current_changed.emit(i))
            self.items.append(item); created.append(item)
            group.addWidget(item)
        self.nav.addLayout(group)
        return label, created

    def set_current(self, index):
        if 0 <= index < len(self.items) and not self.items[index].isChecked():
            self.items[index].setChecked(True)

    def current_index(self):
        for i, item in enumerate(self.items):
            if item.isChecked(): return i
        return -1

    def set_user(self, name, role):
        self.user_name.setText(name); self.user_role.setText(role)
        self.avatar.setText((name or 'O').strip()[:1].upper())

    def set_scale(self, scale=1):
        for item in self.items:
            item.setFixedHeight(round(METRICS['nav_item_height'] * scale))

    def apply_mode(self, mode, scale=1):
        pixmap = QPixmap(str(symbol_path(mode)))
        side = round(54 * scale)
        self.symbol.setFixedSize(side, side)
        if not pixmap.isNull():
            self.symbol.setPixmap(pixmap.scaled(side - 6, side - 6, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.setFixedWidth(round(METRICS['sidebar_width'] * scale))
        self.set_scale(scale)
        for label in self.group_titles:
            font = QFont(label.font()); font.setCapitalization(QFont.AllUppercase); label.setFont(font)
            _tracked(label, 1.5 * scale)


class PageHeader(QWidget):
    """Kicker / title / description on the left, status controls on the right."""

    def __init__(self):
        super().__init__()
        root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(18)
        top = QHBoxLayout(); top.setSpacing(24)
        heading = QVBoxLayout(); heading.setSpacing(2)
        self.kicker = QLabel(); self.kicker.setObjectName('pageKicker'); _tracked(self.kicker, 1.4)
        self.title = QLabel(); self.title.setObjectName('pageTitle')
        self.description = QLabel(); self.description.setObjectName('pageDesc')
        self.description.setWordWrap(True)
        heading.addWidget(self.kicker); heading.addWidget(self.title); heading.addWidget(self.description)
        top.addLayout(heading, 1)
        self.actions = QHBoxLayout(); self.actions.setSpacing(10)
        self.actions.setAlignment(Qt.AlignTop | Qt.AlignRight)
        top.addLayout(self.actions)
        root.addLayout(top)
        self.body = QVBoxLayout(); self.body.setSpacing(0)
        root.addLayout(self.body)

    def set_page(self, kicker, title, description=''):
        self.kicker.setText(kicker.upper()); self.title.setText(title)
        self.description.setText(description); self.description.setVisible(bool(description))


class ContextBar(QWidget):
    """The five-cell production context strip from the MES header."""

    STATES = {'': 'ctxValue', 'ok': 'ctxValueOk', 'warn': 'ctxValueWarn'}

    def __init__(self, keys):
        super().__init__()
        styled(self); self.setObjectName('contextBar')
        self.values = {}
        row = QHBoxLayout(self); row.setContentsMargins(0, 0, 0, 0); row.setSpacing(0)
        for position, key in enumerate(keys):
            cell = QVBoxLayout(); cell.setContentsMargins(12, 8, 12, 8); cell.setSpacing(3)
            name = QLabel(key.upper()); name.setObjectName('ctxKey'); _tracked(name, 1)
            value = QLabel('—'); value.setObjectName('ctxValue')
            value.setTextInteractionFlags(Qt.TextSelectableByMouse)
            cell.addWidget(name); cell.addWidget(value)
            row.addLayout(cell, 1)
            self.values[key] = value
            if position < len(keys) - 1: row.addWidget(divider(vertical=True))

    def set_value(self, key, text, state=''):
        label = self.values.get(key)
        if label is None: return
        label.setText(text)
        label.setObjectName(self.STATES.get(state, 'ctxValue'))
        label.style().unpolish(label); label.style().polish(label)


class VerdictBanner(QWidget):
    """Full-width judgement strip above the image panes."""

    OBJECTS = {'ok': 'verdictOk', 'ng': 'verdictNg', 'review': 'verdictReview', 'idle': 'verdictIdle'}

    def __init__(self):
        super().__init__()
        styled(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(self); row.setContentsMargins(12, 7, 12, 7); row.setSpacing(12)
        self.mark = QLabel(); self.mark.setObjectName('verdictMark')
        self.main = QLabel(); self.main.setObjectName('verdictMain')
        self.sub = QLabel(); self.sub.setObjectName('verdictSub')
        self.stamp = QLabel(); self.stamp.setObjectName('verdictSub')
        for label in (self.mark, self.main, self.sub, self.stamp):
            label.setWordWrap(False); label.setTextFormat(Qt.PlainText)
            label.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Preferred)
        row.addWidget(self.mark); row.addWidget(self.main); row.addWidget(self.sub)
        row.addStretch(1); row.addWidget(self.stamp)
        self.set_idle('물체를 선택하면 판정이 표시됩니다.')

    def _restyle(self, state):
        self.setObjectName(self.OBJECTS.get(state, 'verdictIdle'))
        self.style().unpolish(self); self.style().polish(self)

    def set_idle(self, text):
        self._restyle('idle')
        self.mark.hide(); self.sub.hide(); self.stamp.clear()
        self.main.setObjectName('verdictIdleText')
        self.main.style().unpolish(self.main); self.main.style().polish(self.main)
        self.main.setText(text); self.main.show()

    def set_verdict(self, state, mark, main, sub='', stamp=''):
        self._restyle(state)
        self.main.setObjectName('verdictMain')
        self.main.style().unpolish(self.main); self.main.style().polish(self.main)
        self.mark.setText(mark); self.mark.show()
        self.main.setText(main)
        self.sub.setText(sub); self.sub.setVisible(bool(sub))
        self.stamp.setText(stamp)


class SquareHolder(QWidget):
    """Keeps one child centred at a 1:1 aspect ratio.

    The workbench is square and the pipeline is fed a square centre crop, so every
    pane shows the frame without letterboxing.
    """

    def __init__(self, child):
        super().__init__()
        self.child = child
        self.child.setParent(self)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(80, 80)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        side = min(self.width(), self.height())
        self.child.setGeometry((self.width() - side) // 2, (self.height() - side) // 2, side, side)

    def side(self):
        return min(self.width(), self.height())


class ViewArea(QWidget):
    """Full-size pane on the left, two stacked panes on the right.

    Widths are derived from the available height because all three panes are
    square: the tall pane is as wide as the area is high, and each stacked pane is
    half of that. Recomputed on resize so the window stays usable when resized.

    The three arguments are ImagePanel instances (or anything with a square body).
    """

    CHROME = 58          # ImagePanel title row + zoom control row + spacing
    GAP = 8

    def __init__(self, primary, secondary, tertiary):
        super().__init__()
        styled(self); self.setObjectName('viewArea')
        self.primary, self.secondary, self.tertiary = primary, secondary, tertiary
        row = QHBoxLayout(self); row.setContentsMargins(8, 8, 8, 8); row.setSpacing(self.GAP)
        row.addWidget(primary)
        column = QVBoxLayout(); column.setSpacing(self.GAP)
        column.addWidget(secondary); column.addWidget(tertiary)
        self.column = QWidget(); self.column.setLayout(column)
        row.addWidget(self.column)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        inner = self.height() - 16
        if inner <= 0: return
        # Fixing the two pane widths is the whole rule. Qt derives this widget's
        # maximum width from its layout and carries that maximum up through the
        # enclosing card, so the card shrinks to the panes on its own; capping the
        # area or the card explicitly changes nothing (measured, not assumed).
        # What does matter is that the judgement column beside the card carries NO
        # maximum width - QBoxLayout centres a widget it cannot stretch, which is
        # what put dead space on both sides of it.
        self.primary.setFixedWidth(max(120, inner - self.CHROME))
        self.column.setFixedWidth(max(80, (inner - self.GAP) // 2 - self.CHROME))

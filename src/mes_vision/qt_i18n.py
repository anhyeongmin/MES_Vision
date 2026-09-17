"""Explicit Qt display bindings; never infer translations from user text.

ui_text wraps the existing Qt call and remembers only DisplayText arguments.
Refreshing changes captions in place without reconstructing widgets, querying
storage, emitting input signals, or restarting workers.
"""
import weakref
from PySide6 import QtWidgets as W
from PySide6.QtCore import QObject, QSignalBlocker
from shiboken6 import isValid
from .i18n import DisplayText, render_text, set_language, install_display_font, install_qt_translator

_targets = weakref.WeakValueDictionary()
_ITEM_BINDING_ROLE = 0x0100 + 617


class _ItemBindings:
    def __init__(self): self.values = {}


def _bindings(target, create=False):
    if isinstance(target, (W.QTableWidgetItem, W.QListWidgetItem)):
        # Qt can discard/recreate Python wrappers for C++-owned items. Store the
        # binding with the item, without retaining wrappers or deleted rows.
        holder = target.data(_ITEM_BINDING_ROLE)
        if holder is None and create:
            holder = _ItemBindings(); target.setData(_ITEM_BINDING_ROLE, holder)
        return holder.values if holder is not None else None
    values = getattr(target, '_mes_text_bindings', None)
    if values is None and create: values = target._mes_text_bindings = {}
    return values
_getters = dict(setText='text', setPlainText='toPlainText', setWindowTitle='windowTitle',
                setToolTip='toolTip', setStatusTip='statusTip', setPlaceholderText='placeholderText',
                setSpecialValueText='specialValueText', setFormat='format', setTitle='title',
                setItemText='itemText', setTabText='tabText')


def bound_text(target, setter='setText', *prefix):
    """Copy a caption to another display without losing its translation source."""
    binding = (_bindings(target) or {}).get((setter, prefix))
    current = getattr(target, _getters[setter])(*prefix)
    return binding[0] if binding and current == binding[1] else current


def _remember(target, setter, value, prefix=()):
    key = (setter, prefix)
    bindings = _bindings(target)
    if bindings is None:
        if not isinstance(value, DisplayText): return
        bindings = _bindings(target, create=True)
    if isinstance(value, DisplayText):
        if isinstance(target, QObject): _targets[id(target)] = target
        bindings[key] = (value, str(value))
    else: bindings.pop(key, None)


def ui_text(call, *args, **kwargs):
    target = getattr(call, '__self__', None)
    name = getattr(call, '__name__', '')
    qt_target = isinstance(target, (QObject, W.QTableWidgetItem, W.QListWidgetItem))
    if qt_target and name == 'addRow' and args and isinstance(args[0], DisplayText):
        # QFormLayout creates a QLabel internally; make its source explicit.
        label = ui_text(W.QLabel, args[0])
        return call(label, *args[1:], **kwargs)
    if isinstance(target, (W.QListWidget, W.QComboBox)) and name == 'addItems':
        for value in args[0]: ui_text(target.addItem, value)
        return None
    if isinstance(target, W.QTableWidget) and name in ('setHorizontalHeaderLabels', 'setVerticalHeaderLabels'):
        values = list(args[0]); result = call([render_text(v) for v in values])
        getter = target.horizontalHeaderItem if name == 'setHorizontalHeaderLabels' else target.verticalHeaderItem
        for index, value in enumerate(values):
            item = getter(index)
            if item is not None: _remember(item, 'setText', value)
        return result
    if qt_target and name == 'clear':
        bindings = getattr(target, '_mes_text_bindings', {})
        for key in list(bindings):
            if key[0] in ('setText', 'setPlainText', 'setItemText', 'setTabText'): bindings.pop(key)
    if isinstance(target, (W.QComboBox, W.QTabWidget)) and name in ('removeItem', 'removeTab', 'insertItem'):
        # Indices move, so move their bindings with the items.
        offset = 1 if name == 'insertItem' else -1
        index = args[0]; updated = {}
        for (setter, prefix), value in getattr(target, '_mes_text_bindings', {}).items():
            if prefix and setter in ('setItemText', 'setTabText'):
                old = prefix[0]
                if offset < 0 and old == index: continue
                prefix = (old + offset if old >= index else old,)
            updated[(setter, prefix)] = value
        target._mes_text_bindings = updated
    result = call(*(render_text(v) for v in args), **kwargs)
    if isinstance(call, type) and args and isinstance(args[0], str):
        if isinstance(result, (W.QLabel, W.QAbstractButton, W.QTableWidgetItem, W.QListWidgetItem)):
            _remember(result, 'setText', args[0])
        elif isinstance(result, W.QGroupBox): _remember(result, 'setTitle', args[0])
    elif qt_target:
        if name in _getters and args:
            _remember(target, name, args[-1], tuple(args[:-1]))
        elif name in ('addItem', 'insertItem') and isinstance(target, W.QComboBox):
            index = args[0] if name == 'insertItem' else target.count()-1
            offset = 1 if name == 'insertItem' else 0
            value = args[offset] if isinstance(args[offset], str) else args[offset+1]
            _remember(target, 'setItemText', value, (index,))
        elif name == 'addItem' and isinstance(target, W.QListWidget) and isinstance(args[0], str):
            _remember(target.item(target.count()-1), 'setText', args[0])
        elif name == 'addTab' and isinstance(target, W.QTabWidget):
            _remember(target, 'setTabText', args[-1], (result,))
    return result


def refresh_translations():
    targets = list(_targets.values())
    for widget in W.QApplication.allWidgets():
        if isinstance(widget, W.QTableWidget):
            targets.extend(widget.horizontalHeaderItem(c) for c in range(widget.columnCount()))
            targets.extend(widget.verticalHeaderItem(r) for r in range(widget.rowCount()))
            targets.extend(widget.item(r,c) for r in range(widget.rowCount()) for c in range(widget.columnCount()))
        elif isinstance(widget, W.QListWidget): targets.extend(widget.item(i) for i in range(widget.count()))
    for target in targets:
        if target is None: continue
        if not isValid(target):
            _targets.pop(id(target), None)
            continue
        bindings = _bindings(target)
        if not bindings: continue
        blockers = []
        owner = target if isinstance(target, QObject) else (
            target.tableWidget() if isinstance(target, W.QTableWidgetItem) else target.listWidget())
        if owner is not None: blockers.append(QSignalBlocker(owner))
        combo_edit = None
        if isinstance(target, W.QComboBox) and target.isEditable():
            edit = target.lineEdit(); blockers.append(QSignalBlocker(edit))
            if target.currentText() != target.itemText(target.currentIndex()):
                combo_edit = (target.currentText(), edit.cursorPosition())
        cursor = (target.textCursor().anchor(), target.textCursor().position()) if isinstance(target, W.QPlainTextEdit) else None
        scroll = (target.verticalScrollBar().value(), target.horizontalScrollBar().value()) if cursor else None
        try:
            for key, (source, previous) in list(bindings.items()):
                setter, prefix = key
                # A later user edit or unbound write always owns its text.
                current = getattr(target, _getters[setter])(*prefix)
                if current != previous:
                    bindings.pop(key, None)
                    continue
                translated = str(source)
                if translated != current: getattr(target, setter)(*prefix, translated)
                bindings[key] = (source, translated)
            if combo_edit:
                target.setEditText(combo_edit[0]); target.lineEdit().setCursorPosition(combo_edit[1])
            if cursor:
                # setPlainText resets cursor/scroll; retain their usable positions.
                from PySide6.QtGui import QTextCursor
                restored = target.textCursor(); end = target.document().characterCount()-1
                restored.setPosition(min(cursor[0], end))
                restored.setPosition(min(cursor[1], end), QTextCursor.KeepAnchor)
                target.setTextCursor(restored)
                target.verticalScrollBar().setValue(scroll[0]); target.horizontalScrollBar().setValue(scroll[1])
        finally:
            for blocker in reversed(blockers): blocker.unblock()


def apply_language(app, locale):
    set_language(locale)
    old = getattr(app, '_mes_translator', None)
    if old is not None:
        app.removeTranslator(old); old.deleteLater()
    install_qt_translator(app)
    install_display_font(app)
    refresh_translations()
    for widget in app.allWidgets():
        widget.updateGeometry(); widget.update()

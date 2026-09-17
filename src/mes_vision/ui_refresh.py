from mes_vision.qt_i18n import ui_text
"""Bounded background reads and incremental updates for operator tables."""
from PySide6.QtCore import QObject, QThread, Signal, Slot, QSignalBlocker
from PySide6.QtWidgets import QTableWidgetItem


class _ReadThread(QThread):
    def __init__(self, read, parent):
        super().__init__(parent)
        self.read = read
        self.value = None
        self.error = None

    def run(self):
        try:
            self.value = self.read()
        except Exception as exc:
            self.error = str(exc)


class BackgroundRead(QObject):
    """At most one read; discard results superseded by an explicit UI action."""
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent):
        super().__init__(parent)
        self.job = None
        self.revision = 0
        self.started_revision = 0

    @property
    def busy(self):
        return self.job is not None

    def invalidate(self):
        self.revision += 1

    def start(self, read):
        if self.busy:
            return False
        self.started_revision = self.revision
        self.job = _ReadThread(read, self)
        self.job.finished.connect(self.finished)
        self.job.start()
        return True

    @Slot()
    def finished(self):
        job = self.job
        current = self.started_revision == self.revision
        self.job = None
        job.deleteLater()
        if current:
            if job.error is None:
                self.succeeded.emit(job.value)
            else:
                self.failed.emit(job.error)


def update_table(table, rows, *, selected_key=None, select_first=False):
    """Rows are (stable key, ((text, tooltip), ...)); retain items and scroll."""
    rows = tuple((key, tuple(cells)) for key, cells in rows)
    selected_row = next((i for i, (key, _) in enumerate(rows) if key == selected_key), None)
    if selected_row is None and select_first and rows:
        selected_row = 0
    if (rows == getattr(table, "_display_rows", None)
            and table.currentRow() == (selected_row if selected_row is not None else -1)):
        return
    scroll = table.verticalScrollBar().value()
    horizontal = table.horizontalScrollBar().value()
    blocker = QSignalBlocker(table)
    updates_enabled = table.updatesEnabled()
    table.setUpdatesEnabled(False)
    try:
        if table.rowCount() != len(rows):
            table.setRowCount(len(rows))
        for index, (key, cells) in enumerate(rows):
            for column, (text, tooltip) in enumerate(cells):
                item = table.item(index, column)
                if item is None:
                    item = ui_text(QTableWidgetItem, text)
                    table.setItem(index, column, item)
                elif item.text() != text:
                    ui_text(item.setText, text)
                if item.toolTip() != tooltip:
                    ui_text(item.setToolTip, tooltip)
        if selected_row is None:
            table.clearSelection()
            table.setCurrentCell(-1, -1)
        elif table.currentRow() != selected_row or not table.selectedItems():
            table.selectRow(selected_row)
        table.verticalScrollBar().setValue(scroll)
        table.horizontalScrollBar().setValue(horizontal)
        table._display_rows = rows
    finally:
        table.setUpdatesEnabled(updates_enabled)
        blocker.unblock()

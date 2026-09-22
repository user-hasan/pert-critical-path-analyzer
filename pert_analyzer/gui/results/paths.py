"""
Critical path list widget: shared by the Overview and Critical Paths tabs.

Read-only presentation of the backend critical paths. Provides selection,
copy-to-clipboard, and a highlighted-selection signal used for cross-
navigation with the network canvas.
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.results.formatting import format_duration
from pert_analyzer.gui.themes.palette import ACCENT, BORDER, SURFACE, TEXT_MUTED
from pert_analyzer.gui.themes.typography import MUTED_FONT, STATUS_FONT


def format_path(path: List[str]) -> str:
    """Render an activity chain as 'A \u2192 B \u2192 C'."""
    return " \u2192 ".join(path) if path else "(empty path)"


class PathList(QWidget):
    """Scrollable list of critical paths with copy and highlight."""

    path_selected = Signal(object)  # list of activity ids

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._paths: List[List[str]] = []
        self._empty_text = "No critical paths found."

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)

        header = QHBoxLayout()
        self._count_label = QLabel("")
        self._count_label.setFont(QFont(*STATUS_FONT))
        header.addWidget(self._count_label)
        header.addStretch()
        self._copy_btn = QPushButton("Copy selected")
        self._copy_btn.setToolTip("Copy the selected path to the clipboard")
        self._copy_btn.setEnabled(False)
        self._copy_btn.clicked.connect(self.copy_selected)
        header.addWidget(self._copy_btn)
        root.addLayout(header)

        self._list = QListWidget()
        self._list.setAlternatingRowColors(False)
        self._list.setStyleSheet(
            f"QListWidget {{ background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 8px;}}"
            f"QListWidget::item {{ padding: 6px 8px;"
            f" border-bottom: 1px solid {BORDER};}}"
            f"QListWidget::item:selected {{ background-color: {ACCENT};"
            f" color: white;}}"
        )
        self._list.currentRowChanged.connect(self._on_selection_changed)
        root.addWidget(self._list, stretch=1)

    def _on_selection_changed(self, row: int) -> None:
        self._copy_btn.setEnabled(0 <= row < len(self._paths))
        if 0 <= row < len(self._paths):
            self.path_selected.emit(list(self._paths[row]))

    def set_paths(self, paths: List[List[str]], project_duration: Optional[float]) -> None:
        """Populate the list. ``paths`` count reflects the backend result."""
        self._paths = [list(p) for p in paths]
        self._list.clear()
        count = len(self._paths)
        self._count_label.setText(
            f"Critical paths: {count}"
            if count
            else "Critical paths: 0"
        )
        for index, path in enumerate(self._paths, start=1):
            chain = format_path(path)
            duration = _path_duration(path, project_duration)
            item = QListWidgetItem(
                f"Path {index}   {chain}\nDuration: {duration}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index - 1)
            self._list.addItem(item)
        self._copy_btn.setEnabled(count > 0)
        if count:
            self._list.setCurrentRow(0)
        else:
            self._show_empty_note()

    def clear(self) -> None:
        self._paths = []
        self._list.clear()
        self._count_label.setText("Critical paths: 0")

    def select_path(self, index: int) -> None:
        if 0 <= index < self._list.count():
            self._list.setCurrentRow(index)

    def reselect_current(self) -> None:
        """Re-emit the current selection after a view rebuild."""
        self._on_selection_changed(self._list.currentRow())

    def selected_row(self) -> int:
        return self._list.currentRow()

    def copy_selected(self) -> None:
        """Copy the selected path chain to the system clipboard."""
        row = self._list.currentRow()
        if not (0 <= row < len(self._paths)):
            return
        text = format_path(self._paths[row])
        try:
            from PySide6.QtWidgets import QApplication

            clipboard = QApplication.clipboard()
            if clipboard is not None:
                clipboard.setText(text)
        except Exception:
            pass

    def _show_empty_note(self) -> None:
        item = QListWidgetItem(self._empty_text)
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        self._list.addItem(item)
        self._copy_btn.setEnabled(False)


def _path_duration(path: List[str], project_duration: Optional[float]) -> str:
    """Show the backend project duration for each critical path.

    Critical paths share the project duration by definition; the backend
    result is authoritative and is never recomputed here.
    """
    return format_duration(project_duration)


class EmptyPathLabel(QLabel):
    """Muted placeholder text for secondary panels."""

    def __init__(self, text: str, parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setFont(QFont(*MUTED_FONT))
        self.setStyleSheet(f"color: {TEXT_MUTED};")
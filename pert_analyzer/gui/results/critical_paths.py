"""
Critical Paths tab: dedicated critical-path analysis view.

Master-detail layout: path list on left, selected path details on right.
Selection propagates to the network canvas via path_selected.
"""

from __future__ import annotations

from typing import Any, List, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.results.formatting import format_duration
from pert_analyzer.gui.results.paths import PathList
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    SURFACE,
    SURFACE_LIGHT,
    TEXT,
    TEXT_MUTED,
    TEXT_SECONDARY,
)
from pert_analyzer.gui.themes.typography import (
    BODY_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    TITLE_FONT,
)


class CriticalPathsTab(QWidget):
    """Dedicated critical path list with master-detail layout."""

    path_selected = Signal(object)  # list of activity ids

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._data: Any = None
        self._paths: List[List[str]] = []

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(8)

        # Header with summary
        header = QHBoxLayout()
        header.setSpacing(12)
        heading = QLabel("Critical Paths")
        heading.setFont(QFont(*TITLE_FONT))
        heading.setStyleSheet(f"color: {TEXT};")
        header.addWidget(heading)
        header.addStretch()

        self._summary_label = QLabel("")
        self._summary_label.setFont(QFont(*STATUS_FONT))
        self._summary_label.setStyleSheet(f"color: {ACCENT};")
        header.addWidget(self._summary_label)
        root.addLayout(header)

        self._duration_hint = QLabel("")
        self._duration_hint.setFont(QFont(*MUTED_FONT))
        self._duration_hint.setStyleSheet(f"color: {TEXT_MUTED};")
        root.addWidget(self._duration_hint)

        # Master-detail splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(
            f"QSplitter::handle {{ background-color: {BORDER};"
            f" border-radius: 3px; }}"
        )

        # LEFT: path list (master)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        left_label = QLabel("Path list")
        left_label.setFont(QFont(*LABEL_FONT))
        left_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        left_layout.addWidget(left_label)

        self._path_list = PathList()
        self._path_list.path_selected.connect(self._on_path_selected)
        left_layout.addWidget(self._path_list, stretch=1)
        splitter.addWidget(left_widget)

        # RIGHT: path details (detail)
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(4)

        right_label = QLabel("Selected path details")
        right_label.setFont(QFont(*LABEL_FONT))
        right_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        right_layout.addWidget(right_label)

        self._detail_frame = QFrame()
        self._detail_frame.setFrameShape(QFrame.Shape.NoFrame)
        self._detail_frame.setStyleSheet(
            f"QFrame {{ background-color: {SURFACE_LIGHT};"
            f" border: 1px solid {BORDER}; border-radius: 8px; }}"
        )
        detail_layout = QVBoxLayout(self._detail_frame)
        detail_layout.setContentsMargins(16, 12, 16, 12)
        detail_layout.setSpacing(8)

        self._path_number = QLabel("")
        self._path_number.setFont(QFont(*TITLE_FONT))
        self._path_number.setStyleSheet(f"color: {TEXT};")
        detail_layout.addWidget(self._path_number)

        self._path_duration = QLabel("")
        self._path_duration.setFont(QFont(*STATUS_FONT))
        self._path_duration.setStyleSheet(f"color: {ACCENT};")
        detail_layout.addWidget(self._path_duration)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        detail_layout.addWidget(sep)

        self._seq_label = QLabel("Activity sequence:")
        self._seq_label.setFont(QFont(*LABEL_FONT))
        self._seq_label.setStyleSheet(f"color: {TEXT_SECONDARY};")
        detail_layout.addWidget(self._seq_label)

        self._path_sequence = QTextEdit()
        self._path_sequence.setReadOnly(True)
        self._path_sequence.setFrameShape(QFrame.Shape.NoFrame)
        self._path_sequence.setStyleSheet(
            f"QTextEdit {{ background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: 6px;"
            f" padding: 8px; color: {TEXT}; font-size: 13px; }}"
        )
        self._path_sequence.setMaximumHeight(200)
        detail_layout.addWidget(self._path_sequence)

        self._path_activity_count = QLabel("")
        self._path_activity_count.setFont(QFont(*MUTED_FONT))
        self._path_activity_count.setStyleSheet(f"color: {TEXT_MUTED};")
        detail_layout.addWidget(self._path_activity_count)

        detail_layout.addStretch()

        right_layout.addWidget(self._detail_frame, stretch=1)
        splitter.addWidget(right_widget)

        splitter.setSizes([280, 420])
        root.addWidget(splitter, stretch=1)

    def _on_path_selected(self, path: List[str]) -> None:
        self.path_selected.emit(path)
        self._show_detail(path)

    def _show_detail(self, path: List[str]) -> None:
        index = self._paths.index(path) + 1 if path in self._paths else 0
        self._path_number.setText(f"Path {index}")
        if self._data is not None:
            self._path_duration.setText(
                f"Duration: {format_duration(self._data.project_duration)}"
            )

        chain = " \u2192 ".join(path)
        self._path_sequence.setText(chain)
        self._path_activity_count.setText(
            f"{len(path)} activities in this critical path"
        )

    def _show_empty(self) -> None:
        self._path_number.setText("Select a path")
        self._path_duration.setText("")
        self._path_sequence.setText(
            "Select a path from the list to view its details."
        )
        self._path_activity_count.setText("")

    def set_data(self, data: Any, auto_select: bool = True) -> None:
        self._data = data
        self._paths = [list(p) for p in data.critical_paths]
        self._path_list.set_paths(data.critical_paths, data.project_duration)

        count = len(data.critical_paths)
        self._summary_label.setText(f"{count} critical path{'s' if count != 1 else ''}")
        if data.project_duration is not None:
            self._duration_hint.setText(
                f"All critical paths share the project duration of "
                f"{format_duration(data.project_duration)}"
            )

        self._selected_index = -1
        if auto_select and self._paths:
            self.select_path(0)
        else:
            self._show_empty()

    def path_list(self) -> PathList:
        return self._path_list

    def select_path(self, index: int) -> None:
        if 0 <= index < len(self._paths):
            self._selected_index = index
            self._path_list.select_path(index)
            self._show_detail(self._paths[index])
            self.path_selected.emit(list(self._paths[index]))

"""
Base class for summary/status pages (Review, Validation, Results).
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QLabel,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.themes.palette import TEXT_MUTED
from pert_analyzer.gui.themes.typography import SUBTITLE_FONT, TITLE_FONT


class SummaryPage(QWidget):
    """Reusable page with title, subtitle, empty state, and summary pane."""

    def __init__(self, title: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)

        self._title_label = QLabel(title)
        self._title_label.setFont(QFont(*TITLE_FONT))
        layout.addWidget(self._title_label)

        self._subtitle_label = QLabel(subtitle)
        self._subtitle_label.setFont(QFont(*SUBTITLE_FONT))
        self._subtitle_label.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self._subtitle_label)

        self._empty_label = QLabel("")
        self._empty_label.setWordWrap(True)
        self._empty_label.setStyleSheet(
            f"color: {TEXT_MUTED}; padding: 16px 0px;"
        )
        layout.addWidget(self._empty_label)

        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        self._detail_edit = QPlainTextEdit()
        self._detail_edit.setReadOnly(True)
        self._detail_edit.setMaximumHeight(300)
        layout.addWidget(self._detail_edit)

        layout.addStretch()

        self._show_empty(
            "No project analyzed. Run an analysis on the Analyze Diagram page."
        )

    def _show_empty(self, message: str) -> None:
        self._empty_label.setText(message)
        self._empty_label.setVisible(True)
        self._summary_label.setVisible(False)
        self._detail_edit.setVisible(False)

    def _show_summary(self, summary_text: str, details: str = "") -> None:
        self._empty_label.setVisible(False)
        self._summary_label.setVisible(True)
        self._summary_label.setText(summary_text)
        self._detail_edit.setVisible(bool(details))
        self._detail_edit.setPlainText(details)

    def refresh(self, session: object) -> None:
        """Refresh the page content from the session."""
        workflow = getattr(session, "workflow", None)
        if workflow is None:
            self._show_empty(
                "No project analyzed. Run an analysis on the Analyze Diagram page."
            )
            return
        self._refresh_content(workflow)

    def _refresh_content(self, workflow: object) -> None:  # noqa: B027
        """Override in subclasses to render workflow data."""

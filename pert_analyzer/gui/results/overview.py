"""
Overview dashboard tab: executive KPI cards + critical path summary.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.results.activities import ActivitiesTab
from pert_analyzer.gui.results.formatting import format_duration, format_number
from pert_analyzer.gui.results.network import NetworkTab
from pert_analyzer.gui.results.paths import PathList
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE,
    TEXT,
    TEXT_MUTED,
    WARNING,
)
from pert_analyzer.gui.themes.spacing import RADIUS_LG
from pert_analyzer.gui.themes.typography import (
    KPI_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    TITLE_FONT,
)


class KpiCard(QFrame):
    """A compact KPI value card used in the Results dashboard."""

    def __init__(self, title: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setStyleSheet(
            f"background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_LG}px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)
        self._title_label = QLabel(title)
        self._title_label.setFont(QFont(*MUTED_FONT))
        self._title_label.setStyleSheet(
            f"color: {TEXT_MUTED}; border: none; background: transparent;"
        )
        layout.addWidget(self._title_label)
        self._value_label = QLabel("Unavailable")
        self._value_label.setFont(QFont(*KPI_FONT))
        self._value_label.setStyleSheet(
            "border: none; background: transparent;"
        )
        layout.addWidget(self._value_label)

    def set_value(self, text: str) -> None:
        self._value_label.setText(text)

    def set_value_color(self, color: str) -> None:
        self._value_label.setStyleSheet(
            f"color: {color}; border: none; background: transparent;"
        )

    def title(self) -> str:
        return self._title_label.text()

    def value(self) -> str:
        return self._value_label.text()


class OverviewTab(QWidget):
    """Executive summary: KPI cards and a compact critical path list."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._data: Any = None
        self._kpi_cards: dict[str, KpiCard] = {}
        self._path_list = PathList()
        self._embedded_network = NetworkTab()
        self._embedded_network.setMinimumHeight(320)
        self._embedded_activities = ActivitiesTab()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)

        heading = QLabel("Overview")
        heading.setFont(QFont(*TITLE_FONT))
        root.addWidget(heading)

        self._cards_row = QHBoxLayout()
        self._cards_row.setSpacing(10)
        for key, title in (
            ("activities", "Activities"),
            ("dependencies", "Dependencies"),
            ("duration", "Project Duration"),
            ("critical_paths", "Critical Paths"),
            ("critical_activities", "Critical Activities"),
        ):
            card = KpiCard(title)
            self._kpi_cards[key] = card
            self._cards_row.addWidget(card, stretch=1)
        root.addLayout(self._cards_row)

        # Compact network summary strip below the KPI row.
        self._network_summary = QLabel("")
        self._network_summary.setWordWrap(True)
        self._network_summary.setFont(QFont(*MUTED_FONT))
        self._network_summary.setStyleSheet(
            f"background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_LG}px;"
            f" padding: 10px 14px; color: {TEXT_MUTED};"
        )
        root.addWidget(self._network_summary)

        # Scrollable analytics surface: embedded network, paths, activities.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        body = QWidget()
        body.setStyleSheet("background: transparent; border: none;")
        body_layout = QVBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(8)
        body_layout.addWidget(self._section_heading("Project network"))
        body_layout.addWidget(self._embedded_network)
        body_layout.addWidget(self._section_heading("Critical paths"))
        body_layout.addWidget(self._path_list)
        body_layout.addWidget(self._section_heading("Activities"))
        body_layout.addWidget(self._embedded_activities)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

    @staticmethod
    def _section_heading(text: str) -> QLabel:
        label = QLabel(text)
        label.setFont(QFont(*LABEL_FONT))
        label.setStyleSheet(f"color: {TEXT}; background: transparent; border: none;")
        return label

    def set_data(self, data: Any) -> None:
        self._data = data
        self._kpi_cards["activities"].set_value(str(data.activity_count))
        self._kpi_cards["activities"].set_value_color(TEXT)
        self._kpi_cards["dependencies"].set_value(str(data.dependency_count))
        self._kpi_cards["dependencies"].set_value_color(TEXT)
        self._kpi_cards["duration"].set_value(format_duration(data.project_duration))
        self._kpi_cards["duration"].set_value_color(ACCENT)
        self._kpi_cards["critical_activities"].set_value(
            str(data.critical_activity_count)
        )
        self._kpi_cards["critical_activities"].set_value_color(WARNING)
        self._kpi_cards["critical_paths"].set_value(
            str(data.critical_path_count)
            if data.critical_path_count is not None
            else "0"
        )
        self._kpi_cards["critical_paths"].set_value_color(WARNING)
        non_critical = max(
            0,
            (data.activity_count or 0) - (data.critical_activity_count or 0),
        )
        self._network_summary.setText(
            f"The reviewed network contains {data.activity_count} activities "
            f"and {data.dependency_count} dependencies with a project duration "
            f"of {format_duration(data.project_duration)}. "
            f"{data.critical_activity_count} critical activities lie on "
            f"{data.critical_path_count or 0} critical path(s); "
            f"{non_critical} non-critical activities complete the network."
        )
        self._path_list.set_paths(data.critical_paths, data.project_duration)
        self._embedded_network.set_data(data)
        self._embedded_activities.set_data(data.activities)

    def path_list(self) -> PathList:
        return self._path_list

    def embedded_network(self) -> NetworkTab:
        return self._embedded_network

    def embedded_activities(self) -> ActivitiesTab:
        return self._embedded_activities
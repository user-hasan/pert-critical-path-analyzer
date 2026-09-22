"""
Results Dashboard (Phase 4): real CPM results page.

Replaces the placeholder with an executive Overview, a Qt-native Network
canvas, an Activities table with detail, and a Critical Paths view.

Readiness is gated on: analyzed workflow + applied candidate + valid graph +
open CPM gate + existing CPM result. When not ready the page explains why
and offers navigation. The backend CPM result is authoritative; this page
never recomputes ES/EF/LS/LF, floats, paths, or duration.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.session import ValidationCenterStatus
from pert_analyzer.gui.results.activities import ActivitiesTab
from pert_analyzer.gui.results.critical_paths import CriticalPathsTab
from pert_analyzer.gui.results.data import (
    CPM_BLOCKED,
    GRAPH_INVALID,
    NO_ANALYSIS,
    RESULT_UNAVAILABLE,
    REVIEW_REQUIRED,
    describe_ready,
    extract,
    extract_pert,
)
from pert_analyzer.gui.results.formatting import format_duration
from pert_analyzer.gui.results.network import NetworkTab
from pert_analyzer.gui.results.overview import OverviewTab
from pert_analyzer.gui.results.pert import PertTab
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE_LIGHT,
    TEXT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
)
from pert_analyzer.gui.themes.spacing import RADIUS_SM
from pert_analyzer.gui.themes.typography import (
    BODY_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    SUBTITLE_FONT,
    TITLE_FONT,
)

_PLACEHOLDER_TOOLTIP = "Coming in a later phase"
_EXPORT_TOOLTIP = "Export the current results (PDF, Excel, JSON, CSV)"
_REPORT_TOOLTIP = "Generate a detailed project report"


def _rgba(hex_color: str, alpha: int) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}, {alpha})"


class ResultsPage(QWidget):
    """Full analytical Results dashboard."""

    go_review = Signal()
    go_validation = Signal()
    calculate_requested = Signal()
    pert_run_requested = Signal()
    export_requested = Signal()
    report_requested = Signal()

    _EMPTY = 0
    _DASHBOARD = 1

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session: Any = None
        self._busy: bool = False
        self._data: Any = None

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        root.setSpacing(8)

        title = QLabel("Results Dashboard")
        title.setFont(QFont(*TITLE_FONT))
        title.setStyleSheet(f"color: {TEXT};")
        root.addWidget(title)

        subtitle = QLabel("Critical path analysis of the reviewed graph.")
        subtitle.setFont(QFont(*SUBTITLE_FONT))
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        root.addWidget(subtitle)

        self._stack = QStackedWidget()
        root.addWidget(self._stack, stretch=1)

        self._stack.addWidget(self._build_empty_state())   # 0
        self._stack.addWidget(self._build_dashboard())     # 1

        self._show_not_ready(NO_ANALYSIS)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_empty_state(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(12)
        layout.addStretch()

        self._not_ready_heading = QLabel("RESULTS NOT READY")
        self._not_ready_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._not_ready_heading.setFont(QFont(*LABEL_FONT))
        self._not_ready_heading.setStyleSheet(f"color: {WARNING};")
        layout.addWidget(self._not_ready_heading)

        self._empty_label = QLabel("")
        self._empty_label.setWordWrap(True)
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._empty_label.setFont(QFont(*BODY_FONT))
        self._empty_label.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(self._empty_label)

        self._preliminary_card = self._build_preliminary_card()
        layout.addWidget(self._preliminary_card)

        self._readiness_label = QLabel("")
        self._readiness_label.setWordWrap(True)
        self._readiness_label.setTextFormat(Qt.TextFormat.RichText)
        self._readiness_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._readiness_label.setFont(QFont(*MUTED_FONT))
        self._readiness_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self._readiness_label, alignment=Qt.AlignmentFlag.AlignCenter)

        hint = QLabel("Resolve the issue below, then come back to Results.")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint.setFont(QFont(*MUTED_FONT))
        hint.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(hint)

        self._progress_label = QLabel("")
        self._progress_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._progress_label.setFont(QFont(*MUTED_FONT))
        self._progress_label.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self._progress_label)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        actions.addStretch()
        self._go_review_btn = QPushButton("Continue Review")
        self._go_review_btn.clicked.connect(self.go_review)
        self._go_validation_btn = QPushButton("Open Validation")
        self._go_validation_btn.clicked.connect(self.go_validation)
        self._calculate_btn = QPushButton("Calculate Results")
        self._calculate_btn.setToolTip(
            "Run the existing CPM service for this valid graph"
        )
        self._calculate_btn.clicked.connect(self._on_calculate_clicked)
        actions.addWidget(self._go_review_btn)
        actions.addWidget(self._go_validation_btn)
        actions.addWidget(self._calculate_btn)
        actions.addStretch()
        layout.addLayout(actions)

        layout.addStretch()
        self._empty_hint = hint
        return widget

    def _build_preliminary_card(self) -> QWidget:
        card = QFrame()
        card.setObjectName("preliminaryCard")
        card.setStyleSheet(
            f"QFrame#preliminaryCard {{ background-color: {SURFACE_LIGHT};"
            f" border: 1px solid {BORDER}; border-radius: 8px; }}"
        )
        inner = QVBoxLayout(card)
        inner.setContentsMargins(18, 12, 18, 12)
        inner.setSpacing(4)

        title = QLabel("PRELIMINARY")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont(*MUTED_FONT))
        title.setStyleSheet(f"color: {WARNING}; font-weight: 600;")
        inner.addWidget(title)

        caption = QLabel("Detected from image \u2014 may change after review.")
        caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        caption.setFont(QFont(*MUTED_FONT))
        caption.setStyleSheet(f"color: {TEXT_MUTED};")
        inner.addWidget(caption)

        kpi_row = QHBoxLayout()
        kpi_row.setSpacing(18)
        self._preliminary_values: dict[str, tuple[QLabel, QLabel]] = {}
        for key, label in (
            ("activities", "Activities detected"),
            ("dependencies", "Dependencies detected"),
            ("review_items", "Review items"),
            ("graph_status", "Graph status"),
            ("cpm_readiness", "CPM readiness"),
        ):
            value = QLabel("\u2014")
            value.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value.setFont(QFont(*LABEL_FONT))
            value.setStyleSheet(f"color: {TEXT};")
            cap = QLabel(label)
            cap.setAlignment(Qt.AlignmentFlag.AlignCenter)
            cap.setFont(QFont(*MUTED_FONT))
            cap.setStyleSheet(f"color: {TEXT_MUTED};")
            col = QVBoxLayout()
            col.setSpacing(0)
            col.addWidget(value)
            col.addWidget(cap)
            cell = QWidget()
            cell.setLayout(col)
            kpi_row.addWidget(cell, stretch=1)
            self._preliminary_values[key] = (value, cap)

        inner.addLayout(kpi_row)
        return card

    def _build_dashboard(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.setSpacing(8)

        # Status banner
        self._status_banner = QFrame()
        banner_layout = QHBoxLayout(self._status_banner)
        banner_layout.setContentsMargins(16, 10, 16, 10)
        banner_layout.setSpacing(10)
        self._status_icon = QLabel("\u2713")
        self._status_icon.setFont(QFont(*TITLE_FONT))
        self._status_icon.setStyleSheet(f"color: {SUCCESS};")
        banner_layout.addWidget(self._status_icon)
        banner_text = QVBoxLayout()
        banner_text.setSpacing(2)
        self._status_label = QLabel("FINAL RESULTS")
        self._status_label.setFont(QFont(*STATUS_FONT))
        self._status_label.setStyleSheet(f"color: {SUCCESS}; font-weight: 600;")
        banner_text.addWidget(self._status_label)
        self._status_hint = QLabel("All review decisions applied. Graph validated. CPM computed.")
        self._status_hint.setFont(QFont(*MUTED_FONT))
        self._status_hint.setStyleSheet(f"color: {TEXT_SECONDARY};")
        banner_text.addWidget(self._status_hint)
        banner_layout.addLayout(banner_text, stretch=1)
        self._status_banner.setStyleSheet(
            f"QFrame {{ background-color: {_rgba(SUCCESS, 20)};"
            f" border: 1px solid {SUCCESS}44; border-radius: 8px; }}"
        )
        layout.addWidget(self._status_banner)

        top = QHBoxLayout()
        top.setSpacing(10)
        header_stack = QVBoxLayout()
        header_stack.setSpacing(2)
        self._summary_label = QLabel("")
        self._summary_label.setWordWrap(True)
        self._summary_label.setFont(QFont(*LABEL_FONT))
        self._summary_label.setStyleSheet(f"color: {ACCENT};")
        header_stack.addWidget(self._summary_label)
        top.addLayout(header_stack, stretch=1)

        self._export_btn = QPushButton("Export")
        self._export_btn.setToolTip(_EXPORT_TOOLTIP)
        self._export_btn.clicked.connect(self._on_export_clicked)
        self._report_btn = QPushButton("Report")
        self._report_btn.setToolTip(_REPORT_TOOLTIP)
        self._report_btn.clicked.connect(self._on_report_clicked)
        top.addWidget(self._export_btn)
        top.addWidget(self._report_btn)
        layout.addLayout(top)

        self._tabs = QTabWidget()
        self._tabs.setStyleSheet(
            f"QTabWidget::pane {{ border: 1px solid {BORDER};"
            f" border-radius: {RADIUS_SM}px; top: -1px; }}"
            f"QTabBar::tab {{ padding: 7px 16px; color: {TEXT_MUTED};"
            " border: none; background: transparent; }"
            f"QTabBar::tab:selected {{ color: {TEXT};"
            f" border-bottom: 2px solid {ACCENT}; }}"
        )
        self._overview = OverviewTab()
        self._network = NetworkTab()
        self._activities = ActivitiesTab()
        self._critical_paths = CriticalPathsTab()
        self._pert = PertTab()
        self._tabs.addTab(self._overview, "Overview")
        self._tabs.addTab(self._network, "Network")
        self._tabs.addTab(self._activities, "Activities")
        self._tabs.addTab(self._critical_paths, "Critical Paths")
        self._tabs.addTab(self._pert, "PERT")
        layout.addWidget(self._tabs, stretch=1)

        self._overview.path_list().path_selected.connect(self._on_path_selected)
        self._critical_paths.path_selected.connect(self._on_path_selected)
        self._activities.activity_selected.connect(self._on_activity_selected)
        self._network.node_selected.connect(self._on_network_node_selected)
        self._overview.embedded_network().node_selected.connect(
            self._on_overview_network_node_selected
        )
        self._overview.embedded_activities().activity_selected.connect(
            self._on_embedded_activity_selected
        )
        self._pert.pert_run_requested.connect(self._on_pert_run_clicked)
        return widget

    # ------------------------------------------------------------------
    # Readiness gate
    # ------------------------------------------------------------------

    def _on_export_clicked(self) -> None:
        if not self._busy:
            self.export_requested.emit()

    def _on_report_clicked(self) -> None:
        if not self._busy:
            self.report_requested.emit()

    def refresh(self, session: Any) -> None:
        self._session = session
        ready, reason = describe_ready(session)
        self._pert.refresh(session)
        if not ready:
            self._show_not_ready(reason)
            return
        self._show_dashboard(session)

    def _show_not_ready(self, reason: str) -> None:
        session = self._session
        message = {
            NO_ANALYSIS: (
                "Results not ready. No project analyzed. "
                "Run an analysis on the Analyze Diagram page."
            ),
            REVIEW_REQUIRED: (
                "Results not ready. Review required \u2014 finish the review "
                "and apply your decisions before CPM can run."
            ),
            GRAPH_INVALID: (
                "Results not ready. The reviewed graph is invalid \u2014 "
                "resolve the validation issues first."
            ),
            CPM_BLOCKED: (
                "Results not ready. CPM is blocked \u2014 the graph is not "
                "in a runnable state."
            ),
            RESULT_UNAVAILABLE: (
                "Results not ready. The graph is valid but no CPM result "
                "exists yet. Click Calculate Results to run the CPM service."
            ),
        }[reason]
        self._empty_label.setText(message)
        self._empty_label.show()
        self._set_calculate_visible(reason == RESULT_UNAVAILABLE)
        self._calculate_btn.setEnabled(reason == RESULT_UNAVAILABLE and not self._busy)
        self._set_readiness(reason)
        self._update_preliminary(session)
        self._update_progress(session)
        self._stack.setCurrentIndex(self._EMPTY)

    def _update_preliminary(self, session: Any) -> None:
        """Fill the PRELIMINARY detected-metrics card when an analysis exists."""
        if session is None or session.workflow is None:
            self._preliminary_card.hide()
            return
        summary = getattr(session, "review_summary", None) or {}
        values = {
            "activities": summary.get("total_activities", "\u2014"),
            "dependencies": summary.get("total_dependencies", "\u2014"),
            "review_items": session.review_item_total() or "\u2014",
            "graph_status": summary.get("graph_validation_status", "\u2014"),
            "cpm_readiness": summary.get("cpm_eligibility", "\u2014"),
        }
        for key, value in values.items():
            value_label, _ = self._preliminary_values[key]
            value_label.setText(str(value) if value not in (None, "") else "\u2014")
        self._preliminary_card.show()

    def _update_progress(self, session: Any) -> None:
        total = session.review_item_total() if session is not None else 0
        pending = session.pending_review_total() if session is not None else 0
        done = max(0, total - pending)
        if session is not None and session.workflow is not None and total > 0:
            self._progress_label.setText(f"{done} / {total} reviews complete")
            self._progress_label.show()
        else:
            self._progress_label.setText("")
            self._progress_label.hide()

    def _set_readiness(self, reason: str) -> None:
        """At-a-glance readiness strip shown in the not-ready state."""
        session = self._session
        if session is None:
            self._readiness_label.setText("")
            return
        workflow = getattr(session, "workflow", None)
        status = session.validation_status
        pending = session.pending_review_total()
        done = session.review_item_total()

        rows = []
        if workflow is None:
            rows.append(("\u25CB", TEXT_MUTED, "Analysis complete"))
        else:
            rows.append(("\u2713", SUCCESS, "Analysis complete"))
        if workflow is None:
            rows.append(("\u25CB", TEXT_MUTED, "Review"))
        elif pending == 0:
            rows.append(("\u2713", SUCCESS, f"Review resolved ({done} items)"))
        else:
            rows.append(("\u26a0", WARNING, f"Review in progress ({pending} pending)"))
        if status == ValidationCenterStatus.VALID:
            rows.append(("\u2713", SUCCESS, "Graph valid"))
        elif status == ValidationCenterStatus.INVALID:
            rows.append(("\u2717", DANGER, "Graph invalid"))
        elif status == ValidationCenterStatus.BLOCKED_REVIEW:
            rows.append(("\u26a0", WARNING, "Validation blocked"))
        else:
            rows.append(("\u25CB", TEXT_MUTED, "Graph not validated yet"))
        if reason == RESULT_UNAVAILABLE:
            rows.append(("\u2713", SUCCESS, "CPM-ready - run Calculate Results"))
        elif reason == CPM_BLOCKED:
            rows.append(("\u26a0", WARNING, "CPM unavailable"))
        elif reason == GRAPH_INVALID:
            rows.append(("\u2717", DANGER, "CPM unavailable"))
        else:
            rows.append(("\u25CB", TEXT_MUTED, "CPM result pending"))

        parts = [
            f'<font color="{color}">{glyph}</font> {label}'
            for glyph, color, label in rows
        ]
        self._readiness_label.setText("  \u00b7  ".join(parts))

    def _show_dashboard(self, session: Any) -> None:
        data = extract(session)
        self._data = data
        duration_text = format_duration(data.project_duration)
        self._summary_label.setText(
            f"Project duration: {duration_text}   \u00b7   "
            f"Critical paths: {data.critical_path_count or 0}   \u00b7   "
            f"Critical activities: {data.critical_activity_count}"
        )
        self._overview.set_data(data)
        self._activities.set_data(data.activities)
        self._critical_paths.set_data(data)
        self._network.set_data(data)
        self._network.set_pert_data(extract_pert(session))
        row = self._overview.path_list().selected_row()
        if row >= 0:
            self._overview.path_list().reselect_current()
        self._set_calculate_visible(False)
        self._empty_label.hide()
        self._not_ready_heading.hide()
        self._preliminary_card.hide()
        self._progress_label.hide()
        self._export_btn.setEnabled(True)
        self._report_btn.setEnabled(True)

        # Update status banner
        self._status_banner.show()
        self._status_icon.setText("\u2713")
        self._status_icon.setStyleSheet(f"color: {SUCCESS};")
        self._status_label.setText("FINAL RESULTS")
        self._status_label.setStyleSheet(f"color: {SUCCESS}; font-weight: 600;")
        self._status_hint.setText(
            "All review decisions applied. Graph validated. CPM computed."
        )
        self._status_banner.setStyleSheet(
            f"QFrame {{ background-color: {_rgba(SUCCESS, 20)};"
            f" border: 1px solid {SUCCESS}44; border-radius: 8px; }}"
        )

        self._stack.setCurrentIndex(self._DASHBOARD)

    def _set_calculate_visible(self, visible: bool) -> None:
        self._calculate_btn.setVisible(visible)

    def _on_calculate_clicked(self) -> None:
        if self._busy:
            return
        self.calculate_requested.emit()

    def _on_pert_run_clicked(self) -> None:
        if self._busy:
            return
        self.pert_run_requested.emit()

    # ------------------------------------------------------------------
    # Cross-navigation (presentation-level only)
    # ------------------------------------------------------------------

    def _on_path_selected(self, path_ids: list[str]) -> None:
        self._network.highlight_path(list(path_ids))
        self._overview.embedded_network().highlight_path(list(path_ids))

    def _on_activity_selected(self, activity_id: str) -> None:
        self._network.set_selected_activity(activity_id)
        self._overview.embedded_network().set_selected_activity(activity_id)

    def _on_network_node_selected(self, activity_id: str) -> None:
        self._overview.embedded_network().set_selected_activity(activity_id)
        self._tabs.setCurrentWidget(self._activities)
        self._activities.select_activity(activity_id)

    def _on_overview_network_node_selected(self, activity_id: str) -> None:
        self._network.set_selected_activity(activity_id)
        self._tabs.setCurrentWidget(self._network)

    def _on_embedded_activity_selected(self, activity_id: str) -> None:
        self._network.set_selected_activity(activity_id)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def set_busy(self, busy: bool) -> None:
        self._busy = bool(busy)
        if self._stack.currentIndex() == self._EMPTY:
            self._calculate_btn.setEnabled(
                not busy and self._calculate_btn.isVisible()
            )
        self._pert.set_busy(busy)

    def stack_index(self) -> int:
        return self._stack.currentIndex()

    @property
    def data(self) -> Any:
        return self._data

    def overview(self) -> OverviewTab:
        return self._overview

    def network(self) -> NetworkTab:
        return self._network

    def activities(self) -> ActivitiesTab:
        return self._activities

    def critical_paths(self) -> CriticalPathsTab:
        return self._critical_paths

    def pert(self) -> PertTab:
        return self._pert

    def tabs(self) -> QTabWidget:
        return self._tabs
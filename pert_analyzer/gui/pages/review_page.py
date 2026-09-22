"""
Review page: interactive Human Review Center for the analyzed diagram.

Presented as three panes (categories, pending items, detail + actions)
with a progress bar and apply action. All decisions are forwarded to the
backend ReviewSession through the GuiSession; the page only renders
state and delegates application to the MainWindow worker.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.reviews import categories as review_categories
from pert_analyzer.gui.reviews.categories import (
    ReviewCategory,
    item_key,
    pending_count,
    pending_items,
)
from pert_analyzer.gui.reviews.detail_panels import (
    ActivityDetailPanel,
    DependencyDetailPanel,
    DurationDetailPanel,
    ReviewActionRequest,
)
from pert_analyzer.gui.reviews.widgets import (
    ReviewCategoriesPanel,
    ReviewItemList,
)
from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BORDER,
    DANGER,
    SUCCESS,
    SURFACE,
    SURFACE_LIGHT,
    TEXT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
)
from pert_analyzer.gui.themes.spacing import LG, MD, RADIUS_MD, RADIUS_SM, SM, XL, XS, XXL
from pert_analyzer.gui.themes.typography import (
    BODY_SMALL_FONT,
    BODY_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    SUBTITLE_FONT,
    TITLE_FONT,
    KPI_FONT,
)

logger = logging.getLogger(__name__)

_CATEGORY_LABELS = {
    ReviewCategory.ACTIVITIES: "Activities",
    ReviewCategory.DEPENDENCIES: "Dependencies",
    ReviewCategory.DURATIONS: "Durations",
}


class EmptyReviewState(QWidget):
    """Full-page empty state with optional action."""

    go_validation = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch()

        self._message = QLabel("")
        self._message.setWordWrap(True)
        self._message.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._message.setFont(QFont(*LABEL_FONT))
        self._message.setStyleSheet(f"color: {TEXT_MUTED};")
        layout.addWidget(self._message)

        self._go_btn = QPushButton("Go to Validation")
        self._go_btn.setObjectName("primary")
        self._go_btn.clicked.connect(self.go_validation.emit)
        layout.addWidget(self._go_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._go_btn.hide()

        layout.addStretch()

    def set_message(self, text: str, show_go_validation: bool = False) -> None:
        self._message.setText(text)
        self._go_btn.setVisible(show_go_validation)


class ReviewCompleteState(QWidget):
    """Shown when every review item has been resolved."""

    apply_requested = Signal()
    go_validation = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(10)
        layout.addStretch()

        heading = QLabel("Review Complete")
        heading.setFont(QFont(*TITLE_FONT))
        heading.setStyleSheet(f"color: {SUCCESS};")
        heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(heading)

        self._sub_heading = QLabel("")
        self._sub_heading.setFont(QFont(*LABEL_FONT))
        self._sub_heading.setStyleSheet(f"color: {TEXT_MUTED};")
        self._sub_heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._sub_heading)

        self._category_rows: dict[ReviewCategory, QLabel] = {}
        for cat in review_categories.CATEGORIES:
            lbl = QLabel("")
            lbl.setFont(QFont(*LABEL_FONT))
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._category_rows[cat] = lbl
            layout.addWidget(lbl)

        layout.addSpacing(8)

        self._apply_btn = QPushButton("Apply & Validate")
        self._apply_btn.setObjectName("primary")
        self._apply_btn.clicked.connect(self.apply_requested.emit)
        layout.addWidget(self._apply_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self._go_btn = QPushButton("Go to Validation")
        self._go_btn.setObjectName("primary")
        self._go_btn.clicked.connect(self.go_validation.emit)
        layout.addWidget(self._go_btn, alignment=Qt.AlignmentFlag.AlignCenter)
        self._go_btn.hide()

        layout.addStretch()

    def populate(self, session: Any) -> None:
        review_session = getattr(session, "review_session", None)
        applied = bool(getattr(session, "has_applied_reviews", False))
        if applied:
            self._sub_heading.setText("All reviews resolved and decisions applied.")
            self._apply_btn.hide()
            self._go_btn.show()
        else:
            self._sub_heading.setText("All reviews resolved. Apply decisions to continue.")
            self._apply_btn.show()
            self._go_btn.hide()

        for cat in review_categories.CATEGORIES:
            items = _category_items(review_session, cat)
            pending = pending_count(review_session, cat) if review_session is not None else 0
            if not items:
                self._category_rows[cat].setText(f"\u2013 {_CATEGORY_LABELS[cat]}: no items")
            elif pending == 0:
                self._category_rows[cat].setText(
                    f"\u2713 {_CATEGORY_LABELS[cat]}: all {len(items)} resolved"
                )
            else:
                self._category_rows[cat].setText(
                    f"{_CATEGORY_LABELS[cat]}: {pending} still pending"
                )

    def set_busy(self, busy: bool) -> None:
        self._apply_btn.setEnabled(not busy)
        self._go_btn.setEnabled(not busy)


def _category_items(review_session: Any, category: ReviewCategory) -> list[Any]:
    if review_session is None:
        return []
    return list(getattr(review_session, _CATEGORY_ATTRIBUTE[category], []) or [])


_CATEGORY_ATTRIBUTE = {
    ReviewCategory.ACTIVITIES: "activities",
    ReviewCategory.DEPENDENCIES: "dependencies",
    ReviewCategory.DURATIONS: "durations",
}


def _find_item_by_key(review_session: Any, category: ReviewCategory, key: str) -> Any:
    """Resolve a review item from its key (as produced by ``item_key``)."""
    if review_session is None:
        return None
    items = list(getattr(review_session, _CATEGORY_ATTRIBUTE[category], []) or [])
    for item in items:
        if item_key(item, category) == key:
            return item
    # Fallback for single-value keys (e.g. a bare geometric node id).
    for item in items:
        if _matches_id_lookup(review_session, category, key, item):
            return item
    return None


def _matches_id_lookup(review_session: Any, category: ReviewCategory, key: str, item: Any) -> bool:
    if category == ReviewCategory.ACTIVITIES:
        return getattr(item, "geometric_node_id", None) == key
    if category == ReviewCategory.DEPENDENCIES:
        return getattr(item, "arrow_id", None) == key
    if category == ReviewCategory.DURATIONS:
        return key in (
            getattr(item, "geometric_node_id", None),
            getattr(item, "activity_id", None),
        )
    return False


class ReviewPage(QWidget):
    """Interactive human review workspace for one analyzed image."""

    apply_requested = Signal()
    validate_requested = Signal()

    _EMPTY = 0
    _WORKSPACE = 1
    _COMPLETE = 2

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._session: Any = None
        self._category: ReviewCategory = ReviewCategory.ACTIVITIES
        self._items: list[Any] = []
        self._index: int = 0
        self._busy: bool = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)

        title = QLabel("Review Center")
        title.setFont(QFont(*TITLE_FONT))
        title.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(title)

        subtitle = QLabel(
            "Confirm or correct detected activities, dependencies, and durations."
        )
        subtitle.setFont(QFont(*SUBTITLE_FONT))
        subtitle.setStyleSheet(f"color: {TEXT_SECONDARY};")
        layout.addWidget(subtitle)

        self._stack = QStackedWidget()
        layout.addWidget(self._stack, stretch=1)

        # state pages
        self._empty_state = EmptyReviewState()
        self._empty_state.go_validation.connect(self.validate_requested.emit)
        self._complete_state = ReviewCompleteState()
        self._complete_state.apply_requested.connect(self.apply_requested.emit)
        self._complete_state.go_validation.connect(self.validate_requested.emit)

        self._stack.addWidget(self._empty_state)          # 0
        self._stack.addWidget(self._build_workspace())    # 1
        self._stack.addWidget(self._complete_state)       # 2

        self._show_empty_state(
            "No analysis available. Run an analysis on the Analyze Diagram page first."
        )

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_workspace(self) -> QWidget:
        workspace = QWidget()
        outer = QVBoxLayout(workspace)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        row = QSplitter(Qt.Orientation.Horizontal)
        row.setChildrenCollapsible(False)
        row.setHandleWidth(6)
        row.setStyleSheet(
            f"QSplitter::handle {{ background-color: {BORDER};"
            f" border-radius: {RADIUS_SM}px; }}"
        )

        self._categories = ReviewCategoriesPanel()
        self._categories.category_selected.connect(self._on_category_selected)
        row.addWidget(self._categories)

        self._item_list = ReviewItemList()
        self._item_list.setMinimumWidth(220)
        self._item_list.connect_selection()
        self._item_list.item_picked.connect(self._on_item_picked)
        row.addWidget(self._item_list)

        self._detail_stack = QStackedWidget()
        self._detail_stack.setMinimumWidth(380)
        self._detail_placeholder = QLabel(
            "Select a review item from the list to begin."
        )
        self._detail_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._detail_placeholder.setFont(QFont(*BODY_FONT))
        self._detail_placeholder.setStyleSheet(f"color: {TEXT_MUTED};")
        self._detail_stack.addWidget(self._detail_placeholder)

        self._activity_panel = ActivityDetailPanel()
        self._activity_panel.decision_made.connect(self._on_detail_decision)
        self._detail_stack.addWidget(self._activity_panel)

        self._dependency_panel = DependencyDetailPanel()
        self._dependency_panel.decision_made.connect(self._on_detail_decision)
        self._detail_stack.addWidget(self._dependency_panel)

        self._duration_panel = DurationDetailPanel()
        self._duration_panel.decision_made.connect(self._on_detail_decision)
        self._detail_stack.addWidget(self._duration_panel)

        row.addWidget(self._detail_stack)
        row.setSizes([200, 300, 500])
        self._splitter = row
        self._user_resized = False
        row.splitterMoved.connect(self._on_splitter_moved)
        outer.addWidget(row, stretch=1)

        # bottom action bar
        bar = QHBoxLayout()
        bar.setContentsMargins(0, 4, 0, 0)
        bar.setSpacing(10)

        self._progress = QProgressBar()
        self._progress.setTextVisible(False)
        self._progress.setFixedWidth(220)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_SM}px; }}"
            f"QProgressBar::chunk {{ background-color: {ACCENT};"
            f" border-radius: {RADIUS_SM}px; }}"
        )
        bar.addWidget(self._progress)

        self._progress_label = QLabel("")
        self._progress_label.setFont(QFont(*MUTED_FONT))
        self._progress_label.setStyleSheet(f"color: {TEXT_MUTED};")
        bar.addWidget(self._progress_label)

        self._percent_label = QLabel("0%")
        self._percent_label.setFont(QFont(*MUTED_FONT))
        self._percent_label.setStyleSheet(f"color: {ACCENT};")
        bar.addWidget(self._percent_label)

        self._breakdown_label = QLabel("")
        self._breakdown_label.setWordWrap(False)
        self._breakdown_label.setFont(QFont(*MUTED_FONT))
        self._breakdown_label.setStyleSheet(f"color: {TEXT_MUTED};")
        bar.addWidget(self._breakdown_label)

        self._feedback = QLabel("")
        self._feedback.setWordWrap(True)
        self._feedback.setFont(QFont(*MUTED_FONT))
        bar.addWidget(self._feedback, stretch=1)

        self._leave_btn = QPushButton("Leave Unresolved")
        self._leave_btn.clicked.connect(self._on_leave_unresolved)
        bar.addWidget(self._leave_btn)

        self._apply_btn = QPushButton("Apply & Validate")
        self._apply_btn.setObjectName("primary")
        self._apply_btn.clicked.connect(self.apply_requested.emit)
        bar.addWidget(self._apply_btn)

        outer.addStretch(1)
        outer.addLayout(bar)
        return workspace

    # ------------------------------------------------------------------
    # Refresh API
    # ------------------------------------------------------------------

    def refresh(self, session: Any) -> None:
        """Rebuild the page from the current session state."""
        self._session = session
        if self._busy:
            logger.info("Review page refresh deferred while applying")
            return
        self._rebuild()

    def _rebuild(self) -> None:
        session = self._session
        workflow = getattr(session, "workflow", None)
        if workflow is None:
            self._show_empty_state(
                "No analysis available. Run an analysis on the Analyze Diagram page first."
            )
            return

        review_session = getattr(session, "review_session", None)
        if review_session is None:
            self._show_empty_state(
                "No review session was created for this analysis.", show_go_validation=True
            )
            return

        total = session.review_item_total()
        if total == 0:
            self._show_empty_state("No review items in this analysis.", show_go_validation=True)
            return

        complete = session.pending_review_total() == 0
        if complete:
            self._stack.setCurrentIndex(self._COMPLETE)
            self._complete_state.populate(session)
            self._sync_buttons()
            return

        self._stack.setCurrentIndex(self._WORKSPACE)
        self._category = self._categories.current()
        self._populate()

    def _show_empty_state(self, message: str, show_go_validation: bool = False) -> None:
        self._stack.setCurrentIndex(self._EMPTY)
        self._empty_state.set_message(message, show_go_validation=show_go_validation)

    # ------------------------------------------------------------------
    # Workspace population
    # ------------------------------------------------------------------

    def _populate(self) -> None:
        self._items = pending_items(self._session.review_session, self._category)
        self._item_list.populate(self._items, self._category)
        self._sync_counts()
        self._present()
        self._sync_buttons()

    def _sync_counts(self) -> None:
        session = self._session
        if session is None:
            return
        rs = session.review_session
        self._categories.set_counts(
            pending_count(rs, ReviewCategory.ACTIVITIES),
            pending_count(rs, ReviewCategory.DEPENDENCIES),
            pending_count(rs, ReviewCategory.DURATIONS),
        )
        total = session.review_item_total()
        pending = session.pending_review_total()
        resolved = total - pending
        self._progress.setRange(0, max(1, total))
        self._progress.setValue(resolved)
        self._progress_label.setText(
            f"Resolved {resolved} / {total}  \u00b7  Pending {pending}"
        )
        self._percent_label.setText(
            f"{int(round(resolved * 100 / max(1, total)))}%"
        )
        parts = []
        for cat in review_categories.CATEGORIES:
            items = _category_items(rs, cat) or []
            cat_total = len(items)
            cat_pending = pending_count(rs, cat) if rs is not None else 0
            cat_resolved = cat_total - cat_pending
            parts.append(f"{_CATEGORY_LABELS[cat]}: {cat_resolved}/{cat_total}")
        self._breakdown_label.setText(" \u00b7 ".join(parts))
        self._breakdown_label.setToolTip(
            "Resolved / total review items per category"
        )

    def _present(self) -> None:
        if not self._items:
            self._detail_stack.setCurrentIndex(0)
            return
        target = min(self._index, len(self._items) - 1)
        self._item_list.select_row(target)
        self._on_item_picked(self._items[target])

    def _on_item_picked(self, item: Any) -> None:
        self._index = self._items.index(item) if item in self._items else 0
        panel = self._panel_for(self._category)
        self._detail_stack.setCurrentWidget(panel)
        panel.show_item(item, self._session)
        self._feedback.setText("")
        self._feedback.setStyleSheet(f"color: {TEXT_MUTED};")

    def _panel_for(self, category: ReviewCategory):
        if category == ReviewCategory.ACTIVITIES:
            return self._activity_panel
        if category == ReviewCategory.DEPENDENCIES:
            return self._dependency_panel
        return self._duration_panel

    # ------------------------------------------------------------------
    # Adaptive splitter
    # ------------------------------------------------------------------

    def _on_splitter_moved(self, pos: int, index: int) -> None:
        """Remember explicit user adjustments so auto-layout hands off."""
        del pos, index  # unused
        self._user_resized = True

    def resizeEvent(self, event) -> None:  # noqa: N802
        """Keep the 20% / 30% / 50% splitter balance on large windows.

        Only applies while the user has not manually dragged a handle, so
        hand-tuned layouts are respected.
        """
        super().resizeEvent(event)
        splitter = getattr(self, "_splitter", None)
        if splitter is None or self._user_resized:
            return
        if self.width() < 1000:
            return
        current = splitter.sizes()
        total = sum(current)
        if total <= 0:
            return
        splitter.setSizes(
            [
                int(round(total * 0.20)),
                int(round(total * 0.30)),
                int(round(total * 0.50)),
            ]
        )

    # ------------------------------------------------------------------
    # Category + action handling
    # ------------------------------------------------------------------

    def _on_category_selected(self, category: ReviewCategory) -> None:
        self._category = category
        self._index = 0
        if self.stack_index == self._WORKSPACE:
            self._populate()

    @property
    def stack_index(self) -> int:
        return self._stack.currentIndex()

    def _show_focused_item(self, category: ReviewCategory, item: Any) -> bool:
        """Show a single review item on its own in the workspace."""
        self._category = category
        self._categories.set_category(category)
        self._items = [item]
        self._index = 0
        self._item_list.populate(self._items, category)
        self._sync_counts()
        self._stack.setCurrentIndex(self._WORKSPACE)
        self._present()
        self._sync_buttons()
        return True

    @property
    def empty_message(self) -> str:
        """Text currently shown in the page-level empty state."""
        return self._empty_state._message.text()

    @property
    def empty_go_validation_visible(self) -> bool:
        """Whether the empty state offers a path to validation."""
        return not self._empty_state._go_btn.isHidden()

    @property
    def is_complete(self) -> bool:
        return self._stack.currentIndex() == self._COMPLETE

    def current_items(self) -> list[Any]:
        return list(self._items)

    def focus_item(self, category: ReviewCategory, key: str) -> bool:
        """Select a specific review item (pending or resolved) in the workspace.

        This is used by the Validation Center to open the item affected by a
        validation issue. Returns True when the item was found and shown.
        """
        session = self._session
        if session is None:
            return False
        review_session = getattr(session, "review_session", None)
        if review_session is None:
            return False

        item = _find_item_by_key(review_session, category, key)
        if item is None:
            return False
        return self._show_focused_item(category, item)

    def _on_leave_unresolved(self) -> None:
        if not self._items:
            return
        self._advance_to_next(unresolved=True)

    def _advance_to_next(self, unresolved: bool = False) -> None:
        """Re-populate after a decision and move selection forward."""
        old_key = item_key(self._current_item(), self._category)
        self._items = pending_items(self._session.review_session, self._category)
        self._item_list.populate(self._items, self._category)
        self._sync_counts()

        if not self._items:
            self._on_all_resolved_or_empty()
            return

        keys = [item_key(i, self._category) for i in self._items]
        if old_key in keys:
            base = keys.index(old_key)
            if unresolved:
                target = (base + 1) % len(self._items)
            else:
                target = min(base, len(self._items) - 1)
        else:
            target = min(self._index, len(self._items) - 1)

        self._index = target
        self._item_list.select_row(target)
        self._on_item_picked(self._items[target])
        self._sync_buttons()

    def _on_all_resolved_or_empty(self) -> None:
        session = self._session
        if session is None:
            return
        if session.review_item_total() > 0 and session.pending_review_total() == 0:
            self._stack.setCurrentIndex(self._COMPLETE)
            self._complete_state.populate(session)
        else:
            self._detail_stack.setCurrentIndex(0)
        self._sync_buttons()

    def _current_item(self) -> Any:
        if 0 <= self._index < len(self._items):
            return self._items[self._index]
        return None

    # ------------------------------------------------------------------
    # Detail-panel decision handling
    # ------------------------------------------------------------------

    def _on_detail_decision(self, request: ReviewActionRequest) -> None:
        session = self._session
        if session is None or request.item is None:
            return
        action = request.action
        try:
            ok = self._apply_decision(session, request)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Decision rejected: %s", exc)
            self._show_feedback(str(exc), error=True)
            return
        if not ok:
            self._show_feedback("That item has already been reviewed.", error=True)
            return

        if action != "LEAVE_UNRESOLVED":
            session.record_review_decision()
        self._advance_to_next(unresolved=(action == "LEAVE_UNRESOLVED"))

    def _apply_decision(self, session: Any, request: ReviewActionRequest) -> bool:
        action = request.action
        if request.category == ReviewCategory.ACTIVITIES:
            if action == "CORRECT":
                return session.decide_activity(
                    request.item.geometric_node_id,
                    "CORRECT",
                    corrected_id=request.corrected_value,
                )
            if action == "ACCEPT":
                return session.decide_activity(request.item.geometric_node_id, "ACCEPT")
            if action == "LEAVE_UNRESOLVED":
                return True
            return False

        if request.category == ReviewCategory.DEPENDENCIES:
            arrow = request.item.arrow_id
            if action in ("ACCEPT", "REJECT", "REVERSE"):
                return session.decide_dependency(arrow, action)
            if action == "CHANGE_SOURCE":
                return session.decide_dependency(
                    arrow, "CHANGE_SOURCE", corrected_source_id=request.corrected_value
                )
            if action == "CHANGE_TARGET":
                return session.decide_dependency(
                    arrow, "CHANGE_TARGET", corrected_target_id=request.corrected_value
                )
            if action == "LEAVE_UNRESOLVED":
                return True
            return False

        if request.category == ReviewCategory.DURATIONS:
            if action == "CORRECT":
                return session.decide_duration(
                    request.item.geometric_node_id,
                    "CORRECT",
                    corrected_duration=request.corrected_value,
                )
            if action == "ACCEPT":
                return session.decide_duration(request.item.geometric_node_id, "ACCEPT")
            if action == "LEAVE_UNRESOLVED":
                return True
            return False

        return False

    # ------------------------------------------------------------------
    # Feedback / busy
    # ------------------------------------------------------------------

    def _show_feedback(self, text: str, error: bool = False) -> None:
        color = DANGER if error else SUCCESS
        self._feedback.setText(text)
        self._feedback.setStyleSheet(f"color: {color};")

    @property
    def is_busy(self) -> bool:
        return self._busy

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._complete_state.set_busy(busy)
        for widget in (
            self._categories,
            self._item_list,
            self._leave_btn,
            self._activity_panel,
            self._dependency_panel,
            self._duration_panel,
        ):
            widget.setEnabled(not busy)
        self._sync_buttons()

    def _apply_allowed(self) -> bool:
        session = self._session
        if session is None or self._busy:
            return False
        if getattr(session, "has_applied_reviews", False) and not getattr(
            session, "reviews_dirty", False
        ):
            return False
        total = session.review_item_total()
        pending = session.pending_review_total()
        return total > 0 and pending == 0

    def _sync_buttons(self) -> None:
        allowed = self._apply_allowed()
        self._apply_btn.setEnabled(allowed and not self._busy)
        self._complete_state._apply_btn.setEnabled(allowed and not self._busy)
        self._leave_btn.setEnabled(not self._busy and bool(self._items))
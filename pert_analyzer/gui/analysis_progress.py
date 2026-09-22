"""
Real-time analysis progress model and panel.

The model tracks the exact stages emitted by the pipeline through
``StageProgress`` events (see ``pert_analyzer.pipeline.progress``); it
never invents percentages. The panel renders the stages as a checklist,
an overall progress bar, the current stage message, and real
detected-object metrics (shapes, arrows, OCR labels, activities,
review items) only when the pipeline reports them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QVBoxLayout,
    QWidget,
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
    WARNING,
)
from pert_analyzer.gui.themes.spacing import RADIUS_MD, RADIUS_SM, LG, MD, SM, XS
from pert_analyzer.gui.themes.typography import (
    BODY_SMALL_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
)
from pert_analyzer.pipeline.progress import (
    COMPLETED,
    FAILED,
    PENDING,
    PIPELINE_STAGES,
    REVIEW_REQUIRED,
    RUNNING,
    SKIPPED,
    STAGE_LABELS,
)


@dataclass
class StageEntry:
    """One stage in the progress model with its current state."""

    stage_id: str
    name: str
    state: str = PENDING
    progress: float = 0.0
    message: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)


class AnalysisProgressModel(QObject):
    """Tracks pipeline stage states and overall (real) progress."""

    updated = Signal(object)
    finished = Signal(str)  # "COMPLETED" | "FAILED"

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._entries: List[StageEntry] = [
            StageEntry(sid, STAGE_LABELS.get(sid, sid)) for sid in PIPELINE_STAGES
        ]
        self._by_id: Dict[str, StageEntry] = {e.stage_id: e for e in self._entries}
        self._overall = 0.0
        self._finished = False

    # ------------------------------------------------------------------
    # Read access
    # ------------------------------------------------------------------

    @property
    def entries(self) -> List[StageEntry]:
        return list(self._entries)

    @property
    def overall_progress(self) -> float:
        return self._overall

    @property
    def is_finished(self) -> bool:
        return self._finished

    def stage_entry(self, stage_id: str) -> Optional[StageEntry]:
        return self._by_id.get(stage_id)

    def running_entry(self) -> Optional[StageEntry]:
        for entry in self._entries:
            if entry.state == RUNNING:
                return entry
        return None

    def collect_metrics(self) -> Dict[str, Any]:
        """Merge real metrics reported by the pipeline (later wins)."""
        merged: Dict[str, Any] = {}
        for entry in self._entries:
            for key, value in (entry.metrics or {}).items():
                merged[key] = value
        return merged

    # ------------------------------------------------------------------
    # Mutations
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Return every stage to PENDING for a new analysis run."""
        for entry in self._entries:
            entry.state = PENDING
            entry.progress = 0.0
            entry.message = ""
            entry.metrics = {}
        self._overall = 0.0
        self._finished = False
        self.updated.emit(self)

    def handle_stage(self, stage_progress: Any) -> None:
        """Apply one ``StageProgress`` event from the pipeline."""
        stage_id = getattr(stage_progress, "stage_id", "") or ""
        if not stage_id:
            return
        entry = self._by_id.get(stage_id)
        if entry is None:
            name = getattr(stage_progress, "name", "") or stage_id
            entry = StageEntry(stage_id, name)
            self._entries.append(entry)
            self._by_id[stage_id] = entry

        entry.state = getattr(stage_progress, "state", RUNNING) or RUNNING
        entry.progress = float(getattr(stage_progress, "progress", 0.0) or 0.0)
        message = getattr(stage_progress, "message", "") or ""
        if message:
            entry.message = message
        metrics = getattr(stage_progress, "metrics", None) or {}
        if metrics:
            entry.metrics = dict(metrics)

        if entry.state == RUNNING:
            # The stage just started; any stage that was still running
            # before it has really finished.
            seen = False
            for other in self._entries:
                if other is entry:
                    seen = True
                    continue
                if not seen and other.state == RUNNING:
                    other.state = COMPLETED
                if seen:
                    break

        self._overall = max(self._overall, entry.progress)
        self.updated.emit(self)

    def note_metric(self, key: str, value: Any) -> None:
        """Attach one real measured metric without fabricating a stage event."""
        if value is None or value == "":
            return
        for entry in self._entries:
            if entry.state == COMPLETED:
                entry.metrics = dict(entry.metrics)
                entry.metrics[key] = value
        self.updated.emit(self)

    def finish(self, ok: bool, message: str = "") -> None:
        """Classify the tail of the pipeline after the worker completes or fails."""
        self._finished = True
        running = [e for e in self._entries if e.state == RUNNING]
        if ok:
            for entry in self._entries:
                if entry.state in (PENDING, RUNNING):
                    entry.state = COMPLETED
                if message and entry.state == COMPLETED:
                    entry.message = message
            self._overall = 1.0
            self.finished.emit("COMPLETED")
        else:
            failed = running[-1] if running else None
            reached_failed = failed is None
            for entry in self._entries:
                if entry is failed:
                    entry.state = FAILED
                    if message:
                        entry.message = message
                    reached_failed = True
                elif entry.state in (PENDING, RUNNING):
                    entry.state = SKIPPED if reached_failed else entry.state
            self.finished.emit("FAILED")
        self.updated.emit(self)


def format_metric(key: str, value: Any) -> str:
    """Human label for a real detected-object metric."""
    return {
        "shapes": f"{value} shapes detected",
        "arrows": f"{value} arrow candidates",
        "ocr": f"{value} OCR labels",
        "activities": f"{value} activities",
        "review_items": f"{value} review items",
        "nodes": f"{value} nodes",
        "edges": f"{value} edges",
    }.get(key, f"{key}: {value}")


_STATE_GLYPHS = {
    COMPLETED: "\u2713",
    RUNNING: "\u25CF",
    PENDING: "\u25CB",
    REVIEW_REQUIRED: "\u26a0",
    FAILED: "\u2717",
    SKIPPED: "\u2013",
}

_STATE_COLORS = {
    COMPLETED: SUCCESS,
    RUNNING: ACCENT,
    PENDING: TEXT_MUTED,
    REVIEW_REQUIRED: WARNING,
    FAILED: DANGER,
    SKIPPED: TEXT_MUTED,
}


class StageProgressPanel(QWidget):
    """Checklist + progress bar + current message + metric chips."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._model: Optional[AnalysisProgressModel] = None

        card = QFrame(self)
        card.setObjectName("progressCard")
        card.setStyleSheet(
            f"QFrame#progressCard {{ background-color: {SURFACE_LIGHT};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_MD}px; }}"
        )
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(LG, MD, LG, MD)
        card_layout.setSpacing(SM)

        header = QHBoxLayout()
        header.setSpacing(MD)
        self._title = QLabel("ANALYZING DIAGRAM")
        self._title.setFont(QFont(*STATUS_FONT))
        self._title.setStyleSheet(f"color: {TEXT}; border: none; background: transparent;")
        header.addWidget(self._title)
        header.addStretch()
        self._percent_label = QLabel("0%")
        self._percent_label.setFont(QFont(*STATUS_FONT))
        self._percent_label.setStyleSheet(
            f"color: {ACCENT}; border: none; background: transparent;"
        )
        header.addWidget(self._percent_label)
        card_layout.addLayout(header)

        summary_row = QHBoxLayout()
        summary_row.setSpacing(LG)
        self._stage_row = QLabel("<b>Current stage:</b> \u2014")
        self._stage_row.setTextFormat(Qt.TextFormat.RichText)
        self._stage_row.setFont(QFont(*BODY_SMALL_FONT))
        self._stage_row.setStyleSheet(
            "color: %s; border: none; background: transparent;" % TEXT
        )
        summary_row.addWidget(self._stage_row)
        summary_row.addStretch()
        self._detected_row = QLabel("<b>Detected:</b> \u2014")
        self._detected_row.setTextFormat(Qt.TextFormat.RichText)
        self._detected_row.setFont(QFont(*BODY_SMALL_FONT))
        self._detected_row.setStyleSheet(
            "color: %s; border: none; background: transparent;" % TEXT
        )
        summary_row.addWidget(self._detected_row)
        card_layout.addLayout(summary_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setTextVisible(False)
        self._progress.setFixedHeight(6)
        self._progress.setStyleSheet(
            f"QProgressBar {{ background-color: {SURFACE};"
            f" border: 1px solid {BORDER}; border-radius: {RADIUS_SM}px; }}"
            f"QProgressBar::chunk {{ background-color: {ACCENT};"
            f" border-radius: {RADIUS_SM}px; }}"
        )
        card_layout.addWidget(self._progress)

        self._message = QLabel("Preparing analysis\u2026")
        self._message.setWordWrap(True)
        self._message.setFont(QFont(*LABEL_FONT))
        self._message.setStyleSheet(f"color: {TEXT_MUTED}; border: none; background: transparent;")
        card_layout.addWidget(self._message)

        self._checklist = QLabel("")
        self._checklist.setTextFormat(Qt.TextFormat.RichText)
        self._checklist.setWordWrap(True)
        self._checklist.setFont(QFont(*BODY_SMALL_FONT))
        self._checklist.setStyleSheet("color: %s; border: none; background: transparent;" % TEXT)
        self._checklist.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        card_layout.addWidget(self._checklist)

        self._metrics = QLabel("")
        self._metrics.setWordWrap(True)
        self._metrics.setFont(QFont(*MUTED_FONT))
        card_layout.addWidget(self._metrics)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, SM, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(card)

    # ------------------------------------------------------------------
    # Model wiring
    # ------------------------------------------------------------------

    def set_model(self, model: AnalysisProgressModel) -> None:
        """Bind the panel to a progress model and render its current state."""
        if self._model is model and self._model is not None:
            return
        if self._model is not None:
            try:
                self._model.updated.disconnect(self._on_updated)
            except (TypeError, RuntimeError):
                pass
        self._model = model
        if model is not None:
            model.updated.connect(self._on_updated)
            self._on_updated(model)

    @property
    def model(self) -> Optional[AnalysisProgressModel]:
        return self._model

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _on_updated(self, model: Any) -> None:
        if model is None:
            return
        percent = int(round(model.overall_progress * 100))
        self._progress.setValue(min(100, max(0, percent)))
        self._percent_label.setText(f"{percent}%")

        running = model.running_entry()
        lines: List[str] = []
        for entry in model.entries:
            glyph = _STATE_GLYPHS.get(entry.state, PENDING)
            color = _STATE_COLORS.get(entry.state, TEXT_MUTED)
            name = entry.name or entry.stage_id
            if entry.state == RUNNING and entry.message:
                name = f"{name} \u2014 {entry.message}"
            lines.append(
                f'<font color="{color}">{glyph}</font> {name}'
            )
        self._checklist.setText("<br/>".join(lines))

        if running is not None:
            stage_name = running.name or running.stage_id
        elif getattr(model, "is_finished", False):
            stage_name = "Complete"
        else:
            stage_name = "\u2014"
        self._stage_row.setText(f"<b>Current stage:</b> {stage_name}")
        self._detected_row.setText(f"<b>Detected:</b> {self._detected_text(model)}")

        if running is not None and running.message:
            self._message.setText(running.message)
        elif running is not None:
            self._message.setText(f"Running: {running.name}")
        else:
            self._message.setText(self._status_line(model))

        self._render_metrics(model.collect_metrics())

    def _status_line(self, model: Any) -> str:
        if getattr(model, "is_finished", False):
            return "Analysis complete."
        done = sum(1 for e in model.entries if e.state == COMPLETED)
        if done == 0:
            return "Preparing analysis\u2026"
        return f"{done} of {len(model.entries)} stages complete."

    def _detected_text(self, model: Any) -> str:
        """Compact human summary of the detection metrics gathered so far."""
        metrics = model.collect_metrics()
        parts: List[str] = []
        for key in ("shapes", "arrows", "ocr", "activities", "review_items"):
            value = metrics.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)) and value == 0:
                continue
            parts.append(format_metric(key, value))
        return "  \u00b7  ".join(parts) if parts else "\u2014"

    def _render_metrics(self, metrics: Dict[str, Any]) -> None:
        parts: List[str] = []
        for key in ("shapes", "arrows", "ocr", "activities", "review_items"):
            value = metrics.get(key)
            if value is None or value == "":
                continue
            if isinstance(value, (int, float)) and value == 0:
                continue
            parts.append(
                f'<span style="color:{ACCENT};">{format_metric(key, value)}</span>'
            )
        self._metrics.setText("  \u00b7  ".join(parts) if parts else "")
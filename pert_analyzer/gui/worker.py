"""
Analysis worker: runs ReviewWorkflow.analyze in a background QThread.
"""

from __future__ import annotations

import inspect
import logging
import os
from typing import Any, Callable, Optional

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)


def default_analyze(
    image_path: str, stage_callback: Optional[Callable[[Any], None]] = None
) -> Any:
    """Default backend that calls ReviewWorkflow.analyze."""
    from pert_analyzer.pipeline.review_api import ReviewWorkflow

    return ReviewWorkflow.analyze(
        image_path,
        source_image_id=os.path.basename(image_path),
        stage_callback=stage_callback,
    )


def _accepts_stage_callback(fn: Any) -> bool:
    """Whether a backend callable accepts a ``stage_callback`` kwarg."""
    try:
        signature = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return any(
        name == "stage_callback"
        and kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
        for name, kind in (
            (param.name, param.kind) for param in signature.parameters.values()
        )
    )


class AnalysisWorker(QThread):
    """Runs analysis in a background thread and delivers results via signals."""

    started = Signal(str)
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(object)

    def __init__(
        self,
        image_path: str,
        analyze_fn: Optional[Callable[[str], Any]] = None,
        parent: Any = None,
    ):
        super().__init__(parent)
        self._image_path = image_path
        self._analyze_fn = analyze_fn or default_analyze

    def _emit_progress(self, stage_progress: Any) -> None:
        self.progress.emit(stage_progress)

    def _analyze(self) -> Any:
        if _accepts_stage_callback(self._analyze_fn):
            return self._analyze_fn(
                self._image_path, stage_callback=self._emit_progress
            )
        return self._analyze_fn(self._image_path)

    def run(self) -> None:
        name = os.path.basename(self._image_path)
        self.started.emit(name)
        logger.info("Worker started: %s", name)
        try:
            workflow = self._analyze()
            self.completed.emit(workflow)
            logger.info("Worker completed: %s", name)
        except Exception as exc:
            logger.exception("Worker failed: %s", name)
            self.failed.emit(str(exc))


class ApplyReviewsWorker(QThread):
    """Runs apply_review_decisions in a background thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workflow: Any,
        apply_fn: Optional[Callable[[], Any]] = None,
        parent: Any = None,
    ):
        super().__init__(parent)
        self._workflow = workflow
        self._apply_fn = apply_fn or (lambda: workflow.apply())

    def run(self) -> None:
        try:
            result = self._apply_fn()
            self.completed.emit(result)
            logger.info("Apply reviews worker completed")
        except Exception as exc:
            logger.exception("Apply reviews worker failed")
            self.failed.emit(str(exc))


class CpmWorker(QThread):
    """Runs the existing backend CPM service in a background thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        workflow: Any,
        run_cpm_fn: Optional[Callable[[], Any]] = None,
        parent: Any = None,
    ):
        super().__init__(parent)
        self._workflow = workflow
        self._run_cpm_fn = run_cpm_fn or (lambda: workflow.run_cpm())

    def run(self) -> None:
        try:
            result = self._run_cpm_fn()
            self.completed.emit(result)
            logger.info("CPM worker completed")
        except Exception as exc:
            logger.exception("CPM worker failed")
            self.failed.emit(str(exc))

class ExportWorker(QThread):
    """Runs ExportManager.export in a background thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        export_fn: Callable[[], Any],
        parent: Any = None,
    ):
        super().__init__(parent)
        self._export_fn = export_fn

    def run(self) -> None:
        try:
            result = self._export_fn()
            self.completed.emit(result)
            logger.info("Export worker completed")
        except Exception as exc:
            logger.exception("Export worker failed")
            self.failed.emit(str(exc))


class ReportWorker(QThread):
    """Runs ReportBuilder.build in a background thread."""

    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        report_fn: Callable[[], Any],
        parent: Any = None,
    ):
        super().__init__(parent)
        self._report_fn = report_fn

    def run(self) -> None:
        try:
            result = self._report_fn()
            self.completed.emit(result)
            logger.info("Report worker completed")
        except Exception as exc:
            logger.exception("Report worker failed")
            self.failed.emit(str(exc))

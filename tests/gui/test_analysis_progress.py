"""
Phase 7.1 tests: real-time analysis progress (model, worker, panel, routing).

The progress layer must only ever reflect real pipeline events; none of these
tests fabricate percentages. They verify the model transitions, the worker's
one-arg vs. stage-callback backend compatibility, the panel rendering, the
main-window routing, and the state-aware workflow indicator/header badges.
"""

from __future__ import annotations

import inspect
import time

import pytest
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.analysis_progress import (
    AnalysisProgressModel,
    StageProgressPanel,
    format_metric,
)
from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.palette import DANGER, SUCCESS, WARNING
from pert_analyzer.gui.worker import (
    AnalysisWorker,
    _accepts_stage_callback,
    default_analyze,
)
from pert_analyzer.pipeline.progress import (
    COMPLETED,
    FAILED,
    PENDING,
    RUNNING,
    SKIPPED,
    StageProgress,
)
from tests.gui.fakes import (
    FakeReviewedCandidate,
    FakeWorkflow,
    reviewed_session,
)


@pytest.fixture()
def window(qapp: QApplication) -> MainWindow:
    w = MainWindow()
    w._confirm_discard_fn = lambda: True
    yield w
    w.close()


def _sp(stage_id: str, state: str, progress: float, **kw) -> StageProgress:
    return StageProgress(stage_id=stage_id, state=state, progress=progress, **kw)


class _EmittingBackend:
    """Two-arg backend that emits real-looking stage events."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, image_path: str, stage_callback=None):
        self.calls.append(image_path)
        stage_callback(_sp("Loading image", RUNNING, 0.0))
        stage_callback(_sp("Preprocessing", RUNNING, 0.1))
        stage_callback(_sp("Detecting shapes", COMPLETED, 0.2, metrics={"shapes": 9}))
        stage_callback(_sp("Complete", RUNNING, 1.0))
        return FakeWorkflow()


class _PlainBackend:
    """Legacy one-arg backend without any progress support."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def __call__(self, image_path: str):
        self.calls.append(image_path)
        return FakeWorkflow()


# ---------------------------------------------------------------------------
# Model transitions
# ---------------------------------------------------------------------------


def test_model_starts_all_pending() -> None:
    model = AnalysisProgressModel()
    assert len(model.entries) >= 11
    assert all(e.state == PENDING for e in model.entries)
    assert model.overall_progress == 0.0
    assert not model.is_finished


def test_model_running_then_completed() -> None:
    model = AnalysisProgressModel()
    model.handle_stage(_sp("Loading image", RUNNING, 0.0))
    entry = model.stage_entry("Loading image")
    assert entry is not None
    assert entry.state == RUNNING
    model.handle_stage(_sp("Loading image", COMPLETED, 0.0))
    assert model.stage_entry("Loading image").state == COMPLETED


def test_model_new_running_completes_previous() -> None:
    model = AnalysisProgressModel()
    model.handle_stage(_sp("Loading image", RUNNING, 0.0))
    model.handle_stage(_sp("Preprocessing", RUNNING, 0.1))
    assert model.stage_entry("Loading image").state == COMPLETED
    assert model.stage_entry("Preprocessing").state == RUNNING
    assert model.overall_progress == 0.1


def test_model_finish_ok_completes_tail() -> None:
    model = AnalysisProgressModel()
    model.reset()
    model.handle_stage(_sp("Complete", RUNNING, 1.0))
    statuses = []
    model.finished.connect(statuses.append)
    model.finish(ok=True)
    assert model.overall_progress == 1.0
    assert model.is_finished
    assert statuses == ["COMPLETED"]
    assert all(e.state == COMPLETED for e in model.entries)


def test_model_finish_failed_marks_failed_and_skipped() -> None:
    model = AnalysisProgressModel()
    model.reset()
    model.handle_stage(_sp("Loading image", COMPLETED, 0.0))
    model.handle_stage(_sp("Detecting arrows", RUNNING, 0.4))
    statuses = []
    model.finished.connect(statuses.append)
    model.finish(ok=False, message="boom")
    assert statuses == ["FAILED"]
    assert model.stage_entry("Loading image").state == COMPLETED
    assert model.stage_entry("Detecting arrows").state == FAILED
    assert model.stage_entry("Detecting arrows").message == "boom"
    from pert_analyzer.pipeline.progress import PIPELINE_STAGES

    later = PIPELINE_STAGES[PIPELINE_STAGES.index("Detecting arrows") + 1 :]
    assert any(model.stage_entry(s).state == SKIPPED for s in later)


def test_model_reset_returns_to_pending() -> None:
    model = AnalysisProgressModel()
    model.handle_stage(_sp("Complete", RUNNING, 1.0))
    model.finish(ok=True)
    model.reset()
    assert model.overall_progress == 0.0
    assert not model.is_finished
    assert all(e.state == PENDING for e in model.entries)


def test_model_unknown_stage_appended() -> None:
    model = AnalysisProgressModel()
    assert model.stage_entry("Future stage") is None
    model.handle_stage(_sp("Future stage", RUNNING, 0.3))
    assert model.stage_entry("Future stage") is not None
    assert model.stage_entry("Future stage").name == "Future stage"
    assert model.overall_progress == 0.3


def test_model_collect_metrics_merges_and_wins_later() -> None:
    model = AnalysisProgressModel()
    model.handle_stage(_sp("Detecting shapes", COMPLETED, 0.2, metrics={"shapes": 5}))
    model.handle_stage(_sp("Detecting arrows", COMPLETED, 0.4, metrics={"arrows": 3}))
    model.handle_stage(_sp("Complete", COMPLETED, 1.0, metrics={"shapes": 7}))
    merged = model.collect_metrics()
    assert merged["shapes"] == 7
    assert merged["arrows"] == 3


def test_model_note_metric_only_on_completed() -> None:
    model = AnalysisProgressModel()
    model.handle_stage(_sp("Complete", RUNNING, 1.0))
    model.note_metric("activities", 0)
    model.finish(ok=True)
    model.note_metric("review_items", 4)
    assert model.collect_metrics().get("review_items") == 4


def test_format_metric_labels() -> None:
    assert format_metric("shapes", 12) == "12 shapes detected"
    assert format_metric("arrows", 2) == "2 arrow candidates"
    assert format_metric("ocr", 5) == "5 OCR labels"
    assert format_metric("activities", 4) == "4 activities"
    assert format_metric("review_items", 3) == "3 review items"
    assert format_metric("custom", "x") == "custom: x"


# ---------------------------------------------------------------------------
# Backend compatibility
# ---------------------------------------------------------------------------


def test_accepts_stage_callback_detection() -> None:
    assert _accepts_stage_callback(lambda path: None) is False
    assert _accepts_stage_callback(lambda path, stage_callback=None: None) is True
    assert _accepts_stage_callback(lambda path, **kw: None) is False
    assert _accepts_stage_callback(lambda path, *args: None) is False
    assert _accepts_stage_callback(lambda path, *, stage_callback=None: None) is True
    assert _accepts_stage_callback(42) is False
    assert _accepts_stage_callback(default_analyze) is True


def test_review_workflow_analyze_accepts_stage_callback() -> None:
    from pert_analyzer.pipeline.review_api import ReviewWorkflow

    signature = inspect.signature(ReviewWorkflow.analyze)
    assert "stage_callback" in signature.parameters


def test_worker_emits_progress_and_completes(qapp: QApplication) -> None:
    backend = _EmittingBackend()
    worker = AnalysisWorker("x.png", analyze_fn=backend)
    events: list[StageProgress] = []
    results: dict[str, object] = {}

    worker.progress.connect(lambda sp: events.append(sp))
    worker.completed.connect(lambda wf: results.__setitem__("wf", wf))
    worker.failed.connect(lambda msg: results.__setitem__("err", msg))
    worker.start()
    deadline = time.time() + 15
    while worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.005)

    assert backend.calls == ["x.png"]
    assert "err" not in results
    assert results.get("wf") is not None
    assert [e.stage_id for e in events] == [
        "Loading image",
        "Preprocessing",
        "Detecting shapes",
        "Complete",
    ]
    assert events[-1].progress == 1.0
    assert events[2].metrics.get("shapes") == 9


def test_worker_legacy_one_arg_backend(qapp: QApplication) -> None:
    backend = _PlainBackend()
    worker = AnalysisWorker("x.png", analyze_fn=backend)
    results: dict[str, object] = {}

    worker.completed.connect(lambda wf: results.__setitem__("wf", wf))
    worker.failed.connect(lambda msg: results.__setitem__("err", msg))
    worker.start()
    deadline = time.time() + 15
    while worker.isRunning() and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)
    worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.005)

    assert backend.calls == ["x.png"]
    assert "err" not in results
    assert results.get("wf") is not None


# ---------------------------------------------------------------------------
# Progress panel rendering
# ---------------------------------------------------------------------------


def test_panel_percent_tracks_model(qapp: QApplication) -> None:
    model = AnalysisProgressModel()
    panel = StageProgressPanel()
    panel.set_model(model)
    assert panel._percent_label.text() == "0%"
    model.handle_stage(_sp("Loading image", RUNNING, 0.0))
    assert panel._percent_label.text() == "0%"
    model.handle_stage(_sp("Preprocessing", RUNNING, 0.1))
    assert panel._percent_label.text() == "10%"
    model.finish(ok=True)
    assert panel._percent_label.text() == "100%"
    assert panel._progress.value() == 100
    panel.deleteLater()


def test_panel_checklist_lists_stages(qapp: QApplication) -> None:
    model = AnalysisProgressModel()
    panel = StageProgressPanel()
    panel.set_model(model)
    model.handle_stage(_sp("Loading image", RUNNING, 0.0))
    model.handle_stage(_sp("Detecting shapes", COMPLETED, 0.2))
    model.finish(ok=True)
    text = panel._checklist.text()
    assert "Loading Image" in text
    assert "Shape Detection" in text
    assert "Review Preparation" in text
    assert "\u2713" in text
    panel.deleteLater()


def test_panel_metric_chips(qapp: QApplication) -> None:
    model = AnalysisProgressModel()
    panel = StageProgressPanel()
    panel.set_model(model)
    model.handle_stage(
        _sp("Detecting shapes", COMPLETED, 0.2, metrics={"shapes": 9})
    )
    model.handle_stage(
        _sp("Detecting arrows", COMPLETED, 0.4, metrics={"arrows": 4})
    )
    model.handle_stage(
        _sp("Extracting text (OCR)", COMPLETED, 0.5, metrics={"ocr": 7})
    )
    model.finish(ok=True)
    text = panel._metrics.text()
    assert "9 shapes detected" in text
    assert "4 arrow candidates" in text
    assert "7 OCR labels" in text
    assert "0 " not in text
    panel.deleteLater()


# ---------------------------------------------------------------------------
# Main window routing
# ---------------------------------------------------------------------------


def test_window_analysis_routes_progress_to_page(
    window: MainWindow, qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtGui import QImage

    img = QImage(200, 100, QImage.Format.Format_RGB32)
    img.fill("#ffffff")
    p = tmp_path / "diagram.png"
    img.save(str(p), "PNG")

    window._backend = _EmittingBackend()
    window._analysis_page.load_image(str(p))
    assert window._analysis_page._progress_panel.isHidden()
    window._on_analyze_requested()

    deadline = time.time() + 15
    while (
        window._worker is not None
        and window._worker.isRunning()
        and time.time() < deadline
    ):
        qapp.processEvents()
        time.sleep(0.005)
    if window._worker is not None:
        window._worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.005)

    model = window._analysis_page._progress_model
    assert model.overall_progress == 1.0
    assert model.is_finished
    assert not window._analysis_page._progress_panel.isHidden()
    assert window._analysis_page._progress_panel._percent_label.text() == "100%"
    assert (
        "9 shapes detected"
        in window._analysis_page._progress_panel._metrics.text()
    )


def test_window_analysis_failure_updates_progress(
    window: MainWindow, qapp: QApplication, tmp_path
) -> None:
    from PySide6.QtGui import QImage

    from tests.gui.fakes import FakeBackend

    img = QImage(200, 100, QImage.Format.Format_RGB32)
    img.fill("#ffffff")
    p = tmp_path / "diagram.png"
    img.save(str(p), "PNG")

    window._backend = FakeBackend(fail=True, msg="simulated error")
    window._analysis_page.load_image(str(p))
    window._on_analyze_requested()

    deadline = time.time() + 15
    while (
        window._worker is not None
        and window._worker.isRunning()
        and time.time() < deadline
    ):
        qapp.processEvents()
        time.sleep(0.005)
    if window._worker is not None:
        window._worker.wait(5000)
    for _ in range(20):
        qapp.processEvents()
        time.sleep(0.005)

    model = window._analysis_page._progress_model
    assert model.is_finished
    assert all(
        e.state not in (PENDING, RUNNING) for e in model.entries
    ), "no stage may stay pending/running after a failed run"
    assert window._analysis_page._progress_panel._percent_label.text() == "0%"


# ---------------------------------------------------------------------------
# Workflow indicator + header status (state-aware)
# ---------------------------------------------------------------------------


def test_indicator_fresh_state(window: MainWindow) -> None:
    window._update_workflow_indicator()
    text = window._workflow_indicator.text()
    assert "Review" in text
    assert "\u25CF" in text
    assert "\u25CB" in text


def test_indicator_blocked_review(window: MainWindow) -> None:
    window._session.complete_analysis(FakeWorkflow(pa=2))
    window._on_nav(NavDestination.ANALYZE)
    window._update_workflow_indicator()
    assert "\u26a0" in window._workflow_indicator.text()


def test_indicator_all_but_results_done(window: MainWindow) -> None:
    window._session.complete_analysis(FakeWorkflow())
    window._session.complete_apply(FakeReviewedCandidate(valid=True))
    window._on_nav(NavDestination.ANALYZE)
    window._update_workflow_indicator()
    text = window._workflow_indicator.text()
    assert text.count("\u2713") >= 2
    assert "\u25CB" in text


def test_header_status_analysing(window: MainWindow) -> None:
    window._session.begin_analysis()
    window._update_header_status()
    assert window._header_status.text() == "ANALYZING"
    assert "font-weight" in window._header_status.styleSheet()


def test_header_status_blocked_review(window: MainWindow) -> None:
    window._session.complete_analysis(FakeWorkflow(pa=1))
    window._session.complete_apply(FakeReviewedCandidate(valid=True))
    window._session.reviews_dirty = True
    window._update_header_status()
    assert window._header_status.text() == "BLOCKED REVIEW"
    assert WARNING in window._header_status.styleSheet()


def test_header_status_valid(window: MainWindow) -> None:
    window._session.complete_analysis(FakeWorkflow())
    window._session.complete_apply(FakeReviewedCandidate(valid=True))
    window._update_header_status()
    assert window._header_status.text() == "VALID"
    assert SUCCESS in window._header_status.styleSheet()


def test_header_status_error(window: MainWindow) -> None:
    window._session.fail_analysis("boom")
    window._update_header_status()
    assert window._header_status.text() == "ERROR"
    assert DANGER in window._header_status.styleSheet()


# ---------------------------------------------------------------------------
# Regression links with existing surface contracts
# ---------------------------------------------------------------------------


def test_review_progress_label_format_preserved(window: MainWindow) -> None:
    window._session = reviewed_session(n_activities=2, n_dependencies=1)
    window._review_page.refresh(window._session)
    label = window._review_page._progress_label.text()
    assert "Resolved" in label and "Pending" in label
    assert window._review_page._percent_label.text() == "0%"
    assert "/" in window._review_page._breakdown_label.text()
    assert "Activities" in window._review_page._breakdown_label.text()


def test_header_keeps_font_weight_in_muted_state(window: MainWindow) -> None:
    window._update_header_status()
    assert "font-weight" in window._header_status.styleSheet()
    assert window._header_status.text() == ""
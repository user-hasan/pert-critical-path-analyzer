"""
GUI regression tests: recoverable (REVIEW_REQUIRED) analysis outcomes.

A pipeline outcome with `review_required=True` (e.g. an unreadable/invalid
activity duration like 0.0) must never become a fatal Analysis ERROR. These
tests lock in the GUI behaviour: the Analysis page offers "Open Review
Center", the header shows BLOCKED_REVIEW as a warning, and the Validation
page explicitly lists the pending invalid duration item.
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.session import AppState, GuiSession
from pert_analyzer.gui.themes.palette import WARNING
from pert_analyzer.pipeline.result import AnalysisStatus
from tests.gui.fakes import (
    FakeBackend,
    FakeReviewedCandidate,
    FakeReviewedWorkflow,
    FakeReviewSession,
    FakeValidatedWorkflow,
    FakeWorkflow,
    make_review_session,
)


@pytest.fixture()
def tmp_image(tmp_path: "Path") -> str:  # noqa: F821
    """Create a minimal valid PNG for the image preview."""
    img = QImage(200, 100, QImage.Format.Format_RGB32)
    img.fill(Qt.GlobalColor.white)
    p = tmp_path / "test.png"
    img.save(str(p), "PNG")
    return str(p)


@pytest.fixture()
def window(qapp: "QApplication") -> "MainWindow":  # noqa: F821
    w = MainWindow()
    yield w
    w.close()


def _pump(qapp: "QApplication", cycles: int = 60) -> None:  # noqa: F821
    """Process GUI events until the analysis worker finishes."""
    for _ in range(cycles):
        qapp.processEvents()
        time.sleep(0.01)


def test_analysis_page_review_required_shows_open_review_button(
    window: MainWindow, tmp_image: str, qapp: "QApplication"  # noqa: F821
) -> None:
    window._analysis_page.load_image(tmp_image)
    window._analysis_page.set_analysis_result_status("REVIEW_REQUIRED", activity_count=3)

    assert not window._analysis_page._open_review_btn.isHidden()
    assert window._analysis_page._open_review_btn.text() == "Open Review Center"
    assert window._analysis_page._status_badge.text() == "REVIEW"
    status = window._analysis_page._status_panel._label.text()
    assert "3 activities detected" in status
    assert "review" in status.lower()


def test_open_review_button_navigates_to_review_center(
    window: MainWindow, tmp_image: str, qapp: "QApplication"  # noqa: F821
) -> None:
    workflow = FakeWorkflow(
        pdr=1,
        status=AnalysisStatus.REVIEW_REQUIRED,
        review_required=True,
    )
    window._backend = FakeBackend(workflow)
    window._analysis_page.load_image(tmp_image)
    window._on_analyze_requested()
    _pump(qapp)

    assert window._session.state == AppState.REVIEW_REQUIRED
    # The worker navigates straight to the Review Center; come back and use
    # the explicit "Open Review Center" button instead.
    window._on_nav(NavDestination.ANALYZE)
    assert not window._analysis_page._open_review_btn.isHidden()
    window._analysis_page._open_review_btn.click()
    _pump(qapp)
    assert window._stack.currentIndex() == NavDestination.REVIEW.value


def test_full_review_to_results_flow(  # noqa: C901
    window: MainWindow, tmp_image: str, qapp: "QApplication"  # noqa: F821
) -> None:
    """A clean REVIEW_REQUIRED recovery must reach read only after apply."""
    workflow = FakeWorkflow(
        pdr=0,
        status=AnalysisStatus.REVIEW_REQUIRED,
        review_required=True,
    )
    window._backend = FakeBackend(workflow)
    window._analysis_page.load_image(tmp_image)
    window._on_analyze_requested()
    _pump(qapp)

    assert window._session.state == AppState.REVIEW_REQUIRED
    assert window._session.validation_status.value == "NOT_AVAILABLE"


def test_header_blocked_review_shows_warning_color(
    window: MainWindow, qapp: "QApplication"  # noqa: F821
) -> None:
    session = GuiSession()
    candidate = FakeReviewedCandidate(valid=True, cpm_gate="BLOCKED_REVIEW")
    session.workflow = FakeValidatedWorkflow(
        candidate=candidate,
        review_session=FakeReviewSession(pdr=1),
    )
    session.complete_apply(candidate)
    window._session = session

    window._update_header_status()

    assert window._header_status.text() == "BLOCKED REVIEW"
    assert WARNING in window._header_status.styleSheet()


def test_validation_page_blocked_review_hint_lists_invalid_duration(
    window: MainWindow, qapp: "QApplication"  # noqa: F821
) -> None:
    review_session = make_review_session(
        n_activities=0,
        n_dependencies=0,
        n_durations=1,
    )
    review_session.durations[0].current_duration = 0.0

    session = GuiSession()
    session.complete_analysis(FakeReviewedWorkflow(review_session, review_required=True))
    candidate = FakeReviewedCandidate(valid=True, cpm_gate="BLOCKED_REVIEW")
    session.complete_apply(candidate)

    window._validation_page.refresh(session)

    assert window._validation_page.stack_index == window._validation_page._WORKSPACE
    assert window._validation_page._status_label.text() == "REVIEW REQUIRED"
    hint = window._validation_page._status_hint.text()
    assert "1 review item(s) are still pending" in hint
    assert "Activity 'A0' has an invalid duration." in hint
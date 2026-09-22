"""
Tests for the Analysis Page: upload, preview, analyze, failure handling.
"""

from __future__ import annotations

import time

import pytest
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from tests.gui.fakes import FakeBackend, FakeWorkflow


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


def test_analyze_disabled_without_image(window: MainWindow) -> None:
    window._on_nav(NavDestination.ANALYZE)
    assert not window._analysis_page._analyze_btn.isEnabled()


def test_analyze_empty_state_shows_placeholder(window: MainWindow) -> None:
    window._on_nav(NavDestination.ANALYZE)
    assert "No image selected" in window._analysis_page._image_view.text()
    assert not window._analysis_page._analyze_btn.isEnabled()
    assert not window._analysis_page._remove_btn.isEnabled()
    assert window._analysis_page._status_badge.text() == "READY"


def test_analyze_enabled_after_image_selected(
    window: MainWindow, tmp_image: str
) -> None:
    window._analysis_page.load_image(tmp_image)
    assert window._analysis_page._analyze_btn.isEnabled()
    assert window._analysis_page._remove_btn.isEnabled()
    assert "test.png" in window._analysis_page._file_info.text()
    assert "200" in window._analysis_page._file_info.text()
    assert "100" in window._analysis_page._file_info.text()


def test_remove_image_disables_analyze(
    window: MainWindow, tmp_image: str
) -> None:
    window._analysis_page.load_image(tmp_image)
    assert window._analysis_page._analyze_btn.isEnabled()
    window._analysis_page._on_remove()
    assert not window._analysis_page._analyze_btn.isEnabled()
    assert not window._analysis_page._remove_btn.isEnabled()
    assert window._analysis_page._image_path is None


def test_image_preview_shows_dimensions(
    window: MainWindow, tmp_image: str, qapp: "QApplication"  # noqa: F821
) -> None:
    w, h = window._analysis_page._image_view.set_image(tmp_image)
    assert w == 200
    assert h == 100
    assert window._analysis_page._image_view._pixmap is not None
    assert not window._analysis_page._image_view._pixmap.isNull()


def test_analysis_failure_does_not_crash(
    window: MainWindow,
    tmp_image: str,
    qapp: "QApplication",  # noqa: F821
) -> None:
    window._backend = FakeBackend(fail=True, msg="simulated error")
    window._analysis_page.load_image(tmp_image)
    window._on_analyze_requested()
    qapp.processEvents()
    # let QThread finish
    for _ in range(50):
        qapp.processEvents()
        time.sleep(0.01)
    assert window._session.state.value == "ERROR"
    assert "simulated error" in window._session.error_message
    assert window._analysis_page._analyze_btn.isEnabled()

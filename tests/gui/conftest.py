"""
Shared fixtures for all GUI tests.

Sets QT_QPA_PLATFORM=offscreen before any PySide6 import so tests never
flash windows or depend on a real display.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")  # noqa: E402

import pytest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    """Session-scoped QApplication shared by all GUI tests."""
    app = QApplication.instance() or QApplication([])
    yield app  # type: ignore[misc]

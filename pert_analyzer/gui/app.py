"""
Application entry point: creates QApplication, applies theme, shows MainWindow.
"""

from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.themes.style import apply_theme

_LOCAL_SIZE_W = 1500
_LOCAL_SIZE_H = 900


def _screen_large_enough(window: MainWindow) -> bool:
    """True when the primary screen is comfortably bigger than the minimum."""
    del window  # kept for future use (multi-monitor primary screen check)
    try:
        screen = QApplication.primaryScreen()
        if screen is not None:
            area = screen.availableGeometry()
            return area.width() >= 1280 and area.height() >= 720
    except Exception:
        pass
    return False


def _center_window(window: MainWindow) -> None:
    """Center the window on the primary screen's available work area."""
    screen = QApplication.primaryScreen()
    if screen is None:
        return
    area = screen.availableGeometry()
    x = area.x() + (area.width() - window.width()) // 2
    y = area.y() + (area.height() - window.height()) // 2
    window.move(max(area.x(), x - area.x() // 2), max(area.y(), y - area.y() // 2))


def _fit_available(window: MainWindow) -> None:
    """Fit the window inside the work area at ~96% when the screen is small."""
    screen = QApplication.primaryScreen()
    if screen is None:
        window.resize(_LOCAL_SIZE_W, _LOCAL_SIZE_H)
        return
    area = screen.availableGeometry()
    width = int(area.width() * 0.96)
    height = int(area.height() * 0.96)
    window.resize(min(width, _LOCAL_SIZE_W), min(height, _LOCAL_SIZE_H))
    _center_window(window)


def main() -> int:
    """Launch the GUI application. Returns the exit code."""
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    if _screen_large_enough(window):
        window.resize(_LOCAL_SIZE_W, _LOCAL_SIZE_H)
        _center_window(window)
    else:
        _fit_available(window)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
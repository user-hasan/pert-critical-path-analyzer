"""Capture a screenshot of the Network Builder page with example data."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.style import apply_theme

OUT_DIR = _ROOT / "docs" / "ui_review" / "final"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.resize(1500, 900)

    window._on_nav(NavDestination.NETWORK_BUILDER)
    window.show()
    app.processEvents()

    page = window._network_builder_page
    page._model.load_example()
    page._reload_from_model()
    app.processEvents()

    pixmap = window.grab()
    path = OUT_DIR / "11_network_builder.png"
    pixmap.save(str(path))
    print(f"Saved: {path}")


if __name__ == "__main__":
    sys.exit(main() or 0)

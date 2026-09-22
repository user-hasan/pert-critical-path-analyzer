"""Capture the full chain: Builder → Analyze → Results dashboard."""

import os, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import QTimer

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.style import apply_theme

OUT_DIR = _ROOT / "docs" / "ui_review" / "final"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.resize(1500, 900)

    # Navigate to Builder
    window._on_nav(NavDestination.NETWORK_BUILDER)
    window.show()
    app.processEvents()

    # Add minimal network
    model = window._network_builder_page.get_model()
    for aid, dur, pred in [("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"), ("D", 3, "B, C")]:
        model.add_activity(aid, duration=dur, predecessors=pred)
    window._network_builder_page._reload_from_model()
    app.processEvents()

    # Screenshot: Builder with data
    pixmap = window.grab()
    pixmap.save(str(OUT_DIR / "11_network_builder.png"))
    print("Saved: 11_network_builder.png")

    # Click Analyze → Results (emit signal directly)
    window._network_builder_page.analyze_requested.emit()
    app.processEvents()

    # Screenshot: Results dashboard from Builder
    pixmap = window.grab()
    pixmap.save(str(OUT_DIR / "12_results_from_builder.png"))
    print("Saved: 12_results_from_builder.png")


if __name__ == "__main__":
    main()

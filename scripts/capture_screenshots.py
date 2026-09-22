"""
Capture screenshots of every GUI page using the reference AON workflow.
Run with: python scripts/capture_screenshots.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.style import apply_theme
from tests.helpers.reference_gold import (
    REFERENCE_AON,
    build_corrected_session,
    load_gold,
    run_reference_analysis,
)

OUT_DIR = _ROOT / "docs" / "ui_review" / "final"
_workflow = None


def _grab(window: MainWindow, name: str) -> None:
    pixmap = window.grab()
    path = OUT_DIR / f"{name}.png"
    pixmap.save(str(path))
    print(f"  saved {path}")


def _set_session_data(window: MainWindow) -> None:
    """Push a fully-resolved reference AON session into the GUI."""
    global _workflow
    print("Running reference analysis...")
    pipeline_result = run_reference_analysis()
    gold = load_gold()
    session = build_corrected_session(pipeline_result, source_image_id=REFERENCE_AON.name)

    from pert_analyzer.pipeline.review_api import ReviewWorkflow
    _workflow = ReviewWorkflow(pipeline_result=pipeline_result, review_session=session)

    gui_session = window._session
    gui_session.complete_analysis(_workflow)
    gui_session.current_image_path = str(REFERENCE_AON)
    gui_session.has_applied_reviews = True
    gui_session.reviews_dirty = False

    candidate = _workflow.apply()
    gui_session.candidate = candidate

    window._refresh_pages()
    window._update_workflow_indicator()

    print(f"Session loaded: {candidate.graph.activity_count} activities, "
          f"{candidate.graph.dependency_count} deps")
    print(f"Validation status: {gui_session.validation_status}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.resize(1500, 900)

    print("Setting up reference AON session...")
    _set_session_data(window)
    window.show()
    app.processEvents()

    # 1. Analyze page - load image and show completed state
    print("\nCapturing: 01_analyze")
    window._on_nav(NavDestination.ANALYZE)
    app.processEvents()
    analyze_page = window._analysis_page
    analyze_page.load_image(str(REFERENCE_AON))
    app.processEvents()
    gui_sess = window._session
    summary = getattr(gui_sess, "review_summary", None) or {}
    count = summary.get("total_activities") or None
    analyze_page.set_analysis_result_status("REVIEW_REQUIRED", activity_count=count)
    analyze_page.show_completed(_workflow, review_item_total=gui_sess.review_item_total())
    app.processEvents()
    _grab(window, "01_analyze")

    # 2. Understanding page
    print("Capturing: 02_understanding")
    window._on_nav(NavDestination.UNDERSTANDING)
    app.processEvents()
    _grab(window, "02_understanding")

    # 3. Review Center
    print("Capturing: 03_review")
    window._on_nav(NavDestination.REVIEW)
    app.processEvents()
    _grab(window, "03_review")

    # 4. Validation page
    print("Capturing: 04_validation")
    window._on_nav(NavDestination.VALIDATE)
    app.processEvents()
    _grab(window, "04_validation")

    # 5. Results Overview
    print("Capturing: 05_results_overview")
    window._on_nav(NavDestination.RESULTS)
    app.processEvents()
    _grab(window, "05_results_overview")

    # 6. Results sub-tabs
    print("\nCapturing Results sub-tabs...")
    results_page = window._results_page

    for idx, name in [
        (0, "06_results_overview_tab"),
        (1, "07_results_network"),
        (2, "08_results_activities"),
        (3, "09_results_critical_paths"),
        (4, "10_results_pert"),
    ]:
        results_page._tabs.setCurrentIndex(idx)
        app.processEvents()
        if idx == 3:
            cp_tab = results_page._critical_paths
            if cp_tab._paths:
                cp_tab._selected_index = -1
                cp_tab.select_path(0)
                cp_tab.repaint()
                app.processEvents()
        img = results_page.grab()
        path = OUT_DIR / f"{name}.png"
        img.save(str(path))
        print(f"  saved {path}")

    print("\nDone. Screenshots in:", OUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

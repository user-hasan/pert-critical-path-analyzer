"""Runtime integration trace for Manual Network Builder."""

import os, sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.themes.style import apply_theme
from pert_analyzer.gui.results.data import describe_ready, extract, NO_ANALYSIS
from pert_analyzer.gui.session import AppState, ValidationCenterStatus

STEP = 0
def log(stage, msg):
    global STEP
    STEP += 1
    print(f"  [{STEP:02d}] [{stage}] {msg}", flush=True)

def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    app.processEvents()

    # ── Navigate to Builder ──────────────────────────
    log("NAV", "Navigating to Network Builder")
    window._on_nav(NavDestination.NETWORK_BUILDER)
    app.processEvents()

    page = window._network_builder_page
    model = page.get_model()

    # ── Add activities ───────────────────────────────
    log("BUILD", "Adding: A(1), B(2,A), C(2,A), D(3,B,C)")
    for aid, dur, pred in [("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"), ("D", 3, "B, C")]:
        err = model.add_activity(aid, duration=dur, predecessors=pred)
        if err:
            log("BUILD", f"  ERROR adding {aid}: {err}")
            return
    log("BUILD", f"Activities={model.activity_count}, Deps={model.dependency_count}")
    app.processEvents()

    # ── Build GraphModel ─────────────────────────────
    log("GRAPH", "Building GraphModel...")
    graph = model.build_graph_model()
    log("GRAPH", f"  activities={len(graph.activities)}, deps={len(graph.dependencies)}")

    # ── Run validation ───────────────────────────────
    log("VALIDATION", "Running validation...")
    result = model.validate()
    log("VALIDATION", f"  status={result.status}, is_valid={model.is_valid}")
    log("VALIDATION", f"  errors={[e.message for e in result.errors]}")

    # ── Run CPM ──────────────────────────────────────
    log("CPM", "Running CPM...")
    err = model.calculate_cpm()
    if err:
        log("CPM", f"  ERROR: {err}")
        return
    cpm = model.cpm_result
    log("CPM", f"  duration={cpm.project_duration}, critical_paths={len(cpm.critical_paths)}")
    log("CPM", f"  critical_path={cpm.critical_path}")

    # ── Create candidate ─────────────────────────────
    log("CANDIDATE", "Creating ReviewedGraphCandidate...")
    candidate = model.create_candidate()
    log("CANDIDATE", f"  candidate is None={candidate is None}")
    if candidate is None:
        return
    log("CANDIDATE", f"  cpm_gate={candidate.cpm_gate}, cpm is None={candidate.cpm is None}")

    # ── Assign to session ────────────────────────────
    log("SESSION", "Assigning candidate to session...")
    session = window._session
    session.state = AppState.RESULTS_AVAILABLE
    session.candidate = candidate
    session.has_applied_reviews = True
    session.reviews_dirty = False
    log("SESSION", f"  workflow is None={session.workflow is None}")
    log("SESSION", f"  candidate is None={session.candidate is None}")
    log("SESSION", f"  current_candidate is None={session.current_candidate is None}")
    log("SESSION", f"  validation_status={session.validation_status}")

    # ── Check describe_ready ─────────────────────────
    log("READYNESS", "Checking describe_ready(session)...")
    ready, reason = describe_ready(session)
    log("READYNESS", f"  ready={ready}, reason={reason}")
    if reason == NO_ANALYSIS:
        log("READYNESS", "  >>> BLOCKED: workflow is None → NO_ANALYSIS")

    # ── Check extract ────────────────────────────────
    log("EXTRACT", "Checking extract(session)...")
    data = extract(session)
    log("EXTRACT", f"  ready={data.ready}, reason={data.reason}")
    log("EXTRACT", f"  project_duration={data.project_duration}")
    log("EXTRACT", f"  activities={len(data.activities)}")

    # ── Navigate to Results ──────────────────────────
    log("NAV", "Navigating to Results...")
    window._on_nav(NavDestination.RESULTS)
    app.processEvents()

    results_page = window._results_page
    idx = results_page._stack.currentIndex()
    log("RESULTS", f"  stack index={idx}, DASHBOARD={results_page._DASHBOARD}, EMPTY={results_page._EMPTY}")
    is_dashboard = idx == results_page._DASHBOARD
    log("RESULTS", f"  Dashboard visible={is_dashboard}")
    if not is_dashboard:
        log("RESULTS", "  >>> FAIL: Results page shows NOT-READY state")

    log("DONE", "Trace complete.")


if __name__ == "__main__":
    main()

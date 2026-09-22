"""Real-GUI widget-state verification for Manual Builder → Results.

Drives the ACTUAL Analyze button (not direct method calls) and asserts
the visible Results widgets are populated:
  - Overview KPI card values (duration 6.0, activities 4, deps 4, crit 2)
  - Activities table row count == 4
  - Network scene node/edge item counts == 4 / 4
  - Critical paths list count == 2
  - ResultsPage instance identity across creation / refresh / navigation
"""

import os, sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from PySide6.QtWidgets import QApplication, QMessageBox

from pert_analyzer.gui.main_window import MainWindow
from pert_analyzer.gui.navigation import NavDestination
from pert_analyzer.gui.pages.results_page import ResultsPage
from pert_analyzer.gui.themes.style import apply_theme
from pert_analyzer.gui.session import AppState, ValidationCenterStatus

STEP = 0
FAILURES = []


def log(stage, msg):
    global STEP
    STEP += 1
    print(f"  [{STEP:02d}] [{stage}] {msg}", flush=True)


def check(name, cond, detail=""):
    mark = "PASS" if cond else "FAIL"
    print(f"  [{step():02d}] [ASSERT] {mark}: {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not cond:
        FAILURES.append(name)


def step():
    global STEP
    STEP += 1
    return STEP


def silence_qmessagebox():
    """Auto-accept any QMessageBox so offscreen runs are non-interactive."""
    QMessageBox.question = staticmethod(
        lambda *a, **k: QMessageBox.StandardButton.Yes
    )
    QMessageBox.warning = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)
    QMessageBox.information = staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    apply_theme(app)
    silence_qmessagebox()
    window = MainWindow()
    window.resize(1500, 900)
    window.show()
    app.processEvents()

    results_page = window._results_page
    log("IDENT", f"ResultsPage created once: {type(results_page).__name__}")
    check("ResultsPage is single instance", isinstance(results_page, ResultsPage))

    # ── Navigate to Builder via the real nav route ──────────────
    window._on_nav(NavDestination.NETWORK_BUILDER)
    app.processEvents()
    log("NAV", "Navigated to Network Builder")

    page = window._network_builder_page
    model = page.get_model()

    # ── Add the minimal acceptance network A/B/C/D ─────────────
    log("BUILD", "Adding A(1), B(2,A), C(2,A), D(3,B,C)")
    for aid, dur, pred in [("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"), ("D", 3, "B, C")]:
        err = model.add_activity(aid, duration=dur, predecessors=pred)
        if err:
            log("BUILD", f"  ERROR adding {aid}: {err}")
            return 1
    log("BUILD", f"  model: {model.activity_count} activities, {model.dependency_count} deps")

    # ── Model state (what the page itself drives) ───────────────
    model.validate()
    log("VALIDATION", f"  is_valid={model.is_valid}")
    check("Model valid after structure change", model.is_valid)
    err = model.calculate_cpm()
    log("CPM", f"  err={err!r}, duration={model.cpm_result.project_duration if model.cpm_result else None}")
    check("CPM computed", err is None and model.cpm_result is not None)

    # Ensure the Analyze button is enabled (real GUI gate)
    page._on_validation_changed()
    page._on_cpm_calculated()
    analyze_btn = page._analyze_btn
    log("BUTTON", f"  analyze_btn enabled={analyze_btn.isEnabled()}")
    check("Analyze button enabled", analyze_btn.isEnabled())

    # ── Click the REAL Analyze button ───────────────────────────
    analyze_btn.click()
    app.processEvents()

    # ── Session state after real click ──────────────────────────
    session = window._session
    log("SESSION", f"  state={session.state.value} validation_status={session.validation_status}")
    check("Session state RESULTS_AVAILABLE", session.state == AppState.RESULTS_AVAILABLE)
    check("validation_status VALID", session.validation_status == ValidationCenterStatus.VALID)
    check("candidate present", session.current_candidate is not None)

    # ── Results page was navigated to (real nav route) ──────────
    log("NAV", "Check main stack shows Results")
    check("Main stack on Results", window._stack.currentIndex() == NavDestination.RESULTS.value)
    check("ResultsPage is the SAME instance after nav",
          window._stack.currentWidget() is results_page)
    check("ResultsPage internal stack on DASHBOARD",
          results_page._stack.currentIndex() == results_page._DASHBOARD)

    # ── Widget-state assertions (the real data) ─────────────────
    overview = results_page.overview()
    data = results_page.data
    log("WIDGETS", f"  extract data: duration={data.project_duration}, "
                   f"activities={len(data.activities)}, deps={len(data.dependencies)}")
    check("extract ready", data.ready is True)
    check("duration == 6.0", data.project_duration == 6.0)
    check("4 activities", len(data.activities) == 4)
    check("4 dependencies", len(data.dependencies) == 4)
    check("critical_path_count == 2", data.critical_path_count == 2)
    check("critical path is A→B→D", data.critical_paths and data.critical_paths[0] == ["A", "B", "D"])

    # KPI cards
    kpi = {k: c.value() for k, c in overview._kpi_cards.items()}
    log("KPI", f"  cards={kpi}")
    check("KPI duration = '6 days'", kpi["duration"] == "6 days")
    check("KPI activities = '4'", kpi["activities"] == "4")
    check("KPI dependencies = '4'", kpi["dependencies"] == "4")
    check("KPI critical_activities = '4'", kpi["critical_activities"] == "4")
    check("KPI critical_paths = '2'", kpi["critical_paths"] == "2")

    # Activities table
    table = results_page.activities()._table
    check("Activities table has 4 rows", table.rowCount() == 4)
    col = {table.horizontalHeaderItem(i).text(): i for i in range(table.columnCount())}
    ids = {table.item(r, col["ID"]).text() for r in range(table.rowCount())}
    check("Activities table ids = {A,B,C,D}", ids == {"A", "B", "C", "D"}, detail=str(ids))

    # Network scene
    net = results_page.network()
    check("Network nodes == 4", len(net._node_items) == 4, detail=str(list(net._node_items)))
    check("Network edges == 4", len(net._edge_items) == 4, detail=str(list(net._edge_items)))

    # Critical paths list (dedicated tab)
    crit = results_page.critical_paths()
    check("Critical paths list count == 2", crit._path_list._list.count() == 2)

    # Overview embedded widgets match too
    check("Overview path list == 2", overview.path_list()._list.count() == 2)
    check("Overview embedded table == 4 rows",
          overview.embedded_activities()._table.rowCount() == 4)

    log("DONE", f"{'ALL PASS' if not FAILURES else f'{len(FAILURES)} FAILURES: {FAILURES}'}")
    return 0 if not FAILURES else 1


if __name__ == "__main__":
    sys.exit(main())
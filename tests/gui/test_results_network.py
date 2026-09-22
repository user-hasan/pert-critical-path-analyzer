"""
Network canvas (Phase 4) tests.

Covers the deterministic AON layout, node/edge rendering, critical
highlighting, zoom/reset behavior, path highlighting, node selection
signals, empty state, and rendering to an offscreen image.
"""

from __future__ import annotations

import pytest
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtWidgets import QApplication

from pert_analyzer.gui.results.layout import NODE_HEIGHT, NODE_WIDTH, build_layout
from pert_analyzer.gui.results.network import (
    MAX_ZOOM,
    MIN_ZOOM,
    ActivityNodeItem,
    NetworkTab,
    _draw_arrowhead,
)
from pert_analyzer.gui.results.data import ResultsData
from tests.gui.fakes import (
    FakeCpmResult,
    FakeGraphActivity,
    FakeGraphDependency,
    make_analysis,
    make_fake_graph,
)


def sample_data() -> ResultsData:
    """Small deterministic graph: A0->A1->A2 plus parallel A3->A1 edge."""
    graph = make_fake_graph(
        activities={
            aid: FakeGraphActivity(aid, 2.0)
            for aid in ("A0", "A1", "A2", "A3")
        },
        dependencies=[
            FakeGraphDependency("E0", "A0", "A1"),
            FakeGraphDependency("E1", "A1", "A2"),
            FakeGraphDependency("E2", "A3", "A1"),
        ],
    )
    cpm = FakeCpmResult(
        project_duration=6.0,
        critical_paths=[["A0", "A1", "A2"]],
        analyses={
            aid: make_analysis(aid, 0.0, 2.0, 0.0, 2.0, 0.0, 0.0, aid in ("A0", "A1", "A2"))
            for aid in ("A0", "A1", "A2", "A3")
        },
    )
    return ResultsData(
        project_duration=6.0,
        critical_path_count=1,
        critical_paths=[["A0", "A1", "A2"]],
        critical_activity_count=3,
        critical_edges={("A0", "A1"), ("A1", "A2")},
        ready=True,
        cpm=cpm,
        graph=graph,
    )


@pytest.fixture()
def tab(qapp: QApplication) -> NetworkTab:
    w = NetworkTab()
    w._view.resize(800, 600)
    return w


def sized(tab: NetworkTab) -> NetworkTab:
    tab._view.resize(800, 600)
    return tab


# ---------------------------------------------------------------------------
# Layout (pure, no widgets)
# ---------------------------------------------------------------------------


def test_build_layout_is_deterministic() -> None:
    ids = ["A0", "A1", "A2", "A3", "A4"]
    deps = [("A0", "A1"), ("A1", "A2"), ("A0", "A3"), ("A3", "A4"), ("A2", "A4")]
    first = build_layout(ids, deps)
    second = build_layout(ids, deps)
    assert first == second


def test_build_layout_levels_and_positions() -> None:
    ids = ["A0", "A1", "A2"]
    deps = [("A0", "A1"), ("A1", "A2")]
    positions = build_layout(ids, deps)
    assert positions["A0"][2] == NODE_WIDTH
    assert positions["A0"][3] == NODE_HEIGHT
    assert positions["A0"][0] < positions["A1"][0] < positions["A2"][0]
    assert positions["A0"][1] == positions["A1"][1] == positions["A2"][1]


def test_build_layout_disconnected_nodes_appended() -> None:
    positions = build_layout(["X0", "X1"], [("X0", "X1")])
    assert set(positions) == {"X0", "X1"}
    assert positions["X0"][0] < positions["X1"][0]
    assert positions["X0"][1] == positions["X1"][1]


def test_build_layout_tolerates_bad_deps() -> None:
    positions = build_layout(["A0", "A1"], [None, ("A0", "MISSING"), 42])
    assert set(positions) == {"A0", "A1"}


# ---------------------------------------------------------------------------
# Network canvas
# ---------------------------------------------------------------------------


def test_network_renders_nodes_and_edges(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    assert set(tab.node_items()) == {"A0", "A1", "A2", "A3"}
    assert len(tab.edge_items()) == 3
    assert tab.is_empty() is False


def test_network_marks_critical_edges(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    assert tab.edge_items()[("A0", "A1")].is_critical
    assert tab.edge_items()[("A1", "A2")].is_critical
    assert not tab.edge_items()[("A3", "A1")].is_critical


def test_network_marks_critical_nodes(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    assert tab.node_items()["A1"].is_critical
    assert not tab.node_items()["A3"].is_critical


def test_network_empty_no_data(tab: NetworkTab) -> None:
    from PySide6.QtWidgets import QGraphicsSimpleTextItem

    sized(tab).set_data(ResultsData(ready=False))
    assert tab.is_empty()
    items = [
        item.text()
        for item in tab._scene.items()
        if isinstance(item, QGraphicsSimpleTextItem)
    ]
    assert items == ["No network data to display"]


def test_network_zoom_bounds(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.reset_zoom()
    base = tab.zoom
    tab.zoom_by(1.2)
    assert tab.zoom > base
    tab.zoom_by(1.0 / 1.2)
    assert abs(tab.zoom - base) < 1e-6


def test_network_zoom_is_clamped(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.reset_zoom()
    tab.zoom_by(1000.0)
    assert tab.zoom <= MAX_ZOOM + 1e-9
    tab.reset_zoom()
    tab.zoom_by(0.0001)
    assert tab.zoom >= MIN_ZOOM - 1e-9


def test_network_reset_zoom_centers(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.zoom_by(2.0)
    tab.reset_zoom()
    assert abs(tab.zoom - 1.0) < 1e-6


def test_network_fit_to_view(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.fit_to_view()
    assert tab.zoom > 0


def test_network_highlight_path(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.highlight_path(["A0", "A1", "A2"])
    assert tab.current_path() == ["A0", "A1", "A2"]
    assert tab.node_items()["A0"]._path_highlight
    assert tab.node_items()["A3"]._path_highlight is False
    assert tab.edge_items()[("A0", "A1")]._path_highlight
    assert tab.edge_items()[("A3", "A1")]._path_highlight is False


def test_network_clear_highlight(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.highlight_path(["A0", "A1", "A2"])
    tab.clear_highlight()
    assert tab.current_path() == []
    assert all(not item._path_highlight for item in tab.node_items().values())
    assert all(not item._path_highlight for item in tab.edge_items().values())


def test_network_node_selected_signal(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    received: list[str] = []
    tab.node_selected.connect(received.append)
    tab.node_items()["A1"].clicked.emit("A1")
    assert received == ["A1"]
    assert tab.selected_activity == "A1"
    assert tab.node_items()["A1"]._selected


def test_network_set_selected_activity(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.set_selected_activity("A2")
    assert tab.selected_activity == "A2"
    assert tab.node_items()["A2"]._selected
    assert tab.node_items()["A0"]._selected is False


def test_network_selection_unknown_id_is_safe(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    tab.set_selected_activity("NOPE")
    assert tab.selected_activity == "NOPE"
    assert not any(item._selected for item in tab.node_items().values())


def test_network_render_to_image(tab: NetworkTab) -> None:
    """Rendering exercises node + arrowhead painting without crashing."""
    sized(tab).set_data(sample_data())
    image = tab._view.grab()
    assert not image.isNull()


def test_activity_node_tooltip(tab: NetworkTab) -> None:
    sized(tab).set_data(sample_data())
    node = tab.node_items()["A0"]
    assert node.toolTip().startswith("A0")
    assert "Duration: 2" in node.toolTip()


def test_arrowhead_tolerates_zero_length() -> None:
    image = QImage(64, 64, QImage.Format.Format_ARGB32)
    image.fill(QColor("transparent"))
    painter = QPainter(image)
    try:
        _draw_arrowhead(painter, 0, 0, 0, 0, QColor("#4f8cff"))
        _draw_arrowhead(painter, 0, 0, 20, 20, QColor("#ffffff"))
    finally:
        painter.end()
    assert not image.isNull()


def test_network_snapshot_uses_backend_values_only(tab: NetworkTab) -> None:
    """The canvas never recomputes graph relationships."""
    data = sample_data()
    sized(tab).set_data(data)
    assert tab._data is not None
    assert tab._data.graph is data.graph
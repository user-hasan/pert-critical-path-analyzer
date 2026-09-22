"""
Focused tests for the Manual Network Builder feature.

Covers spec section 39 acceptance criteria:
1-20: Activity CRUD, validation, CPM integration, inspector, graph sync.
"""

import pytest

from pert_analyzer.gui.builder.model import (
    ActivityEntry,
    ManualNetworkModel,
    RelationshipEntry,
)


# ────────────────────────────────────────────────────────────
# Fixtures
# ────────────────────────────────────────────────────────────


@pytest.fixture
def model() -> ManualNetworkModel:
    return ManualNetworkModel()


# ────────────────────────────────────────────────────────────
# 1. Add activity
# ────────────────────────────────────────────────────────────


class TestAddActivity:
    def test_add_single(self, model: ManualNetworkModel) -> None:
        err = model.add_activity("A", duration=1)
        assert err is None
        assert model.activity_count == 1
        assert model.get_activity("A") is not None
        assert model.get_activity("A").duration == 1.0

    def test_add_multiple(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_activity("C", duration=3)
        assert model.activity_count == 3

    def test_add_with_predecessors_string(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.add_activity("B", duration=2, predecessors="A")
        assert err is None
        preds = model.get_predecessors("B")
        assert "A" in preds


# ────────────────────────────────────────────────────────────
# 2. Reject duplicate activity
# ────────────────────────────────────────────────────────────


class TestRejectDuplicate:
    def test_duplicate_rejected(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.add_activity("A", duration=2)
        assert err is not None
        assert "already exists" in err
        assert model.activity_count == 1

    def test_empty_id_rejected(self, model: ManualNetworkModel) -> None:
        err = model.add_activity("", duration=1)
        assert err is not None
        assert "empty" in err.lower()


# ────────────────────────────────────────────────────────────
# 3. Edit duration
# ────────────────────────────────────────────────────────────


class TestEditDuration:
    def test_update_duration(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.update_activity("A", duration=5)
        assert err is None
        assert model.get_activity("A").duration == 5.0


# ────────────────────────────────────────────────────────────
# 4. Reject invalid duration
# ────────────────────────────────────────────────────────────


class TestRejectInvalidDuration:
    def test_negative_duration(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.update_activity("A", duration=-3)
        assert err is not None
        assert "non-negative" in err.lower()

    def test_non_numeric_duration(self, model: ManualNetworkModel) -> None:
        err = model.add_activity("A", duration="abc")
        assert err is not None
        assert "numeric" in err.lower()


# ────────────────────────────────────────────────────────────
# 5. Add predecessor via update
# ────────────────────────────────────────────────────────────


class TestAddPredecessor:
    def test_predecessor_creates_relationship(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        err = model.update_activity("B", predecessors="A")
        assert err is None
        assert model.has_relationship("A", "B")

    def test_multiple_predecessors(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=1)
        model.add_activity("C", duration=2, predecessors="A, B")
        assert model.has_relationship("A", "C")
        assert model.has_relationship("B", "C")


# ────────────────────────────────────────────────────────────
# 6. Reject missing predecessor
# ────────────────────────────────────────────────────────────


class TestRejectMissingPredecessor:
    def test_missing_pred_in_add(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.add_activity("B", duration=1, predecessors="Z")
        assert err is not None
        assert "does not exist" in err.lower()

    def test_missing_pred_in_update(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.update_activity("A", predecessors="Z")
        assert err is not None


# ────────────────────────────────────────────────────────────
# 7. Add relationship explicitly
# ────────────────────────────────────────────────────────────


class TestAddRelationship:
    def test_explicit_add(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        err = model.add_relationship("A", "B")
        assert err is None
        assert model.dependency_count == 1
        assert model.has_relationship("A", "B")


# ────────────────────────────────────────────────────────────
# 8. Reject duplicate relationship
# ────────────────────────────────────────────────────────────


class TestRejectDuplicateRelationship:
    def test_duplicate_edge(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        err = model.add_relationship("A", "B")
        assert err is not None
        assert "already exists" in err


# ────────────────────────────────────────────────────────────
# 9. Reject self-loop
# ────────────────────────────────────────────────────────────


class TestRejectSelfLoop:
    def test_self_loop(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.add_relationship("A", "A")
        assert err is not None
        assert "self-loop" in err.lower()


# ────────────────────────────────────────────────────────────
# 10. Detect cycle
# ────────────────────────────────────────────────────────────


class TestDetectCycle:
    def test_cycle_detected(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=1)
        model.add_activity("C", duration=1)
        model.add_relationship("A", "B")
        model.add_relationship("B", "C")
        model.add_relationship("C", "A")
        result = model.validate()
        assert result.status.value in ("invalid", "INVALID")
        cycle_msgs = [e.message.lower() for e in result.errors]
        assert any("cycle" in m for m in cycle_msgs)


# ────────────────────────────────────────────────────────────
# 11. Disconnected components
# ────────────────────────────────────────────────────────────


class TestDisconnectedComponents:
    def test_disconnected_reported(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=1)
        model.add_activity("C", duration=1)
        model.add_activity("D", duration=1)
        model.add_relationship("A", "B")
        model.add_relationship("C", "D")
        result = model.validate()
        assert result.component_count >= 2


# ────────────────────────────────────────────────────────────
# 12. Delete activity and associated relationships
# ────────────────────────────────────────────────────────────


class TestDeleteActivity:
    def test_delete_removes_edges(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=1)
        model.add_activity("C", duration=1)
        model.add_relationship("A", "B")
        model.add_relationship("B", "C")
        model.remove_activity("B")
        assert model.activity_count == 2
        assert not model.has_relationship("A", "B")
        assert not model.has_relationship("B", "C")

    def test_delete_nonexistent(self, model: ManualNetworkModel) -> None:
        err = model.remove_activity("Z")
        assert err is not None


# ────────────────────────────────────────────────────────────
# 13. Table/graph model synchronization
# ────────────────────────────────────────────────────────────


class TestSync:
    def test_graph_matches_model(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        graph = model.build_graph_model()
        assert len(graph.activities) == 2
        assert len(graph.dependencies) == 1
        assert graph.activities["A"].duration == 1.0
        assert graph.activities["B"].duration == 2.0
        assert graph.dependencies[0].source == "A"
        assert graph.dependencies[0].target == "B"


# ────────────────────────────────────────────────────────────
# 14. CPM stale state
# ────────────────────────────────────────────────────────────


class TestCpmStaleState:
    def test_cpm_invalidated_on_change(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        model.calculate_cpm()
        assert not model.cpm_is_stale
        model.update_activity("B", duration=5)
        assert model.cpm_is_stale


# ────────────────────────────────────────────────────────────
# 15. Valid network reaches existing CPM engine
# ────────────────────────────────────────────────────────────


class TestCpmIntegration:
    def test_simple_cpm(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_activity("C", duration=3)
        model.add_relationship("A", "B")
        model.add_relationship("B", "C")
        err = model.calculate_cpm()
        assert err is None
        assert model.cpm_result is not None
        assert model.cpm_result.project_duration == 6.0

    def test_parallel_cpm(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_activity("C", duration=2)
        model.add_activity("D", duration=3)
        model.add_relationship("A", "B")
        model.add_relationship("A", "C")
        model.add_relationship("B", "D")
        model.add_relationship("C", "D")
        err = model.calculate_cpm()
        assert err is None
        assert model.cpm_result.project_duration == 6.0
        critical = model.cpm_result.critical_paths
        assert len(critical) > 0


# ────────────────────────────────────────────────────────────
# 16. Manual network reaches existing Results Dashboard
# ────────────────────────────────────────────────────────────


class TestResultsIntegration:
    def test_candidate_creation(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        candidate = model.create_candidate()
        assert candidate is not None
        assert candidate.graph is not None
        assert candidate.validation is not None
        assert candidate.cpm is not None
        assert candidate.cpm_gate.value in ("RUNNABLE", "runnable")


# ────────────────────────────────────────────────────────────
# 17. Critical paths shown from actual CPM data
# ────────────────────────────────────────────────────────────


class TestCriticalPaths:
    def test_critical_paths_available(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_activity("C", duration=3)
        model.add_relationship("A", "B")
        model.add_relationship("B", "C")
        model.calculate_cpm()
        assert len(model.cpm_result.critical_paths) > 0
        assert model.cpm_result.critical_path == ["A", "B", "C"]


# ────────────────────────────────────────────────────────────
# 18. Node selection updates inspector data
# ────────────────────────────────────────────────────────────


class TestNodeSelection:
    def test_select_activity(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        model.select_activity("A")
        assert model.selected_activity == "A"
        assert model.selected_relationship is None
        assert model.get_predecessors("A") == []
        assert model.get_successors("A") == ["B"]

    def test_select_relationship(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        model.select_relationship("A", "B")
        assert model.selected_relationship == ("A", "B")
        assert model.selected_activity is None


# ────────────────────────────────────────────────────────────
# 19. Relationship selection updates inspector
# ────────────────────────────────────────────────────────────


class TestRelationshipSelection:
    def test_relationship_data(self, model: ManualNetworkModel) -> None:
        model.add_activity("X", duration=5)
        model.add_activity("Y", duration=3)
        model.add_relationship("X", "Y")
        rels = model.get_all_relationships()
        assert len(rels) == 1
        assert rels[0].source == "X"
        assert rels[0].target == "Y"


# ────────────────────────────────────────────────────────────
# 20. Reference-like fixture produces correct CPM
# ────────────────────────────────────────────────────────────


class TestReferenceFixture:
    def test_reference_aon_via_model(self, model: ManualNetworkModel) -> None:
        activities = [
            ("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"),
            ("D", 3, "B, C"), ("E", 4, "D"), ("F", 4, "D"),
            ("G", 4, "D"), ("H", 4, "D"),
            ("I", 5, "E, F, G, H"), ("J", 5, "I"), ("K", 3, "J"),
            ("L", 6, "K"), ("M", 6, "K"),
            ("N", 8, "L, M"), ("O", 3, "N"), ("P", 4, "O"),
            ("Q", 3, "P"), ("R", 2, "Q"), ("S", 1, "R"),
            ("T", 2, "S"), ("U", 1, "T"), ("V", 1, "U"),
        ]
        for aid, dur, pred in activities:
            err = model.add_activity(aid, duration=dur, predecessors=pred)
            assert err is None, f"Failed to add {aid}: {err}"

        assert model.activity_count == 22
        result = model.validate()
        assert result.status.value in ("valid", "VALID")
        assert result.component_count == 1

        err = model.calculate_cpm()
        assert err is None
        assert model.cpm_result.project_duration == 54.0
        assert len(model.cpm_result.critical_paths) == 16
        assert len(model.cpm_result.critical_path) == 17


# ────────────────────────────────────────────────────────────
# Additional: next_activity_id generation
# ────────────────────────────────────────────────────────────


class TestNextActivityId:
    def test_first_is_a(self, model: ManualNetworkModel) -> None:
        assert model.next_activity_id == "A"

    def test_after_a_is_b(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        assert model.next_activity_id == "B"

    def test_after_all_letters(self, model: ManualNetworkModel) -> None:
        for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            model.add_activity(c, duration=1)
        assert model.next_activity_id == "A1"


# ────────────────────────────────────────────────────────────
# Additional: remove relationship
# ────────────────────────────────────────────────────────────


class TestRemoveRelationship:
    def test_remove_explicit(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=1)
        model.add_relationship("A", "B")
        err = model.remove_relationship("A", "B")
        assert err is None
        assert model.dependency_count == 0

    def test_remove_nonexistent(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        err = model.remove_relationship("A", "B")
        assert err is not None


# ────────────────────────────────────────────────────────────
# Additional: clear
# ────────────────────────────────────────────────────────────


class TestClear:
    def test_clear_resets_all(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        model.calculate_cpm()
        model.clear()
        assert model.activity_count == 0
        assert model.dependency_count == 0
        assert model.cpm_result is None
        assert model.validation_result is None


# ────────────────────────────────────────────────────────────
# Additional: rename activity
# ────────────────────────────────────────────────────────────


class TestRenameActivity:
    def test_rename_updates_relationships(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        model.add_relationship("A", "B")
        err = model.rename_activity("A", "X")
        assert err is None
        assert model.has_relationship("X", "B")
        assert not model.has_relationship("A", "B")

    def test_rename_duplicate(self, model: ManualNetworkModel) -> None:
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        err = model.rename_activity("A", "B")
        assert err is not None


# ────────────────────────────────────────────────────────────
# REGRESSION: Full integration chain produces ready Results
# ────────────────────────────────────────────────────────────


class TestIntegrationChainRegression:
    """
    Regression test for the bug where manual builder → Results
    integration silently failed because describe_ready() gated
    on session.workflow being None.

    Verifies the complete chain:
    manual input → GraphModel → validation → CPM → candidate
    → session assignment → describe_ready → extract → Results
    """

    def test_full_chain_describe_ready(self) -> None:
        from pert_analyzer.gui.results.data import describe_ready, extract
        from pert_analyzer.gui.session import GuiSession, AppState
        from pert_analyzer.pipeline.human_review import CpmGateStatus

        model = ManualNetworkModel()
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2, predecessors="A")
        model.add_activity("C", duration=2, predecessors="A")
        model.add_activity("D", duration=3, predecessors="B, C")

        # Run full chain
        candidate = model.create_candidate()
        assert candidate is not None
        assert candidate.cpm_gate == CpmGateStatus.RUNNABLE
        assert candidate.cpm is not None

        # Assign to session WITHOUT setting workflow
        session = GuiSession()
        session.state = AppState.RESULTS_AVAILABLE
        session.candidate = candidate
        session.has_applied_reviews = True
        session.reviews_dirty = False
        session.workflow = None  # manual builder has no workflow

        # describe_ready must return True even with workflow=None
        ready, reason = describe_ready(session)
        assert ready is True, f"describe_ready failed: reason={reason}"
        assert reason == ""

        # extract must produce valid data
        data = extract(session)
        assert data.ready is True
        assert data.project_duration == 6.0
        assert len(data.activities) == 4

    def test_full_chain_cpm_data_populated(self) -> None:
        from pert_analyzer.gui.results.data import extract
        from pert_analyzer.gui.session import GuiSession, AppState

        model = ManualNetworkModel()
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2, predecessors="A")
        model.add_activity("C", duration=2, predecessors="A")
        model.add_activity("D", duration=3, predecessors="B, C")

        candidate = model.create_candidate()
        session = GuiSession()
        session.candidate = candidate
        session.has_applied_reviews = True
        session.workflow = None

        data = extract(session)
        assert data.ready is True
        a_ids = {a.activity_id for a in data.activities}
        assert a_ids == {"A", "B", "C", "D"}
        # All activities must have CPM values
        for act in data.activities:
            assert act.early_start is not None
            assert act.early_finish is not None
            assert act.late_start is not None
            assert act.late_finish is not None
            assert act.total_float is not None
            assert act.free_float is not None

    def test_validation_status_valid_without_workflow(self) -> None:
        from pert_analyzer.gui.session import GuiSession, AppState, ValidationCenterStatus

        model = ManualNetworkModel()
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2, predecessors="A")

        candidate = model.create_candidate()
        session = GuiSession()
        session.candidate = candidate
        session.has_applied_reviews = True
        session.workflow = None

        assert session.validation_status == ValidationCenterStatus.VALID


class TestResultsWidgetsPopulatedFromAnalyzeButton:
    """
    Real-GUI regression: clicking the ACTUAL Analyze button in the
    Network Builder must populate the visible Results widgets
    (KPI cards, activities table, network scene, critical paths list),
    routed through the public analyze_requested signal chain.

    Guards against any future fix that keeps backend data correct but
    leaves the visible widgets empty (the reported real-GUI bug).
    """

    def _populate(self, window) -> None:
        page = window._network_builder_page
        model = page.get_model()
        for aid, dur, pred in [
            ("A", 1, ""),
            ("B", 2, "A"),
            ("C", 2, "A"),
            ("D", 3, "B, C"),
        ]:
            err = model.add_activity(aid, duration=dur, predecessors=pred)
            assert err is None
        model.validate()
        assert model.is_valid
        assert page._analyze_btn.isEnabled()

    def test_analyze_button_populates_results_widgets(
        self, qapp, monkeypatch
    ) -> None:
        from PySide6.QtWidgets import QMessageBox
        from pert_analyzer.gui.main_window import MainWindow
        from pert_analyzer.gui.navigation import NavDestination
        from pert_analyzer.gui.session import AppState, ValidationCenterStatus

        monkeypatch.setattr(
            QMessageBox, "question",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Yes),
        )
        monkeypatch.setattr(
            QMessageBox, "warning",
            staticmethod(lambda *a, **k: QMessageBox.StandardButton.Ok),
        )

        window = MainWindow()
        results_page = window._results_page

        window._on_nav(NavDestination.NETWORK_BUILDER)
        self._populate(window)

        # Click the real button → emits analyze_requested → _on_manual_analyze
        window._network_builder_page._analyze_btn.click()
        qapp.processEvents()

        # Session routed through the real chain
        session = window._session
        assert session.state == AppState.RESULTS_AVAILABLE
        assert session.validation_status == ValidationCenterStatus.VALID
        assert session.current_candidate is not None

        # Main window navigated and same ResultsPage instance is live
        assert window._stack.currentIndex() == NavDestination.RESULTS.value
        assert window._stack.currentWidget() is results_page
        assert results_page._stack.currentIndex() == results_page._DASHBOARD

        # Widget state (not just backend data)
        data = results_page.data
        assert data.ready is True
        assert data.project_duration == 6.0
        assert len(data.activities) == 4
        assert len(data.dependencies) == 4
        assert data.critical_path_count == 2
        assert data.critical_paths[0] == ["A", "B", "D"]

        kpi = {k: c.value() for k, c in results_page.overview()._kpi_cards.items()}
        assert kpi["duration"] == "6 days"
        assert kpi["activities"] == "4"
        assert kpi["dependencies"] == "4"
        assert kpi["critical_paths"] == "2"

        table = results_page.activities()._table
        assert table.rowCount() == 4
        col = {table.horizontalHeaderItem(i).text(): i for i in range(table.columnCount())}
        ids = {table.item(r, col["ID"]).text() for r in range(table.rowCount())}
        assert ids == {"A", "B", "C", "D"}

        net = results_page.network()
        assert len(net._node_items) == 4
        assert len(net._edge_items) == 4

        assert results_page.critical_paths()._path_list._list.count() == 2
        assert results_page.overview().path_list()._list.count() == 2
        assert results_page.overview().embedded_activities()._table.rowCount() == 4


class TestBuilderStateBadgeAndGraph:
    """
    UI-state refinement regression:
    - State badge derives from model (DRAFT/EDITING/VALID/CPM READY/
      CPM OUTDATED/ERROR)
    - Validation message never shows "0 issues: Unknown issue."
    - Real network canvas renders nodes + edges with clickable selection
    - Relationship dropdowns list only existing activities
    """

    def _make_page(self, qapp):
        from pert_analyzer.gui.pages.network_builder_page import NetworkBuilderPage

        return NetworkBuilderPage()

    def test_badge_tracks_network_states(self, qapp) -> None:
        page = self._make_page(qapp)
        assert page._state_badge.text() == "DRAFT"
        model = page.get_model()

        # A single isolated activity is an invalid AON network
        model.add_activity("Z", duration=1)
        page._on_validation_changed()
        assert page._state_badge.text() == "ERROR"
        model.remove_activity("Z")

        # A→B is a valid connected network
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2, predecessors="A")
        model.validate()
        page._on_validation_changed()
        assert page._state_badge.text() == "VALID"

        model.calculate_cpm()
        page._on_cpm_calculated()
        assert page._state_badge.text() == "CPM READY"

        # Editing after CPM marks the result outdated
        model.add_activity("C", duration=1, predecessors="A")
        page._on_validation_changed()
        assert page._state_badge.text() == "CPM OUTDATED"

        model.clear()
        page._on_validation_changed()
        assert page._state_badge.text() == "DRAFT"

    def test_validation_message_never_unknown_issue(self, qapp) -> None:
        page = self._make_page(qapp)
        model = page.get_model()
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2, predecessors="X")  # missing pred
        page._on_validation_changed()
        text = page._status_text.text()
        assert "Unknown issue" not in text
        assert "X" in text or "issue" in text

    def test_real_canvas_renders_nodes_and_edges(self, qapp) -> None:
        page = self._make_page(qapp)
        model = page.get_model()
        for aid, dur, pred in [
            ("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"), ("D", 3, "B, C"),
        ]:
            model.add_activity(aid, duration=dur, predecessors=pred)
        page._update_graph_display()

        assert len(page._graph_nodes) == 4
        assert len(page._graph_edges) == 4
        assert set(page._graph_nodes) == {"A", "B", "C", "D"}

        # Clicking a node updates the model selection (visual sync)
        page._graph_nodes["B"]._emit_clicked()
        assert model.selected_activity == "B"

        # Clicking an edge selects the relationship
        edge = next(e for e in page._graph_edges
                    if e._source == "A" and e._target == "B")
        edge._emit_clicked()
        assert model.selected_relationship == ("A", "B")

    def test_relationship_dropdowns_list_only_existing(self, qapp) -> None:
        page = self._make_page(qapp)
        model = page.get_model()
        model.add_activity("A", duration=1)
        model.add_activity("B", duration=2)
        self._values = [page._rel_from.itemText(i)
                        for i in range(page._rel_from.count())]
        assert self._values == ["\u2014", "A", "B"]

    def test_results_ready_and_network_modified_badges(self, qapp) -> None:
        page = self._make_page(qapp)
        model = page.get_model()
        for aid, dur, pred in [
            ("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"), ("D", 3, "B, C"),
        ]:
            model.add_activity(aid, duration=dur, predecessors=pred)
        model.validate()
        assert model.is_valid

        page.mark_results_ready()
        assert page._state_badge.text() == "RESULTS READY"

        model.add_activity("E", 1, "D")
        page._on_model_changed()
        assert page._state_badge.text() == "NETWORK MODIFIED"

    def test_cpm_stale_hides_inspector_metrics(self, qapp) -> None:
        page = self._make_page(qapp)
        model = page.get_model()
        model.add_activity("A", 1)
        model.add_activity("B", 2, "A")
        model.calculate_cpm()
        page._update_inspector_activity("A")
        text = page._inspector_content.text()
        assert "ES:" in text and "LF:" in text

        model.add_activity("C", 1, "B")  # invalidates CPM
        page._update_inspector_activity("A")
        text = page._inspector_content.text()
        assert "ES:" not in text and "LF:" not in text
        assert "outdated" in text.lower()

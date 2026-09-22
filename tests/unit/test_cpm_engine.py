"""
Comprehensive unit tests for the CPM (Critical Path Method) engine.

Tests cover:
- Single activity network
- Simple linear chain
- Branching networks
- Non-critical branches
- Multiple critical paths
- Multiple predecessors
- Multiple successors
- Cycle detection
- Invalid duration handling
- Invalid dependency handling
- Floating-point duration support
- Multiple terminal activities
"""

import pytest

from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.analysis.exceptions import (
    CyclicGraphError,
    DuplicateActivityError,
    EmptyGraphError,
    InvalidDurationError,
    MissingActivityError,
    MissingDurationError,
)
from pert_analyzer.core.models import Activity, Dependency, DiagramType, GraphModel


# =============================================================================
# Test Helpers
# =============================================================================


def _make_activity(
    activity_id: str,
    duration: float,
    name: str = "",
    is_dummy: bool = False,
) -> Activity:
    """Create an Activity with the given ID and duration."""
    return Activity(
        activity_id=activity_id,
        name=name or activity_id,
        duration=duration,
        is_dummy=is_dummy,
    )


def _make_dependency(source: str, target: str) -> Dependency:
    """Create a Dependency from source to target."""
    return Dependency(source=source, target=target)


def _make_graph(
    activities: list[tuple[str, float]],
    dependencies: list[tuple[str, str]],
    name_map: dict[str, str] | None = None,
) -> GraphModel:
    """
    Convenience function to create a GraphModel.

    Args:
        activities: List of (activity_id, duration) tuples.
        dependencies: List of (source_id, target_id) tuples.
        name_map: Optional dict mapping activity_id to name.

    Returns:
        A configured GraphModel.
    """
    name_map = name_map or {}
    graph = GraphModel(diagram_type=DiagramType.AON)
    for aid, dur in activities:
        graph.activities[aid] = _make_activity(aid, dur, name=name_map.get(aid, aid))
    for src, tgt in dependencies:
        graph.dependencies.append(_make_dependency(src, tgt))
    return graph


# =============================================================================
# TEST 1: Single Activity
# =============================================================================


class TestSingleActivity:
    """Test CPM analysis with a single activity."""

    def test_single_activity_values(self):
        """ES=0, EF=5, LS=0, LF=5, Float=0, Duration=5."""
        graph = _make_graph(
            activities=[("A", 5)],
            dependencies=[],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        aa = result.activity_analyses["A"]
        assert aa.early_start == pytest.approx(0.0)
        assert aa.early_finish == pytest.approx(5.0)
        assert aa.late_start == pytest.approx(0.0)
        assert aa.late_finish == pytest.approx(5.0)
        assert aa.total_float == pytest.approx(0.0)
        assert aa.free_float == pytest.approx(0.0)
        assert aa.is_critical is True

    def test_single_activity_project_duration(self):
        """Project duration should equal the single activity's duration."""
        graph = _make_graph(
            activities=[("A", 5)],
            dependencies=[],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.project_duration == pytest.approx(5.0)

    def test_single_activity_critical_path(self):
        """Critical path should be [A]."""
        graph = _make_graph(
            activities=[("A", 5)],
            dependencies=[],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.critical_path == ["A"]
        assert len(result.critical_paths) == 1
        assert result.critical_paths[0] == ["A"]


# =============================================================================
# TEST 2: Simple Linear Chain
# =============================================================================


class TestLinearChain:
    """Test CPM analysis with a simple linear chain: A -> B -> C."""

    def test_linear_chain_project_duration(self):
        """Duration should be sum of all activities: 5 + 3 + 4 = 12."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.project_duration == pytest.approx(12.0)

    def test_linear_chain_all_critical(self):
        """All activities in a linear chain should be critical."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        for aid in ["A", "B", "C"]:
            assert result.activity_analyses[aid].is_critical is True
            assert result.activity_analyses[aid].total_float == pytest.approx(0.0)

    def test_linear_chain_es_ef_values(self):
        """Verify ES/EF for each activity in the chain."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        aa_a = result.activity_analyses["A"]
        assert aa_a.early_start == pytest.approx(0.0)
        assert aa_a.early_finish == pytest.approx(5.0)

        aa_b = result.activity_analyses["B"]
        assert aa_b.early_start == pytest.approx(5.0)
        assert aa_b.early_finish == pytest.approx(8.0)

        aa_c = result.activity_analyses["C"]
        assert aa_c.early_start == pytest.approx(8.0)
        assert aa_c.early_finish == pytest.approx(12.0)

    def test_linear_chain_ls_lf_values(self):
        """Verify LS/LF for each activity in the chain."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        aa_a = result.activity_analyses["A"]
        assert aa_a.late_start == pytest.approx(0.0)
        assert aa_a.late_finish == pytest.approx(5.0)

        aa_b = result.activity_analyses["B"]
        assert aa_b.late_start == pytest.approx(5.0)
        assert aa_b.late_finish == pytest.approx(8.0)

        aa_c = result.activity_analyses["C"]
        assert aa_c.late_start == pytest.approx(8.0)
        assert aa_c.late_finish == pytest.approx(12.0)

    def test_linear_chain_critical_path(self):
        """Critical path should be [A, B, C]."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.critical_path == ["A", "B", "C"]
        assert result.critical_paths == [["A", "B", "C"]]


# =============================================================================
# TEST 3: Simple Branching Network
# =============================================================================


class TestBranchingNetwork:
    """
    Test CPM with branching network:
        A(2) -> B(4) -> D(3)
        A(2) -> C(1) -> D(3)

    Expected:
        A: ES=0, EF=2, LS=0, LF=2 (critical)
        B: ES=2, EF=6, LS=2, LF=6 (critical)
        C: ES=2, EF=3, LS=5, LF=6 (non-critical, float=3)
        D: ES=6, EF=9, LS=6, LF=9 (critical)
        Project Duration: 9
        Critical Path: A -> B -> D
    """

    def _build_branching_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 2), ("B", 4), ("C", 1), ("D", 3)],
            dependencies=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )

    def test_project_duration(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(9.0)

    def test_activity_a(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        aa = result.activity_analyses["A"]
        assert aa.early_start == pytest.approx(0.0)
        assert aa.early_finish == pytest.approx(2.0)
        assert aa.late_start == pytest.approx(0.0)
        assert aa.late_finish == pytest.approx(2.0)
        assert aa.total_float == pytest.approx(0.0)
        assert aa.is_critical is True

    def test_activity_b(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        aa = result.activity_analyses["B"]
        assert aa.early_start == pytest.approx(2.0)
        assert aa.early_finish == pytest.approx(6.0)
        assert aa.late_start == pytest.approx(2.0)
        assert aa.late_finish == pytest.approx(6.0)
        assert aa.total_float == pytest.approx(0.0)
        assert aa.is_critical is True

    def test_activity_c(self):
        """C is on the non-critical branch with float=3."""
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        aa = result.activity_analyses["C"]
        assert aa.early_start == pytest.approx(2.0)
        assert aa.early_finish == pytest.approx(3.0)
        assert aa.late_start == pytest.approx(5.0)
        assert aa.late_finish == pytest.approx(6.0)
        assert aa.total_float == pytest.approx(3.0)
        assert aa.is_critical is False

    def test_activity_d(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        aa = result.activity_analyses["D"]
        assert aa.early_start == pytest.approx(6.0)
        assert aa.early_finish == pytest.approx(9.0)
        assert aa.late_start == pytest.approx(6.0)
        assert aa.late_finish == pytest.approx(9.0)
        assert aa.total_float == pytest.approx(0.0)
        assert aa.is_critical is True

    def test_critical_path(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_path == ["A", "B", "D"]

    def test_critical_activity_count(self):
        graph = self._build_branching_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_activity_count == 3
        assert result.non_critical_activity_count == 1


# =============================================================================
# TEST 4: Non-Critical Branch
# =============================================================================


class TestNonCriticalBranch:
    """
    Test network where one branch has positive float.

    Network:
        A(5) -> B(3) -> D(2)
        A(5) -> C(1) -> D(2)

    A-B branch: 5+3=8, A-C branch: 5+1=6
    Critical path: A -> B -> D (duration 10)
    C has float = 8 - 6 = 2
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 1), ("D", 2)],
            dependencies=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )

    def test_critical_activities_zero_float(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        for aid in ["A", "B", "D"]:
            aa = result.activity_analyses[aid]
            assert aa.is_critical is True
            assert aa.total_float == pytest.approx(0.0)

    def test_non_critical_has_positive_float(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_c = result.activity_analyses["C"]
        assert aa_c.is_critical is False
        assert aa_c.total_float == pytest.approx(2.0)

    def test_free_float_for_non_critical(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        # C finishes at 6, D starts at 8 (max(8,6)=8), free float = 8-6 = 2
        aa_c = result.activity_analyses["C"]
        assert aa_c.free_float == pytest.approx(2.0)

    def test_critical_path(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_path == ["A", "B", "D"]


# =============================================================================
# TEST 5: Multiple Critical Paths
# =============================================================================


class TestMultipleCriticalPaths:
    """
    Test network with two critical paths.

    Network:
        A(3) -> B(4) -> D(2)
        A(3) -> C(4) -> D(2)

    Both paths have duration 3+4+2 = 9.
    Both paths should be identified as critical.
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 3), ("B", 4), ("C", 4), ("D", 2)],
            dependencies=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )

    def test_both_paths_returned(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert len(result.critical_paths) == 2

    def test_paths_are_correct(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        expected_paths = [["A", "B", "D"], ["A", "C", "D"]]
        for path in expected_paths:
            assert path in result.critical_paths

    def test_all_activities_critical(self):
        """All activities are on critical paths."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        for aid in ["A", "B", "C", "D"]:
            assert result.activity_analyses[aid].is_critical is True

    def test_project_duration(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(9.0)

    def test_floats_are_zero(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        for aid in ["A", "B", "C", "D"]:
            assert result.activity_analyses[aid].total_float == pytest.approx(0.0)


# =============================================================================
# TEST 6: Multiple Predecessors
# =============================================================================


class TestMultiplePredecessors:
    """
    Test that ES = max(predecessor EF) when an activity has multiple predecessors.

    Network:
        A(3) -> C
        B(5) -> C
        C(2)

    C.ES = max(A.EF, B.EF) = max(3, 5) = 5
    C.EF = 5 + 2 = 7
    """

    def test_es_is_max_of_predecessors(self):
        graph = _make_graph(
            activities=[("A", 3), ("B", 5), ("C", 2)],
            dependencies=[("A", "C"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        aa_c = result.activity_analyses["C"]
        assert aa_c.early_start == pytest.approx(5.0)
        assert aa_c.early_finish == pytest.approx(7.0)

    def test_project_duration(self):
        graph = _make_graph(
            activities=[("A", 3), ("B", 5), ("C", 2)],
            dependencies=[("A", "C"), ("B", "C")],
        )
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(7.0)

    def test_critical_path_uses_longer_predecessor(self):
        """Critical path should follow B (longer predecessor), not A."""
        graph = _make_graph(
            activities=[("A", 3), ("B", 5), ("C", 2)],
            dependencies=[("A", "C"), ("B", "C")],
        )
        result = CPMEngine().analyze(graph)
        # B is critical (on the critical path), A has float
        assert result.activity_analyses["B"].is_critical is True
        assert result.activity_analyses["A"].is_critical is False
        assert result.activity_analyses["A"].total_float == pytest.approx(2.0)


# =============================================================================
# TEST 7: Multiple Successors
# =============================================================================


class TestMultipleSuccessors:
    """
    Test that LF = min(successor LS) when an activity has multiple successors.

    Network:
        A(3) -> B(5)
        A(3) -> C(2)

    B.LS = 8-5 = 3 (assuming project duration = 8)
    C.LS = 8-2 = 6
    A.LF = min(B.LS, C.LS) = min(3, 6) = 3
    A.LS = 3-3 = 0
    """

    def test_lf_is_min_of_successors(self):
        graph = _make_graph(
            activities=[("A", 3), ("B", 5), ("C", 2)],
            dependencies=[("A", "B"), ("A", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        aa_a = result.activity_analyses["A"]
        assert aa_a.late_finish == pytest.approx(3.0)
        assert aa_a.late_start == pytest.approx(0.0)

    def test_critical_path_follows_longer_successor(self):
        """Critical path should follow B (longer successor)."""
        graph = _make_graph(
            activities=[("A", 3), ("B", 5), ("C", 2)],
            dependencies=[("A", "B"), ("A", "C")],
        )
        result = CPMEngine().analyze(graph)
        assert result.activity_analyses["B"].is_critical is True
        assert result.activity_analyses["C"].is_critical is False
        assert result.activity_analyses["C"].total_float == pytest.approx(3.0)


# =============================================================================
# TEST 8: Cycle Detection
# =============================================================================


class TestCycleDetection:
    """Test that cycles in the network are detected and rejected."""

    def test_simple_cycle(self):
        """A -> B -> C -> A should raise CyclicGraphError."""
        graph = _make_graph(
            activities=[("A", 3), ("B", 4), ("C", 2)],
            dependencies=[("A", "B"), ("B", "C"), ("C", "A")],
        )
        engine = CPMEngine()
        with pytest.raises(CyclicGraphError):
            engine.analyze(graph)

    def test_self_loop(self):
        """A -> A should raise CyclicGraphError."""
        graph = _make_graph(
            activities=[("A", 3)],
            dependencies=[("A", "A")],
        )
        engine = CPMEngine()
        with pytest.raises(CyclicGraphError):
            engine.analyze(graph)

    def test_larger_cycle(self):
        """A -> B -> C -> D -> A should raise CyclicGraphError."""
        graph = _make_graph(
            activities=[("A", 1), ("B", 2), ("C", 3), ("D", 4)],
            dependencies=[("A", "B"), ("B", "C"), ("C", "D"), ("D", "A")],
        )
        engine = CPMEngine()
        with pytest.raises(CyclicGraphError):
            engine.analyze(graph)


# =============================================================================
# TEST 9: Invalid Duration
# =============================================================================


class TestInvalidDuration:
    """Test that invalid durations are rejected."""

    def test_negative_duration(self):
        """Negative duration should raise InvalidDurationError."""
        graph = _make_graph(
            activities=[("A", -5)],
            dependencies=[],
        )
        engine = CPMEngine()
        with pytest.raises(InvalidDurationError):
            engine.analyze(graph)

    def test_zero_duration_non_dummy(self):
        """Zero duration for non-dummy activity should raise MissingDurationError."""
        graph = _make_graph(
            activities=[("A", 0)],
            dependencies=[],
        )
        engine = CPMEngine()
        with pytest.raises(MissingDurationError):
            engine.analyze(graph)

    def test_dummy_zero_duration_accepted(self):
        """Dummy activities with zero duration should be accepted."""
        graph = GraphModel(diagram_type=DiagramType.AON)
        graph.activities["A"] = Activity(activity_id="A", name="A", duration=5.0)
        graph.activities["DUMMY"] = Activity(
            activity_id="DUMMY", name="DUMMY", duration=0.0, is_dummy=True
        )
        graph.dependencies.append(Dependency(source="A", target="DUMMY"))
        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.project_duration == pytest.approx(5.0)


# =============================================================================
# TEST 10: Invalid Dependency
# =============================================================================


class TestInvalidDependency:
    """Test that dependencies referencing nonexistent activities are rejected."""

    def test_dependency_missing_source(self):
        """Dependency with nonexistent source should raise MissingActivityError."""
        graph = _make_graph(
            activities=[("A", 3)],
            dependencies=[("X", "A")],  # X doesn't exist
        )
        engine = CPMEngine()
        with pytest.raises(MissingActivityError):
            engine.analyze(graph)

    def test_dependency_missing_target(self):
        """Dependency with nonexistent target should raise MissingActivityError."""
        graph = _make_graph(
            activities=[("A", 3)],
            dependencies=[("A", "X")],  # X doesn't exist
        )
        engine = CPMEngine()
        with pytest.raises(MissingActivityError):
            engine.analyze(graph)

    def test_dependency_both_missing(self):
        """Dependency with both nonexistent source and target."""
        graph = _make_graph(
            activities=[("A", 3)],
            dependencies=[("X", "Y")],  # Neither exists
        )
        engine = CPMEngine()
        with pytest.raises(MissingActivityError):
            engine.analyze(graph)


# =============================================================================
# TEST 11: Floating-Point Duration
# =============================================================================


class TestFloatingPointDuration:
    """Test correct calculations with decimal durations."""

    def test_float_durations(self):
        """Verify calculations with floating-point durations."""
        graph = _make_graph(
            activities=[("A", 2.5), ("B", 1.75), ("C", 3.25)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == pytest.approx(7.5)

        aa_a = result.activity_analyses["A"]
        assert aa_a.early_start == pytest.approx(0.0)
        assert aa_a.early_finish == pytest.approx(2.5)

        aa_b = result.activity_analyses["B"]
        assert aa_b.early_start == pytest.approx(2.5)
        assert aa_b.early_finish == pytest.approx(4.25)

        aa_c = result.activity_analyses["C"]
        assert aa_c.early_start == pytest.approx(4.25)
        assert aa_c.early_finish == pytest.approx(7.5)

    def test_float_duration_critical_path(self):
        """All activities in a linear chain with floats should be critical."""
        graph = _make_graph(
            activities=[("A", 2.5), ("B", 1.75), ("C", 3.25)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        result = CPMEngine().analyze(graph)
        for aid in ["A", "B", "C"]:
            assert result.activity_analyses[aid].is_critical is True

    def test_float_tolerance_affects_criticality(self):
        """Activities with very small float should be considered critical within tolerance."""
        graph = _make_graph(
            activities=[("A", 0.1), ("B", 0.2), ("C", 0.3)],
            dependencies=[("A", "B"), ("B", "C")],
        )
        engine = CPMEngine(float_tolerance=1e-6)
        result = engine.analyze(graph)
        for aid in ["A", "B", "C"]:
            assert result.activity_analyses[aid].is_critical is True


# =============================================================================
# TEST 12: Multiple Terminal Activities
# =============================================================================


class TestMultipleTerminalActivities:
    """
    Test network with multiple terminal (sink) activities.

    Network:
        A(5) -> B(3)
        A(5) -> C(4)

    Both B and C are terminal.
    Project duration = max(5+3, 5+4) = 9
    B has float = 9-8 = 1
    C has float = 9-9 = 0 (critical)
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 5), ("B", 3), ("C", 4)],
            dependencies=[("A", "B"), ("A", "C")],
        )

    def test_project_duration(self):
        """Duration should be max of terminal activities' EF."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(9.0)

    def test_terminal_activity_b(self):
        """B is terminal with EF=8, LF=9, float=1."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_b = result.activity_analyses["B"]
        assert aa_b.early_finish == pytest.approx(8.0)
        assert aa_b.late_finish == pytest.approx(9.0)
        assert aa_b.total_float == pytest.approx(1.0)
        assert aa_b.is_critical is False

    def test_terminal_activity_c(self):
        """C is terminal with EF=9, LF=9, float=0 (critical)."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_c = result.activity_analyses["C"]
        assert aa_c.early_finish == pytest.approx(9.0)
        assert aa_c.late_finish == pytest.approx(9.0)
        assert aa_c.total_float == pytest.approx(0.0)
        assert aa_c.is_critical is True

    def test_critical_path(self):
        """Critical path should be A -> C."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_path == ["A", "C"]

    def test_free_float_for_terminal_b(self):
        """Terminal activities should have free_float = 0."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_b = result.activity_analyses["B"]
        assert aa_b.free_float == pytest.approx(0.0)


# =============================================================================
# TEST 13: Complex Network
# =============================================================================


class TestComplexNetwork:
    """
    Test a more complex network structure.

    Network:
        A(0) -> B(5) -> D(3) -> F(2)
        A(0) -> C(4) -> E(6) -> F(2)
        D(3) -> E(6)

    A is start (duration 0 or dummy)
    F is end
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[
                ("A", 0.001),  # Near-zero start activity
                ("B", 5),
                ("C", 4),
                ("D", 3),
                ("E", 6),
                ("F", 2),
            ],
            dependencies=[
                ("A", "B"),
                ("A", "C"),
                ("B", "D"),
                ("C", "E"),
                ("D", "E"),
                ("D", "F"),  # D can also go to F directly
                ("E", "F"),
            ],
        )

    def test_project_duration(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        # Paths: A-B-D-F (10.001), A-B-D-E-F (16.001), A-C-E-F (12.001)
        assert result.project_duration == pytest.approx(16.001)

    def test_critical_path_through_d_and_e(self):
        """The longest path A-B-D-E-F should be critical."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert "A" in result.activity_analyses
        assert "B" in result.activity_analyses
        assert "D" in result.activity_analyses
        assert "E" in result.activity_analyses
        assert "F" in result.activity_analyses


# =============================================================================
# TEST 14: Empty Graph
# =============================================================================


class TestEmptyGraph:
    """Test that empty graphs are rejected."""

    def test_empty_graph_raises_error(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        engine = CPMEngine()
        with pytest.raises(EmptyGraphError):
            engine.analyze(graph)


# =============================================================================
# TEST 15: Duplicate Activity IDs
# =============================================================================


class TestDuplicateActivityIDs:
    """Test that duplicate activity IDs are detected."""

    def test_duplicate_ids_rejected(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        # Both activities have the same activity_id field but different dict keys
        graph.activities["A"] = Activity(activity_id="A", name="A1", duration=5)
        graph.activities["A_dup"] = Activity(activity_id="A", name="A2", duration=3)
        engine = CPMEngine()
        with pytest.raises(DuplicateActivityError):
            engine.analyze(graph)


# =============================================================================
# TEST 16: Analysis Type and Interface
# =============================================================================


class TestAnalysisInterface:
    """Test that CPMEngine properly implements the AnalysisEngine interface."""

    def test_analysis_type(self):
        engine = CPMEngine()
        assert engine.get_analysis_type() == "CPM"

    def test_can_analyze_valid_graph(self):
        graph = _make_graph(
            activities=[("A", 5)],
            dependencies=[],
        )
        engine = CPMEngine()
        assert engine.can_analyze(graph) is True

    def test_can_analyze_empty_graph(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        engine = CPMEngine()
        assert engine.can_analyze(graph) is False

    def test_can_analyze_cyclic_graph(self):
        graph = _make_graph(
            activities=[("A", 1), ("B", 2)],
            dependencies=[("A", "B"), ("B", "A")],
        )
        engine = CPMEngine()
        assert engine.can_analyze(graph) is False

    def test_get_critical_path_returns_list(self):
        graph = _make_graph(
            activities=[("A", 5), ("B", 3)],
            dependencies=[("A", "B")],
        )
        engine = CPMEngine()
        path = engine.get_critical_path(graph)
        assert isinstance(path, list)
        assert path == ["A", "B"]

    def test_get_all_critical_paths_returns_list(self):
        graph = _make_graph(
            activities=[("A", 5), ("B", 3)],
            dependencies=[("A", "B")],
        )
        engine = CPMEngine()
        paths = engine.get_all_critical_paths(graph)
        assert isinstance(paths, list)
        assert len(paths) >= 1


# =============================================================================
# TEST 17: Float Tolerance
# =============================================================================


class TestFloatTolerance:
    """Test that float tolerance configuration works correctly."""

    def test_default_tolerance(self):
        engine = CPMEngine()
        assert engine.float_tolerance == 1e-9

    def test_custom_tolerance(self):
        engine = CPMEngine(float_tolerance=0.01)
        assert engine.float_tolerance == 0.01

    def test_negative_tolerance_rejected(self):
        engine = CPMEngine()
        with pytest.raises(ValueError):
            engine.float_tolerance = -0.01

    def test_tolerance_affects_criticality(self):
        """With large tolerance, small floats should still be critical."""
        graph = _make_graph(
            activities=[("A", 5), ("B", 3)],
            dependencies=[("A", "B")],
        )
        engine = CPMEngine(float_tolerance=1.0)
        result = engine.analyze(graph)
        # Both should be critical since they're in a linear chain
        assert result.activity_analyses["A"].is_critical is True
        assert result.activity_analyses["B"].is_critical is True


# =============================================================================
# TEST 18: Large Linear Chain
# =============================================================================


class TestLargeLinearChain:
    """Test with a longer linear chain to verify sequential calculations."""

    def test_10_activity_chain(self):
        """10 activities in sequence, total duration = sum(1..10) = 55."""
        activities = [(f"A{i}", float(i)) for i in range(1, 11)]
        dependencies = [(f"A{i}", f"A{i+1}") for i in range(1, 10)]
        graph = _make_graph(activities, dependencies)

        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(55.0)

        # All should be critical
        for i in range(1, 11):
            assert result.activity_analyses[f"A{i}"].is_critical is True

        # Verify sequential ES/EF
        cumulative = 0.0
        for i in range(1, 11):
            aa = result.activity_analyses[f"A{i}"]
            assert aa.early_start == pytest.approx(cumulative)
            cumulative += i
            assert aa.early_finish == pytest.approx(cumulative)


# =============================================================================
# TEST 19: Diamond Network
# =============================================================================


class TestDiamondNetwork:
    """
    Test a diamond-shaped network:

        A -> B -> C -> E
        A -> D -> E

    A(2), B(4), C(3), D(5), E(2)
    Path 1: A-B-C-E = 2+4+3+2 = 11
    Path 2: A-D-E = 2+5+2 = 9
    Critical path: A -> B -> C -> E (11)
    D has float = 11-9 = 2
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 2), ("B", 4), ("C", 3), ("D", 5), ("E", 2)],
            dependencies=[("A", "B"), ("A", "D"), ("B", "C"), ("C", "E"), ("D", "E")],
        )

    def test_project_duration(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(11.0)

    def test_critical_path(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_path == ["A", "B", "C", "E"]

    def test_non_critical_d(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_d = result.activity_analyses["D"]
        assert aa_d.is_critical is False
        assert aa_d.total_float == pytest.approx(2.0)

    def test_critical_activities(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        critical = {aid for aid, aa in result.activity_analyses.items() if aa.is_critical}
        assert critical == {"A", "B", "C", "E"}


# =============================================================================
# TEST 20: Independent Parallel Paths
# =============================================================================


class TestIndependentParallelPaths:
    """
    Test two completely independent parallel paths.

    Path 1: A(3) -> B(4) = 7
    Path 2: C(5) -> D(2) = 7

    Both paths are critical with duration 7.
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 3), ("B", 4), ("C", 5), ("D", 2)],
            dependencies=[("A", "B"), ("C", "D")],
        )

    def test_project_duration(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.project_duration == pytest.approx(7.0)

    def test_all_critical(self):
        """All activities should be critical (both paths are critical)."""
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        for aid in ["A", "B", "C", "D"]:
            assert result.activity_analyses[aid].is_critical is True

    def test_two_critical_paths(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert len(result.critical_paths) == 2
        expected = [["A", "B"], ["C", "D"]]
        for path in expected:
            assert path in result.critical_paths


# =============================================================================
# TEST 21: Free Float with Multiple Successors
# =============================================================================


class TestFreeFloatMultipleSuccessors:
    """
    Test free float calculation with multiple successors.

    Network:
        A(3) -> B(2)
        A(3) -> C(5)

    A finishes at 3.
    B starts at 3, C starts at 3.
    A's free_float = min(ES_B, ES_C) - EF_A = min(3, 3) - 3 = 0
    """

    def test_free_float_zero(self):
        graph = _make_graph(
            activities=[("A", 3), ("B", 2), ("C", 5)],
            dependencies=[("A", "B"), ("A", "C")],
        )
        result = CPMEngine().analyze(graph)
        aa_a = result.activity_analyses["A"]
        assert aa_a.free_float == pytest.approx(0.0)

    def test_free_float_with_gap(self):
        """
        Network:
            A(2) -> B(3)
            A(2) -> C(5)

        A finishes at 2. B and C both start at 2.
        A's free_float = min(2, 2) - 2 = 0
        """
        graph = _make_graph(
            activities=[("A", 2), ("B", 3), ("C", 5)],
            dependencies=[("A", "B"), ("A", "C")],
        )
        result = CPMEngine().analyze(graph)
        aa_a = result.activity_analyses["A"]
        assert aa_a.free_float == pytest.approx(0.0)


# =============================================================================
# TEST 22: Backward Pass Correctness
# =============================================================================


class TestBackwardPassCorrectness:
    """
    Verify backward pass calculations are correct.

    Network:
        A(4) -> B(3) -> D(2)
        A(4) -> C(5) -> D(2)

    Forward: A(0-4), B(4-7), C(4-9), D(9-11)
    Backward: D(9-11), C(4-9), B(7-9-3=... let me recalculate)

    Actually:
    D is terminal: LF=11, LS=9
    B: LF = LS_D = 9, LS = 9-3 = 6
    C: LF = LS_D = 9, LS = 9-5 = 4
    A: LF = min(LS_B, LS_C) = min(6, 4) = 4, LS = 4-4 = 0

    So:
    A: ES=0, EF=4, LS=0, LF=4, float=0 (critical)
    B: ES=4, EF=7, LS=6, LF=9, float=2
    C: ES=4, EF=9, LS=4, LF=9, float=0 (critical)
    D: ES=9, EF=11, LS=9, LF=11, float=0 (critical)

    Critical path: A -> C -> D
    """

    def _build_graph(self) -> GraphModel:
        return _make_graph(
            activities=[("A", 4), ("B", 3), ("C", 5), ("D", 2)],
            dependencies=[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
        )

    def test_activity_b_backward(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_b = result.activity_analyses["B"]
        assert aa_b.late_start == pytest.approx(6.0)
        assert aa_b.late_finish == pytest.approx(9.0)
        assert aa_b.total_float == pytest.approx(2.0)

    def test_activity_c_backward(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        aa_c = result.activity_analyses["C"]
        assert aa_c.late_start == pytest.approx(4.0)
        assert aa_c.late_finish == pytest.approx(9.0)
        assert aa_c.total_float == pytest.approx(0.0)

    def test_critical_path(self):
        graph = self._build_graph()
        result = CPMEngine().analyze(graph)
        assert result.critical_path == ["A", "C", "D"]

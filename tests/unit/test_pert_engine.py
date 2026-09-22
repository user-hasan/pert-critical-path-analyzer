"""
Unit tests for the PERT engine (Program Evaluation and Review Technique).

Covers:
- Expected time TE = (O + 4M + P) / 6
- Variance var = ((P - O) / 6)^2 and standard deviation
- Validation: valid / ordering (O<=M<=P) / negative / missing
- Project scheduling under expected times (duration, variance, std dev)
- Critical paths determined under TE (not activity durations)
- Completion probability: Z-score, boundaries, determinism
- Analysis stability: repeatable results, graph never mutated
- Explicit manual-entry mapping is authoritative

The GraphModel and the backend Activity objects are the only fakes-free
dependencies; no GUI code is touched here.
"""

import math

import pytest

from pert_analyzer.analysis.pert_engine import (
    PertEngine,
    PertEstimate,
    PertEstimateError,
    PertStatus,
    completion_probability,
)
from pert_analyzer.core.models import Activity, Dependency, GraphModel


# =============================================================================
# Test helpers
# =============================================================================


def make_graph(
    activities: dict[str, tuple[float, float, float, float]],
    dependencies: list[tuple[str, str]],
) -> GraphModel:
    """Build a GraphModel; activity spec is id -> (duration, O, M, P).

    Values of ``None`` for O/M/P leave the estimate field unset, which lets
    tests exercise the graph-fallback (fixed durations only) path.
    """
    graph = GraphModel()
    for aid, (duration, o, m, p) in activities.items():
        graph.activities[aid] = Activity(
            activity_id=aid,
            name=aid,
            duration=duration,
            optimistic_time=o,
            most_likely_time=m,
            pessimistic_time=p,
        )
    for i, (source, target) in enumerate(dependencies):
        graph.dependencies.append(
            Dependency(source=source, target=target, dependency_id=f"d{i}")
        )
    graph.source_node_id = "A0"
    graph.sink_node_id = "A1"
    return graph


def linear_estimates() -> dict[str, PertEstimate]:
    """The deterministic A0(O=2,M=4,P=6) -> A1(O=4,M=6,P=8) fixture."""
    return {
        "A0": PertEstimate(optimistic=2.0, most_likely=4.0, pessimistic=6.0),
        "A1": PertEstimate(optimistic=4.0, most_likely=6.0, pessimistic=8.0),
    }


def linear_graph() -> GraphModel:
    return make_graph(
        {
            "A0": (5.0, 2.0, 4.0, 6.0),
            "A1": (5.0, 4.0, 6.0, 8.0),
        },
        [("A0", "A1")],
    )


# =============================================================================
# Formulas
# =============================================================================


def test_expected_time_formula() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    assert result.activity_results["A0"].expected_time == pytest.approx(4.0)
    assert result.activity_results["A1"].expected_time == pytest.approx(6.0)
    # TE = (O + 4M + P) / 6
    assert result.activity_results["A0"].expected_time == pytest.approx(
        (2.0 + 4 * 4.0 + 6.0) / 6.0
    )


def test_variance_formula() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    assert result.activity_results["A0"].variance == pytest.approx(16.0 / 36.0)
    # var = ((P - O) / 6)^2
    assert result.activity_results["A0"].variance == pytest.approx(
        ((6.0 - 2.0) / 6.0) ** 2
    )


def test_std_dev_formula() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    assert result.activity_results["A1"].std_dev == pytest.approx(
        math.sqrt(16.0 / 36.0)
    )


# =============================================================================
# Validation / status
# =============================================================================


def test_valid_estimates_are_ready() -> None:
    graph = linear_graph()
    validation = PertEngine().validate(graph, linear_estimates())
    assert validation.status == PertStatus.PERT_READY
    assert validation.is_valid
    assert validation.errors == []
    assert validation.warnings == []


def test_optimistic_above_most_likely_is_invalid() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=8.0, most_likely=4.0, pessimistic=9.0),
        "A1": linear_estimates()["A1"],
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_INVALID
    assert any(e.code == "pert_ordering" for e in validation.errors)


def test_most_likely_above_pessimistic_is_invalid() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=2.0, most_likely=9.0, pessimistic=4.0),
        "A1": linear_estimates()["A1"],
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_INVALID
    assert any(e.code == "pert_ordering" for e in validation.errors)


def test_optimistic_above_pessimistic_is_invalid() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=10.0, most_likely=4.0, pessimistic=2.0),
        "A1": linear_estimates()["A1"],
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_INVALID


def test_negative_estimate_is_invalid() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=-2.0, most_likely=4.0, pessimistic=6.0),
        "A1": linear_estimates()["A1"],
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_INVALID
    assert any(e.code == "pert_estimate_invalid" for e in validation.errors)


def test_non_numeric_estimate_is_invalid() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(
            optimistic="two", most_likely=4.0, pessimistic=6.0
        ),
        "A1": linear_estimates()["A1"],
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_INVALID
    assert any(e.code == "pert_estimate_not_numeric" for e in validation.errors)


def test_missing_estimates_require_review() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=2.0, most_likely=None, pessimistic=None),
        "A1": PertEstimate(optimistic=4.0, most_likely=6.0, pessimistic=8.0),
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_REVIEW_REQUIRED
    assert not validation.is_valid
    assert any(e.code == "missing_pert_estimate" for e in validation.warnings)


def test_partial_missing_estimate_requires_review() -> None:
    graph = linear_graph()
    estimates = {
        "A0": PertEstimate(optimistic=2.0, most_likely=4.0, pessimistic=8.0),
        "A1": PertEstimate(optimistic=4.0, most_likely=6.0, pessimistic=None),
    }
    validation = PertEngine().validate(graph, estimates)
    assert validation.status == PertStatus.PERT_REVIEW_REQUIRED


def test_no_estimate_data_is_no_pert_data() -> None:
    graph = make_graph({"A0": (5.0, None, None, None)}, [])
    engine = PertEngine()
    assert engine.status(graph, None) == PertStatus.NO_PERT_DATA
    assert PertEngine().status(None) == PertStatus.NO_PERT_DATA


# =============================================================================
# Project scheduling under expected times
# =============================================================================


def test_project_duration_is_sum_of_te_on_critical_path() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    assert result.project_duration == pytest.approx(10.0)  # 4.0 + 6.0
    assert result.critical_path == ["A0", "A1"]
    assert result.critical_activity_count == 2


def test_project_variance_sums_along_primary_critical_path() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    assert result.project_variance == pytest.approx(32.0 / 36.0)
    assert result.project_std_dev == pytest.approx(math.sqrt(32.0 / 36.0))


def test_critical_path_determined_under_expected_times() -> None:
    # CPM durations: Start+A(8)=10 beats Start+B(5)=7  -> A would be critical.
    # PERT TE:       Start+A(4)=6   loses  Start+B(5)=7 -> B is critical.
    graph = make_graph(
        {
            "Start": (2.0, 2.0, 2.0, 2.0),
            "A": (8.0, 2.0, 4.0, 6.0),      # TE 4.0
            "B": (5.0, 4.0, 5.0, 6.0),      # TE 5.0
            "Finish": (2.0, 0.0, 0.0, 0.0),  # TE 0.0
        },
        [("Start", "A"), ("Start", "B"), ("A", "Finish"), ("B", "Finish")],
    )
    result = PertEngine().analyze(graph, None)
    assert result.critical_path == ["Start", "B", "Finish"]
    assert result.activity_results["B"].is_critical
    assert not result.activity_results["A"].is_critical
    assert result.activity_results["A"].total_float == pytest.approx(1.0)


def test_multiple_critical_paths_are_all_reported() -> None:
    graph = make_graph(
        {
            "A": (1.0, 1.0, 1.0, 1.0),
            "B": (1.0, 1.0, 1.0, 1.0),
            "C": (1.0, 1.0, 1.0, 1.0),
        },
        [("A", "B"), ("A", "C")],  # fork: B and C both critical
    )
    result = PertEngine().analyze(graph, None)
    assert {"A", "B"} <= {x for p in result.critical_paths for x in p}
    assert {"A", "C"} <= {x for p in result.critical_paths for x in p}


def test_analyze_rejects_cyclic_graph() -> None:
    graph = make_graph(
        {"A": (1.0, 1.0, 1.0, 1.0), "B": (1.0, 1.0, 1.0, 1.0)},
        [("A", "B"), ("B", "A")],
    )
    with pytest.raises(PertEstimateError):
        PertEngine().analyze(graph, None)


# =============================================================================
# Completion probability
# =============================================================================


def test_probability_z_score() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, linear_estimates())
    probability = result.probability_for(12.0)
    assert probability.z_score == pytest.approx(
        (12.0 - 10.0) / result.project_std_dev
    )
    assert 0.0 < probability.probability < 1.0


def test_probability_boundaries() -> None:
    deterministic = make_graph({"A": (3.0, 3.0, 3.0, 3.0)}, [])
    result = PertEngine().analyze(deterministic, None)
    assert result.project_std_dev == pytest.approx(0.0)
    assert result.probability_for(3.0).probability == pytest.approx(1.0)
    assert result.probability_for(2.9).probability == pytest.approx(0.0)

    uncertain = linear_graph()
    result = PertEngine().analyze(uncertain, linear_estimates())
    assert result.probability_for(10.0).probability == pytest.approx(0.5)
    assert result.probability_for(10_000.0).probability > 0.999
    assert result.probability_for(-100.0).probability < 0.001


def test_completion_probability_function_direct() -> None:
    with pytest.raises(ValueError):
        completion_probability(10.0, 10.0, -1.0)


# =============================================================================
# Stability and isolation
# =============================================================================


def test_analyze_is_deterministic_and_repeatable() -> None:
    graph = linear_graph()
    engine = PertEngine()
    first = engine.analyze(graph, linear_estimates())
    second = engine.analyze(graph, linear_estimates())
    assert first.project_duration == second.project_duration
    assert first.project_variance == second.project_variance
    assert first.critical_paths == second.critical_paths
    assert first.activity_results["A0"] == second.activity_results["A0"]


def test_analyze_never_mutates_graph_or_graph_estimates() -> None:
    graph = linear_graph()
    original_duration = graph.activities["A0"].duration
    PertEngine().analyze(graph, linear_estimates())
    assert graph.activities["A0"].duration == original_duration
    assert graph.activities["A0"].optimistic_time == 2.0
    assert graph.activities["A0"].most_likely_time == 4.0
    assert graph.activities["A0"].pessimistic_time == 6.0


def test_explicit_mapping_is_authoritative_over_graph_fields() -> None:
    # Graph fields say TE=4/A0, but the explicit mapping must win.
    graph = linear_graph()
    override = {
        "A0": PertEstimate(optimistic=1.0, most_likely=2.0, pessimistic=3.0),
        "A1": PertEstimate(optimistic=4.0, most_likely=6.0, pessimistic=8.0),
    }
    result = PertEngine().analyze(graph, override)
    assert result.activity_results["A0"].expected_time == pytest.approx(2.0)
    assert result.activity_results["A0"].variance == pytest.approx(
        ((3.0 - 1.0) / 6.0) ** 2
    )
    # The authoritative mapping did not alter the stored graph fields.
    assert graph.activities["A0"].optimistic_time == 2.0


def test_graph_field_estimates_used_when_no_mapping() -> None:
    graph = linear_graph()
    result = PertEngine().analyze(graph, None)
    assert result.activity_results["A0"].expected_time == pytest.approx(4.0)
    assert result.activity_results["A0"].optimistic == 2.0
    assert result.activity_results["A0"].pessimistic == 6.0
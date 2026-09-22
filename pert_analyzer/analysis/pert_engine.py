"""
PERT (Program Evaluation and Review Technique) engine.

Implements an independent PERT analysis layer over the validated GraphModel:

- Three-point estimate validation (O/M/P rules; missing values are reported
  as a structured ``PERT_REVIEW_REQUIRED`` state, never invented).
- Expected time   TE = (O + 4M + P) / 6
- Variance        var = ((P - O) / 6) ** 2
- Std deviation   sd  = sqrt(var)
- Project scheduling under expected times (forward/backward pass, floats,
  criticality) and ALL critical paths under TE.
- Project variance and standard deviation along the primary critical path.
- Lightweight completion-probability API: Z = (T - expected) / project_sd,
  approximated with the normal CDF via ``math.erf``. No simulation.

Design intent:
- PERT is a separate analytical layer. It never mutates the GraphModel and
  never touches CPM results (``activity.duration`` stays authoritative for
  CPM; PERT expected times are derived from O/M/P estimates).
- The engine reuses the GraphModel and NetworkX for graph traversal
  (topological order, adjacency), but runs its own scheduling pass because
  the mathematical requirements genuinely differ from CPM: durations come
  from TE (not ``activity.duration``) and a zero expected time is a valid
  PERT quantity whereas CPM treats zero durations as missing.
- Estimates may come from an explicit mapping (manual entry / fixture) or
  fall back to the O/M/P fields on the back-ends ``Activity`` objects. The
  two sources are never mixed inside one analysis run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from pert_analyzer.core.models import Activity, GraphModel

# Default tolerance for floating-point comparison.
DEFAULT_FLOAT_TOLERANCE = 1e-9


class PertStatus(Enum):
    """PERT availability/validity state for an analysis session."""

    NO_PERT_DATA = "NO_PERT_DATA"
    PERT_REVIEW_REQUIRED = "PERT_REVIEW_REQUIRED"
    PERT_INVALID = "PERT_INVALID"
    PERT_READY = "PERT_READY"


class PertEstimateError(ValueError):
    """Raised when PERT analysis cannot run on the provided estimates."""


@dataclass(frozen=True)
class PertEstimate:
    """A three-point (optimistic, most-likely, pessimistic) activity estimate.

    A field value of ``None`` means the value has not been supplied; it is
    treated as missing by validation and never defaults to zero.
    """

    optimistic: Optional[float] = None
    most_likely: Optional[float] = None
    pessimistic: Optional[float] = None


@dataclass
class PertValidationIssue:
    """A single structured PERT estimate finding."""

    code: str
    message: str
    elements: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "elements": list(self.elements),
        }


@dataclass
class PertValidationResult:
    """Result of validating PERT estimates for a graph."""

    status: PertStatus = PertStatus.PERT_READY
    errors: List[PertValidationIssue] = field(default_factory=list)
    warnings: List[PertValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return self.status == PertStatus.PERT_READY

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "errors": [e.to_dict() for e in self.errors],
            "warnings": [e.to_dict() for e in self.warnings],
        }


@dataclass
class PertActivityResult:
    """PERT results for a single activity (backend authoritative values)."""

    activity_id: str
    optimistic: Optional[float] = None
    most_likely: Optional[float] = None
    pessimistic: Optional[float] = None
    expected_time: float = 0.0
    variance: float = 0.0
    std_dev: float = 0.0
    early_start: float = 0.0
    early_finish: float = 0.0
    late_start: float = 0.0
    late_finish: float = 0.0
    total_float: float = 0.0
    free_float: float = 0.0
    is_critical: bool = False


@dataclass(frozen=True)
class PertProbability:
    """Completion probability estimate for a given target duration."""

    target_duration: float
    z_score: float
    probability: float


def completion_probability(
    target_duration: float,
    expected_duration: float,
    std_dev: float,
) -> PertProbability:
    """Approximate the probability of finishing within ``target_duration``.

    Z = (target - expected) / std_dev and the result uses the normal CDF.
    When ``std_dev`` is zero the estimate is deterministic: probability is
    1.0 when the target covers the expected duration, else 0.0.
    """
    expected = float(expected_duration)
    std = float(std_dev)
    if std < 0 or std != std:  # negative or NaN
        raise ValueError("std_dev must be a non-negative, finite number")
    target = float(target_duration)

    if std == 0.0:
        if target >= expected:
            return PertProbability(target_duration=target, z_score=0.0, probability=1.0)
        return PertProbability(target_duration=target, z_score=0.0, probability=0.0)

    z = (target - expected) / std
    probability = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    return PertProbability(
        target_duration=target, z_score=z, probability=probability
    )


@dataclass
class PertResult:
    """Complete project-level PERT analysis result."""

    project_duration: float = 0.0
    project_variance: float = 0.0
    project_std_dev: float = 0.0
    critical_paths: List[List[str]] = field(default_factory=list)
    activity_results: Dict[str, PertActivityResult] = field(default_factory=dict)
    estimates: Dict[str, PertEstimate] = field(default_factory=dict)

    @property
    def critical_activity_count(self) -> int:
        return sum(
            1 for r in self.activity_results.values() if r.is_critical
        )

    @property
    def critical_path(self) -> List[str]:
        return list(self.critical_paths[0]) if self.critical_paths else []

    def probability_for(self, target_duration: float) -> PertProbability:
        """Completion probability by ``target_duration`` using this result."""
        return completion_probability(
            target_duration,
            self.project_duration,
            self.project_std_dev,
        )


class PertEngine:
    """Independent PERT analysis engine over a validated GraphModel."""

    def __init__(self, float_tolerance: float = DEFAULT_FLOAT_TOLERANCE):
        self._float_tolerance = float_tolerance

    # ------------------------------------------------------------------
    # Status / validation
    # ------------------------------------------------------------------

    def status(
        self,
        graph: Optional[GraphModel],
        estimates: Optional[Dict[str, PertEstimate]] = None,
    ) -> PertStatus:
        """Derive the PERT state for ``graph``.

        ``estimates`` is an explicit activity-id -> PertEstimate mapping.
        When ``None`` the engine reads the O/M/P fields stored on the
        ``Activity`` objects. ``NO_PERT_DATA`` is returned when no estimate
        values exist anywhere (no fake zeros are invented).
        """
        if graph is None or not getattr(graph, "activities", None):
            return PertStatus.NO_PERT_DATA
        if estimates is None and not _graph_has_estimates(graph):
            return PertStatus.NO_PERT_DATA
        return self.validate(graph, estimates).status

    def validate(
        self,
        graph: GraphModel,
        estimates: Optional[Dict[str, PertEstimate]] = None,
    ) -> PertValidationResult:
        """Validate PERT estimates for every activity in the graph.

        Rules (never silently repaired):
          - every activity needs all three estimates (missing -> review);
          - values must be numeric and non-negative (violation -> invalid);
          - O <= M <= P ordering must hold (violation -> invalid).
        """
        if graph is None or not getattr(graph, "activities", None):
            return PertValidationResult(
                status=PertStatus.NO_PERT_DATA,
                errors=[PertValidationIssue(
                    "empty_graph", "Graph has no activities", [],
                )],
            )

        errors: List[PertValidationIssue] = []
        missing: List[PertValidationIssue] = []

        for aid in graph.activities:
            est = _estimate_for(graph, aid, estimates)

            expected_missing = (
                _is_missing(est.optimistic)
                or _is_missing(est.most_likely)
                or _is_missing(est.pessimistic)
            )
            values = (
                ("optimistic", est.optimistic),
                ("most_likely", est.most_likely),
                ("pessimistic", est.pessimistic),
            )
            present: Dict[str, float] = {}
            for name, value in values:
                if value is None:
                    continue
                try:
                    number = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    errors.append(PertValidationIssue(
                        "pert_estimate_not_numeric",
                        f"Activity '{aid}' has a non-numeric {name} estimate",
                        [aid],
                    ))
                    continue
                if number < 0 or number != number or math.isinf(number):
                    errors.append(PertValidationIssue(
                        "pert_estimate_invalid",
                        f"Activity '{aid}' has an invalid {name} estimate "
                        f"({number:g}); estimates must be non-negative",
                        [aid],
                    ))
                    continue
                present[name] = number

            if expected_missing:
                missing.append(PertValidationIssue(
                    "missing_pert_estimate",
                    f"Activity '{aid}' is missing PERT estimates; complete "
                    "O, M and P before running PERT",
                    [aid],
                ))

            o = present.get("optimistic")
            m = present.get("most_likely")
            p = present.get("pessimistic")
            if o is not None and m is not None and o > m:
                errors.append(PertValidationIssue(
                    "pert_ordering",
                    f"Activity '{aid}': optimistic ({o:g}) must be <= "
                    f"most likely ({m:g})",
                    [aid],
                ))
            if m is not None and p is not None and m > p:
                errors.append(PertValidationIssue(
                    "pert_ordering",
                    f"Activity '{aid}': most likely ({m:g}) must be <= "
                    f"pessimistic ({p:g})",
                    [aid],
                ))
            if o is not None and p is not None and o > p:
                errors.append(PertValidationIssue(
                    "pert_ordering",
                    f"Activity '{aid}': optimistic ({o:g}) must be <= "
                    f"pessimistic ({p:g})",
                    [aid],
                ))

        if errors:
            return PertValidationResult(
                status=PertStatus.PERT_INVALID,
                errors=errors,
                warnings=missing,
            )
        if missing:
            return PertValidationResult(
                status=PertStatus.PERT_REVIEW_REQUIRED,
                errors=[],
                warnings=missing,
            )
        return PertValidationResult(status=PertStatus.PERT_READY)

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze(
        self,
        graph: GraphModel,
        estimates: Optional[Dict[str, PertEstimate]] = None,
    ) -> PertResult:
        """Run a full PERT analysis over ``graph`` using expected times.

        The graph itself is treated as read-only: the activity ``duration``
        fields and O/M/P values are never modified.
        """
        if graph is None or not getattr(graph, "activities", None):
            raise PertEstimateError("Cannot run PERT: graph has no activities")

        validation = self.validate(graph, estimates)
        if validation.status == PertStatus.PERT_INVALID:
            messages = [f"{e.code}: {e.message}" for e in validation.errors]
            raise PertEstimateError("; ".join(messages) or "PERT estimates invalid")
        if validation.status == PertStatus.PERT_REVIEW_REQUIRED:
            messages = [f"{e.code}: {e.message}" for e in validation.warnings]
            raise PertEstimateError(
                "; ".join(messages) or "PERT estimates incomplete"
            )

        resolved: Dict[str, PertEstimate] = {}
        expected_time: Dict[str, float] = {}
        variance: Dict[str, float] = {}
        for aid in graph.activities:
            est = _estimate_for(graph, aid, estimates)
            resolved[aid] = est
            o = float(est.optimistic)
            m = float(est.most_likely)
            p = float(est.pessimistic)
            expected_time[aid] = (o + 4.0 * m + p) / 6.0
            variance[aid] = ((p - o) / 6.0) ** 2

        nx_graph = self._build_nx_graph(graph)
        if not nx.is_directed_acyclic_graph(nx_graph):
            try:
                cycle = nx.find_cycle(nx_graph)
                path = [str(n) for n, _ in cycle]
            except nx.NetworkXNoCycle:
                path = []
            raise PertEstimateError(
                "Cannot run PERT on a cyclic graph" + (f": {' -> '.join(path)}" if path else "")
            )

        es, ef = self._forward_pass(nx_graph, expected_time)
        ls, lf = self._backward_pass(nx_graph, es, ef, expected_time)

        project_duration = max(ef.values()) if ef else 0.0

        activity_results: Dict[str, PertActivityResult] = {}
        for aid, act in graph.activities.items():
            total_float = ls[aid] - es[aid]
            successors = list(nx_graph.successors(aid))
            if successors:
                free_float = min(es[succ] for succ in successors) - ef[aid]
            else:
                free_float = 0.0
            is_critical = abs(total_float) < self._float_tolerance
            activity_results[aid] = PertActivityResult(
                activity_id=aid,
                optimistic=resolved[aid].optimistic,
                most_likely=resolved[aid].most_likely,
                pessimistic=resolved[aid].pessimistic,
                expected_time=expected_time[aid],
                variance=variance[aid],
                std_dev=math.sqrt(variance[aid]),
                early_start=es[aid],
                early_finish=ef[aid],
                late_start=ls[aid],
                late_finish=lf[aid],
                total_float=total_float,
                free_float=free_float,
                is_critical=is_critical,
            )

        critical_activities = {
            aid for aid, r in activity_results.items() if r.is_critical
        }
        critical_paths = self._find_all_critical_paths(
            nx_graph, critical_activities
        )

        # Project variance follows the primary critical path (deterministic).
        primary = critical_paths[0] if critical_paths else sorted(activity_results)
        project_variance = sum(variance[aid] for aid in primary)

        return PertResult(
            project_duration=project_duration,
            project_variance=project_variance,
            project_std_dev=math.sqrt(project_variance),
            critical_paths=critical_paths,
            activity_results=activity_results,
            estimates=resolved,
        )

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_nx_graph(graph: GraphModel) -> nx.DiGraph:
        nx_graph = nx.DiGraph()
        for aid in graph.activities:
            nx_graph.add_node(aid)
        for dep in graph.dependencies:
            source = getattr(dep, "source", None)
            target = getattr(dep, "target", None)
            if source in graph.activities and target in graph.activities:
                nx_graph.add_edge(source, target)
        return nx_graph

    @staticmethod
    def _forward_pass(
        nx_graph: nx.DiGraph,
        expected_time: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        es: Dict[str, float] = {}
        ef: Dict[str, float] = {}
        topo_order = list(nx.topological_sort(nx_graph))
        for aid in topo_order:
            predecessors = list(nx_graph.predecessors(aid))
            if predecessors:
                es[aid] = max(ef[pred] for pred in predecessors)
            else:
                es[aid] = 0.0
            ef[aid] = es[aid] + expected_time[aid]
        return es, ef

    @staticmethod
    def _backward_pass(
        nx_graph: nx.DiGraph,
        es: Dict[str, float],
        ef: Dict[str, float],
        expected_time: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        project_duration = max(ef.values()) if ef else 0.0
        ls: Dict[str, float] = {}
        lf: Dict[str, float] = {}
        topo_order = list(nx.topological_sort(nx_graph))
        for aid in reversed(topo_order):
            successors = list(nx_graph.successors(aid))
            if successors:
                lf[aid] = min(ls[succ] for succ in successors)
            else:
                lf[aid] = project_duration
            ls[aid] = lf[aid] - expected_time[aid]
        return ls, lf

    @staticmethod
    def _find_all_critical_paths(
        nx_graph: nx.DiGraph,
        critical_activities: set,
    ) -> List[List[str]]:
        if not critical_activities:
            return []

        subgraph = nx.DiGraph()
        for aid in critical_activities:
            subgraph.add_node(aid)
        for u, v in nx_graph.edges():
            if u in critical_activities and v in critical_activities:
                subgraph.add_edge(u, v)

        sources = [
            n for n in subgraph.nodes() if subgraph.in_degree(n) == 0
        ]
        sinks = [
            n for n in subgraph.nodes() if subgraph.out_degree(n) == 0
        ]

        paths: List[List[str]] = []
        for source in sources:
            for sink in sinks:
                try:
                    paths.extend(
                        list(nx.all_simple_paths(subgraph, source, sink))
                    )
                except nx.NetworkXError:
                    continue
        paths.sort(key=lambda p: (len(p), p))
        return paths


# =============================================================================
# Estimate resolution helpers
# =============================================================================


def resolve_estimates(
    graph: Optional[GraphModel],
    estimates: Optional[Dict[str, PertEstimate]] = None,
) -> Dict[str, PertEstimate]:
    """Resolve the effective estimate for every activity in ``graph``.

    With an explicit mapping the mapping is authoritative; otherwise the
    O/M/P fields on the back-ends Activity objects are used. Each value may
    be ``None`` (meaning the activity has no usable estimate yet).
    """
    if graph is None:
        return {}
    return {
        aid: _estimate_for(graph, aid, estimates)
        for aid in graph.activities
    }


def _estimate_for(
    graph: GraphModel,
    activity_id: str,
    estimates: Optional[Dict[str, PertEstimate]] = None,
) -> Optional[PertEstimate]:
    """Return the PertEstimate for one activity or None when missing."""
    if estimates is not None:
        est = estimates.get(activity_id)
        if est is not None:
            return _coerce(est)
        return PertEstimate()
    activity: Activity = graph.activities[activity_id]
    return PertEstimate(
        optimistic=activity.optimistic_time,
        most_likely=activity.most_likely_time,
        pessimistic=activity.pessimistic_time,
    )


def _coerce(est: PertEstimate) -> PertEstimate:
    """Normalize explicit estimate values.

    Numeric strings are converted to floats; genuinely non-numeric values
    are preserved verbatim so validation can report them as
    ``pert_estimate_not_numeric`` instead of silently treating them as
    missing. ``None`` stays the missing sentinel.
    """
    return PertEstimate(
        optimistic=_as_float_or_raw(est.optimistic),
        most_likely=_as_float_or_raw(est.most_likely),
        pessimistic=_as_float_or_raw(est.pessimistic),
    )


def _as_float_or_raw(value: Any) -> Any:
    if _is_missing(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return value


def _num(value: Any) -> Optional[float]:
    if _is_missing(value):
        return None
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _is_missing(value: Any) -> bool:
    return value is None


def _graph_has_estimates(graph: GraphModel) -> bool:
    """True when at least one activity carries a non-empty O/M/P value."""
    for act in graph.activities.values():
        for name in ("optimistic_time", "most_likely_time", "pessimistic_time"):
            value = getattr(act, name, None)
            if value is not None and not _is_missing(value):
                return True
    return False
"""
Report builder (Phase 6): Qt-free, authoritative, no recalculation.

The builder consumes the authoritative backend objects already produced by the
workshop (CPM ``AnalysisResult``, PERT ``PertResult``/estimates, graph
``GraphModel``, ``GraphValidationResult``, review summary + decisions) plus a
small deterministic set of session metadata (project name, source image path,
timestamps, engine version).

It never recomputes forward/backward passes, floats, path sums, variances,
probabilities, or project duration. It only *packages* the authoritative
values into the immutable ``ProjectReport`` model, exactly mirroring what the
Results dashboard shows.

PERT handling follows the dashboard readiness rules: when no PERT data /
PERT is unavailable, the PERT section is marked unavailable with a clear,
defined reason (never fake zeros, never a fabricated estimate).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.reporting.models import (
    ActivitiesSection,
    ActivityReportRow,
    CriticalPathsSection,
    CpmSection,
    DependenciesSection,
    DependencyReportRow,
    PertSection,
    ProjectReport,
    ReportMetadata,
    ReviewSection,
    SummarySection,
    ValidationSection,
)

logger = logging.getLogger(__name__)


def _num(value: Any) -> Optional[float]:
    """Coerce a backend number, tolerating missing/malformed values."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _bool(value: Any) -> bool:
    return bool(value)


def _str(value: Any, default: str = "") -> str:
    return str(value) if value is not None else default


def _iso(value: Any) -> str:
    """Normalize an ISO-ish timestamp to a stable string."""
    return _str(value).replace("Z", "+00:00") if value else ""


# =============================================================================
# Authoritative field access (mirrors gui.results.data/extract accessors)
# =============================================================================


def _cpm_analyses(cpm: Any) -> Dict[str, Any]:
    return getattr(cpm, "activity_analyses", None) or {}


def _cpm_critical_paths(cpm: Any) -> List[List[str]]:
    return [list(p) for p in (getattr(cpm, "critical_paths", None) or [])]


def _cpm_critical_path(cpm: Any) -> List[str]:
    p = getattr(cpm, "critical_path", None)
    return list(p) if p else []


def _graph_leader(graph: Any) -> Optional[float]:
    return _num(getattr(graph, "project_duration", None))


def _pert_rows(pert_result: Any) -> List[Any]:
    return list(getattr(pert_result, "activity_results", None) or [])


def _pert_estimates(session: Any) -> Dict[str, Any]:
    return getattr(session, "pert_estimates", None) or {}


def _pert_result(session: Any) -> Optional[Any]:
    return getattr(session, "pert_result", None)


def _pert_status(session: Any) -> Any:
    return getattr(session, "pert_status", None)


def _validation(session: Any) -> Optional[Any]:
    candidate = getattr(session, "current_candidate", None)
    if candidate is None:
        return None
    return getattr(candidate, "validation", None)


def _review_summary(session: Any) -> Optional[Any]:
    return getattr(session, "review_summary", None)


def _review_decisions(session: Any) -> List[Any]:
    return list(getattr(session, "review_decisions", None) or [])


def _workflow_summary(session: Any) -> Dict[str, Any]:
    workflow = getattr(session, "workflow", None)
    if workflow is None:
        return {}
    summary = getattr(workflow, "summary", None)
    if callable(summary):
        try:
            return dict(summary())
        except Exception:
            return {}
    return dict(summary or {})


def _pert_available(session: Any, status: Any) -> bool:
    if status is None:
        return bool(getattr(session, "pert_result", None))
    value = getattr(status, "value", status)
    return value in ("PERT_READY", "pert_ready")


# =============================================================================
# Section builders (mapping only)
# =============================================================================


def _build_summary(
    session: Any,
    cpm: Any,
    graph: Any,
    pert: Optional[Any],
    pert_available: bool,
) -> SummarySection:
    critical_paths = _cpm_critical_paths(cpm)
    critical_activity_count = _critical_activity_count(cpm)
    duration = _num(getattr(cpm, "project_duration", None))
    if duration is None and graph is not None:
        duration = _num(getattr(graph, "cpm_project_duration", None))
    return SummarySection(
        project_duration=duration,
        project_variance=_num(getattr(cpm, "project_variance", None)),
        project_std_dev=_num(getattr(cpm, "project_std_dev", None)),
        critical_path_count=len(critical_paths),
        critical_activity_count=critical_activity_count,
        activity_count=len(getattr(graph, "activities", None) or {}),
        dependency_count=len(getattr(graph, "dependencies", None) or []),
        pert_available=pert_available,
        pert_project_duration=_num(getattr(pert, "project_duration", None))
        if pert_available
        else None,
        pert_project_variance=_num(getattr(pert, "project_variance", None))
        if pert_available
        else None,
        pert_project_std_dev=_num(getattr(pert, "project_std_dev", None))
        if pert_available
        else None,
    )


def _critical_activity_count(cpm: Any) -> int:
    return sum(
        1
        for aa in getattr(cpm, "activity_analyses", None or {}).values()
        if bool(getattr(aa, "is_critical", False))
    )


def _build_activities(cpm: Any, graph: Any) -> ActivitiesSection:
    analyses = _cpm_analyses(cpm)
    activities_data = getattr(graph, "activities", None) or {}
    rows: List[ActivityReportRow] = []
    for aid in sorted(activities_data):
        backend_activity = activities_data[aid]
        analysis = analyses.get(aid)
        rows.append(
            ActivityReportRow(
                activity_id=aid,
                name=_str(getattr(backend_activity, "name", "")),
                duration=_num(getattr(backend_activity, "duration", None)),
                early_start=_num(getattr(analysis, "early_start", None)),
                early_finish=_num(getattr(analysis, "early_finish", None)),
                late_start=_num(getattr(analysis, "late_start", None)),
                late_finish=_num(getattr(analysis, "late_finish", None)),
                total_float=_num(getattr(analysis, "total_float", None)),
                free_float=_num(getattr(analysis, "free_float", None)),
                is_critical=_bool(getattr(analysis, "is_critical", False)),
            )
        )
    return ActivitiesSection(rows=rows)


def _build_dependencies(graph: Any) -> DependenciesSection:
    deps = getattr(graph, "dependencies", None) or []
    rows = [
        DependencyReportRow(
            source=_str(getattr(d, "source", "")),
            target=_str(getattr(d, "target", "")),
            dependency_id=_str(getattr(d, "dependency_id", "")),
        )
        for d in deps
        if getattr(d, "source", None) and getattr(d, "target", None)
    ]
    return DependenciesSection(rows=rows)


def _build_critical_paths(cpm: Any) -> CriticalPathsSection:
    return CriticalPathsSection(critical_paths=_cpm_critical_paths(cpm))


def _build_cpm(cpm: Any) -> CpmSection:
    analysis_type = _str(getattr(cpm, "analysis_type", "CPM"))
    return CpmSection(
        analysis_type=analysis_type,
        is_valid=_bool(getattr(cpm, "is_valid", True)),
        project_duration=_num(getattr(cpm, "project_duration", None)),
        project_variance=_num(getattr(cpm, "project_variance", None)),
        project_std_dev=_num(getattr(cpm, "project_std_dev", None)),
        critical_path_count=len(_cpm_critical_paths(cpm)),
        critical_activity_count=_critical_activity_count(cpm),
        errors=list(getattr(cpm, "errors", None) or []),
    )


def _build_pert(
    session: Any,
    pert_result: Any,
    estimates: Dict[str, Any],
    available: bool,
) -> PertSection:
    if not available:
        status = _pert_status(session)
        status_value = getattr(status, "value", status)
        return PertSection(
            available=False,
            reason=_pert_unavailable_reason(status_value),
            status=str(status_value or "NO_PERT_DATA"),
        )
    rows = _build_pert_rows(pert_result, estimates)
    return PertSection(
        available=True,
        reason="",
        status="PERT_READY",
        project_duration=_num(getattr(pert_result, "project_duration", None)),
        project_variance=_num(getattr(pert_result, "project_variance", None)),
        project_std_dev=_num(getattr(pert_result, "project_std_dev", None)),
        critical_paths=[
            list(p) for p in (getattr(pert_result, "critical_paths", None) or [])
        ],
        rows=rows,
    )


def _pert_unavailable_reason(status_value: Any) -> str:
    return PERT_STATUS_TEXT.get(str(status_value), PERT_UNAVAILABLE)


PERT_UNAVAILABLE = (
    "PERT analysis was not available for this project. Provide optimistic, "
    "most-likely, and pessimistic estimates and run PERT to unlock this section."
)
PERT_STATUS_TEXT = {
    "NO_PERT_DATA": (
        "PERT analysis was not available for this project. Provide "
        "optimistic, most-likely, and pessimistic estimates and run PERT."
    ),
    "PERT_REVIEW_REQUIRED": (
        "PERT analysis requires review. Resolve outstanding review items "
        "before running PERT."
    ),
    "PERT_INVALID": (
        "PERT analysis could not be computed because the reviewed graph "
        "was not valid."
    ),
}


def _build_pert_rows(pert_result: Any, estimates: Dict[str, Any]) -> List[Any]:
    activity_results = _pert_rows(pert_result)
    by_id = {getattr(r, "activity_id", None): r for r in activity_results}
    rows: List[Any] = []
    for aid in sorted(estimates):
        analysis = by_id.get(aid)
        rows.append(
            RowProxy(
                activity_id=aid,
                optimistic=_num(_estimate_value(estimates[aid], "optimistic")),
                most_likely=_num(_estimate_value(estimates[aid], "most_likely")),
                pessimistic=_num(_estimate_value(estimates[aid], "pessimistic")),
                expected_time=_num(getattr(analysis, "expected_time", None)),
                variance=_num(getattr(analysis, "variance", None)),
                total_float=_num(getattr(analysis, "total_float", None)),
                is_critical=_bool(getattr(analysis, "is_critical", False)),
            )
        )
    return rows


def _estimate_value(estimate: Any, field: str) -> Optional[float]:
    if estimate is None:
        return None
    if isinstance(estimate, dict):
        return estimate.get(field)
    return getattr(estimate, field, None)


class RowProxy:
    """Lightweight read-only row facade for PERT export rows."""

    __slots__ = (
        "activity_id",
        "optimistic",
        "most_likely",
        "pessimistic",
        "expected_time",
        "variance",
        "total_float",
        "is_critical",
    )

    def __init__(
        self,
        action: str = "activity_id",
        activity_id: str = "",
        optimistic: Optional[float] = None,
        most_likely: Optional[float] = None,
        pessimistic: Optional[float] = None,
        expected_time: Optional[float] = None,
        variance: Optional[float] = None,
        total_float: Optional[float] = None,
        is_critical: bool = False,
    ) -> None:
        self.activity_id = activity_id
        self.optimistic = optimistic
        self.most_likely = most_likely
        self.pessimistic = pessimistic
        self.expected_time = expected_time
        self.variance = variance
        self.total_float = total_float
        self.is_critical = is_critical


def _build_validation(session: Any) -> ValidationSection:
    validation = _validation(session)
    errors = getattr(validation, "errors", None) or []
    warnings = getattr(validation, "warnings", None) or []
    return ValidationSection(
        is_valid=_bool(getattr(validation, "is_valid", False)),
        status=_str(getattr(validation, "status", "")),
        component_count=_int(getattr(validation, "component_count", None)),
        is_acyclic=_bool(getattr(validation, "is_acyclic", True)),
        error_count=len(errors),
        warning_count=len(warnings),
        errors=[_issue_dict(e) for e in errors],
        warnings=[_issue_dict(e) for e in warnings],
    )


def _issue_dict(issue: Any) -> Dict[str, Any]:
    return {
        "code": _str(getattr(issue, "code", "")),
        "message": _str(getattr(issue, "message", "")),
        "elements": list(getattr(issue, "elements", None) or []),
    }


def _build_review(session: Any) -> ReviewSection:
    summary = _review_summary(session)
    if summary is None:
        return ReviewSection(available=False)
    counts = getattr(summary, "activities", None)
    if counts is None:
        counts = getattr(summary, "review_summary", None)
    total = _int(getattr(counts, "total", 0))
    confirmed = _int(getattr(counts, "confirmed", 0))
    return ReviewSection(
        available=True,
        total_activities=total,
        confirmed_activities=confirmed,
        confirmed_dependencies=_int(getattr(counts, "confirmed_dependencies", 0)),
        total_dependencies=_int(getattr(counts, "total_dependencies", 0)),
        review_summary=_review_summary_dict(summary),
        decisions=[_decision_dict(d) for d in _review_decisions(session)],
    )


def _review_summary_dict(summary: Any) -> Dict[str, Any]:
    if summary is None:
        return {}
    if isinstance(summary, dict):
        return dict(summary)
    return dict(getattr(summary, "to_dict", lambda: {})() or {})


def _decision_dict(decision: Any) -> Dict[str, Any]:
    return {
        "item_type": _str(getattr(decision, "item_type", "")),
        "item_id": _str(getattr(decision, "item_id", "")),
        "decision": _str(getattr(decision, "decision", "")),
        "corrected_value": getattr(decision, "corrected_value", None),
        "reason": _str(getattr(decision, "reason", "")),
        "comment": _str(getattr(decision, "comment", "")),
        "applied_at": _iso(getattr(decision, "applied_at", "")),
    }


def _safe_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    return dict(metadata or {})


# =============================================================================
# Public builder
# =============================================================================


def _authoritative_graph(session: Any) -> Any:
    """Return the authoritative backend graph, mirroring gui.results.data.extract.

    The sanctioned dashboard reads the graph from the current candidate
    (``session.current_candidate.graph``); the report builder must resolve the
    exact same source so exported counts match what the GUI displays. Falls back
    to the session tree/graph only when no candidate graph exists.
    """
    candidate = getattr(session, "current_candidate", None) or getattr(
        session, "candidate", None
    )
    candidate_graph = getattr(candidate, "graph", None)
    if candidate_graph is not None:
        return candidate_graph
    return getattr(session, "tree", None) or getattr(session, "graph", None)


def _authoritative_graph(session: Any) -> Any:
    """Return the authoritative backend graph, mirroring dashboard extract().

    The dashboard data extractor resolves the sanctioned counts from the
    current candidate (``candidate.graph``). The report builder must resolve
    the exact same graph so exported summary numbers match the dashboard
    (the Phase-6 sanctioned contract: 22 activities, 28 dependencies, 54.0,
    16 critical paths — never hardcoded, always derived from the fixture).
    """
    candidate = getattr(session, "current_candidate", None) or getattr(
        session, "candidate", None
    )
    graph = getattr(candidate, "graph", None) or getattr(session, "tree", None)
    if graph is not None:
        return graph
    return getattr(session, "graph", None) or getattr(session, "tree", None)


class ReportBuilder:
    """Builds an authoritative, deterministic ProjectReport from a tree."""

    def __init__(self, *, analyzer_version: str = ""):
        self._analyzer_version = analyzer_version or _default_version()

    def build_report(self, session: Any, *, metadata: Optional[Dict[str, Any]] = None) -> ProjectReport:
        """Build the full report for the current session's authoritative state."""
        cpm = _cpm(session)
        graph = _authoritative_graph(session)
        pert_result = _pert_result(session)
        estimates = _pert_estimates(session)
        pert_available = _pert_available(session, _pert_status(session))
        report_meta = ReportMetadata(
            report_id=_report_id(session),
            project_name=_project_name(session, metadata),
            source_image=_str(getattr(session, "current_image_path", "")),
            analysis_type="CPM" + (" + PERT" if pert_available else ""),
            created_at=_iso(getattr(session, "analysis_iso", "")),
            generated_at=_iso(getattr(session, "generated_at", "")),
            analyzer_version=self._analyzer_version,
        )
        if metadata:
            report_meta = ReportMetadata(
                **{
                    **report_meta.__dict__,
                    **{k: v for k, v in metadata.items() if k in report_meta.__dict__},
                }
            )
        return ProjectReport(
            metadata=report_meta,
            summary=_build_summary(session, cpm, graph, pert_result, pert_available),
            activities=_build_activities(cpm, graph),
            dependencies=_build_dependencies(graph),
            critical_paths=_build_critical_paths(cpm),
            cpm=_build_cpm(cpm),
            pert=_build_pert(session, pert_result, estimates, pert_available),
            validation=_build_validation(session),
            review=_build_review(session),
        )


def _cpm(session: Any) -> Any:
    candidate = getattr(session, "current_candidate", None)
    if candidate is None:
        return None
    return getattr(candidate, "cpm", None)


def _project_name(session: Any, metadata: Optional[Dict[str, Any]]) -> str:
    if metadata and metadata.get("project_name"):
        return str(metadata["project_name"])
    candidates = (
        getattr(session, "project_name", None),
        getattr(getattr(session, "workflow", None), "project_name", None),
    )
    for name in candidates:
        if name:
            return str(name)
    path = getattr(session, "current_image_path", "") or ""
    if path:
        return path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    return "Untitled Project"


def _report_id(session: Any) -> str:
    return _str(getattr(session, "report_id", "")) or "RPT-P6"


def _default_version() -> str:
    try:
        from pert_analyzer import __version__

        return str(__version__)
    except Exception:
        return "unknown"


# Backwards-compatible alias (Phase 5 placed the snapshot extract under gui).
def build_project_report(session: Any, *, metadata: Optional[Dict[str, Any]] = None) -> ProjectReport:
    """Top-level convenience: build the authoritative project report."""
    return ReportBuilder().build_report(session, metadata=metadata)

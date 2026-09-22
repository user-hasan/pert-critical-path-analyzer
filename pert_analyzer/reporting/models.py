"""
Report presentation models (Phase 6).

Immutable dataclasses describing a packaged project report. They are built
by ``ReportBuilder`` from the authoritative backend snapshot and consumed by
the exporters. No analysis math lives here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Optional

# Importing from the Qt-free snapshot layer keeps a single extraction path:
# the report rows are exactly what the dashboard shows.

SECTIONS_ORDER = (
    "summary",
    "activities",
    "dependencies",
    "critical_paths",
    "cpm",
    "pert",
    "validation",
    "review",
)


@dataclass(frozen=True)
class ReportMetadata:
    """Fixed provenance for a generated report."""

    report_id: str
    project_name: str = "Untitled Project"
    source_image: str = ""          # image path used for the analysis
    analysis_type: str = "CPM + PERT"  # incorporated engines
    created_at: str = ""            # analysis/review timestamp (authoritative)
    generated_at: str = ""          # report build timestamp (injectable)
    analyzer_version: str = "unknown"


@dataclass(frozen=True)
class SummarySection:
    """Top-level KPIs for the whole project."""

    project_name: str = ""
    project_duration: Optional[float] = None
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    critical_path_count: Optional[int] = None
    critical_activity_count: Optional[int] = None
    activity_count: int = 0
    dependency_count: int = 0
    pert_available: bool = False
    pert_project_duration: Optional[float] = None
    pert_project_variance: Optional[float] = None
    pert_project_std_dev: Optional[float] = None


@dataclass(frozen=True)
class ActivityReportRow:
    """One activity in the report."""

    activity_id: str = ""
    name: str = ""
    duration: Optional[float] = None
    early_start: Optional[float] = None
    early_finish: Optional[float] = None
    late_start: Optional[float] = None
    late_finish: Optional[float] = None
    total_float: Optional[float] = None
    free_float: Optional[float] = None
    is_critical: bool = False
    expected_time: Optional[float] = None   # PERT expected time, when available
    variance: Optional[float] = None        # PERT variance, when available


@dataclass(frozen=True)
class ActivitiesSection:
    """Full activity breakdown."""

    rows: List[ActivityReportRow] = dc_field(default_factory=list)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self.rows]


@dataclass(frozen=True)
class DependencyReportRow:
    """One dependency edge."""

    source: str = ""
    target: str = ""
    dependency_id: str = ""


@dataclass(frozen=True)
class DependenciesSection:
    """The dependency list."""

    rows: List[DependencyReportRow] = dc_field(default_factory=list)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [r.__dict__ for r in self.rows]


@dataclass(frozen=True)
class CriticalPathsSection:
    """The critical path(s)."""

    project_duration: Optional[float] = None
    critical_paths: List[List[str]] = dc_field(default_factory=list)

    def to_rows(self) -> List[Dict[str, Any]]:
        return [
            {"path": list(path), "length": len(path)}
            for path in self.critical_paths
        ]


@dataclass(frozen=True)
class CpmSection:
    """CPM results header block (authoritative)."""

    analysis_type: str = "CPM"
    is_valid: bool = True
    project_duration: Optional[float] = None
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    critical_path_count: Optional[int] = None
    critical_activity_count: Optional[int] = None
    errors: List[str] = dc_field(default_factory=list)


@dataclass(frozen=True)
class PertSection:
    """PERT analysis results, or a clear marker when PERT was unavailable."""

    available: bool = False
    reason: str = "PERT analysis was not part of this analysis."
    status: str = "NO_PERT_DATA"
    project_duration: Optional[float] = None
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    critical_paths: List[List[str]] = dc_field(default_factory=list)
    rows: List[Dict[str, Any]] = dc_field(default_factory=list)


@dataclass(frozen=True)
class ValidationSection:
    """Graph validation summary."""

    is_valid: bool = False
    status: str = "INVALID"
    errors: List[Dict[str, Any]] = dc_field(default_factory=list)
    warnings: List[Dict[str, Any]] = dc_field(default_factory=list)
    error_count: int = 0
    warning_count: int = 0
    component_count: Optional[int] = None
    is_acyclic: Optional[bool] = None
    error_count: int = 0
    warning_count: int = 0


@dataclass(frozen=True)
class ReviewSection:
    """Human review summary."""

    total_activities: int = 0
    confirmed_activities: int = 0
    activity_reviews_pending: int = 0
    total_dependencies: int = 0
    accepted_dependencies: int = 0
    rejected_dependencies: int = 0
    pending_dependency_reviews: int = 0
    duration_reviews_pending: int = 0
    duration_corrected: int = 0
    graph_validation_status: str = "NOT_VALIDATED"
    available: bool = False
    confirmed_dependencies: int = 0
    review_summary: Optional[Dict[str, Any]] = None
    decisions: List[Dict[str, Any]] = dc_field(default_factory=list)


@dataclass(frozen=True)
class ProjectReport:
    """A complete, immutable report document."""

    metadata: ReportMetadata = dc_field(default_factory=ReportMetadata)
    summary: SummarySection = dc_field(default_factory=SummarySection)
    activities: ActivitiesSection = dc_field(default_factory=ActivitiesSection)
    dependencies: DependenciesSection = dc_field(default_factory=DependenciesSection)
    critical_paths: CriticalPathsSection = dc_field(
        default_factory=CriticalPathsSection
    )
    cpm: CpmSection = dc_field(default_factory=CpmSection)
    pert: PertSection = dc_field(default_factory=PertSection)
    validation: ValidationSection = dc_field(default_factory=ValidationSection)
    review: ReviewSection = dc_field(default_factory=ReviewSection)

    def section_names(self) -> List[str]:
        return list(SECTIONS_ORDER)

    def sections(self) -> List[Any]:
        """Return the real section objects in the authoritative render order."""
        return [getattr(self, name) for name in self.section_names()]

    def to_dict(self) -> Dict[str, Any]:
        """Stable ordered dictionary for JSON/dict export."""
        return {
            "metadata": {
                "report_id": self.metadata.report_id,
                "project_name": self.metadata.project_name,
                "source_image": self.metadata.source_image,
                "analysis_type": self.metadata.analysis_type,
                "created_at": self.metadata.created_at,
                "generated_at": self.metadata.generated_at,
                "analyzer_version": self.metadata.analyzer_version,
            },
            "summary": {
                "project_name": self.summary.project_name,
                "project_duration": self.summary.project_duration,
                "project_variance": self.summary.project_variance,
                "project_std_dev": self.summary.project_std_dev,
                "critical_path_count": self.summary.critical_path_count,
                "critical_activity_count": self.summary.critical_activity_count,
                "activity_count": self.summary.activity_count,
                "dependency_count": self.summary.dependency_count,
                "pert_available": self.summary.pert_available,
                "pert_project_duration": self.summary.pert_project_duration,
                "pert_project_variance": self.summary.pert_project_variance,
            },
            "activities": self.activities.to_rows(),
            "dependencies": self.dependencies.to_rows(),
            "critical_paths": self.critical_paths.to_rows(),
            "cpm": {
                "analysis_type": self.cpm.analysis_type,
                "is_valid": self.cpm.is_valid,
                "project_duration": self.cpm.project_duration,
                "project_variance": self.cpm.project_variance,
                "project_std_dev": self.cpm.project_std_dev,
                "critical_path_count": self.cpm.critical_path_count,
                "critical_activity_count": self.cpm.critical_activity_count,
                "errors": list(self.cpm.errors),
            },
            "pert": {
                "available": self.pert.available,
                "reason": self.pert.reason,
                "status": self.pert.status,
                "project_duration": self.pert.project_duration,
                "project_variance": self.pert.project_variance,
                "project_std_dev": self.pert.project_std_dev,
                "critical_paths": [list(p) for p in self.pert.critical_paths],
                "rows": list(self.pert.rows),
            },
            "validation": {
                "is_valid": self.validation.is_valid,
                "status": self.validation.status,
                "errors": list(self.validation.errors),
                "warnings": list(self.validation.warnings),
                "component_count": self.validation.component_count,
                "is_acyclic": self.validation.is_acyclic,
            },
            "review": {
                "total_activities": self.review.total_activities,
                "confirmed_activities": self.review.confirmed_activities,
                "activity_reviews_pending": self.review.activity_reviews_pending,
                "total_dependencies": self.review.total_dependencies,
                "accepted_dependencies": self.review.accepted_dependencies,
                "rejected_dependencies": self.review.rejected_dependencies,
                "pending_dependency_reviews": self.review.pending_dependency_reviews,
                "duration_reviews_pending": self.review.duration_reviews_pending,
                "duration_corrected": self.review.duration_corrected,
                "graph_validation_status": self.review.graph_validation_status,
            },
        }


def make_report_id(seed: str = "") -> str:
    """Derive a short deterministic report id from a content seed."""
    digest = hashlib.sha256((seed or "report").encode("utf-8")).hexdigest()
    return f"RPT-{digest[:12].upper()}"

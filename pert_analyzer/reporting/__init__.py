"""
Reporting & Export subsystem (Phase 6).

A Qt-free package that packages the authoritative analysis snapshot into
human-readable project reports and exports them as PDF, Excel, JSON, and
CSV.

Design contract
---------------
* **Authoritative, never re-computed**: the report model is built by mapping
  the authoritative backend values already computed by the analysis pipeline
  (CPM `AnalysisResult`, `PertResult`, `GraphValidationResult`, review
  summary) — exactly the snapshot the Results dashboard presents. No CPM,
  PERT, or validation math is ever recalculated in this package.
* **Qt-free**: this package contains no PySide6 imports and is testable
  without a QApplication. The GUI layer calls into `ReportBuilder` /
  `ExportManager`; these classes never touch widgets or the event loop.
* **Deterministic**: exporters accept an injectable timestamp and produce
  stable, reproducible output (stable key ordering, no locale surprises,
  no non-deterministic metadata).
* **Sections with unavailable data are clearly marked**, never fabricated
  (e.g. a PERT section is omitted/marked when no PERT analysis was run),
  so reports never invent values the dashboard does not show.
"""

from __future__ import annotations

from pert_analyzer.reporting.builder import ReportBuilder
from pert_analyzer.reporting.export import ExportManager, ExportResult
from pert_analyzer.reporting.models import (
    ActivitiesSection,
    ActivityReportRow,
    CpmSection,
    CriticalPathsSection,
    DependenciesSection,
    DependencyReportRow,
    PertSection,
    ProjectReport,
    ReportMetadata,
    ReviewSection,
    SummarySection,
    ValidationSection,
)

__all__ = [
    "ActivitiesSection",
    "ActivityReportRow",
    "CpmSection",
    "CriticalPathsSection",
    "DependenciesSection",
    "DependencyReportRow",
    "ExportManager",
    "ExportResult",
    "PertSection",
    "ProjectReport",
    "ReportBuilder",
    "ReportMetadata",
    "ReviewSection",
    "SummarySection",
    "ValidationSection",
]

"""
Data models for semantic diagram reconstruction.

Defines structures that bridge CV/OCR detection evidence with
the semantic GraphModel. Each reconstructed element carries evidence
provenance and confidence scores, enabling the next integration phase
(CPM calculation) to assess reconstruction quality.

These models are passive data holders — the reconstruction logic
lives in cv/reconstruction.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class ActivityStatus(Enum):
    """Status of a reconstructed activity's identification."""

    CONFIRMED = "confirmed"
    INFERRED = "inferred"
    REVIEW_REQUIRED = "review_required"


class IDSource(Enum):
    """Source of an activity's identifier."""

    OCR_HIGH_CONFIDENCE = "ocr_high_confidence"
    OCR_ACCEPTABLE = "ocr_acceptable"
    REGION_OCR = "region_ocr"
    SPATIAL_INFERENCE = "spatial_inference"
    INTERNAL_GENERATED = "internal_generated"


class AmbiguityType(Enum):
    """Types of ambiguity that can occur during diagram reconstruction."""

    TEXT_NO_MATCH = "text_no_match"
    TEXT_MULTIPLE_MATCHES = "text_multiple_matches"
    ARROW_NO_SOURCE = "arrow_no_source"
    ARROW_NO_TARGET = "arrow_no_target"
    ARROW_AMBIGUOUS = "arrow_ambiguous"
    SHAPE_NO_TEXT = "shape_no_text"
    SHAPE_ROLE_UNKNOWN = "shape_role_unknown"
    DURATION_CONFLICT = "duration_conflict"
    DUPLICATE_LABEL = "duplicate_label"
    MISSING_DURATION = "missing_duration"
    MISSING_ACTIVITY_ID = "missing_activity_id"
    ISOLATED_NODE = "isolated_node"
    ORPHANED_ARROW = "orphaned_arrow"


class ValidationSeverity(Enum):
    """Severity levels for validation findings."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class EvidenceTrace:
    """
    Provenance for a single piece of evidence used in reconstruction.

    Tracks which detection phase produced the evidence, the specific
    source IDs involved, and the confidence contribution.
    """

    source_phase: str
    source_ids: List[str] = field(default_factory=list)
    confidence_contribution: float = 0.0
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructedActivity:
    """
    A reconstructed activity node (AON) or activity arrow (AOA).

    Carries the semantic interpretation plus all evidence that
    contributed to the reconstruction.

    Dual identity model:
        geometric_node_id: The stable geometric identifier from shape detection.
            This never changes, regardless of OCR success or failure.
        activity_id: The semantic activity label (e.g., "A", "B").
            May be None when OCR fails or returns invalid text.
        semantic_status: Whether the semantic identity is confirmed or needs review.
    """

    activity_id: str
    label: str = ""
    duration: float = 0.0
    is_dummy: bool = False
    confidence: float = 0.0
    evidence: List[EvidenceTrace] = field(default_factory=list)
    source_shape_id: Optional[str] = None
    source_node_id: Optional[str] = None
    source_arrow_id: Optional[str] = None
    source_text_region_ids: List[str] = field(default_factory=list)
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    position: Optional[Tuple[float, float]] = None
    bounding_box: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ActivityStatus = ActivityStatus.REVIEW_REQUIRED
    id_source: IDSource = IDSource.INTERNAL_GENERATED
    needs_review: bool = True

    # Dual identity fields
    geometric_node_id: Optional[str] = None
    semantic_activity_id: Optional[str] = None
    semantic_status: ActivityStatus = ActivityStatus.REVIEW_REQUIRED


@dataclass
class ReconstructedEvent:
    """
    A reconstructed event node (circle/end-point).

    In AON diagrams, events are implicit. In AOA diagrams, events
    are explicit nodes (circles) that connect arrows.
    """

    event_id: str
    label: str = ""
    confidence: float = 0.0
    evidence: List[EvidenceTrace] = field(default_factory=list)
    source_shape_id: Optional[str] = None
    source_text_region_ids: List[str] = field(default_factory=list)
    position: Optional[Tuple[float, float]] = None
    bounding_box: Optional[Any] = None
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReconstructedDependency:
    """
    A reconstructed dependency (arrow) between two elements.

    Maps a DetectedArrow to a semantic precedence relationship.
    """

    source_id: str
    target_id: str
    dependency_type: str = "finish_to_start"
    confidence: float = 0.0
    evidence: List[EvidenceTrace] = field(default_factory=list)
    source_arrow_id: Optional[str] = None
    source_text_region_ids: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AmbiguityIssue:
    """
    An ambiguity or uncertainty discovered during reconstruction.

    Captures what is ambiguous, which elements are involved,
    what alternatives exist, and a suggested resolution.
    """

    ambiguity_type: AmbiguityType
    description: str
    involved_ids: List[str] = field(default_factory=list)
    alternatives: List[str] = field(default_factory=list)
    suggested_resolution: str = ""
    severity: ValidationSeverity = ValidationSeverity.WARNING
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """
    Result of validating a reconstructed diagram.

    Contains pass/fail status, error and warning counts, and all
    individual findings.
    """

    is_valid: bool = True
    errors: List[AmbiguityIssue] = field(default_factory=list)
    warnings: List[AmbiguityIssue] = field(default_factory=list)
    info: List[AmbiguityIssue] = field(default_factory=list)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def warning_count(self) -> int:
        return len(self.warnings)

    @property
    def total_issues(self) -> int:
        return self.error_count + self.warning_count + len(self.info)

    def add_issue(self, issue: AmbiguityIssue) -> None:
        """Add a validation issue, categorizing by severity."""
        if issue.severity == ValidationSeverity.ERROR:
            self.errors.append(issue)
            self.is_valid = False
        elif issue.severity == ValidationSeverity.WARNING:
            self.warnings.append(issue)
        else:
            self.info.append(issue)


@dataclass
class ReconstructedDiagram:
    """
    Complete result of semantic diagram reconstruction.

    Contains all reconstructed elements, ambiguities, validation
    result, and can produce a GraphModel via to_graph_model().
    """

    diagram_type: str = "UNKNOWN"
    activities: List[ReconstructedActivity] = field(default_factory=list)
    events: List[ReconstructedEvent] = field(default_factory=list)
    dependencies: List[ReconstructedDependency] = field(default_factory=list)
    ambiguities: List[AmbiguityIssue] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    overall_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def activity_count(self) -> int:
        return len(self.activities)

    @property
    def event_count(self) -> int:
        return len(self.events)

    @property
    def dependency_count(self) -> int:
        return len(self.dependencies)

    @property
    def ambiguity_count(self) -> int:
        return len(self.ambiguities)

    def get_activity_by_id(self, activity_id: str) -> Optional[ReconstructedActivity]:
        """Find an activity by its ID."""
        for act in self.activities:
            if act.activity_id == activity_id:
                return act
        return None

    def get_event_by_id(self, event_id: str) -> Optional[ReconstructedEvent]:
        """Find an event by its ID."""
        for evt in self.events:
            if evt.event_id == event_id:
                return evt
        return None

    def get_dependencies_for(self, element_id: str) -> List[ReconstructedDependency]:
        """Get all dependencies where the given element is source or target."""
        return [
            dep for dep in self.dependencies
            if dep.source_id == element_id or dep.target_id == element_id
        ]

    def to_graph_model(self) -> "GraphModel":
        """
        Convert reconstructed diagram to a GraphModel.

        Returns:
            A GraphModel suitable for CPM analysis.
        """
        from pert_analyzer.core.models import DiagramType as DT, GraphModel
        from pert_analyzer.graph.builder import GraphBuilder

        diagram_type = DT.AON if self.diagram_type == "AON" else DT.AOA
        builder = GraphBuilder(diagram_type=diagram_type)

        if self.diagram_type == "AON":
            for act in self.activities:
                builder.add_activity(
                    activity_id=act.activity_id,
                    duration=act.duration,
                    name=act.label or act.activity_id,
                    is_dummy=act.is_dummy,
                )
            for dep in self.dependencies:
                if dep.source_id and dep.target_id:
                    try:
                        builder.add_dependency(dep.source_id, dep.target_id)
                    except Exception:
                        pass
        else:
            for evt in self.events:
                builder.add_node(
                    node_id=evt.event_id,
                    label=evt.label or evt.event_id,
                )
            for act in self.activities:
                builder.add_activity(
                    activity_id=act.activity_id,
                    duration=act.duration,
                    name=act.label or act.activity_id,
                    is_dummy=act.is_dummy,
                    source_node=act.source_node,
                    target_node=act.target_node,
                )
            for dep in self.dependencies:
                if dep.source_id and dep.target_id:
                    try:
                        builder.add_dependency(dep.source_id, dep.target_id)
                    except Exception:
                        pass

        try:
            return builder.build()
        except Exception:
            return GraphModel(diagram_type=diagram_type)

    def reconcile(self) -> "ReconciliationResult":
        """
        Run graph-based reconciliation on this diagram.

        Returns:
            ReconciliationResult with full diagnostic info.
        """
        from pert_analyzer.cv.reconciliation import ReconciliationEngine

        engine = ReconciliationEngine()
        return engine.reconcile(self)

"""
Graph-consistency validation for reconstructed AON diagrams.

Analyzes the reconstructed graph using node positions, arrow geometry,
graph connectivity, and structural properties to detect inconsistencies
and provide evidence for semantic correction.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pert_analyzer.cv.reconstruction_models import (
    ActivityStatus,
    EvidenceTrace,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
)
from pert_analyzer.cv.semantic_resolution import (
    CandidateResolution,
    DurationResolution,
    ResolutionEvidence,
    ResolutionSource,
    ResolutionStatus,
    SemanticResolutionResult,
)

logger = logging.getLogger(__name__)


class TopologyIssueType(Enum):
    """Types of topology issues."""
    SELF_LOOP = "self_loop"
    DUPLICATE_ID = "duplicate_id"
    DISCONNECTED_COMPONENT = "disconnected_component"
    ISOLATED_NODE = "isolated_node"
    MISSING_DURATION = "missing_duration"
    SUSPICIOUS_ID = "suspicious_id"
    SUSPICIOUS_DURATION = "suspicious_duration"
    NO_SOURCES = "no_sources"
    NO_SINKS = "no_sinks"
    CYCLE_DETECTED = "cycle_detected"


class TopologySeverity(Enum):
    """Severity of topology issues."""
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass
class TopologyIssue:
    """A topology issue found during validation."""
    issue_type: TopologyIssueType
    description: str
    severity: TopologySeverity
    involved_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TopologyValidationReport:
    """Structured report of graph-consistency validation."""
    activity_count: int = 0
    dependency_count: int = 0
    duplicate_ids: List[str] = field(default_factory=list)
    missing_or_unknown_ids: List[str] = field(default_factory=list)
    suspicious_ids: List[str] = field(default_factory=list)
    suspicious_durations: List[str] = field(default_factory=list)
    isolated_nodes: List[str] = field(default_factory=list)
    disconnected_components: List[List[str]] = field(default_factory=list)
    invalid_edges: List[Tuple[str, str]] = field(default_factory=list)
    self_loops: List[str] = field(default_factory=list)
    unresolved_ambiguities: List[str] = field(default_factory=list)
    confidence_summary: Dict[str, float] = field(default_factory=dict)
    issues: List[TopologyIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not any(
            i.severity == TopologySeverity.ERROR for i in self.issues
        )


class TopologyValidator:
    """
    Validates graph consistency of reconstructed AON diagrams.

    Analyzes structural properties without assuming specific activity IDs
    or reference topology. Provides evidence for semantic correction.
    """

    def validate(
        self,
        diagram: ReconstructedDiagram,
        resolution_result: Optional[SemanticResolutionResult] = None,
    ) -> TopologyValidationReport:
        """
        Run full topology validation on a reconstructed diagram.

        Args:
            diagram: Reconstructed diagram with activities and dependencies.
            resolution_result: Optional semantic resolution result.

        Returns:
            TopologyValidationReport with all findings.
        """
        report = TopologyValidationReport()
        report.activity_count = len(diagram.activities)
        report.dependency_count = len(diagram.dependencies)

        # Build adjacency structures
        act_ids = {a.activity_id for a in diagram.activities}
        id_map = {a.activity_id: a for a in diagram.activities}

        # 1. Self-loops
        self._check_self_loops(diagram, act_ids, report)

        # 2. Duplicate IDs
        self._check_duplicate_ids(diagram, report)

        # 3. Isolated nodes
        self._check_isolated_nodes(diagram, act_ids, report)

        # 4. Disconnected components
        self._check_disconnected_components(diagram, act_ids, report)

        # 5. Invalid edges (references to non-existent activities)
        self._check_invalid_edges(diagram, act_ids, report)

        # 6. Missing durations
        self._check_missing_durations(diagram, act_ids, report)

        # 7. Suspicious IDs (INFERRED, review required)
        self._check_suspicious_ids(diagram, resolution_result, report)

        # 8. Suspicious durations (no duration, very high/low)
        self._check_suspicious_durations(diagram, id_map, report)

        # 9. Source/sink analysis
        self._check_sources_and_sinks(diagram, act_ids, report)

        # 10. Cycles
        self._check_cycles(diagram, act_ids, report)

        # 11. Confidence summary
        self._compute_confidence_summary(diagram, report)

        # 12. Unresolved ambiguities
        self._check_unresolved_ambiguities(diagram, report)

        return report

    def _check_self_loops(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for self-loop dependencies."""
        for dep in diagram.dependencies:
            if dep.source_id == dep.target_id and dep.source_id in act_ids:
                report.self_loops.append(dep.source_id)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.SELF_LOOP,
                    description=f"Self-loop: {dep.source_id} -> {dep.target_id}",
                    severity=TopologySeverity.ERROR,
                    involved_ids=[dep.source_id],
                ))

    def _check_duplicate_ids(
        self,
        diagram: ReconstructedDiagram,
        report: TopologyValidationReport,
    ) -> None:
        """Check for duplicate activity IDs."""
        seen: Dict[str, int] = defaultdict(int)
        for act in diagram.activities:
            seen[act.activity_id] += 1
        for aid, count in seen.items():
            if count > 1:
                report.duplicate_ids.append(aid)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.DUPLICATE_ID,
                    description=f"Duplicate ID '{aid}' appears {count} times",
                    severity=TopologySeverity.ERROR,
                    involved_ids=[aid],
                ))

    def _check_isolated_nodes(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for nodes with no dependencies."""
        connected: Set[str] = set()
        for dep in diagram.dependencies:
            connected.add(dep.source_id)
            connected.add(dep.target_id)
        for aid in act_ids:
            if aid not in connected:
                report.isolated_nodes.append(aid)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.ISOLATED_NODE,
                    description=f"Node '{aid}' has no dependencies",
                    severity=TopologySeverity.WARNING,
                    involved_ids=[aid],
                ))

    def _check_disconnected_components(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for disconnected graph components."""
        adj: Dict[str, Set[str]] = defaultdict(set)
        for dep in diagram.dependencies:
            if dep.source_id in act_ids and dep.target_id in act_ids:
                adj[dep.source_id].add(dep.target_id)
                adj[dep.target_id].add(dep.source_id)

        visited: Set[str] = set()
        components: List[List[str]] = []

        for start in act_ids:
            if start in visited:
                continue
            component = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if node in visited:
                    continue
                visited.add(node)
                component.append(node)
                for neighbor in adj.get(node, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            if component:
                components.append(sorted(component))

        if len(components) > 1:
            report.disconnected_components = components
            for comp in components:
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.DISCONNECTED_COMPONENT,
                    description=f"Disconnected component: {comp}",
                    severity=TopologySeverity.WARNING,
                    involved_ids=comp,
                ))

    def _check_invalid_edges(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for edges referencing non-existent activities."""
        for dep in diagram.dependencies:
            if dep.source_id and dep.source_id not in act_ids:
                report.invalid_edges.append((dep.source_id, dep.target_id))
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.SELF_LOOP,
                    description=f"Invalid edge: source '{dep.source_id}' not in activities",
                    severity=TopologySeverity.ERROR,
                    involved_ids=[dep.source_id],
                ))
            if dep.target_id and dep.target_id not in act_ids:
                report.invalid_edges.append((dep.source_id, dep.target_id))
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.SELF_LOOP,
                    description=f"Invalid edge: target '{dep.target_id}' not in activities",
                    severity=TopologySeverity.ERROR,
                    involved_ids=[dep.target_id],
                ))

    def _check_missing_durations(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for activities with zero or missing durations."""
        for act in diagram.activities:
            if act.duration <= 0 and not act.is_dummy:
                report.missing_or_unknown_ids.append(act.activity_id)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.MISSING_DURATION,
                    description=f"Activity '{act.activity_id}' has no duration",
                    severity=TopologySeverity.WARNING,
                    involved_ids=[act.activity_id],
                ))

    def _check_suspicious_ids(
        self,
        diagram: ReconstructedDiagram,
        resolution_result: Optional[SemanticResolutionResult],
        report: TopologyValidationReport,
    ) -> None:
        """Check for suspicious activity IDs."""
        for act in diagram.activities:
            is_suspicious = False
            reason = ""

            if act.activity_id.startswith("INFERRED_"):
                is_suspicious = True
                reason = "Inferred ID (no OCR)"
            elif act.status == ActivityStatus.REVIEW_REQUIRED:
                is_suspicious = True
                reason = "Review required"
            elif resolution_result:
                cr = resolution_result.id_resolutions.get(act.source_node_id)
                if cr and cr.source == ResolutionSource.REVIEW_REQUIRED:
                    is_suspicious = True
                    reason = "Resolution review required"

            if is_suspicious:
                report.suspicious_ids.append(act.activity_id)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.SUSPICIOUS_ID,
                    description=f"Suspicious ID '{act.activity_id}': {reason}",
                    severity=TopologySeverity.WARNING,
                    involved_ids=[act.activity_id],
                    metadata={"reason": reason},
                ))

    def _check_suspicious_durations(
        self,
        diagram: ReconstructedDiagram,
        id_map: Dict[str, ReconstructedActivity],
        report: TopologyValidationReport,
    ) -> None:
        """Check for suspicious durations."""
        for act in diagram.activities:
            is_suspicious = False
            reason = ""

            if act.duration <= 0 and not act.is_dummy:
                is_suspicious = True
                reason = "Missing duration"
            elif act.duration > 50:
                is_suspicious = True
                reason = f"Unusually high duration ({act.duration})"
            elif act.duration < 0:
                is_suspicious = True
                reason = f"Negative duration ({act.duration})"

            if is_suspicious:
                report.suspicious_durations.append(act.activity_id)
                report.issues.append(TopologyIssue(
                    issue_type=TopologyIssueType.SUSPICIOUS_DURATION,
                    description=f"Suspicious duration for '{act.activity_id}': {reason}",
                    severity=TopologySeverity.WARNING,
                    involved_ids=[act.activity_id],
                    metadata={"reason": reason, "duration": act.duration},
                ))

    def _check_sources_and_sinks(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for source and sink nodes."""
        has_out: Set[str] = set()
        has_in: Set[str] = set()
        for dep in diagram.dependencies:
            if dep.source_id in act_ids:
                has_out.add(dep.source_id)
            if dep.target_id in act_ids:
                has_in.add(dep.target_id)

        sources = act_ids - has_in
        sinks = act_ids - has_out

        if not sources:
            report.issues.append(TopologyIssue(
                issue_type=TopologyIssueType.NO_SOURCES,
                description="No source nodes (every node has an incoming dependency)",
                severity=TopologySeverity.WARNING,
            ))
        if not sinks:
            report.issues.append(TopologyIssue(
                issue_type=TopologyIssueType.NO_SINKS,
                description="No sink nodes (every node has an outgoing dependency)",
                severity=TopologySeverity.WARNING,
            ))

    def _check_cycles(
        self,
        diagram: ReconstructedDiagram,
        act_ids: Set[str],
        report: TopologyValidationReport,
    ) -> None:
        """Check for cycles using DFS."""
        adj: Dict[str, List[str]] = defaultdict(list)
        for dep in diagram.dependencies:
            if dep.source_id in act_ids and dep.target_id in act_ids:
                adj[dep.source_id].append(dep.target_id)

        WHITE, GRAY, BLACK = 0, 1, 2
        color = {a: WHITE for a in act_ids}
        has_cycle = False

        def dfs(node: str) -> bool:
            nonlocal has_cycle
            color[node] = GRAY
            for neighbor in adj.get(node, []):
                if color[neighbor] == GRAY:
                    has_cycle = True
                    return True
                if color[neighbor] == WHITE:
                    if dfs(neighbor):
                        return True
            color[node] = BLACK
            return False

        for act in act_ids:
            if color[act] == WHITE:
                if dfs(act):
                    break

        if has_cycle:
            report.issues.append(TopologyIssue(
                issue_type=TopologyIssueType.CYCLE_DETECTED,
                description="Cycle detected in dependency graph",
                severity=TopologySeverity.ERROR,
            ))

    def _compute_confidence_summary(
        self,
        diagram: ReconstructedDiagram,
        report: TopologyValidationReport,
    ) -> None:
        """Compute confidence statistics."""
        if not diagram.activities:
            return
        confs = [a.confidence for a in diagram.activities]
        report.confidence_summary = {
            "mean": sum(confs) / len(confs),
            "min": min(confs),
            "max": max(confs),
            "below_0.3": sum(1 for c in confs if c < 0.3),
            "above_0.7": sum(1 for c in confs if c >= 0.7),
        }

    def _check_unresolved_ambiguities(
        self,
        diagram: ReconstructedDiagram,
        report: TopologyValidationReport,
    ) -> None:
        """Check for unresolved ambiguities."""
        for amb in diagram.ambiguities:
            report.unresolved_ambiguities.append(amb.description)

"""
Comprehensive test suite for graph-based activity reconciliation.

Tests: START/FINISH exclusion, false-positive detection, isolated nodes,
duplicate IDs, inferred IDs, missing durations, graph consistency,
dependency preservation, self-arrow regression, and diagnostic output.
"""

from __future__ import annotations

from typing import List, Optional

import pytest

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import (
    CandidateNode,
    DetectedArrow,
    ShapeDetectionResult,
    ShapeType,
)
from pert_analyzer.cv.ocr_models import OCRProcessingResult, OCRTextRegion, TextType
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.reconstruction_models import (
    ActivityStatus,
    AmbiguityType,
    EvidenceTrace,
    IDSource,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
)
from pert_analyzer.cv.reconciliation import (
    CandidateActivity,
    CandidateVerdict,
    GraphConsistencyStatus,
    ReconciliationEngine,
    ReconciliationStatus,
)


# =============================================================================
# Helper functions
# =============================================================================


def _make_candidate(
    node_id: str = "c1",
    shape_type: ShapeType = ShapeType.RECTANGLE,
    source_shape_id: str = "s1",
    x: float = 100,
    y: float = 100,
    w: float = 200,
    h: float = 100,
    confidence: float = 0.85,
) -> CandidateNode:
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    return CandidateNode(
        node_id=node_id,
        shape_type=shape_type,
        source_shape_id=source_shape_id,
        label="",
        position=Point(x + w / 2, y + h / 2),
        bounding_box=bbox,
        confidence=confidence,
    )


def _make_shape_result(
    candidates: List[CandidateNode],
) -> ShapeDetectionResult:
    return ShapeDetectionResult(
        candidate_nodes=candidates,
        detected_shapes=[],
        image_dimensions=(800, 600),
    )


def _make_dependency(
    source_id: str, target_id: str, confidence: float = 0.8
) -> ReconstructedDependency:
    return ReconstructedDependency(
        source_id=source_id,
        target_id=target_id,
        confidence=confidence,
        evidence=[],
    )


def _make_activity(
    activity_id: str,
    duration: float = 5.0,
    label: str = "",
    status: ActivityStatus = ActivityStatus.CONFIRMED,
    source_node_id: Optional[str] = None,
    position: Optional[tuple] = None,
    bounding_box: Optional[BoundingBox] = None,
    confidence: float = 0.85,
) -> ReconstructedActivity:
    return ReconstructedActivity(
        activity_id=activity_id,
        label=label or activity_id,
        duration=duration,
        status=status,
        source_node_id=source_node_id,
        position=position,
        bounding_box=bounding_box,
        confidence=confidence,
    )


def _make_diagram(
    activities: List[ReconstructedActivity],
    dependencies: List[ReconstructedDependency],
) -> ReconstructedDiagram:
    diagram = ReconstructedDiagram(diagram_type="AON")
    diagram.activities = activities
    diagram.dependencies = dependencies
    return diagram


# =============================================================================
# Test Class
# =============================================================================


class TestReconciliation:
    """Tests for graph-based activity reconciliation."""

    def test_start_finish_exclusion(self) -> None:
        """START/FINISH nodes should be identified and excluded from normal activities."""
        start = _make_activity("START", duration=0, label="START", position=(50, 50))
        finish = _make_activity("FINISH", duration=0, label="FINISH", position=(750, 550))
        a1 = _make_activity("A", duration=5, source_node_id="c1")
        a2 = _make_activity("B", duration=3, source_node_id="c2")

        deps = [
            _make_dependency("START", "A"),
            _make_dependency("A", "B"),
            _make_dependency("B", "FINISH"),
        ]

        diagram = _make_diagram([start, finish, a1, a2], deps)
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        start_cands = [
            c for c in result.candidates if c.is_start_candidate
        ]
        finish_cands = [
            c for c in result.candidates if c.is_finish_candidate
        ]
        assert len(start_cands) >= 1
        assert len(finish_cands) >= 1

    def test_false_positive_candidate(self) -> None:
        """Extremely small rectangle that doesn't match cluster should be FP."""
        # Create several normal-sized activities to form a cluster
        activities = [
            _make_activity(f"ACT_{i}", duration=5.0, source_node_id=f"c{i}",
                          bounding_box=BoundingBox(x=100+i*200, y=100, width=150, height=80))
            for i in range(5)
        ]
        # Noise: extremely small (area < 100) and different dimensions
        noise = _make_activity(
            "NOISE", duration=0, source_node_id="c_noise",
            bounding_box=BoundingBox(x=400, y=300, width=5, height=3),
        )

        diagram = _make_diagram(activities + [noise], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        noise_cand = next(
            (c for c in result.candidates if c.activity.activity_id == "NOISE"),
            None,
        )
        assert noise_cand is not None
        # 2 negative signals: cluster mismatch + extremely small → FP
        assert noise_cand.verdict == CandidateVerdict.LIKELY_FALSE_POSITIVE

    def test_isolated_rectangle_with_cluster(self) -> None:
        """Isolated rectangle matching cluster should be REVIEW_REQUIRED, not FP."""
        # Create a cluster of normal activities
        activities = [
            _make_activity(f"ACT_{i}", duration=5.0, source_node_id=f"c{i}",
                          bounding_box=BoundingBox(x=100+i*200, y=100, width=150, height=80))
            for i in range(5)
        ]
        # Isolated but matching the cluster dimensions
        isolated = _make_activity(
            "ISO", duration=0, source_node_id="c_iso",
            bounding_box=BoundingBox(x=400, y=300, width=150, height=80),
        )

        diagram = _make_diagram(activities + [isolated], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        iso_cand = next(
            (c for c in result.candidates if c.activity.activity_id == "ISO"),
            None,
        )
        assert iso_cand is not None
        # Should be REVIEW_REQUIRED or LIKELY_REAL, NOT LIKELY_FALSE_POSITIVE
        assert iso_cand.verdict != CandidateVerdict.LIKELY_FALSE_POSITIVE

    def test_real_rectangle_no_arrows_retained(self) -> None:
        """A real rectangle with no detected arrows should be retained (not FP)."""
        # Simulate F/H/R/U scenario: normal-sized activity with OCR ID but no arrows
        base_activities = [
            _make_activity(f"ACT_{i}", duration=5.0, source_node_id=f"c{i}",
                          bounding_box=BoundingBox(x=100+i*200, y=100, width=150, height=80))
            for i in range(5)
        ]
        # F-style: real activity with OCR but no arrows
        f_activity = _make_activity(
            "F", duration=4.0, source_node_id="c_f",
            bounding_box=BoundingBox(x=500, y=300, width=150, height=80),
            status=ActivityStatus.CONFIRMED,
        )

        diagram = _make_diagram(base_activities + [f_activity], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        f_cand = next(
            (c for c in result.candidates if c.activity.activity_id == "F"),
            None,
        )
        assert f_cand is not None
        # F is a real activity — should NOT be FP even with no arrows
        assert f_cand.verdict == CandidateVerdict.LIKELY_REAL

    def test_rectangle_with_valid_dependencies(self) -> None:
        """Rectangle with valid dependencies should be marked as LIKELY_REAL."""
        start = _make_activity("START", duration=0, label="START", position=(50, 50))
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=3, source_node_id="c2")
        c = _make_activity("C", duration=4, source_node_id="c3")
        finish = _make_activity("FINISH", duration=0, label="FINISH", position=(750, 550))

        deps = [
            _make_dependency("START", "A"),
            _make_dependency("A", "B"),
            _make_dependency("A", "C"),
            _make_dependency("B", "FINISH"),
            _make_dependency("C", "FINISH"),
        ]

        diagram = _make_diagram([start, a, b, c, finish], deps)
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        for cand in result.candidates:
            if cand.activity.activity_id in ("A", "B", "C"):
                assert cand.verdict == CandidateVerdict.LIKELY_REAL

    def test_duplicate_id_detection(self) -> None:
        """Two candidates with the same ID should trigger review."""
        a1 = _make_activity("A", duration=5, source_node_id="c1")
        a2 = _make_activity("A", duration=3, source_node_id="c2")

        diagram = _make_diagram([a1, a2], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        dup_cands = [
            c for c in result.candidates if len(c.duplicate_ids) > 0
        ]
        assert len(dup_cands) == 2

    def test_inferred_id_preserved(self) -> None:
        """INFERRED_XXX IDs should be preserved with inferred status."""
        inferred = _make_activity(
            "INFERRED_001", duration=0, status=ActivityStatus.INFERRED,
            source_node_id="c1",
        )

        diagram = _make_diagram([inferred], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        cand = result.candidates[0]
        assert cand.activity.activity_id == "INFERRED_001"
        assert cand.activity.status == ActivityStatus.INFERRED

    def test_missing_duration_review_required(self) -> None:
        """Activity with missing duration should be REVIEW_REQUIRED."""
        start = _make_activity("START", duration=0, label="START", position=(50, 50))
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=0, source_node_id="c2")  # missing duration
        finish = _make_activity("FINISH", duration=0, label="FINISH", position=(750, 550))

        deps = [
            _make_dependency("START", "A"),
            _make_dependency("A", "B"),
            _make_dependency("B", "FINISH"),
        ]

        diagram = _make_diagram([start, a, b, finish], deps)
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        # CPM should not run because B has no duration
        assert result.cpm_status == "REVIEW_REQUIRED"

    def test_graph_based_reconciliation(self) -> None:
        """Full reconciliation should produce correct counts."""
        activities = [
            _make_activity(f"ACT_{i}", duration=float(i + 1), source_node_id=f"c{i}")
            for i in range(5)
        ]
        deps = [
            _make_dependency(f"ACT_{i}", f"ACT_{i+1}")
            for i in range(4)
        ]

        diagram = _make_diagram(activities, deps)
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        assert result.candidates_before == 5
        assert result.confirmed_activities + result.inferred_activities >= 4

    def test_review_required_state(self) -> None:
        """Mixed activities should produce appropriate review states."""
        start = _make_activity("START", duration=0, label="START", position=(50, 50))
        confirmed = _make_activity(
            "A", duration=5, status=ActivityStatus.CONFIRMED, source_node_id="c1"
        )
        inferred = _make_activity(
            "INFERRED_001", duration=0, status=ActivityStatus.INFERRED,
            source_node_id="c2",
        )
        finish = _make_activity("FINISH", duration=0, label="FINISH", position=(750, 550))

        deps = [
            _make_dependency("START", "A"),
            _make_dependency("A", "INFERRED_001"),
            _make_dependency("INFERRED_001", "FINISH"),
        ]

        diagram = _make_diagram([start, confirmed, inferred, finish], deps)
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        assert result.confirmed_activities >= 1
        assert result.inferred_activities >= 1

    def test_dependency_preservation(self) -> None:
        """Valid dependencies should be preserved through reconciliation."""
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=3, source_node_id="c2")

        deps = [_make_dependency("A", "B")]
        diagram = _make_diagram([a, b], deps)

        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        assert result.dependency_reconciliation.preserved_deps == 1
        assert len(result.final_dependencies) == 1

    def test_self_arrow_regression(self) -> None:
        """Self-arrows should remain 0 after reconciliation."""
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=3, source_node_id="c2")

        deps = [_make_dependency("A", "B")]
        diagram = _make_diagram([a, b], deps)

        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        self_arrows = [
            d for d in result.final_dependencies
            if d.source_id == d.target_id
        ]
        assert len(self_arrows) == 0

    def test_graph_consistency_valid(self) -> None:
        """Valid acyclic graph should pass consistency checks."""
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=3, source_node_id="c2")
        c = _make_activity("C", duration=4, source_node_id="c3")

        deps = [
            _make_dependency("A", "B"),
            _make_dependency("B", "C"),
        ]
        diagram = _make_diagram([a, b, c], deps)

        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        assert result.graph_consistency.status == GraphConsistencyStatus.VALID

    def test_diagnostic_table_generated(self) -> None:
        """Diagnostic table should be generated with header and data."""
        a = _make_activity("A", duration=5, source_node_id="c1")
        b = _make_activity("B", duration=3, source_node_id="c2")

        deps = [_make_dependency("A", "B")]
        diagram = _make_diagram([a, b], deps)

        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        assert "DIAGNOSTIC TABLE" in result.diagnostic_table
        assert "A" in result.diagnostic_table
        assert "B" in result.diagnostic_table

    def test_filtered_candidates_stored(self) -> None:
        """Filtered candidates should be stored for human review."""
        # Create a cluster of normal activities first
        activities = [
            _make_activity(f"ACT_{i}", duration=5.0, source_node_id=f"c{i}",
                          bounding_box=BoundingBox(x=100+i*200, y=100, width=150, height=80))
            for i in range(5)
        ]
        # Noise: extremely small (area < 100) AND different from cluster → 2 negative signals → FP
        noise = _make_activity(
            "X", duration=0, source_node_id="c_noise",
            bounding_box=BoundingBox(x=400, y=300, width=5, height=3),
        )

        diagram = _make_diagram(activities + [noise], [])
        engine = ReconciliationEngine()
        result = engine.reconcile(diagram)

        filtered = [
            f for f in result.filtered_candidates
            if f.activity_id == "X"
        ]
        assert len(filtered) == 1
        assert filtered[0].reason != ""

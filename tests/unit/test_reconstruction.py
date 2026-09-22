"""
Comprehensive test suite for Phase 7: Semantic Diagram Reconstruction.

Covers: data models, AON reconstruction, AOA reconstruction, confidence
aggregation, validation, GraphModel conversion, ambiguity detection,
edge cases, and integration with existing CV pipeline.
"""

from __future__ import annotations

from typing import List, Optional
from unittest.mock import MagicMock

import pytest

from pert_analyzer.core.models import (
    BoundingBox,
    DiagramType,
    GraphModel,
    Point,
)
from pert_analyzer.cv.models import (
    CandidateNode,
    DetectedArrow,
    ShapeDetectionResult,
    ShapeType,
)
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    OCRProcessingResult,
    OCRTextRegion,
    TextAssociation,
    TextAssociationResult,
    TextType,
)
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.reconstruction_models import (
    AmbiguityIssue,
    AmbiguityType,
    EvidenceTrace,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
    ValidationResult,
    ValidationSeverity,
)
from pert_analyzer.cv.exceptions import ReconstructionError
from pert_analyzer.graph.builder import GraphBuilder


# =============================================================================
# Helper functions
# =============================================================================


def _make_region(
    text: str = "test",
    x: float = 100,
    y: float = 100,
    w: float = 50,
    h: float = 30,
    confidence: float = 0.9,
    text_type: TextType = TextType.UNKNOWN,
    region_id: str = "",
) -> OCRTextRegion:
    """Create an OCRTextRegion for testing."""
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    region = OCRTextRegion(
        text=text,
        raw_text=text,
        normalized_text=text.strip(),
        bounding_box=bbox,
        center=bbox.center,
        confidence=confidence,
        text_type=text_type,
    )
    if region_id:
        region.region_id = region_id
    return region


def _make_candidate(
    node_id: str = "cnode_1",
    shape_type: ShapeType = ShapeType.RECTANGLE,
    source_shape_id: str = "shape_1",
    x: float = 100,
    y: float = 100,
    w: float = 200,
    h: float = 100,
    confidence: float = 0.85,
) -> CandidateNode:
    """Create a CandidateNode for testing."""
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


def _make_arrow(
    arrow_id: str = "arrow_1",
    start_x: float = 0,
    start_y: float = 0,
    end_x: float = 100,
    end_y: float = 0,
    confidence: float = 0.8,
    source_candidate_id: Optional[str] = None,
    target_candidate_id: Optional[str] = None,
) -> DetectedArrow:
    """Create a DetectedArrow for testing."""
    start = Point(start_x, start_y)
    end = Point(end_x, end_y)
    dx = end_x - start_x
    dy = end_y - start_y
    length = (dx**2 + dy**2) ** 0.5
    evidence = {}
    if source_candidate_id:
        evidence["node_at_endpoint_1"] = source_candidate_id
    if target_candidate_id:
        evidence["node_at_endpoint_2"] = target_candidate_id
    return DetectedArrow(
        arrow_id=arrow_id,
        start=start,
        end=end,
        length=length,
        confidence=confidence,
        evidence=evidence,
    )


def _make_shape_result(
    candidates: List[CandidateNode],
    shapes: Optional[List] = None,
) -> ShapeDetectionResult:
    """Create a ShapeDetectionResult for testing."""
    return ShapeDetectionResult(
        candidate_nodes=candidates,
        detected_shapes=shapes or [],
        image_dimensions=(800, 600),
    )


def _make_association_result(
    associations: List[TextAssociation],
) -> TextAssociationResult:
    """Create a TextAssociationResult for testing."""
    best = associations[0] if associations else None
    if best:
        best.is_best_candidate = True
    return TextAssociationResult(
        text_region_id=associations[0].text_region_id if associations else "",
        associations=associations,
        best_association=best,
        is_ambiguous=False,
        has_no_match=len(associations) == 0,
    )


def _make_ocr_result(
    regions: List[OCRTextRegion],
) -> OCRProcessingResult:
    """Create an OCRProcessingResult for testing."""
    return OCRProcessingResult(regions=regions)


# =============================================================================
# EvidenceTrace Tests
# =============================================================================


class TestEvidenceTrace:
    """Tests for EvidenceTrace data model."""

    def test_default_values(self) -> None:
        trace = EvidenceTrace(source_phase="shape_detection")
        assert trace.source_phase == "shape_detection"
        assert trace.source_ids == []
        assert trace.confidence_contribution == 0.0
        assert trace.description == ""
        assert trace.metadata == {}

    def test_with_values(self) -> None:
        trace = EvidenceTrace(
            source_phase="ocr_association",
            source_ids=["r1", "r2"],
            confidence_contribution=0.85,
            description="Text found inside rectangle",
            metadata={"key": "value"},
        )
        assert trace.source_phase == "ocr_association"
        assert trace.source_ids == ["r1", "r2"]
        assert trace.confidence_contribution == 0.85
        assert trace.description == "Text found inside rectangle"
        assert trace.metadata == {"key": "value"}


# =============================================================================
# ReconstructedActivity Tests
# =============================================================================


class TestReconstructedActivity:
    """Tests for ReconstructedActivity data model."""

    def test_default_values(self) -> None:
        act = ReconstructedActivity(activity_id="A1")
        assert act.activity_id == "A1"
        assert act.label == ""
        assert act.duration == 0.0
        assert act.is_dummy is False
        assert act.confidence == 0.0
        assert act.evidence == []
        assert act.warnings == []

    def test_with_evidence(self) -> None:
        evidence = [EvidenceTrace(source_phase="shape_detection")]
        act = ReconstructedActivity(
            activity_id="A1",
            label="Design",
            duration=5.0,
            confidence=0.9,
            evidence=evidence,
        )
        assert act.activity_id == "A1"
        assert act.label == "Design"
        assert act.duration == 5.0
        assert len(act.evidence) == 1

    def test_position_and_bbox(self) -> None:
        bbox = BoundingBox(10, 20, 100, 50)
        act = ReconstructedActivity(
            activity_id="A1",
            position=(60.0, 45.0),
            bounding_box=bbox,
        )
        assert act.position == (60.0, 45.0)
        assert act.bounding_box is not None


# =============================================================================
# ReconstructedEvent Tests
# =============================================================================


class TestReconstructedEvent:
    """Tests for ReconstructedEvent data model."""

    def test_default_values(self) -> None:
        evt = ReconstructedEvent(event_id="E1")
        assert evt.event_id == "E1"
        assert evt.label == ""
        assert evt.confidence == 0.0
        assert evt.evidence == []

    def test_with_values(self) -> None:
        evt = ReconstructedEvent(
            event_id="E1",
            label="Start",
            confidence=0.95,
            source_shape_id="shape_5",
        )
        assert evt.event_id == "E1"
        assert evt.label == "Start"
        assert evt.confidence == 0.95
        assert evt.source_shape_id == "shape_5"


# =============================================================================
# ReconstructedDependency Tests
# =============================================================================


class TestReconstructedDependency:
    """Tests for ReconstructedDependency data model."""

    def test_default_values(self) -> None:
        dep = ReconstructedDependency(source_id="A1", target_id="A2")
        assert dep.source_id == "A1"
        assert dep.target_id == "A2"
        assert dep.dependency_type == "finish_to_start"
        assert dep.confidence == 0.0

    def test_with_confidence(self) -> None:
        dep = ReconstructedDependency(
            source_id="A1",
            target_id="A2",
            confidence=0.88,
            source_arrow_id="arrow_1",
        )
        assert dep.confidence == 0.88
        assert dep.source_arrow_id == "arrow_1"


# =============================================================================
# AmbiguityIssue Tests
# =============================================================================


class TestAmbiguityIssue:
    """Tests for AmbiguityIssue data model."""

    def test_default_values(self) -> None:
        issue = AmbiguityIssue(
            ambiguity_type=AmbiguityType.TEXT_NO_MATCH,
            description="Text region has no matching shape",
        )
        assert issue.ambiguity_type == AmbiguityType.TEXT_NO_MATCH
        assert issue.description == "Text region has no matching shape"
        assert issue.severity == ValidationSeverity.WARNING

    def test_error_severity(self) -> None:
        issue = AmbiguityIssue(
            ambiguity_type=AmbiguityType.DUPLICATE_LABEL,
            description="Duplicate ID",
            severity=ValidationSeverity.ERROR,
        )
        assert issue.severity == ValidationSeverity.ERROR


# =============================================================================
# ValidationResult Tests
# =============================================================================


class TestValidationResult:
    """Tests for ValidationResult data model."""

    def test_default_valid(self) -> None:
        result = ValidationResult()
        assert result.is_valid is True
        assert result.error_count == 0
        assert result.warning_count == 0
        assert result.total_issues == 0

    def test_add_error(self) -> None:
        result = ValidationResult()
        issue = AmbiguityIssue(
            ambiguity_type=AmbiguityType.MISSING_DURATION,
            description="No duration",
            severity=ValidationSeverity.ERROR,
        )
        result.add_issue(issue)
        assert result.is_valid is False
        assert result.error_count == 1
        assert result.total_issues == 1

    def test_add_warning(self) -> None:
        result = ValidationResult()
        issue = AmbiguityIssue(
            ambiguity_type=AmbiguityType.ISOLATED_NODE,
            description="Isolated node",
            severity=ValidationSeverity.WARNING,
        )
        result.add_issue(issue)
        assert result.is_valid is True
        assert result.warning_count == 1

    def test_add_info(self) -> None:
        result = ValidationResult()
        issue = AmbiguityIssue(
            ambiguity_type=AmbiguityType.SHAPE_NO_TEXT,
            description="No text",
            severity=ValidationSeverity.INFO,
        )
        result.add_issue(issue)
        assert result.is_valid is True
        assert len(result.info) == 1


# =============================================================================
# ReconstructedDiagram Tests
# =============================================================================


class TestReconstructedDiagram:
    """Tests for ReconstructedDiagram data model."""

    def test_default_values(self) -> None:
        diagram = ReconstructedDiagram()
        assert diagram.diagram_type == "UNKNOWN"
        assert diagram.activity_count == 0
        assert diagram.event_count == 0
        assert diagram.dependency_count == 0
        assert diagram.ambiguity_count == 0

    def test_get_activity_by_id_found(self) -> None:
        diagram = ReconstructedDiagram()
        act = ReconstructedActivity(activity_id="A1")
        diagram.activities.append(act)
        assert diagram.get_activity_by_id("A1") is act

    def test_get_activity_by_id_not_found(self) -> None:
        diagram = ReconstructedDiagram()
        assert diagram.get_activity_by_id("A1") is None

    def test_get_event_by_id_found(self) -> None:
        diagram = ReconstructedDiagram()
        evt = ReconstructedEvent(event_id="E1")
        diagram.events.append(evt)
        assert diagram.get_event_by_id("E1") is evt

    def test_get_event_by_id_not_found(self) -> None:
        diagram = ReconstructedDiagram()
        assert diagram.get_event_by_id("E1") is None

    def test_get_dependencies_for(self) -> None:
        diagram = ReconstructedDiagram()
        d1 = ReconstructedDependency(source_id="A1", target_id="A2")
        d2 = ReconstructedDependency(source_id="A2", target_id="A3")
        d3 = ReconstructedDependency(source_id="A1", target_id="A3")
        diagram.dependencies.extend([d1, d2, d3])
        deps = diagram.get_dependencies_for("A1")
        assert len(deps) == 2
        assert d1 in deps
        assert d3 in deps

    def test_get_dependencies_for_none(self) -> None:
        diagram = ReconstructedDiagram()
        assert diagram.get_dependencies_for("A1") == []


# =============================================================================
# ReconstructionEngine Initialization Tests
# =============================================================================


class TestReconstructionEngineInit:
    """Tests for ReconstructionEngine initialization."""

    def test_default_init(self) -> None:
        engine = ReconstructionEngine()
        assert engine.confidence_threshold == 0.1

    def test_custom_threshold(self) -> None:
        engine = ReconstructionEngine(confidence_threshold=0.5)
        assert engine.confidence_threshold == 0.5


# =============================================================================
# AON Reconstruction Tests
# =============================================================================


class TestAONReconstruction:
    """Tests for AON diagram reconstruction."""

    def test_empty_shapes(self) -> None:
        engine = ReconstructionEngine()
        shape_result = ShapeDetectionResult(
            candidate_nodes=[],
            detected_shapes=[],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.diagram_type == "AON"
        assert diagram.activity_count == 0
        assert diagram.event_count == 0
        assert diagram.dependency_count == 0

    def test_single_rectangle_creates_activity(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 1
        assert diagram.activities[0].activity_id == "INFERRED_001"
        assert diagram.activities[0].source_shape_id == "s1"
        assert diagram.activities[0].source_node_id == "c1"
        assert diagram.activities[0].needs_review is True

    def test_multiple_rectangles_create_activities(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(node_id="c1", source_shape_id="s1"),
            _make_candidate(node_id="c2", source_shape_id="s2",
                           x=300, y=100),
            _make_candidate(node_id="c3", source_shape_id="s3",
                           x=500, y=100),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 3
        ids = [a.activity_id for a in diagram.activities]
        assert "INFERRED_001" in ids
        assert "INFERRED_002" in ids
        assert "INFERRED_003" in ids

    def test_circles_create_events(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(
                node_id="c1",
                shape_type=ShapeType.CIRCLE,
                source_shape_id="s1",
                w=80, h=80,
            ),
            _make_candidate(
                node_id="c2",
                shape_type=ShapeType.CIRCLE,
                source_shape_id="s2",
                x=400, y=100, w=80, h=80,
            ),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.event_count == 2
        assert diagram.activity_count == 0

    def test_mixed_shapes(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(
                node_id="c1",
                shape_type=ShapeType.RECTANGLE,
                source_shape_id="s1",
            ),
            _make_candidate(
                node_id="c2",
                shape_type=ShapeType.CIRCLE,
                source_shape_id="s2",
                x=300, y=100, w=80, h=80,
            ),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 1
        assert diagram.event_count == 1

    def test_arrows_create_dependencies(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(
                node_id="c1",
                shape_type=ShapeType.RECTANGLE,
                source_shape_id="s1",
                x=50, y=100, w=100, h=50,
            ),
            _make_candidate(
                node_id="c2",
                shape_type=ShapeType.RECTANGLE,
                source_shape_id="s2",
                x=300, y=100, w=100, h=50,
            ),
        ]
        shape_result = _make_shape_result(candidates)
        arrow = _make_arrow(
            arrow_id="a1",
            start_x=150, start_y=125,
            end_x=300, end_y=125,
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aon(shape_result, arrow_result)
        assert diagram.dependency_count == 1
        dep = diagram.dependencies[0]
        assert dep.source_arrow_id == "a1"

    def test_ocr_text_provides_label(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            x=50, y=50, w=200, h=100,
        )
        shape_result = _make_shape_result([candidate])
        region = _make_region(
            text="A",
            x=100, y=75, w=80, h=30,
            text_type=TextType.TEXT_LABEL_CANDIDATE,
            region_id="r1",
        )
        ocr_result = _make_ocr_result([region])
        assoc = TextAssociation(
            text_region_id="r1",
            candidate_target_id="c1",
            target_type=AssociationTargetType.NODE,
            association_score=0.9,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        diagram = engine.reconstruct_aon(
            shape_result, ocr_result=ocr_result,
            association_result=assoc_result,
        )
        assert diagram.activity_count == 1
        assert diagram.activities[0].label == "A"

    def test_ocr_text_provides_duration(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            x=50, y=50, w=200, h=100,
        )
        shape_result = _make_shape_result([candidate])
        region = _make_region(
            text="5",
            x=120, y=80, w=30, h=20,
            text_type=TextType.NUMERIC_CANDIDATE,
            region_id="r1",
        )
        ocr_result = _make_ocr_result([region])
        assoc = TextAssociation(
            text_region_id="r1",
            candidate_target_id="c1",
            target_type=AssociationTargetType.NODE,
            association_score=0.95,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        diagram = engine.reconstruct_aon(
            shape_result, ocr_result=ocr_result,
            association_result=assoc_result,
        )
        assert diagram.activities[0].duration == 5.0

    def test_evidence_trace_populated(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        act = diagram.activities[0]
        assert len(act.evidence) >= 1
        assert act.evidence[0].source_phase == "shape_detection"

    def test_confidence_computed(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            confidence=0.85,
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.overall_confidence > 0
        assert diagram.activities[0].confidence > 0

    def test_validation_runs(self) -> None:
        engine = ReconstructionEngine()
        shape_result = _make_shape_result([])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.validation is not None
        assert isinstance(diagram.validation, ValidationResult)

    def test_validation_catches_empty_diagram(self) -> None:
        engine = ReconstructionEngine()
        shape_result = _make_shape_result([])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.validation.is_valid is False
        assert diagram.validation.error_count >= 1


# =============================================================================
# AOA Reconstruction Tests
# =============================================================================


class TestAOAReconstruction:
    """Tests for AOA diagram reconstruction."""

    def test_empty_shapes(self) -> None:
        engine = ReconstructionEngine()
        shape_result = ShapeDetectionResult(
            candidate_nodes=[],
            detected_shapes=[],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aoa(shape_result)
        assert diagram.diagram_type == "AOA"
        assert diagram.activity_count == 0
        assert diagram.event_count == 0

    def test_circles_create_events(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(
                node_id="c1",
                shape_type=ShapeType.CIRCLE,
                source_shape_id="s1",
                x=50, y=100, w=80, h=80,
            ),
            _make_candidate(
                node_id="c2",
                shape_type=ShapeType.CIRCLE,
                source_shape_id="s2",
                x=400, y=100, w=80, h=80,
            ),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aoa(shape_result)
        assert diagram.event_count == 2
        assert diagram.activity_count == 0

    def test_arrows_create_activities(self) -> None:
        engine = ReconstructionEngine()
        circle1 = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.CIRCLE,
            source_shape_id="s1",
            x=50, y=100, w=80, h=80,
        )
        circle2 = _make_candidate(
            node_id="c2",
            shape_type=ShapeType.CIRCLE,
            source_shape_id="s2",
            x=400, y=100, w=80, h=80,
        )
        shape_result = _make_shape_result([circle1, circle2])
        arrow = _make_arrow(
            arrow_id="a1",
            start_x=130, start_y=140,
            end_x=400, end_y=140,
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aoa(shape_result, arrow_result)
        assert diagram.activity_count == 1
        assert diagram.activities[0].source_arrow_id == "a1"

    def test_aoa_dependency_via_events(self) -> None:
        engine = ReconstructionEngine()
        circle1 = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.CIRCLE,
            source_shape_id="s1",
            x=50, y=100, w=80, h=80,
        )
        circle2 = _make_candidate(
            node_id="c2",
            shape_type=ShapeType.CIRCLE,
            source_shape_id="s2",
            x=400, y=100, w=80, h=80,
        )
        circle3 = _make_candidate(
            node_id="c3",
            shape_type=ShapeType.CIRCLE,
            source_shape_id="s3",
            x=750, y=100, w=80, h=80,
        )
        shape_result = _make_shape_result([circle1, circle2, circle3])
        arrow1 = _make_arrow(
            arrow_id="a1",
            start_x=90, start_y=140,
            end_x=400, end_y=140,
        )
        arrow2 = _make_arrow(
            arrow_id="a2",
            start_x=440, start_y=140,
            end_x=750, end_y=140,
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow1, arrow2],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aoa(shape_result, arrow_result)
        assert diagram.activity_count == 2
        # AOA precedence: activity a1 ends at the middle event, a2 starts
        # there, so a1→a2 is a dependency between ACTIVITY ids (resolvable
        # by the downstream AON graph) rather than an event-pair edge.
        assert diagram.dependency_count == 1
        assert diagram.dependencies[0].source_id == diagram.activities[0].activity_id
        assert diagram.dependencies[0].target_id == diagram.activities[1].activity_id

    def test_aoa_rectangles_as_labels(self) -> None:
        engine = ReconstructionEngine()
        rect = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            x=200, y=50, w=120, h=40,
        )
        shape_result = _make_shape_result([rect])
        diagram = engine.reconstruct_aoa(shape_result)
        assert diagram.activity_count == 0
        assert diagram.event_count == 0


# =============================================================================
# Validation Tests
# =============================================================================


class TestValidation:
    """Tests for diagram validation."""

    def test_valid_diagram(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=5.0, confidence=0.9,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", duration=3.0, confidence=0.85,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="A1", target_id="A2", confidence=0.9,
        ))
        result = engine.validate(diagram)
        assert result.is_valid is True
        assert result.error_count == 0

    def test_missing_duration_warning(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=0.0, is_dummy=False,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", duration=5.0,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="A1", target_id="A2",
        ))
        result = engine.validate(diagram)
        warnings = [
            i for i in result.warnings
            if i.ambiguity_type == AmbiguityType.MISSING_DURATION
        ]
        assert len(warnings) >= 1

    def test_isolated_node_warning(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=5.0,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", duration=3.0,
        ))
        # No dependencies — A2 is isolated
        result = engine.validate(diagram)
        isolated = [
            i for i in result.warnings
            if i.ambiguity_type == AmbiguityType.ISOLATED_NODE
        ]
        assert len(isolated) >= 1

    def test_duplicate_activity_id_error(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=5.0,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=3.0,
        ))
        result = engine.validate(diagram)
        assert result.is_valid is False
        dup_errors = [
            i for i in result.errors
            if i.ambiguity_type == AmbiguityType.DUPLICATE_LABEL
        ]
        assert len(dup_errors) >= 1

    def test_duplicate_event_id_error(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AOA")
        diagram.events.append(ReconstructedEvent(event_id="E1"))
        diagram.events.append(ReconstructedEvent(event_id="E1"))
        result = engine.validate(diagram)
        assert result.is_valid is False

    def test_empty_diagram_error(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        result = engine.validate(diagram)
        assert result.is_valid is False
        assert result.error_count >= 1

    def test_dummy_activity_no_duration_warning(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="D1", duration=0.0, is_dummy=True,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=5.0,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="D1", target_id="A1",
        ))
        result = engine.validate(diagram)
        # Dummy with 0 duration should not trigger missing duration warning
        duration_warnings = [
            i for i in result.warnings
            if i.ambiguity_type == AmbiguityType.MISSING_DURATION
            and "D1" in (i.involved_ids or [])
        ]
        assert len(duration_warnings) == 0

    def test_low_confidence_info(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", duration=5.0, confidence=0.2,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", duration=3.0, confidence=0.9,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="A1", target_id="A2",
        ))
        result = engine.validate(diagram)
        low_conf = [
            i for i in result.info
            if i.ambiguity_type == AmbiguityType.SHAPE_ROLE_UNKNOWN
        ]
        assert len(low_conf) >= 1


# =============================================================================
# Confidence Aggregation Tests
# =============================================================================


class TestConfidenceAggregation:
    """Tests for confidence aggregation logic."""

    def test_empty_confidence(self) -> None:
        engine = ReconstructionEngine()
        assert engine._aggregate_confidence([]) == 0.0

    def test_single_confidence(self) -> None:
        engine = ReconstructionEngine()
        assert engine._aggregate_confidence([0.8]) == 0.8

    def test_multiple_confidence(self) -> None:
        engine = ReconstructionEngine()
        result = engine._aggregate_confidence([0.9, 0.7, 0.8])
        assert 0.7 <= result <= 0.9

    def test_all_zeros(self) -> None:
        engine = ReconstructionEngine()
        assert engine._aggregate_confidence([0.0, 0.0, 0.0]) == 0.0

    def test_overall_confidence_averages_all(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram()
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", confidence=0.8,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", confidence=0.6,
        ))
        diagram.events.append(ReconstructedEvent(
            event_id="E1", confidence=0.9,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="A1", target_id="A2", confidence=0.7,
        ))
        overall = engine._compute_overall_confidence(diagram)
        expected = (0.8 + 0.6 + 0.9 + 0.7) / 4
        assert abs(overall - expected) < 0.01


# =============================================================================
# Ambiguity Detection Tests
# =============================================================================


class TestAmbiguityDetection:
    """Tests for ambiguity detection."""

    def test_duplicate_label_detected(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", label="Design",
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", label="Design",
        ))
        engine._detect_ambiguities(diagram)
        dupes = [
            a for a in diagram.ambiguities
            if a.ambiguity_type == AmbiguityType.DUPLICATE_LABEL
        ]
        assert len(dupes) >= 1

    def test_no_text_detected(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1",
            source_text_region_ids=[],
        ))
        engine._detect_ambiguities(diagram)
        no_text = [
            a for a in diagram.ambiguities
            if a.ambiguity_type == AmbiguityType.SHAPE_NO_TEXT
        ]
        assert len(no_text) >= 1

    def test_no_ambiguities_for_clean_diagram(self) -> None:
        engine = ReconstructionEngine()
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", label="A",
            source_text_region_ids=["r1"],
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", label="B",
            source_text_region_ids=["r2"],
        ))
        engine._detect_ambiguities(diagram)
        assert len(diagram.ambiguities) == 0


# =============================================================================
# GraphModel Conversion Tests
# =============================================================================


class TestGraphModelConversion:
    """Tests for converting ReconstructedDiagram to GraphModel."""

    def test_to_graph_model_aon_basic(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", label="Design", duration=5.0,
        ))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A2", label="Build", duration=3.0,
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="A1", target_id="A2",
        ))
        graph = diagram.to_graph_model()
        assert isinstance(graph, GraphModel)
        assert graph.diagram_type == DiagramType.AON
        assert "A1" in graph.activities
        assert "A2" in graph.activities
        assert graph.activities["A1"].duration == 5.0
        assert graph.activities["A2"].duration == 3.0
        assert len(graph.dependencies) == 1
        assert graph.dependencies[0].source == "A1"
        assert graph.dependencies[0].target == "A2"

    def test_to_graph_model_aoa(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AOA")
        diagram.events.append(ReconstructedEvent(event_id="1"))
        diagram.events.append(ReconstructedEvent(event_id="2"))
        diagram.activities.append(ReconstructedActivity(
            activity_id="A1", label="Activity A",
            duration=5.0, is_dummy=False,
            source_node="1", target_node="2",
        ))
        diagram.dependencies.append(ReconstructedDependency(
            source_id="1", target_id="2",
        ))
        graph = diagram.to_graph_model()
        assert graph.diagram_type == DiagramType.AOA
        assert "1" in graph.nodes
        assert "2" in graph.nodes
        assert "A1" in graph.activities

    def test_to_graph_model_empty(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AON")
        graph = diagram.to_graph_model()
        assert isinstance(graph, GraphModel)
        assert len(graph.activities) == 0


# =============================================================================
# Utility Method Tests
# =============================================================================


class TestUtilityMethods:
    """Tests for private utility methods."""

    def test_extract_activity_id_letter_digit(self) -> None:
        engine = ReconstructionEngine()
        assert engine._extract_activity_id("A1") == "A1"
        assert engine._extract_activity_id("B2") == "B2"
        assert engine._extract_activity_id("T10") == "T10"

    def test_extract_activity_id_single_letter(self) -> None:
        engine = ReconstructionEngine()
        assert engine._extract_activity_id("A") == "A"
        assert engine._extract_activity_id("x") == "X"

    def test_extract_activity_id_short_alphanumeric(self) -> None:
        engine = ReconstructionEngine()
        assert engine._extract_activity_id("AB") == "AB"
        assert engine._extract_activity_id("abc") == "ABC"

    def test_extract_activity_id_long_text_returns_none(self) -> None:
        engine = ReconstructionEngine()
        assert engine._extract_activity_id("This is a long label") is None

    def test_parse_numeric_integer(self) -> None:
        engine = ReconstructionEngine()
        assert engine._parse_numeric("5") == 5.0

    def test_parse_numeric_float(self) -> None:
        engine = ReconstructionEngine()
        assert engine._parse_numeric("3.5") == 3.5

    def test_parse_numeric_negative(self) -> None:
        engine = ReconstructionEngine()
        assert engine._parse_numeric("-2") == -2.0

    def test_parse_numeric_in_text(self) -> None:
        engine = ReconstructionEngine()
        assert engine._parse_numeric("duration: 7 days") == 7.0

    def test_parse_numeric_no_number(self) -> None:
        engine = ReconstructionEngine()
        assert engine._parse_numeric("no number") is None

    def test_build_text_region_map(self) -> None:
        engine = ReconstructionEngine()
        r1 = _make_region(text="A", region_id="r1")
        r2 = _make_region(text="B", x=200, region_id="r2")
        ocr_result = _make_ocr_result([r1, r2])
        text_map = engine._build_text_region_map(ocr_result)
        assert "r1" in text_map
        assert "r2" in text_map
        assert text_map["r1"].raw_text == "A"

    def test_build_text_region_map_none(self) -> None:
        engine = ReconstructionEngine()
        text_map = engine._build_text_region_map(None)
        assert text_map == {}


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_single_activity_no_dependencies(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 1
        assert diagram.dependency_count == 0
        # Single node is isolated
        isolated = [
            i for i in diagram.validation.warnings
            if i.ambiguity_type == AmbiguityType.ISOLATED_NODE
        ]
        assert len(isolated) >= 1

    def test_many_activities(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(
                node_id=f"c{i}",
                shape_type=ShapeType.RECTANGLE,
                source_shape_id=f"s{i}",
                x=50 + i * 150, y=100,
            )
            for i in range(10)
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 10

    def test_very_long_text_label(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            x=50, y=50, w=300, h=100,
        )
        shape_result = _make_shape_result([candidate])
        long_label = "A" * 200
        region = _make_region(
            text=long_label,
            x=100, y=75, w=200, h=30,
            text_type=TextType.TEXT_LABEL_CANDIDATE,
            region_id="r1",
        )
        ocr_result = _make_ocr_result([region])
        assoc = TextAssociation(
            text_region_id="r1",
            candidate_target_id="c1",
            target_type=AssociationTargetType.NODE,
            association_score=0.9,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        diagram = engine.reconstruct_aon(
            shape_result, ocr_result=ocr_result,
            association_result=assoc_result,
        )
        # Long text (>6 chars) is rejected as implausible activity ID
        assert diagram.activities[0].activity_id == "INFERRED_001"
        assert diagram.activity_count == 1

    def test_zero_confidence_candidate(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            confidence=0.0,
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 1
        assert diagram.activities[0].confidence == 0.0

    def test_negative_duration(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1",
            shape_type=ShapeType.RECTANGLE,
            source_shape_id="s1",
            x=50, y=50, w=200, h=100,
        )
        shape_result = _make_shape_result([candidate])
        region = _make_region(
            text="-5",
            x=120, y=80, w=30, h=20,
            text_type=TextType.NUMERIC_CANDIDATE,
            region_id="r1",
        )
        ocr_result = _make_ocr_result([region])
        assoc = TextAssociation(
            text_region_id="r1",
            candidate_target_id="c1",
            target_type=AssociationTargetType.NODE,
            association_score=0.95,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        diagram = engine.reconstruct_aon(
            shape_result, ocr_result=ocr_result,
            association_result=assoc_result,
        )
        assert diagram.activities[0].duration == -5.0

    def test_diagram_with_only_arrows_no_shapes(self) -> None:
        engine = ReconstructionEngine()
        shape_result = ShapeDetectionResult(
            candidate_nodes=[],
            detected_shapes=[],
            image_dimensions=(800, 600),
        )
        arrow = _make_arrow(
            arrow_id="a1",
            start_x=100, start_y=100,
            end_x=300, end_y=100,
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow],
            image_dimensions=(800, 600),
        )
        diagram = engine.reconstruct_aon(shape_result, arrow_result)
        # Arrow without shapes produces ambiguity
        assert diagram.ambiguity_count >= 1


# =============================================================================
# Exception Tests
# =============================================================================


class TestExceptions:
    """Tests for reconstruction exceptions."""

    def test_reconstruction_error_creation(self) -> None:
        err = ReconstructionError("reconstruct_aon", "no shapes found")
        assert err.operation == "reconstruct_aon"
        assert err.reason == "no shapes found"
        assert "reconstruct_aon" in str(err)
        assert "no shapes found" in str(err)

    def test_reconstruction_error_inheritance(self) -> None:
        from pert_analyzer.cv.exceptions import CVError
        err = ReconstructionError("test", "msg")
        assert isinstance(err, CVError)
        assert isinstance(err, Exception)


# =============================================================================
# Deterministic ID Generation Tests (per specification)
# =============================================================================


class TestDeterministicInferredIDs:
    """Tests for deterministic INFERRED_XXX ID generation."""

    def test_rectangle_no_ocr_gets_inferred_id(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1", shape_type=ShapeType.RECTANGLE, source_shape_id="s1"
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 1
        act = diagram.activities[0]
        assert act.activity_id.startswith("INFERRED_")
        assert act.needs_review is True
        assert act.status.value == "inferred"

    def test_spatial_ordering_top_to_bottom_left_to_right(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(node_id="c1", source_shape_id="s1", x=300, y=300),
            _make_candidate(node_id="c2", source_shape_id="s2", x=100, y=100),
            _make_candidate(node_id="c3", source_shape_id="s3", x=500, y=100),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        # Verify spatial ordering: c2 (y=100,x=100) -> first, c3 (y=100,x=500) -> second, c1 (y=300) -> third
        shape_to_id = {a.source_shape_id: a.activity_id for a in diagram.activities}
        assert shape_to_id["s2"] == "INFERRED_001"
        assert shape_to_id["s3"] == "INFERRED_002"
        assert shape_to_id["s1"] == "INFERRED_003"

    def test_rectangle_with_ocr_id_uses_ocr(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1", shape_type=ShapeType.RECTANGLE, source_shape_id="s1"
        )
        shape_result = _make_shape_result([candidate])

        region = _make_region(
            text="X7", x=120, y=120, w=40, h=20,
            text_type=TextType.ACTIVITY_ID_CANDIDATE,
        )
        assoc = TextAssociation(
            text_region_id=region.region_id,
            candidate_target_id=candidate.node_id,
            target_type=AssociationTargetType.NODE,
            association_score=0.9,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        ocr_result = OCRProcessingResult(
            regions=[region], engine="mock", image_dimensions=(800, 600)
        )

        diagram = engine.reconstruct_aon(shape_result, ocr_result=ocr_result, association_result=assoc_result)
        act = diagram.activities[0]
        assert act.activity_id == "X7"
        assert act.status.value == "review_required"
        assert act.needs_review is True
        assert act.duration == 0.0

    def test_rectangle_with_ocr_id_no_duration_gets_review_status(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1", shape_type=ShapeType.RECTANGLE, source_shape_id="s1"
        )
        shape_result = _make_shape_result([candidate])

        region = _make_region(
            text="K3", x=120, y=120, w=40, h=20,
            text_type=TextType.ACTIVITY_ID_CANDIDATE,
        )
        assoc = TextAssociation(
            text_region_id=region.region_id,
            candidate_target_id=candidate.node_id,
            target_type=AssociationTargetType.NODE,
            association_score=0.9,
            is_best_candidate=True,
        )
        assoc_result = _make_association_result([assoc])
        ocr_result = OCRProcessingResult(
            regions=[region], engine="mock", image_dimensions=(800, 600)
        )

        diagram = engine.reconstruct_aon(shape_result, ocr_result=ocr_result, association_result=assoc_result)
        act = diagram.activities[0]
        assert act.activity_id == "K3"
        assert act.status.value == "review_required"
        assert act.needs_review is True
        assert any("No duration" in w for w in act.warnings)

    def test_rectangle_with_ocr_id_and_duration_is_confirmed(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1", shape_type=ShapeType.RECTANGLE, source_shape_id="s1"
        )
        shape_result = _make_shape_result([candidate])

        id_region = _make_region(
            text="M5", x=120, y=110, w=40, h=15,
            text_type=TextType.ACTIVITY_ID_CANDIDATE,
        )
        dur_region = _make_region(
            text="7", x=120, y=130, w=30, h=15,
            text_type=TextType.NUMERIC_CANDIDATE,
        )
        assoc1 = TextAssociation(
            text_region_id=id_region.region_id,
            candidate_target_id=candidate.node_id,
            target_type=AssociationTargetType.NODE,
            association_score=0.9,
            is_best_candidate=True,
        )
        assoc2 = TextAssociation(
            text_region_id=dur_region.region_id,
            candidate_target_id=candidate.node_id,
            target_type=AssociationTargetType.NODE,
            association_score=0.8,
            is_best_candidate=False,
        )
        assoc_result = _make_association_result([assoc1, assoc2])
        ocr_result = OCRProcessingResult(
            regions=[id_region, dur_region], engine="mock", image_dimensions=(800, 600)
        )

        diagram = engine.reconstruct_aon(shape_result, ocr_result=ocr_result, association_result=assoc_result)
        act = diagram.activities[0]
        assert act.activity_id == "M5"
        assert act.duration == 7.0
        assert act.status.value == "confirmed"
        assert act.needs_review is False

    def test_all_rectangles_produce_activities(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(node_id=f"c{i}", source_shape_id=f"s{i}", x=100 + i * 200, y=100)
            for i in range(5)
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        assert diagram.activity_count == 5
        for act in diagram.activities:
            assert act.activity_id.startswith("INFERRED_")

    def test_dependency_uses_node_id_mapping(self) -> None:
        engine = ReconstructionEngine()
        c1 = _make_candidate(node_id="c1", source_shape_id="s1", x=50, y=50, w=100, h=60)
        c2 = _make_candidate(node_id="c2", source_shape_id="s2", x=300, y=50, w=100, h=60)
        shape_result = _make_shape_result([c1, c2])

        arrow = _make_arrow(
            arrow_id="a1",
            start_x=150, start_y=80,
            end_x=300, end_y=80,
        )
        arrow_result = ShapeDetectionResult(
            candidate_nodes=[c1, c2],
            detected_shapes=[],
            image_dimensions=(800, 600),
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow],
            image_dimensions=(800, 600),
        )

        diagram = engine.reconstruct_aon(shape_result, arrow_result=arrow_result)
        resolved_deps = [d for d in diagram.dependencies if d.source_id and d.target_id]
        assert len(resolved_deps) >= 1
        src_ids = {d.source_id for d in resolved_deps}
        tgt_ids = {d.target_id for d in resolved_deps}
        act_ids = {a.activity_id for a in diagram.activities}
        assert src_ids.issubset(act_ids)
        assert tgt_ids.issubset(act_ids)

    def test_self_arrows_rejected(self) -> None:
        engine = ReconstructionEngine()
        c1 = _make_candidate(node_id="c1", source_shape_id="s1", x=50, y=50, w=100, h=60)
        shape_result = _make_shape_result([c1])

        arrow = _make_arrow(
            arrow_id="self1",
            start_x=100, start_y=80,
            end_x=120, end_y=80,
        )
        from pert_analyzer.cv.models import ArrowDetectionResult
        arrow_result = ArrowDetectionResult(
            arrows=[arrow],
            image_dimensions=(800, 600),
        )

        diagram = engine.reconstruct_aon(shape_result, arrow_result=arrow_result)
        self_arrows = [d for d in diagram.dependencies if d.source_id == d.target_id]
        assert len(self_arrows) == 0

    def test_activity_source_node_id_set(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="cnode_xyz", shape_type=ShapeType.RECTANGLE, source_shape_id="shape_abc"
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        act = diagram.activities[0]
        assert act.source_node_id == "cnode_xyz"
        assert act.source_shape_id == "shape_abc"

    def test_activity_status_and_id_source_fields(self) -> None:
        engine = ReconstructionEngine()
        candidate = _make_candidate(
            node_id="c1", shape_type=ShapeType.RECTANGLE, source_shape_id="s1"
        )
        shape_result = _make_shape_result([candidate])
        diagram = engine.reconstruct_aon(shape_result)
        act = diagram.activities[0]
        assert hasattr(act, 'status')
        assert hasattr(act, 'id_source')
        assert hasattr(act, 'needs_review')
        assert act.id_source.value == "spatial_inference"

    def test_inferred_activities_generate_ambiguity_issues(self) -> None:
        engine = ReconstructionEngine()
        candidates = [
            _make_candidate(node_id="c1", source_shape_id="s1"),
            _make_candidate(node_id="c2", source_shape_id="s2", x=300, y=100),
        ]
        shape_result = _make_shape_result(candidates)
        diagram = engine.reconstruct_aon(shape_result)
        inferred_issues = [
            a for a in diagram.ambiguities
            if a.ambiguity_type == AmbiguityType.MISSING_ACTIVITY_ID
        ]
        assert len(inferred_issues) == 2

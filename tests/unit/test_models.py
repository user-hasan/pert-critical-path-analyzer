"""
Tests for core data models.
"""

import pytest

from pert_analyzer.core.models import (
    Activity,
    ActivityAnalysis,
    AnalysisResult,
    Arrow,
    BoundingBox,
    Dependency,
    DependencyType,
    DetectedShape,
    DiagramAnalysis,
    DiagramType,
    GraphModel,
    Node,
    OCRResult,
    Point,
    Project,
    ValidationIssue,
    ValidationResult,
)


class TestPoint:
    """Tests for the Point dataclass."""

    def test_creation(self):
        p = Point(10.0, 20.0)
        assert p.x == 10.0
        assert p.y == 20.0

    def test_distance_to(self):
        p1 = Point(0, 0)
        p2 = Point(3, 4)
        assert p1.distance_to(p2) == 5.0

    def test_distance_to_same_point(self):
        p1 = Point(5, 5)
        assert p1.distance_to(p1) == 0.0

    def test_to_tuple(self):
        p = Point(1.5, 2.5)
        assert p.to_tuple() == (1.5, 2.5)


class TestBoundingBox:
    """Tests for the BoundingBox dataclass."""

    def test_creation(self):
        bb = BoundingBox(10, 20, 100, 50)
        assert bb.x == 10
        assert bb.y == 20
        assert bb.width == 100
        assert bb.height == 50

    def test_center(self):
        bb = BoundingBox(0, 0, 100, 50)
        center = bb.center
        assert center.x == 50.0
        assert center.y == 25.0

    def test_area(self):
        bb = BoundingBox(0, 0, 10, 20)
        assert bb.area == 200

    def test_contains_point_inside(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert bb.contains_point(Point(50, 50))

    def test_contains_point_outside(self):
        bb = BoundingBox(0, 0, 100, 100)
        assert not bb.contains_point(Point(150, 50))

    def test_contains_point_with_margin(self):
        bb = BoundingBox(10, 10, 100, 100)
        assert bb.contains_point(Point(5, 5), margin=10)
        assert not bb.contains_point(Point(5, 5), margin=0)

    def test_overlaps(self):
        bb1 = BoundingBox(0, 0, 100, 100)
        bb2 = BoundingBox(50, 50, 100, 100)
        assert bb1.overlaps(bb2)

    def test_no_overlap(self):
        bb1 = BoundingBox(0, 0, 100, 100)
        bb2 = BoundingBox(200, 200, 100, 100)
        assert not bb1.overlaps(bb2)


class TestArrow:
    """Tests for the Arrow dataclass."""

    def test_creation(self):
        arrow = Arrow(
            start=Point(0, 0),
            end=Point(100, 50),
            has_arrowhead=True,
            bounding_box=BoundingBox(0, 0, 100, 50),
            confidence=0.9,
        )
        assert arrow.has_arrowhead
        assert arrow.confidence == 0.9

    def test_length(self):
        arrow = Arrow(
            start=Point(0, 0),
            end=Point(3, 4),
            has_arrowhead=True,
            bounding_box=BoundingBox(0, 0, 3, 4),
            confidence=0.9,
        )
        assert arrow.length == 5.0

    def test_midpoint(self):
        arrow = Arrow(
            start=Point(0, 0),
            end=Point(10, 20),
            has_arrowhead=True,
            bounding_box=BoundingBox(0, 0, 10, 20),
            confidence=0.9,
        )
        mid = arrow.midpoint
        assert mid.x == 5.0
        assert mid.y == 10.0


class TestOCRResult:
    """Tests for the OCRResult dataclass."""

    def test_creation(self):
        result = OCRResult(
            text="Hello",
            bounding_box=BoundingBox(0, 0, 100, 30),
            confidence=0.95,
        )
        assert result.text == "Hello"
        assert not result.is_numeric

    def test_numeric_detection(self):
        result = OCRResult(
            text="42",
            bounding_box=BoundingBox(0, 0, 50, 30),
            confidence=0.9,
        )
        assert result.is_numeric
        assert result.parsed_value == 42.0

    def test_float_detection(self):
        result = OCRResult(
            text="3.14",
            bounding_box=BoundingBox(0, 0, 50, 30),
            confidence=0.9,
        )
        assert result.is_numeric
        assert result.parsed_value == 3.14


class TestActivity:
    """Tests for the Activity dataclass."""

    def test_creation(self):
        activity = Activity(
            activity_id="A",
            name="Design",
            duration=5.0,
        )
        assert activity.activity_id == "A"
        assert activity.duration == 5.0

    def test_pert_expected_time(self):
        activity = Activity(
            activity_id="A",
            duration=5.0,
            optimistic_time=3.0,
            most_likely_time=5.0,
            pessimistic_time=8.0,
        )
        expected = (3 + 4 * 5 + 8) / 6
        assert activity.expected_time == pytest.approx(expected)

    def test_pert_variance(self):
        activity = Activity(
            activity_id="A",
            duration=5.0,
            optimistic_time=3.0,
            most_likely_time=5.0,
            pessimistic_time=8.0,
        )
        expected = ((8 - 3) / 6) ** 2
        assert activity.variance == pytest.approx(expected)

    def test_pert_expected_time_none_when_incomplete(self):
        activity = Activity(
            activity_id="A",
            duration=5.0,
            optimistic_time=3.0,
        )
        assert activity.expected_time is None


class TestGraphModel:
    """Tests for the GraphModel dataclass."""

    def test_creation(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        assert graph.diagram_type == DiagramType.AON
        assert graph.node_count == 0

    def test_node_count(self):
        graph = GraphModel()
        graph.nodes["n1"] = Node(node_id="n1")
        graph.nodes["n2"] = Node(node_id="n2")
        assert graph.node_count == 2

    def test_get_node(self):
        graph = GraphModel()
        node = Node(node_id="n1", label="Start")
        graph.nodes["n1"] = node
        assert graph.get_node("n1") == node
        assert graph.get_node("nonexistent") is None

    def test_get_successors(self):
        graph = GraphModel()
        graph.dependencies.append(Dependency(source="n1", target="n2"))
        graph.dependencies.append(Dependency(source="n1", target="n3"))
        successors = graph.get_successors("n1")
        assert "n2" in successors
        assert "n3" in successors


class TestAnalysisResult:
    """Tests for the AnalysisResult dataclass."""

    def test_creation(self):
        result = AnalysisResult(project_duration=10.0)
        assert result.project_duration == 10.0
        assert result.analysis_type == "CPM"

    def test_critical_activity_count(self):
        result = AnalysisResult()
        result.activity_analyses["A"] = ActivityAnalysis(
            activity_id="A", is_critical=True
        )
        result.activity_analyses["B"] = ActivityAnalysis(
            activity_id="B", is_critical=False
        )
        assert result.critical_activity_count == 1
        assert result.non_critical_activity_count == 1


class TestValidationResult:
    """Tests for the ValidationResult dataclass."""

    def test_creation(self):
        result = ValidationResult()
        assert result.is_valid
        assert result.error_count == 0

    def test_add_error(self):
        result = ValidationResult()
        issue = ValidationIssue(severity="error", message="Test error")
        result.add_issue(issue)
        assert not result.is_valid
        assert result.error_count == 1

    def test_add_warning(self):
        result = ValidationResult()
        issue = ValidationIssue(severity="warning", message="Test warning")
        result.add_issue(issue)
        assert result.is_valid
        assert result.warning_count == 1


class TestProject:
    """Tests for the Project dataclass."""

    def test_creation(self):
        project = Project(name="Test Project")
        assert project.name == "Test Project"
        assert project.status == "draft"

    def test_update_modified(self):
        project = Project(name="Test")
        original = project.modified_at
        project.update_modified()
        assert project.modified_at >= original

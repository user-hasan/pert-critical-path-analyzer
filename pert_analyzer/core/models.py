"""
Core data models for the PERT & Critical Path Analyzer.

All data structures are defined as typed Python dataclasses.
These are pure data containers with no business logic.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


def _generate_id(prefix: str = "") -> str:
    """Generate a unique identifier."""
    short_uuid = uuid.uuid4().hex[:8]
    return f"{prefix}_{short_uuid}" if prefix else short_uuid


# =============================================================================
# Enums
# =============================================================================


class DiagramType(Enum):
    """Type of network diagram representation."""

    AON = "AON"  # Activity-on-Node
    AOA = "AOA"  # Activity-on-Arrow
    UNKNOWN = "UNKNOWN"


class DependencyType(Enum):
    """Type of precedence dependency."""

    FINISH_TO_START = "finish_to_start"
    START_TO_START = "start_to_start"
    FINISH_TO_FINISH = "finish_to_finish"
    START_TO_FINISH = "start_to_finish"


# =============================================================================
# Geometry Models
# =============================================================================


@dataclass
class Point:
    """A 2D point in image coordinates."""

    x: float
    y: float

    def distance_to(self, other: Point) -> float:
        """Calculate Euclidean distance to another point."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

    def to_tuple(self) -> Tuple[float, float]:
        """Convert to tuple (x, y)."""
        return (self.x, self.y)


@dataclass
class BoundingBox:
    """Axis-aligned bounding box defined by top-left corner and dimensions."""

    x: float
    y: float
    width: float
    height: float

    @property
    def center(self) -> Point:
        """Get the center point of the bounding box."""
        return Point(self.x + self.width / 2, self.y + self.height / 2)

    @property
    def top_left(self) -> Point:
        """Get the top-left corner."""
        return Point(self.x, self.y)

    @property
    def top_right(self) -> Point:
        """Get the top-right corner."""
        return Point(self.x + self.width, self.y)

    @property
    def bottom_left(self) -> Point:
        """Get the bottom-left corner."""
        return Point(self.x, self.y + self.height)

    @property
    def bottom_right(self) -> Point:
        """Get the bottom-right corner."""
        return Point(self.x + self.width, self.y + self.height)

    @property
    def area(self) -> float:
        """Calculate the area of the bounding box."""
        return self.width * self.height

    def contains_point(self, point: Point, margin: float = 0) -> bool:
        """Check if a point is inside the bounding box."""
        return (
            self.x - margin <= point.x <= self.x + self.width + margin
            and self.y - margin <= point.y <= self.y + self.height + margin
        )

    def overlaps(self, other: BoundingBox, threshold: float = 0.0) -> bool:
        """Check if two bounding boxes overlap."""
        x_overlap = max(
            0,
            min(self.x + self.width, other.x + other.width)
            - max(self.x, other.x),
        )
        y_overlap = max(
            0,
            min(self.y + self.height, other.y + other.height)
            - max(self.y, other.y),
        )
        overlap_area = x_overlap * y_overlap
        if overlap_area == 0:
            return False
        min_area = min(self.area, other.area)
        if min_area == 0:
            return False
        return (overlap_area / min_area) >= threshold


@dataclass
class Arrow:
    """A detected arrow or line segment in the image."""

    start: Point
    end: Point
    has_arrowhead: bool
    bounding_box: BoundingBox
    confidence: float
    is_dummy: bool = False
    line_thickness: int = 1
    color: Tuple[int, int, int] = (0, 0, 0)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def length(self) -> float:
        """Calculate the length of the arrow."""
        return self.start.distance_to(self.end)

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of the arrow."""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
        )


# =============================================================================
# Visual Element Models
# =============================================================================


@dataclass
class DetectedShape:
    """A shape detected in the image (circle, rectangle, polygon, etc.)."""

    shape_id: str = field(default_factory=lambda: _generate_id("shape"))
    shape_type: str = "unknown"  # "circle", "rectangle", "polygon", "ellipse"
    contour: Any = None  # numpy array of contour points
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    centroid: Point = field(default_factory=lambda: Point(0, 0))
    confidence: float = 0.0
    color: Tuple[int, int, int] = (255, 255, 255)
    area: float = 0.0
    is_filled: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OCRResult:
    """A text region detected and recognized by the OCR engine."""

    text: str
    bounding_box: BoundingBox
    confidence: float
    language: str = "en"
    raw_data: Dict[str, Any] = field(default_factory=dict)
    word_confidences: List[float] = field(default_factory=list)
    is_numeric: bool = False
    parsed_value: Optional[float] = None

    def __post_init__(self):
        """Parse numeric values after initialization."""
        if self.text and not self.parsed_value:
            try:
                self.parsed_value = float(self.text)
                self.is_numeric = True
            except ValueError:
                self.is_numeric = False


# =============================================================================
# Semantic Models
# =============================================================================


@dataclass
class Node:
    """A node in the network diagram (event or activity depending on diagram type)."""

    node_id: str = field(default_factory=lambda: _generate_id("node"))
    label: str = ""
    position: Point = field(default_factory=lambda: Point(0, 0))
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    node_type: str = "unknown"  # "event", "activity", "start", "end"
    confidence: float = 0.0
    shape: Optional[DetectedShape] = None
    associated_texts: List[OCRResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """An activity in the project network."""

    activity_id: str = field(default_factory=lambda: _generate_id("act"))
    name: str = ""
    duration: float = 0.0
    optimistic_time: Optional[float] = None
    most_likely_time: Optional[float] = None
    pessimistic_time: Optional[float] = None
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    is_dummy: bool = False
    associated_texts: List[OCRResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def expected_time(self) -> Optional[float]:
        """Calculate PERT expected time if all PERT values are available."""
        if (
            self.optimistic_time is not None
            and self.most_likely_time is not None
            and self.pessimistic_time is not None
        ):
            return (
                self.optimistic_time
                + 4 * self.most_likely_time
                + self.pessimistic_time
            ) / 6
        return None

    @property
    def variance(self) -> Optional[float]:
        """Calculate PERT variance if all PERT values are available."""
        if (
            self.optimistic_time is not None
            and self.pessimistic_time is not None
        ):
            return ((self.pessimistic_time - self.optimistic_time) / 6) ** 2
        return None


@dataclass
class Dependency:
    """A dependency relationship between two nodes or activities."""

    source: str
    target: str
    dependency_id: str = field(default_factory=lambda: _generate_id("dep"))
    activity_id: Optional[str] = None
    dependency_type: str = "finish_to_start"
    confidence: float = 0.0
    arrow: Optional[Arrow] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Graph Model
# =============================================================================


@dataclass
class GraphModel:
    """
    The reconstructed network graph model.

    Supports both AON (Activity-on-Node) and AOA (Activity-on-Arrow)
    representations. In AON, activities are nodes and dependencies are
    edges. In AOA, events are nodes and activities are edges.

    The diagram_type field indicates which representation is currently
    stored in this graph.
    """

    diagram_type: DiagramType = DiagramType.UNKNOWN
    nodes: Dict[str, Node] = field(default_factory=dict)
    activities: Dict[str, Activity] = field(default_factory=dict)
    dependencies: List[Dependency] = field(default_factory=list)
    source_node_id: Optional[str] = None
    sink_node_id: Optional[str] = None
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def node_count(self) -> int:
        """Get the number of nodes."""
        return len(self.nodes)

    @property
    def activity_count(self) -> int:
        """Get the number of activities."""
        return len(self.activities)

    @property
    def dependency_count(self) -> int:
        """Get the number of dependencies."""
        return len(self.dependencies)

    def get_node(self, node_id: str) -> Optional[Node]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_activity(self, activity_id: str) -> Optional[Activity]:
        """Get an activity by ID."""
        return self.activities.get(activity_id)

    def get_successors(self, node_id: str) -> List[str]:
        """Get all successor node IDs."""
        return [dep.target for dep in self.dependencies if dep.source == node_id]

    def get_predecessors(self, node_id: str) -> List[str]:
        """Get all predecessor node IDs."""
        return [dep.source for dep in self.dependencies if dep.target == node_id]

    def add_activity(self, activity: Activity) -> None:
        """
        Add an activity to the graph.

        Raises:
            ValueError: If an activity with the same ID already exists.
        """
        if activity.activity_id in self.activities:
            raise ValueError(
                f"Activity with ID '{activity.activity_id}' already exists"
            )
        self.activities[activity.activity_id] = activity

    def remove_activity(self, activity_id: str) -> Optional[Activity]:
        """
        Remove an activity and all its associated dependencies.

        Returns:
            The removed activity, or None if not found.
        """
        activity = self.activities.pop(activity_id, None)
        if activity is not None:
            self.dependencies = [
                dep for dep in self.dependencies
                if dep.source != activity_id and dep.target != activity_id
            ]
        return activity

    def add_node(self, node: Node) -> None:
        """
        Add a node/event to the graph.

        Raises:
            ValueError: If a node with the same ID already exists.
        """
        if node.node_id in self.nodes:
            raise ValueError(f"Node with ID '{node.node_id}' already exists")
        self.nodes[node.node_id] = node

    def remove_node(self, node_id: str) -> Optional[Node]:
        """
        Remove a node and all its associated dependencies.

        Returns:
            The removed node, or None if not found.
        """
        node = self.nodes.pop(node_id, None)
        if node is not None:
            self.dependencies = [
                dep for dep in self.dependencies
                if dep.source != node_id and dep.target != node_id
            ]
        return node

    def add_dependency(self, dependency: Dependency) -> None:
        """
        Add a dependency to the graph.

        Raises:
            ValueError: If a duplicate dependency exists (same source and target).
        """
        for dep in self.dependencies:
            if dep.source == dependency.source and dep.target == dependency.target:
                raise ValueError(
                    f"Dependency from '{dependency.source}' to '{dependency.target}' "
                    f"already exists"
                )
        self.dependencies.append(dependency)

    def remove_dependency(self, source: str, target: str) -> bool:
        """
        Remove a dependency by source and target.

        Returns:
            True if the dependency was found and removed, False otherwise.
        """
        original_count = len(self.dependencies)
        self.dependencies = [
            dep for dep in self.dependencies
            if not (dep.source == source and dep.target == target)
        ]
        return len(self.dependencies) < original_count

    def get_dependency(self, source: str, target: str) -> Optional[Dependency]:
        """Get a dependency by source and target."""
        for dep in self.dependencies:
            if dep.source == source and dep.target == target:
                return dep
        return None

    def get_all_dependencies_for(self, activity_id: str) -> List[Dependency]:
        """Get all dependencies where the activity is source or target."""
        return [
            dep for dep in self.dependencies
            if dep.source == activity_id or dep.target == activity_id
        ]

    def has_activity(self, activity_id: str) -> bool:
        """Check if an activity exists in the graph."""
        return activity_id in self.activities

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return node_id in self.nodes

    def activity_ids(self) -> List[str]:
        """Get all activity IDs."""
        return list(self.activities.keys())

    def node_ids(self) -> List[str]:
        """Get all node IDs."""
        return list(self.nodes.keys())

    def clear(self) -> None:
        """Clear all activities, nodes, and dependencies."""
        self.activities.clear()
        self.nodes.clear()
        self.dependencies.clear()
        self.source_node_id = None
        self.sink_node_id = None


# =============================================================================
# Analysis Models
# =============================================================================


@dataclass
class ActivityAnalysis:
    """Analysis results for a single activity."""

    activity_id: str
    early_start: float = 0.0
    early_finish: float = 0.0
    late_start: float = 0.0
    late_finish: float = 0.0
    total_float: float = 0.0
    free_float: float = 0.0
    is_critical: bool = False

    @property
    def duration(self) -> float:
        """Calculate duration from early start and finish."""
        return self.early_finish - self.early_start


@dataclass
class AnalysisResult:
    """Complete analysis results for the project."""

    project_duration: float = 0.0
    critical_path: List[str] = field(default_factory=list)
    critical_paths: List[List[str]] = field(default_factory=list)
    activity_analyses: Dict[str, ActivityAnalysis] = field(default_factory=dict)
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    analysis_type: str = "CPM"  # "CPM" or "PERT"
    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def critical_activity_count(self) -> int:
        """Get the number of critical activities."""
        return sum(
            1 for aa in self.activity_analyses.values() if aa.is_critical
        )

    @property
    def non_critical_activity_count(self) -> int:
        """Get the number of non-critical activities."""
        return sum(
            1 for aa in self.activity_analyses.values() if not aa.is_critical
        )


# =============================================================================
# Validation Models
# =============================================================================


@dataclass
class ValidationIssue:
    """A single validation issue."""

    issue_id: str = field(default_factory=lambda: _generate_id("issue"))
    severity: str = "info"  # "error", "warning", "info"
    category: str = "general"  # "structural", "data", "confidence", "general"
    message: str = ""
    element_id: Optional[str] = None
    element_type: Optional[str] = None  # "node", "activity", "dependency"
    suggestion: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Complete validation results."""

    is_valid: bool = True
    issues: List[ValidationIssue] = field(default_factory=list)
    confidence_score: float = 1.0
    summary: str = ""
    checked_at: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def error_count(self) -> int:
        """Get the number of errors."""
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        """Get the number of warnings."""
        return sum(1 for i in self.issues if i.severity == "warning")

    @property
    def info_count(self) -> int:
        """Get the number of info messages."""
        return sum(1 for i in self.issues if i.severity == "info")

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue."""
        self.issues.append(issue)
        if issue.severity == "error":
            self.is_valid = False

    def has_errors(self) -> bool:
        """Check if there are any errors."""
        return self.error_count > 0


# =============================================================================
# Pipeline Models
# =============================================================================


@dataclass
class DiagramAnalysis:
    """Results from analyzing a diagram image."""

    diagram_type: DiagramType = DiagramType.UNKNOWN
    detection_confidence: float = 0.0
    detected_shapes: List[DetectedShape] = field(default_factory=list)
    detected_arrows: List[Arrow] = field(default_factory=list)
    ocr_results: List[OCRResult] = field(default_factory=list)
    preprocessed_image: Any = None  # numpy array (not serialized)
    original_image: Any = None  # numpy array (not serialized)
    image_path: Optional[str] = None
    processing_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def shape_count(self) -> int:
        """Get the number of detected shapes."""
        return len(self.detected_shapes)

    @property
    def arrow_count(self) -> int:
        """Get the number of detected arrows."""
        return len(self.detected_arrows)

    @property
    def ocr_result_count(self) -> int:
        """Get the number of OCR results."""
        return len(self.ocr_results)


@dataclass
class Project:
    """A complete project containing all analysis data."""

    project_id: str = field(default_factory=lambda: _generate_id("proj"))
    name: str = "Untitled Project"
    description: str = ""
    diagram: Optional[DiagramAnalysis] = None
    graph: Optional[GraphModel] = None
    analysis: Optional[AnalysisResult] = None
    validation: Optional[ValidationResult] = None
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    modified_at: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "draft"  # "draft", "analyzing", "review", "complete", "error"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def update_modified(self) -> None:
        """Update the modification timestamp."""
        self.modified_at = datetime.now().isoformat()

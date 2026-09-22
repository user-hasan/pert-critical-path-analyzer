"""
Data models for image preprocessing.

These models represent input images, preprocessing configurations,
preprocessing results, and coordinate transformations for the CV pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pert_analyzer.core.models import BoundingBox, Point, _generate_id


class BlurMethod(Enum):
    """Blur/noise reduction methods."""

    NONE = "none"
    GAUSSIAN = "gaussian"
    MEDIAN = "median"
    BILATERAL = "bilateral"


class ContrastMethod(Enum):
    """Contrast enhancement methods."""

    NONE = "none"
    HISTOGRAM_EQUALIZATION = "histogram_equalization"
    CLAHE = "clahe"


class ThresholdMethod(Enum):
    """Thresholding methods."""

    NONE = "none"
    GLOBAL = "global"
    OTSU = "otsu"
    ADAPTIVE_MEAN = "adaptive_mean"
    ADAPTIVE_GAUSSIAN = "adaptive_gaussian"


class EdgeMethod(Enum):
    """Edge detection methods."""

    NONE = "none"
    CANNY = "canny"
    SOBEL = "sobel"


class ResizeStrategy(Enum):
    """Image resize strategies."""

    NONE = "none"
    MAX_DIMENSION = "max_dimension"
    SCALE_FACTOR = "scale_factor"
    FIXED_SIZE = "fixed_size"


@dataclass
class PreprocessingConfig:
    """
    Configuration for the image preprocessing pipeline.

    Controls all aspects of preprocessing behavior with sensible defaults
    suitable for typical project network diagrams.
    """

    # Resize settings
    resize_strategy: ResizeStrategy = ResizeStrategy.MAX_DIMENSION
    max_width: int = 2000
    max_height: int = 2000
    min_width: int = 100
    min_height: int = 100
    scale_factor: float = 1.0
    target_width: int = 1000
    target_height: int = 800

    # Grayscale settings
    convert_to_grayscale: bool = True

    # Noise reduction settings
    blur_method: BlurMethod = BlurMethod.GAUSSIAN
    blur_kernel_size: int = 5
    median_blur_kernel_size: int = 5
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Contrast enhancement settings
    contrast_method: ContrastMethod = ContrastMethod.CLAHE
    clahe_clip_limit: float = 2.0
    clahe_tile_size: Tuple[int, int] = (8, 8)

    # Thresholding settings
    threshold_method: ThresholdMethod = ThresholdMethod.ADAPTIVE_GAUSSIAN
    global_threshold_value: int = 128
    adaptive_block_size: int = 11
    adaptive_c: int = 2

    # Edge detection settings
    edge_method: EdgeMethod = EdgeMethod.CANNY
    canny_low_threshold: int = 50
    canny_high_threshold: int = 150
    canny_aperture_size: int = 3
    sobel_kernel_size: int = 3

    # Deskew settings
    enable_deskew: bool = True
    deskew_max_angle: float = 15.0

    # Output control
    preserve_original: bool = True
    return_multiple_representations: bool = True

    def validate(self) -> List[str]:
        """
        Validate the configuration and return a list of issues.

        Returns:
            List of validation error messages. Empty if valid.
        """
        issues = []

        if self.max_width < self.min_width:
            issues.append(
                f"max_width ({self.max_width}) must be >= min_width ({self.min_width})"
            )
        if self.max_height < self.min_height:
            issues.append(
                f"max_height ({self.max_height}) must be >= min_height ({self.min_height})"
            )
        if self.scale_factor <= 0:
            issues.append(f"scale_factor ({self.scale_factor}) must be > 0")
        if self.blur_kernel_size < 1:
            issues.append(f"blur_kernel_size ({self.blur_kernel_size}) must be >= 1")
        if self.blur_kernel_size % 2 == 0:
            issues.append(f"blur_kernel_size ({self.blur_kernel_size}) must be odd")
        if self.median_blur_kernel_size < 1:
            issues.append(
                f"median_blur_kernel_size ({self.median_blur_kernel_size}) must be >= 1"
            )
        if self.median_blur_kernel_size % 2 == 0:
            issues.append(
                f"median_blur_kernel_size ({self.median_blur_kernel_size}) must be odd"
            )
        if self.adaptive_block_size < 3:
            issues.append(
                f"adaptive_block_size ({self.adaptive_block_size}) must be >= 3"
            )
        if self.adaptive_block_size % 2 == 0:
            issues.append(
                f"adaptive_block_size ({self.adaptive_block_size}) must be odd"
            )
        if self.canny_low_threshold < 0:
            issues.append(
                f"canny_low_threshold ({self.canny_low_threshold}) must be >= 0"
            )
        if self.canny_high_threshold < self.canny_low_threshold:
            issues.append(
                f"canny_high_threshold ({self.canny_high_threshold}) must be >= "
                f"canny_low_threshold ({self.canny_low_threshold})"
            )
        if self.clahe_clip_limit <= 0:
            issues.append(
                f"clahe_clip_limit ({self.clahe_clip_limit}) must be > 0"
            )
        if self.deskew_max_angle <= 0:
            issues.append(
                f"deskew_max_angle ({self.deskew_max_angle}) must be > 0"
            )

        return issues


@dataclass
class CoordinateMapping:
    """
    Maps coordinates between original and processed images.

    This is essential for converting detected shapes/arrows/OCR back
    to original image coordinates after preprocessing.
    """

    original_width: int
    original_height: int
    processed_width: int
    processed_height: int
    scale_x: float = 1.0
    scale_y: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    rotation_angle: float = 0.0

    def __post_init__(self):
        """Compute scale factors if not explicitly set."""
        if self.original_width > 0 and self.original_height > 0:
            self.scale_x = self.processed_width / self.original_width
            self.scale_y = self.processed_height / self.original_height

    def to_original(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert a coordinate from processed image back to original.

        Args:
            x: X coordinate in processed image.
            y: Y coordinate in processed image.

        Returns:
            (x, y) in original image coordinates.
        """
        orig_x = (x - self.offset_x) / self.scale_x
        orig_y = (y - self.offset_y) / self.scale_y
        return (orig_x, orig_y)

    def to_original_point(self, x: float, y: float) -> Tuple[float, float]:
        """
        Alias for to_original (for clarity in different contexts).

        Converts a point from processed to original coordinates.
        """
        return self.to_original(x, y)

    def to_processed(self, x: float, y: float) -> Tuple[float, float]:
        """
        Convert a coordinate from original image to processed.

        Args:
            x: X coordinate in original image.
            y: Y coordinate in original image.

        Returns:
            (x, y) in processed image coordinates.
        """
        proc_x = x * self.scale_x + self.offset_x
        proc_y = y * self.scale_y + self.offset_y
        return (proc_x, proc_y)

    def scale_bbox(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
    ) -> Tuple[float, float, float, float]:
        """
        Scale a bounding box from processed to original coordinates.

        Args:
            x, y: Top-left corner in processed coordinates.
            width, height: Dimensions in processed coordinates.

        Returns:
            (x, y, width, height) in original coordinates.
        """
        orig_x, orig_y = self.to_original(x, y)
        orig_w = width / self.scale_x
        orig_h = height / self.scale_y
        return (orig_x, orig_y, orig_w, orig_h)


@dataclass
class PreprocessingResult:
    """
    Result of image preprocessing containing all representations.

    The preprocessing pipeline produces multiple image representations
    to support different downstream tasks (shape detection, OCR, arrow
    detection). Each representation is optimized for its use case.

    Representations:
        original: The unmodified input image.
        normalized: Color-normalized and resized image.
        grayscale: Single-channel grayscale image.
        denoised: Noise-reduced grayscale image.
        contrast_enhanced: Contrast-enhanced image.
        binary: Global/Otsu thresholded binary image.
        adaptive_binary: Adaptive thresholded binary image.
        edges: Edge-detected image (e.g., Canny).
    """

    # Image representations (numpy arrays, not serialized)
    original: Optional[np.ndarray] = None
    normalized: Optional[np.ndarray] = None
    grayscale: Optional[np.ndarray] = None
    denoised: Optional[np.ndarray] = None
    contrast_enhanced: Optional[np.ndarray] = None
    binary: Optional[np.ndarray] = None
    adaptive_binary: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None

    # Coordinate mapping for coordinate transformation
    coordinate_mapping: Optional[CoordinateMapping] = None

    # Metadata
    original_dimensions: Tuple[int, int] = (0, 0)
    processed_dimensions: Tuple[int, int] = (0, 0)
    scale_factor: float = 1.0
    was_grayscale_input: bool = False
    rotation_angle: float = 0.0
    was_deskewed: bool = False

    # Processing info
    processing_config: Optional[PreprocessingConfig] = None
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0

    @property
    def has_original(self) -> bool:
        """Check if original image is available."""
        return self.original is not None

    @property
    def has_grayscale(self) -> bool:
        """Check if grayscale representation is available."""
        return self.grayscale is not None

    @property
    def has_binary(self) -> bool:
        """Check if binary representation is available."""
        return self.binary is not None

    @property
    def has_adaptive_binary(self) -> bool:
        """Check if adaptive binary representation is available."""
        return self.adaptive_binary is not None

    @property
    def has_edges(self) -> bool:
        """Check if edge representation is available."""
        return self.edges is not None

    def get_representation(self, name: str) -> Optional[np.ndarray]:
        """
        Get a specific image representation by name.

        Args:
            name: One of: 'original', 'normalized', 'grayscale', 'denoised',
                  'contrast_enhanced', 'binary', 'adaptive_binary', 'edges'.

        Returns:
            The requested image representation, or None if not available.
        """
        representations = {
            "original": self.original,
            "normalized": self.normalized,
            "grayscale": self.grayscale,
            "denoised": self.denoised,
            "contrast_enhanced": self.contrast_enhanced,
            "binary": self.binary,
            "adaptive_binary": self.adaptive_binary,
            "edges": self.edges,
        }
        return representations.get(name)

    def get_available_representations(self) -> List[str]:
        """
        Get names of all available image representations.

        Returns:
            List of available representation names.
        """
        available = []
        for name in [
            "original",
            "normalized",
            "grayscale",
            "denoised",
            "contrast_enhanced",
            "binary",
            "adaptive_binary",
            "edges",
        ]:
            if self.get_representation(name) is not None:
                available.append(name)
        return available

    def to_original_coordinates(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """
        Convert processed coordinates to original image coordinates.

        Args:
            x: X coordinate in processed image.
            y: Y coordinate in processed image.

        Returns:
            (x, y) in original image coordinates.
        """
        if self.coordinate_mapping is None:
            return (x, y)
        return self.coordinate_mapping.to_original(x, y)

    def to_processed_coordinates(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """
        Convert original coordinates to processed image coordinates.

        Args:
            x: X coordinate in original image.
            y: Y coordinate in original image.

        Returns:
            (x, y) in processed image coordinates.
        """
        if self.coordinate_mapping is None:
            return (x, y)
        return self.coordinate_mapping.to_processed(x, y)


# =============================================================================
# Shape Detection Models
# =============================================================================


class ShapeType(Enum):
    """Geometric shape types for diagram node detection."""

    RECTANGLE = "rectangle"
    SQUARE = "square"
    CIRCLE = "circle"
    ELLIPSE = "ellipse"
    POLYGON = "polygon"
    UNKNOWN = "unknown"


@dataclass
class ShapeDetectionConfig:
    """
    Configuration for shape detection.

    Controls contour analysis, filtering, and confidence thresholds.
    """

    # Rectangle detection
    min_rect_area: int = 500
    max_rect_area: int = 200000
    min_rect_side: int = 20
    rect_approx_epsilon: float = 0.04
    min_rectangularity: float = 0.7
    rect_aspect_ratio_range: Tuple[float, float] = (0.1, 10.0)

    # Circle detection
    min_circle_area: int = 200
    max_circle_area: int = 150000
    min_circularity: float = 0.5
    hough_dp: float = 1.2
    hough_min_dist: int = 30
    hough_param1: int = 100
    hough_param2: int = 30
    hough_min_radius: int = 10
    hough_max_radius: int = 200

    # Polygon detection
    min_polygon_area: int = 500
    polygon_min_vertices: int = 5
    polygon_max_vertices: int = 12

    # General filtering
    min_shape_area: int = 200
    max_shape_area: int = 300000
    min_width: int = 15
    min_height: int = 15
    border_margin: int = 5
    overlap_threshold: float = 0.5

    # Confidence thresholds
    min_confidence: float = 0.1
    high_confidence: float = 0.7

    # Duplicate suppression
    enable_duplicate_suppression: bool = True
    duplicate_distance_threshold: float = 20.0

    def validate(self) -> List[str]:
        """Validate configuration and return issues."""
        issues = []
        if self.min_rect_area < 0:
            issues.append(f"min_rect_area must be >= 0, got {self.min_rect_area}")
        if self.min_circularity < 0 or self.min_circularity > 1:
            issues.append(f"min_circularity must be in [0, 1], got {self.min_circularity}")
        if self.min_shape_area < 0:
            issues.append(f"min_shape_area must be >= 0, got {self.min_shape_area}")
        if self.rect_aspect_ratio_range[0] <= 0:
            issues.append("rect_aspect_ratio_range[0] must be > 0")
        return issues


@dataclass
class CandidateNode:
    """
    A candidate diagram node interpreted from a detected shape.

    This is distinct from DetectedShape — it represents a shape that
    has been interpreted as a potential diagram node (activity or event).
    """

    node_id: str = field(default_factory=lambda: _generate_id("cnode"))
    shape_type: ShapeType = ShapeType.UNKNOWN
    source_shape_id: str = ""
    label: str = ""
    position: Point = field(default_factory=lambda: Point(0, 0))
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    confidence: float = 0.0
    node_role: str = "unknown"  # "activity", "event", "unknown"
    area: float = 0.0
    perimeter: float = 0.0
    aspect_ratio: float = 1.0
    nearest_neighbors: List[str] = field(default_factory=list)
    distance_to_border: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShapeDetectionResult:
    """
    Result of shape detection containing all detected shapes and candidates.

    Each detected shape preserves the raw geometric information.
    Candidate nodes provide interpreted results suitable for graph construction.
    """

    detected_shapes: List = field(default_factory=list)  # List[DetectedShape]
    candidate_nodes: List[CandidateNode] = field(default_factory=list)
    image_dimensions: Tuple[int, int] = (0, 0)
    coordinate_mapping: Optional[CoordinateMapping] = None
    source_representation: str = ""
    contours_analyzed: int = 0
    shapes_filtered: int = 0
    duplicates_suppressed: int = 0
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def rectangle_count(self) -> int:
        """Count detected rectangles."""
        return sum(
            1 for s in self.detected_shapes
            if hasattr(s, "shape_type") and s.shape_type in ("rectangle", "square")
        )

    @property
    def circle_count(self) -> int:
        """Count detected circles."""
        return sum(
            1 for s in self.detected_shapes
            if hasattr(s, "shape_type") and s.shape_type in ("circle", "ellipse")
        )

    @property
    def polygon_count(self) -> int:
        """Count detected polygons."""
        return sum(
            1 for s in self.detected_shapes
            if hasattr(s, "shape_type") and s.shape_type == "polygon"
        )

    @property
    def total_candidates(self) -> int:
        """Total number of candidate nodes."""
        return len(self.candidate_nodes)

    def get_candidates_by_type(
        self, shape_type: ShapeType
    ) -> List[CandidateNode]:
        """Get candidates of a specific shape type."""
        return [c for c in self.candidate_nodes if c.shape_type == shape_type]

    def to_original_coordinates(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """Convert detection coordinates to original image coordinates."""
        if self.coordinate_mapping is None:
            return (x, y)
        return self.coordinate_mapping.to_original(x, y)


@dataclass
class DiagramClassificationResult:
    """
    Result of AON vs AOA diagram classification.

    Contains the predicted type, confidence, evidence, and warnings.
    """

    diagram_type: str = "UNKNOWN"  # "AON", "AOA", "UNKNOWN"
    confidence: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    alternative_type: Optional[str] = None
    alternative_confidence: float = 0.0
    warnings: List[str] = field(default_factory=list)
    rectangle_count: int = 0
    circle_count: int = 0
    polygon_count: int = 0

    def __post_init__(self):
        """Normalize diagram_type to uppercase."""
        if self.diagram_type:
            self.diagram_type = self.diagram_type.upper()


# =============================================================================
# Arrow Detection Models
# =============================================================================


class LineStyle(Enum):
    """Line style for detected lines."""

    SOLID = "solid"
    DASHED = "dashed"
    DOTTED = "dotted"
    UNKNOWN = "unknown"


class ArrowheadType(Enum):
    """Type of arrowhead detected."""

    NONE = "none"
    FILLED_TRIANGLE = "filled_triangle"
    OPEN_V = "open_v"
    SIMPLE_TWO_LINE = "simple_two_line"
    UNKNOWN = "unknown"


@dataclass
class ArrowDetectionConfig:
    """
    Configuration for arrow and line detection.

    Controls line detection, filtering, arrowhead detection,
    segment merging, and node association.
    """

    # Line detection (HoughLinesP)
    hough_rho: int = 1
    hough_theta_resolution: int = 180
    hough_threshold: int = 30
    hough_min_line_length: int = 15
    hough_max_line_gap: int = 15

    # Line filtering
    min_line_length: int = 10
    max_line_length: int = 2000
    min_line_angle_deg: float = 0.0
    max_line_angle_deg: float = 180.0
    filter_rectangle_borders: bool = True
    rectangle_border_tolerance: int = 15

    # Arrowhead detection
    arrowhead_detection_radius: int = 30
    arrowhead_min_triangularity: float = 0.3
    arrowhead_min_area: int = 30
    arrowhead_max_area: int = 5000
    arrowhead_min_confidence: float = 0.10

    # Segment merging
    merge_angle_tolerance_deg: float = 15.0
    merge_distance_tolerance: int = 20
    merge_collinearity_tolerance: float = 0.85

    # Logical-arrow consolidation (merge transitive closure)
    enable_transitive_segment_merge: bool = True

    # Post-assembly duplicate suppression (one visual arrow -> one logical arrow)
    dedup_angle_tolerance_deg: float = 8.0
    dedup_lateral_tolerance: int = 6
    dedup_projection_overlap: float = 0.5

    # AOA event-boundary validation (only when circle events dominate)
    enable_aoa_event_validation: bool = True
    aoa_min_event_circles: int = 2
    aoa_circle_dominance: float = 0.5
    event_boundary_tolerance: float = 20.0
    aoa_arrowhead_required_confidence: float = 0.20

    # Node association
    node_connection_tolerance: int = 30
    source_endpoint_tolerance: int = 40
    target_endpoint_tolerance: int = 40

    # Confidence thresholds
    min_line_confidence: float = 0.1
    min_arrow_confidence: float = 0.10

    def validate(self) -> List[str]:
        """Validate configuration and return issues."""
        issues = []
        if self.hough_min_line_length < 1:
            issues.append(f"hough_min_line_length must be >= 1, got {self.hough_min_line_length}")
        if self.hough_max_line_gap < 0:
            issues.append(f"hough_max_line_gap must be >= 0, got {self.hough_max_line_gap}")
        if self.min_line_length < 1:
            issues.append(f"min_line_length must be >= 1, got {self.min_line_length}")
        if self.arrowhead_detection_radius < 1:
            issues.append(f"arrowhead_detection_radius must be >= 1, got {self.arrowhead_detection_radius}")
        if self.merge_angle_tolerance_deg <= 0:
            issues.append(f"merge_angle_tolerance_deg must be > 0, got {self.merge_angle_tolerance_deg}")
        if self.dedup_angle_tolerance_deg <= 0:
            issues.append(f"dedup_angle_tolerance_deg must be > 0, got {self.dedup_angle_tolerance_deg}")
        if self.dedup_lateral_tolerance < 0:
            issues.append(f"dedup_lateral_tolerance must be >= 0, got {self.dedup_lateral_tolerance}")
        if not (0 < self.dedup_projection_overlap <= 1):
            issues.append(
                f"dedup_projection_overlap must be in (0,1], got {self.dedup_projection_overlap}"
            )
        if self.event_boundary_tolerance <= 0:
            issues.append(f"event_boundary_tolerance must be > 0, got {self.event_boundary_tolerance}")
        if self.aoa_min_event_circles < 1:
            issues.append(f"aoa_min_event_circles must be >= 1, got {self.aoa_min_event_circles}")
        if self.aoa_circle_dominance <= 0 or self.aoa_circle_dominance > 1:
            issues.append(f"aoa_circle_dominance must be in (0,1], got {self.aoa_circle_dominance}")
        if self.node_connection_tolerance < 1:
            issues.append(f"node_connection_tolerance must be >= 1, got {self.node_connection_tolerance}")
        if self.arrowhead_min_confidence < 0 or self.arrowhead_min_confidence > 1:
            issues.append(f"arrowhead_min_confidence must be in [0,1], got {self.arrowhead_min_confidence}")
        return issues


@dataclass
class DetectedLineSegment:
    """
    A raw line segment detected by Hough transform or contour analysis.

    This is an intermediate result before arrow assembly.
    """

    segment_id: str = field(default_factory=lambda: _generate_id("line"))
    start: Point = field(default_factory=lambda: Point(0, 0))
    end: Point = field(default_factory=lambda: Point(0, 0))
    angle_deg: float = 0.0
    length: float = 0.0
    line_thickness: int = 1
    confidence: float = 0.0
    line_style: LineStyle = LineStyle.UNKNOWN
    is_part_of_shape_border: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of the line segment."""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
        )

    @property
    def direction_vector(self) -> Tuple[float, float]:
        """Get the unit direction vector from start to end."""
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return (0.0, 0.0)
        return (dx / length, dy / length)


@dataclass
class ArrowheadEvidence:
    """
    Evidence for the presence and type of an arrowhead at a line endpoint.
    """

    endpoint: Point = field(default_factory=lambda: Point(0, 0))
    arrowhead_type: ArrowheadType = ArrowheadType.NONE
    confidence: float = 0.0
    triangularity: float = 0.0
    detected_contour: Any = None  # numpy contour if available
    angle_span_deg: float = 0.0
    size: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectedArrow:
    """
    A detected arrow with geometric and arrowhead evidence.

    This is the primary output of the arrow detection pipeline.
    It represents a visual arrow detected in the image, NOT a semantic
    dependency. Endpoint ordering is purely geometric (sorted by x, then y)
    and must NOT be interpreted as source → target.

    Semantic direction is determined later by DirectionResolver using
    arrowhead evidence and boundary geometry.
    """

    arrow_id: str = field(default_factory=lambda: _generate_id("arrow"))
    start: Point = field(default_factory=lambda: Point(0, 0))
    end: Point = field(default_factory=lambda: Point(0, 0))
    direction_vector: Tuple[float, float] = (0.0, 0.0)
    angle_deg: float = 0.0
    length: float = 0.0
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )

    # Arrowhead info — this is the ONLY reliable source of semantic direction
    arrowhead_point: Optional[Point] = None
    arrowhead_type: ArrowheadType = ArrowheadType.NONE
    arrowhead_confidence: float = 0.0
    direction_confidence: float = 0.0

    # Line style
    line_style: LineStyle = LineStyle.UNKNOWN
    line_thickness: int = 1

    # Overall confidence
    confidence: float = 0.0

    # Source segments (which raw line segments were merged)
    source_segment_ids: List[str] = field(default_factory=list)

    # Evidence
    evidence: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize endpoint ordering to be purely geometric.

        Sort by x-coordinate, then y-coordinate. This removes any
        semantic source/target implication from Hough endpoint ordering.
        The arrowhead_point is preserved independently.
        """
        if (self.start.x > self.end.x or
                (self.start.x == self.end.x and self.start.y > self.end.y)):
            self.start, self.end = self.end, self.start

        # Always recompute direction vector from normalized endpoints
        dx = self.end.x - self.start.x
        dy = self.end.y - self.start.y
        dl = (dx**2 + dy**2) ** 0.5
        if dl > 0:
            self.direction_vector = (dx / dl, dy / dl)
        else:
            self.direction_vector = (0.0, 0.0)
        self.angle_deg = math.degrees(math.atan2(dy, dx)) % 360

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of the arrow."""
        return Point(
            (self.start.x + self.end.x) / 2,
            (self.start.y + self.end.y) / 2,
        )


@dataclass
class ArrowDetectionResult:
    """
    Complete result of arrow detection on an image.

    Contains all detected arrows, intermediate line segments,
    rejected candidates, warnings, and metadata.
    """

    arrows: List[DetectedArrow] = field(default_factory=list)
    raw_line_segments: List[DetectedLineSegment] = field(default_factory=list)
    rejected_candidates: List[DetectedLineSegment] = field(default_factory=list)
    image_dimensions: Tuple[int, int] = (0, 0)
    coordinate_mapping: Optional[CoordinateMapping] = None
    source_representation: str = ""
    lines_detected: int = 0
    lines_filtered: int = 0
    segments_merged: int = 0
    arrows_detected: int = 0
    rejected_arrows: List[DetectedArrow] = field(default_factory=list)
    arrow_candidates: int = 0
    deduplicated_arrow_count: int = 0
    validated_arrow_count: int = 0
    aoa_context: bool = False
    segments_per_logical_arrow: List[int] = field(default_factory=list)
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)

    @property
    def arrow_count(self) -> int:
        """Number of detected arrows."""
        return len(self.arrows)

    @property
    def has_arrows(self) -> bool:
        """Check if any arrows were detected."""
        return len(self.arrows) > 0

    def get_arrows_by_confidence(
        self, min_confidence: float = 0.0
    ) -> List[DetectedArrow]:
        """Get arrows above a confidence threshold."""
        return [a for a in self.arrows if a.confidence >= min_confidence]

    def get_arrows_involving_candidate(
        self, candidate_id: str
    ) -> List[DetectedArrow]:
        """Get arrows where the given candidate is source or target."""
        return [
            a for a in self.arrows
            if a.evidence.get("node_at_endpoint_1") == candidate_id
            or a.evidence.get("node_at_endpoint_2") == candidate_id
        ]

    def to_original_coordinates(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """Convert detection coordinates to original image coordinates."""
        if self.coordinate_mapping is None:
            return (x, y)
        return self.coordinate_mapping.to_original(x, y)

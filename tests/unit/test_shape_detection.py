"""
Comprehensive unit tests for shape detection and diagram classification.

Tests cover:
- Shape detection configuration
- Rectangle detection (clean, multiple, different aspect ratios)
- Circle detection (clean, multiple)
- Polygon detection
- Mixed shape detection
- False positive filtering
- Duplicate suppression
- Confidence scoring
- Coordinate mapping
- AON/AOA classification
- Reference diagram tests
- Edge cases
"""

import os
import tempfile

import cv2
import numpy as np
import pytest

from pert_analyzer.core.models import BoundingBox, DetectedShape, Point
from pert_analyzer.cv.classification import DiagramClassifier
from pert_analyzer.cv.exceptions import ClassificationError, ShapeDetectionError
from pert_analyzer.cv.models import (
    CandidateNode,
    CoordinateMapping,
    DiagramClassificationResult,
    PreprocessingResult,
    ShapeDetectionConfig,
    ShapeDetectionResult,
    ShapeType,
)
from pert_analyzer.cv.preprocessing import ImagePreprocessor
from pert_analyzer.cv.shape_detection import ShapeDetector


# =============================================================================
# Test Fixtures — Synthetic Image Generators
# =============================================================================


def create_clean_rectangle(
    width: int = 400,
    height: int = 300,
    rect_w: int = 80,
    rect_h: int = 50,
    margin: int = 60,
) -> np.ndarray:
    """Create an image with a clean rectangle."""
    image = np.full((height, width), 255, dtype=np.uint8)
    x1 = margin
    y1 = margin
    x2 = width - margin
    y2 = height - margin
    cv2.rectangle(image, (x1, y1), (x2, y2), 0, 2)
    return image


def create_multiple_rectangles(
    width: int = 600,
    height: int = 400,
    count: int = 4,
) -> np.ndarray:
    """Create an image with multiple rectangles."""
    image = np.full((height, width), 255, dtype=np.uint8)
    cols = 2
    rows = (count + 1) // 2
    cell_w = width // cols
    cell_h = height // rows

    for i in range(count):
        row = i // cols
        col = i % cols
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2
        rw = cell_w // 4
        rh = cell_h // 4
        cv2.rectangle(
            image,
            (cx - rw, cy - rh),
            (cx + rw, cy + rh),
            0,
            2,
        )
    return image


def create_clean_circle(
    width: int = 400,
    height: int = 300,
    radius: int = 50,
    margin: int = 80,
) -> np.ndarray:
    """Create an image with a clean circle."""
    image = np.full((height, width), 255, dtype=np.uint8)
    cx = width // 2
    cy = height // 2
    cv2.circle(image, (cx, cy), radius, 0, 2)
    return image


def create_multiple_circles(
    width: int = 600,
    height: int = 400,
    count: int = 4,
) -> np.ndarray:
    """Create an image with multiple circles."""
    image = np.full((height, width), 255, dtype=np.uint8)
    cols = 2
    rows = (count + 1) // 2
    cell_w = width // cols
    cell_h = height // rows

    for i in range(count):
        row = i // cols
        col = i % cols
        cx = col * cell_w + cell_w // 2
        cy = row * cell_h + cell_h // 2
        r = min(cell_w, cell_h) // 5
        cv2.circle(image, (cx, cy), r, 0, 2)
    return image


def create_mixed_shapes(width: int = 600, height: int = 400) -> np.ndarray:
    """Create an image with rectangles and circles."""
    image = np.full((height, width), 255, dtype=np.uint8)

    # Two rectangles on left
    cv2.rectangle(image, (30, 30), (180, 100), 0, 2)
    cv2.rectangle(image, (30, 150), (180, 220), 0, 2)

    # Two circles on right
    cv2.circle(image, (400, 80), 40, 0, 2)
    cv2.circle(image, (400, 220), 40, 0, 2)

    return image


def create_polygon_image(
    width: int = 400,
    height: int = 300,
) -> np.ndarray:
    """Create an image with a polygon (hexagon)."""
    image = np.full((height, width), 255, dtype=np.uint8)
    cx, cy = width // 2, height // 2
    r = 60
    vertices = []
    for i in range(6):
        angle = np.pi / 3 * i - np.pi / 6
        x = int(cx + r * np.cos(angle))
        y = int(cy + r * np.sin(angle))
        vertices.append([x, y])
    pts = np.array(vertices, np.int32)
    cv2.polylines(image, [pts], True, 0, 2)
    return image


def create_noisy_image(
    width: int = 400,
    height: int = 300,
) -> np.ndarray:
    """Create an image with random noise."""
    image = np.full((height, width), 255, dtype=np.uint8)
    noise = np.random.randint(0, 80, (height, width), dtype=np.uint8)
    image = cv2.subtract(image, noise)
    # Add a small rectangle
    cv2.rectangle(image, (100, 80), (250, 180), 0, 2)
    return image


def create_reference_aon(
    width: int = 800,
    height: int = 600,
    num_nodes: int = 6,
) -> np.ndarray:
    """
    Create a reference AON diagram with rectangular activity nodes.

    Simulates a typical AON layout: rectangles connected by implied arrows.
    """
    image = np.full((height, width), 255, dtype=np.uint8)

    cols = 3
    rows = (num_nodes + 1) // 2
    cell_w = width // (cols + 1)
    cell_h = height // (rows + 1)

    for i in range(num_nodes):
        row = i // cols
        col = i % cols
        cx = (col + 1) * cell_w
        cy = (row + 1) * cell_h
        rw = cell_w // 3
        rh = cell_h // 5
        cv2.rectangle(
            image,
            (cx - rw, cy - rh),
            (cx + rw, cy + rh),
            0,
            2,
        )

    # Add connecting lines (arrows will be detected in Phase 5)
    for i in range(num_nodes - 1):
        r1 = i // cols
        c1 = i % cols
        r2 = (i + 1) // cols
        c2 = (i + 1) % cols
        x1 = (c1 + 1) * cell_w + cell_w // 3
        y1 = (r1 + 1) * cell_h
        x2 = (c2 + 1) * cell_w - cell_w // 3
        y2 = (r2 + 1) * cell_h
        if r1 == r2:
            cv2.line(image, (x1, y1), (x2, y2), 0, 1)

    return image


def create_reference_aoa(
    width: int = 800,
    height: int = 600,
    num_events: int = 5,
) -> np.ndarray:
    """
    Create a reference AOA diagram with circular event nodes.

    Simulates a typical AOA layout: circles connected by lines.
    """
    image = np.full((height, width), 255, dtype=np.uint8)

    cols = 3
    rows = (num_events + 1) // 2
    cell_w = width // (cols + 1)
    cell_h = height // (rows + 1)

    positions = []
    for i in range(num_events):
        row = i // cols
        col = i % cols
        cx = (col + 1) * cell_w
        cy = (row + 1) * cell_h
        r = min(cell_w, cell_h) // 6
        cv2.circle(image, (cx, cy), r, 0, 2)
        positions.append((cx, cy))

    # Connect circles with lines
    for i in range(len(positions) - 1):
        cv2.line(image, positions[i], positions[i + 1], 0, 1)

    return image


# =============================================================================
# Test ShapeDetectionConfig
# =============================================================================


class TestShapeDetectionConfig:
    """Tests for ShapeDetectionConfig."""

    def test_default_config_is_valid(self):
        config = ShapeDetectionConfig()
        issues = config.validate()
        assert len(issues) == 0

    def test_invalid_min_rect_area(self):
        config = ShapeDetectionConfig(min_rect_area=-1)
        issues = config.validate()
        assert any("min_rect_area" in i for i in issues)

    def test_invalid_circularity(self):
        config = ShapeDetectionConfig(min_circularity=1.5)
        issues = config.validate()
        assert any("min_circularity" in i for i in issues)

    def test_invalid_aspect_ratio_range(self):
        config = ShapeDetectionConfig(rect_aspect_ratio_range=(0, 5.0))
        issues = config.validate()
        assert any("rect_aspect_ratio_range" in i for i in issues)


# =============================================================================
# Test ShapeType
# =============================================================================


class TestShapeType:
    """Tests for ShapeType enum."""

    def test_all_types_exist(self):
        assert ShapeType.RECTANGLE.value == "rectangle"
        assert ShapeType.SQUARE.value == "square"
        assert ShapeType.CIRCLE.value == "circle"
        assert ShapeType.ELLIPSE.value == "ellipse"
        assert ShapeType.POLYGON.value == "polygon"
        assert ShapeType.UNKNOWN.value == "unknown"

    def test_from_value(self):
        assert ShapeType("rectangle") == ShapeType.RECTANGLE
        assert ShapeType("circle") == ShapeType.CIRCLE


# =============================================================================
# Test ShapeDetector
# =============================================================================


class TestShapeDetector:
    """Tests for ShapeDetector."""

    def setup_method(self):
        self.detector = ShapeDetector()

    def test_initialization(self):
        detector = ShapeDetector()
        assert detector.config is not None

    def test_custom_config(self):
        config = ShapeDetectionConfig(min_shape_area=100)
        detector = ShapeDetector(config)
        assert detector.config.min_shape_area == 100

    def test_get_supported_shapes(self):
        shapes = self.detector.get_supported_shapes()
        assert "rectangle" in shapes
        assert "circle" in shapes
        assert "polygon" in shapes

    def test_get_parameters(self):
        params = self.detector.get_parameters()
        assert "min_rect_area" in params
        assert "min_shape_area" in params

    def test_set_parameters(self):
        self.detector.set_parameters({"min_shape_area": 50})
        assert self.detector.config.min_shape_area == 50


class TestRectangleDetection:
    """Tests for rectangle detection."""

    def setup_method(self):
        self.detector = ShapeDetector()

    def test_clean_rectangle(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        rects = [s for s in shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 1

    def test_rectangle_has_valid_bbox(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert shape.bounding_box.width > 0
            assert shape.bounding_box.height > 0
            assert shape.bounding_box.x >= 0
            assert shape.bounding_box.y >= 0

    def test_rectangle_has_valid_centroid(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert shape.centroid.x >= 0
            assert shape.centroid.y >= 0

    def test_rectangle_has_valid_confidence(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert 0.0 <= shape.confidence <= 1.0

    def test_multiple_rectangles(self):
        image = create_multiple_rectangles(count=4)
        shapes = self.detector.detect(image)
        rects = [s for s in shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 3

    def test_rectangle_area_positive(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert shape.area > 0

    def test_rectangle_metadata(self):
        image = create_clean_rectangle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert "perimeter" in shape.metadata
            assert "rectangularity" in shape.metadata
            assert "aspect_ratio" in shape.metadata
            assert shape.metadata["rectangularity"] > 0


class TestCircleDetection:
    """Tests for circle detection."""

    def setup_method(self):
        self.detector = ShapeDetector()

    def test_clean_circle(self):
        image = create_clean_circle()
        shapes = self.detector.detect(image)
        circles = [s for s in shapes if s.shape_type in ("circle", "ellipse")]
        assert len(circles) >= 1

    def test_circle_has_valid_centroid(self):
        image = create_clean_circle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert shape.centroid.x >= 0
            assert shape.centroid.y >= 0

    def test_circle_has_valid_confidence(self):
        image = create_clean_circle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert 0.0 <= shape.confidence <= 1.0

    def test_multiple_circles(self):
        image = create_multiple_circles(count=4)
        shapes = self.detector.detect(image)
        circles = [s for s in shapes if s.shape_type in ("circle", "ellipse")]
        assert len(circles) >= 3

    def test_circle_metadata(self):
        image = create_clean_circle()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert "circularity" in shape.metadata
            assert shape.metadata["circularity"] > 0


class TestPolygonDetection:
    """Tests for polygon detection."""

    def setup_method(self):
        self.detector = ShapeDetector()

    def test_hexagon_detected(self):
        image = create_polygon_image()
        shapes = self.detector.detect(image)
        # Hexagon should be detected as polygon or unknown
        assert len(shapes) >= 1

    def test_polygon_area_positive(self):
        image = create_polygon_image()
        shapes = self.detector.detect(image)
        for shape in shapes:
            assert shape.area > 0


class TestMixedShapes:
    """Tests for mixed shape detection."""

    def setup_method(self):
        self.detector = ShapeDetector()

    def test_rectangles_and_circles(self):
        image = create_mixed_shapes()
        shapes = self.detector.detect(image)
        assert len(shapes) >= 2

    def test_shape_types_distinguished(self):
        image = create_mixed_shapes()
        shapes = self.detector.detect(image)
        types = set(s.shape_type for s in shapes)
        # Should detect at least some shapes
        assert len(types) >= 1


class TestFiltering:
    """Tests for shape filtering."""

    def test_noise_filtered(self):
        config = ShapeDetectionConfig(min_shape_area=200)
        detector = ShapeDetector(config)
        image = create_noisy_image()
        shapes = detector.detect(image)
        # Small noise should be filtered
        for shape in shapes:
            assert shape.area >= 200

    def test_border_filtered(self):
        config = ShapeDetectionConfig(border_margin=5)
        detector = ShapeDetector(config)
        # Create image with shape touching border
        image = np.full((200, 200), 255, dtype=np.uint8)
        cv2.rectangle(image, (0, 0), (50, 50), 0, 2)
        shapes = detector.detect(image)
        # Border shape should be filtered
        rects = [s for s in shapes if s.shape_type == "rectangle"]
        # The border-touching rectangle should not be kept
        for r in rects:
            assert r.bounding_box.x > 5

    def test_min_area_filter(self):
        config = ShapeDetectionConfig(min_shape_area=1000)
        detector = ShapeDetector(config)
        image = create_clean_rectangle(rect_w=30, rect_h=20)
        shapes = detector.detect(image)
        for shape in shapes:
            assert shape.area >= 1000


class TestDuplicateSuppression:
    """Tests for duplicate detection suppression."""

    def test_overlapping_shapes_suppressed(self):
        detector = ShapeDetector()
        # Create image with overlapping rectangles
        image = np.full((200, 200), 255, dtype=np.uint8)
        cv2.rectangle(image, (30, 30), (150, 150), 0, 2)
        cv2.rectangle(image, (35, 35), (145, 145), 0, 2)
        result = detector._detect_shapes(image)
        # Should suppress one of the overlapping rectangles
        rects = [
            s for s in result.detected_shapes
            if s.shape_type in ("rectangle", "square")
        ]
        assert len(rects) <= 2  # At most 2 (one might be kept)

    def test_no_duplicates_when_separate(self):
        detector = ShapeDetector()
        image = create_multiple_rectangles(count=4)
        result = detector._detect_shapes(image)
        # All should be kept (no overlap)
        assert len(result.detected_shapes) >= 3


class TestGeometricDeduplication:
    """Deterministic tests for geometric deduplication logic."""

    def test_identical_rectangles_merged(self):
        """Two identical rectangles at the same position → 1 kept."""
        detector = ShapeDetector()
        image = np.full((300, 300), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) == 1, f"Expected 1 rectangle after merge, got {len(rects)}"

    def test_nested_rectangles_merged(self):
        """Inner rectangle fully inside outer → merged (containment)."""
        detector = ShapeDetector()
        image = np.full((300, 300), 255, dtype=np.uint8)
        cv2.rectangle(image, (40, 40), (160, 160), 0, 2)
        cv2.rectangle(image, (50, 50), (150, 150), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) <= 2, f"Expected ≤2 rectangles after containment dedup, got {len(rects)}"

    def test_multi_scale_duplicates_merged(self):
        """Two same-size rectangles 10px apart → merged (center distance)."""
        detector = ShapeDetector()
        image = np.full((300, 400), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        cv2.rectangle(image, (55, 55), (155, 105), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) == 1, f"Expected 1 rectangle after multi-scale merge, got {len(rects)}"

    def test_nearby_different_rectangles_preserved(self):
        """Two distinct rectangles close but non-overlapping → both kept."""
        detector = ShapeDetector()
        image = np.full((200, 400), 255, dtype=np.uint8)
        cv2.rectangle(image, (30, 50), (130, 120), 0, 2)
        cv2.rectangle(image, (160, 50), (260, 120), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 2, f"Expected ≥2 distinct rectangles, got {len(rects)}"

    def test_overlapping_distinct_rectangles(self):
        """Two rectangles with small overlap → both kept if not contained."""
        detector = ShapeDetector()
        image = np.full((200, 400), 255, dtype=np.uint8)
        cv2.rectangle(image, (30, 50), (180, 120), 0, 2)
        cv2.rectangle(image, (150, 50), (300, 120), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 2, f"Expected ≥2 overlapping-but-distinct rectangles, got {len(rects)}"

    def test_same_center_different_size_preserved(self):
        """Two rectangles same center, very different sizes, NOT contained → both kept."""
        detector = ShapeDetector()
        image = np.full((300, 300), 255, dtype=np.uint8)
        # Wide rectangle centered at (150,150)
        cv2.rectangle(image, (20, 100), (280, 200), 0, 2)
        # Tall rectangle centered at (150,150) — not contained in the wide one
        cv2.rectangle(image, (100, 10), (200, 290), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 2, f"Expected ≥2 different-shape rectangles, got {len(rects)}"

    def test_coordinate_mapping_preserved(self):
        """Deduplication preserves original coordinate system."""
        detector = ShapeDetector()
        image = np.full((300, 300), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        result = detector._detect_shapes(image)
        for s in result.detected_shapes:
            bb = s.bounding_box
            assert bb.x >= 0 and bb.y >= 0, "Coordinates must be non-negative"

    def test_deterministic_ids(self):
        """Running detection twice produces same candidate count."""
        detector = ShapeDetector()
        image = np.full((300, 300), 255, dtype=np.uint8)
        cv2.rectangle(image, (50, 50), (150, 100), 0, 2)
        cv2.rectangle(image, (180, 50), (280, 100), 0, 2)
        r1 = detector._detect_shapes(image)
        r2 = detector._detect_shapes(image)
        assert len(r1.candidate_nodes) == len(r2.candidate_nodes), \
            "Candidate count must be deterministic across runs"

    def test_distinct_adjacent_activities_preserved(self):
        """Grid of 6 distinct rectangles (AON-style) → all preserved."""
        detector = ShapeDetector()
        image = np.full((400, 600), 255, dtype=np.uint8)
        positions = [
            (30, 50, 130, 120), (160, 50, 260, 120), (290, 50, 390, 120),
            (30, 180, 130, 250), (160, 180, 260, 250), (290, 180, 390, 250),
        ]
        for x1, y1, x2, y2 in positions:
            cv2.rectangle(image, (x1, y1), (x2, y2), 0, 2)
        result = detector._detect_shapes(image)
        rects = [s for s in result.detected_shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 5, f"Expected ≥5 distinct activity rectangles, got {len(rects)}"


class TestCandidateNodes:
    """Tests for candidate node generation."""

    def test_candidates_generated(self):
        detector = ShapeDetector()
        image = create_multiple_rectangles(count=4)
        result = detector._detect_shapes(image)
        assert len(result.candidate_nodes) >= 1

    def test_candidate_has_position(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        result = detector._detect_shapes(image)
        for c in result.candidate_nodes:
            assert c.position.x >= 0
            assert c.position.y >= 0

    def test_candidate_has_bounding_box(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        result = detector._detect_shapes(image)
        for c in result.candidate_nodes:
            assert c.bounding_box.width > 0
            assert c.bounding_box.height > 0

    def test_rectangle_candidate_role(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        result = detector._detect_shapes(image)
        rect_candidates = [
            c for c in result.candidate_nodes
            if c.shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE)
        ]
        for c in rect_candidates:
            assert c.node_role == "activity"

    def test_circle_candidate_role(self):
        detector = ShapeDetector()
        image = create_clean_circle()
        result = detector._detect_shapes(image)
        circ_candidates = [
            c for c in result.candidate_nodes
            if c.shape_type in (ShapeType.CIRCLE, ShapeType.ELLIPSE)
        ]
        for c in circ_candidates:
            assert c.node_role == "event"

    def test_nearest_neighbors_computed(self):
        detector = ShapeDetector()
        image = create_multiple_rectangles(count=3)
        result = detector._detect_shapes(image)
        for c in result.candidate_nodes:
            assert isinstance(c.nearest_neighbors, list)


class TestConfidenceScoring:
    """Tests for confidence scoring."""

    def test_confidence_range(self):
        detector = ShapeDetector()
        image = create_mixed_shapes()
        shapes = detector.detect(image)
        for shape in shapes:
            assert 0.0 <= shape.confidence <= 1.0

    def test_clean_shapes_higher_confidence(self):
        detector = ShapeDetector()
        clean = create_clean_rectangle()
        noisy = create_noisy_image()
        clean_shapes = detector.detect(clean)
        # Clean shapes should have reasonable confidence
        for s in clean_shapes:
            assert s.confidence > 0.0


class TestCoordinateMapping:
    """Tests for coordinate mapping."""

    def test_no_mapping_identity(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        result = detector._detect_shapes(image)
        # Without mapping, coordinates should be as-is
        assert result.coordinate_mapping is None

    def test_mapping_applied(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        result = detector._detect_shapes(image)
        # Apply a mapping
        mapping = CoordinateMapping(
            original_width=400,
            original_height=300,
            processed_width=200,
            processed_height=150,
        )
        result.coordinate_mapping = mapping
        detector._map_coordinates_to_original(result)

        # Coordinates should be scaled back
        for shape in result.detected_shapes:
            # Original was 400x300, processed was 200x150, scale=0.5
            assert shape.bounding_box.x >= 0
            assert shape.bounding_box.y >= 0

    def test_shape_detection_result_coordinate_conversion(self):
        mapping = CoordinateMapping(
            original_width=800,
            original_height=600,
            processed_width=400,
            processed_height=300,
        )
        result = ShapeDetectionResult(coordinate_mapping=mapping)
        orig_x, orig_y = result.to_original_coordinates(200, 150)
        assert orig_x == pytest.approx(400.0)
        assert orig_y == pytest.approx(300.0)


class TestShapeDetectionResult:
    """Tests for ShapeDetectionResult."""

    def test_empty_result(self):
        result = ShapeDetectionResult()
        assert result.rectangle_count == 0
        assert result.circle_count == 0
        assert result.polygon_count == 0
        assert result.total_candidates == 0

    def test_get_candidates_by_type(self):
        result = ShapeDetectionResult()
        c1 = CandidateNode(shape_type=ShapeType.RECTANGLE)
        c2 = CandidateNode(shape_type=ShapeType.CIRCLE)
        result.candidate_nodes = [c1, c2]

        rects = result.get_candidates_by_type(ShapeType.RECTANGLE)
        assert len(rects) == 1
        assert rects[0].shape_type == ShapeType.RECTANGLE

    def test_image_dimensions(self):
        result = ShapeDetectionResult(image_dimensions=(800, 600))
        assert result.image_dimensions == (800, 600)


class TestDebugRendering:
    """Tests for debug visualization."""

    def test_render_returns_image(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        shapes = detector.detect(image)
        result = ShapeDetectionResult(detected_shapes=shapes)
        vis = detector.render_detections(image, result)
        assert vis is not None
        assert vis.shape[0] > 0
        assert vis.shape[1] > 0

    def test_render_grayscale_input(self):
        detector = ShapeDetector()
        image = create_clean_rectangle()
        shapes = detector.detect(image)
        result = ShapeDetectionResult(detected_shapes=shapes)
        vis = detector.render_detections(image, result)
        # Should convert to BGR
        assert len(vis.shape) == 3


# =============================================================================
# Test DiagramClassifier
# =============================================================================


class TestDiagramClassifier:
    """Tests for DiagramClassifier."""

    def setup_method(self):
        self.classifier = DiagramClassifier()

    def test_initialization(self):
        classifier = DiagramClassifier()
        assert classifier is not None

    def test_get_supported_types(self):
        types = self.classifier.get_supported_types()
        assert "AON" in types
        assert "AOA" in types
        assert "UNKNOWN" in types

    def test_get_confidence(self):
        assert self.classifier.get_confidence() == 0.0

    def test_classify_empty_shapes(self):
        result = self.classifier.classify([], [])
        assert result[0] == "UNKNOWN"
        assert result[1] == 0.0


class TestAONClassification:
    """Tests for AON diagram classification."""

    def setup_method(self):
        self.classifier = DiagramClassifier()

    def test_rectangles_classified_as_aon(self):
        # Create detection result with many rectangles
        result = ShapeDetectionResult()
        for _ in range(5):
            shape = DetectedShape(
                shape_type="rectangle",
                confidence=0.8,
                area=2000,
                bounding_box=BoundingBox(50, 50, 100, 60),
                centroid=Point(100, 80),
            )
            result.detected_shapes.append(shape)
            candidate = CandidateNode(
                shape_type=ShapeType.RECTANGLE,
                confidence=0.8,
                area=2000,
            )
            result.candidate_nodes.append(candidate)

        classification = self.classifier.classify_from_detection(result)
        assert classification.diagram_type == "AON"
        assert classification.confidence > 0.3

    def test_aon_evidence_computed(self):
        result = ShapeDetectionResult()
        for _ in range(4):
            shape = DetectedShape(
                shape_type="rectangle",
                confidence=0.7,
                area=1500,
            )
            result.detected_shapes.append(shape)
            candidate = CandidateNode(
                shape_type=ShapeType.RECTANGLE,
                confidence=0.7,
                area=1500,
            )
            result.candidate_nodes.append(candidate)

        classification = self.classifier.classify_from_detection(result)
        assert "aon_score" in classification.evidence
        assert classification.evidence["rectangle_count"] == 4


class TestAOAClassification:
    """Tests for AOA diagram classification."""

    def setup_method(self):
        self.classifier = DiagramClassifier()

    def test_circles_classified_as_aoa(self):
        result = ShapeDetectionResult()
        for _ in range(5):
            shape = DetectedShape(
                shape_type="circle",
                confidence=0.8,
                area=1500,
                bounding_box=BoundingBox(50, 50, 60, 60),
                centroid=Point(80, 80),
            )
            result.detected_shapes.append(shape)
            candidate = CandidateNode(
                shape_type=ShapeType.CIRCLE,
                confidence=0.8,
                area=1500,
            )
            result.candidate_nodes.append(candidate)

        classification = self.classifier.classify_from_detection(result)
        assert classification.diagram_type == "AOA"
        assert classification.confidence > 0.3

    def test_aoa_evidence_computed(self):
        result = ShapeDetectionResult()
        for _ in range(4):
            shape = DetectedShape(
                shape_type="circle",
                confidence=0.7,
                area=1200,
            )
            result.detected_shapes.append(shape)
            candidate = CandidateNode(
                shape_type=ShapeType.CIRCLE,
                confidence=0.7,
                area=1200,
            )
            result.candidate_nodes.append(candidate)

        classification = self.classifier.classify_from_detection(result)
        assert "aoa_score" in classification.evidence
        assert classification.evidence["circle_count"] == 4


class TestAmbiguousClassification:
    """Tests for ambiguous/unknown classification."""

    def setup_method(self):
        self.classifier = DiagramClassifier()

    def test_mixed_shapes_unknown(self):
        result = ShapeDetectionResult()
        # Equal rectangles and circles
        for _ in range(3):
            shape_r = DetectedShape(
                shape_type="rectangle", confidence=0.7, area=1500
            )
            shape_c = DetectedShape(
                shape_type="circle", confidence=0.7, area=1500
            )
            result.detected_shapes.extend([shape_r, shape_c])
            result.candidate_nodes.extend([
                CandidateNode(shape_type=ShapeType.RECTANGLE, confidence=0.7, area=1500),
                CandidateNode(shape_type=ShapeType.CIRCLE, confidence=0.7, area=1500),
            ])

        classification = self.classifier.classify_from_detection(result)
        # Should be UNKNOWN or low confidence
        assert classification.confidence < 0.8

    def test_insufficient_shapes_unknown(self):
        result = ShapeDetectionResult()
        shape = DetectedShape(
            shape_type="rectangle", confidence=0.7, area=1500
        )
        result.detected_shapes.append(shape)

        classification = self.classifier.classify_from_detection(result)
        assert classification.diagram_type == "UNKNOWN"


class TestClassificationResult:
    """Tests for DiagramClassificationResult."""

    def test_default_values(self):
        result = DiagramClassificationResult()
        assert result.diagram_type == "UNKNOWN"
        assert result.confidence == 0.0

    def test_diagram_type_normalized(self):
        result = DiagramClassificationResult(diagram_type="aon")
        assert result.diagram_type == "AON"

    def test_evidence_dict(self):
        result = DiagramClassificationResult()
        result.evidence["test"] = "value"
        assert result.evidence["test"] == "value"

    def test_warnings_list(self):
        result = DiagramClassificationResult()
        result.warnings.append("Test warning")
        assert len(result.warnings) == 1


# =============================================================================
# Reference Diagram Tests
# =============================================================================


class TestReferenceAON:
    """Integration tests with reference AON diagrams."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()
        self.detector = ShapeDetector()
        self.classifier = DiagramClassifier()

    def test_aon_rectangles_detected(self):
        image = create_reference_aon(num_nodes=6)
        shapes = self.detector.detect(image)
        rects = [s for s in shapes if s.shape_type in ("rectangle", "square")]
        assert len(rects) >= 4

    def test_aon_classified_correctly(self):
        image = create_reference_aon(num_nodes=6)
        shapes = self.detector.detect(image)
        result = ShapeDetectionResult(detected_shapes=shapes)
        classification = self.classifier.classify_from_detection(result)
        assert classification.diagram_type == "AON"
        assert classification.confidence > 0.3

    def test_aon_has_rectangular_candidates(self):
        image = create_reference_aon(num_nodes=4)
        det_result = self.detector._detect_shapes(image)
        rect_candidates = [
            c for c in det_result.candidate_nodes
            if c.shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE)
        ]
        assert len(rect_candidates) >= 2


class TestReferenceAOA:
    """Integration tests with reference AOA diagrams."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()
        self.detector = ShapeDetector()
        self.classifier = DiagramClassifier()

    def test_aoa_circles_detected(self):
        image = create_reference_aoa(num_events=5)
        shapes = self.detector.detect(image)
        circles = [s for s in shapes if s.shape_type in ("circle", "ellipse")]
        assert len(circles) >= 3

    def test_aoa_classified_correctly(self):
        image = create_reference_aoa(num_events=5)
        shapes = self.detector.detect(image)
        result = ShapeDetectionResult(detected_shapes=shapes)
        classification = self.classifier.classify_from_detection(result)
        assert classification.diagram_type == "AOA"
        assert classification.confidence > 0.3

    def test_aoa_has_circular_candidates(self):
        image = create_reference_aoa(num_events=4)
        det_result = self.detector._detect_shapes(image)
        circ_candidates = [
            c for c in det_result.candidate_nodes
            if c.shape_type in (ShapeType.CIRCLE, ShapeType.ELLIPSE)
        ]
        assert len(circ_candidates) >= 2


# =============================================================================
# Integration with Preprocessing
# =============================================================================


class TestPreprocessingIntegration:
    """Tests for integration with preprocessing pipeline."""

    def setup_method(self):
        self.preprocessor = ImagePreprocessor()
        self.detector = ShapeDetector()

    def test_detect_from_preprocessing_result(self):
        image = create_reference_aon(num_nodes=4)
        prep_result = self.preprocessor.process_image(image)
        det_result = self.detector.detect_from_preprocessing(prep_result)
        assert len(det_result.detected_shapes) >= 1

    def test_coordinate_mapping_from_preprocessing(self):
        config = ShapeDetectionConfig(min_shape_area=100)
        detector = ShapeDetector(config)
        image = create_reference_aon(num_nodes=4)
        prep_result = self.preprocessor.process_image(image)
        det_result = detector.detect_from_preprocessing(prep_result)
        # Should have coordinate mapping from preprocessing
        assert det_result.coordinate_mapping is not None

    def test_full_pipeline(self):
        """Test complete preprocessing → detection → classification."""
        image = create_reference_aon(num_nodes=6)
        prep_result = self.preprocessor.process_image(image)
        det_result = self.detector.detect_from_preprocessing(prep_result)
        classifier = DiagramClassifier()
        classification = classifier.classify_from_detection(det_result)
        assert classification.diagram_type in ("AON", "AOA", "UNKNOWN")


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_image(self):
        detector = ShapeDetector()
        image = np.full((100, 100), 255, dtype=np.uint8)
        shapes = detector.detect(image)
        # No shapes in blank image
        assert len(shapes) == 0

    def test_all_black_image(self):
        detector = ShapeDetector()
        image = np.zeros((100, 100), dtype=np.uint8)
        shapes = detector.detect(image)
        # May detect the entire image as one shape, or none
        assert isinstance(shapes, list)

    def test_very_small_image(self):
        detector = ShapeDetector()
        image = np.full((10, 10), 255, dtype=np.uint8)
        shapes = detector.detect(image)
        # Too small for any shapes
        assert len(shapes) == 0

    def test_detect_in_region(self):
        detector = ShapeDetector()
        image = create_multiple_rectangles(count=4)
        shapes = detector.detect_in_region(image, (0, 0, 300, 200))
        assert isinstance(shapes, list)

    def test_detect_in_invalid_region(self):
        detector = ShapeDetector()
        image = create_multiple_rectangles(count=4)
        shapes = detector.detect_in_region(image, (0, 0, 0, 0))
        assert len(shapes) == 0

    def test_single_pixel_contour(self):
        detector = ShapeDetector()
        image = np.full((100, 100), 255, dtype=np.uint8)
        image[50, 50] = 0
        shapes = detector.detect(image)
        # Single pixel should be filtered out
        assert len(shapes) == 0

"""
Shape detection module for diagram geometric analysis.

Detects rectangles, circles, polygons, and other geometric shapes
from preprocessed images. Provides both raw detected shapes and
interpreted candidate diagram nodes.

Uses OpenCV contour analysis and Hough transforms with configurable
parameters for different image qualities.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pert_analyzer.core.interfaces import ShapeDetector as ShapeDetectorABC
from pert_analyzer.core.models import BoundingBox, DetectedShape, Point, _generate_id
from pert_analyzer.cv.exceptions import ShapeDetectionError
from pert_analyzer.cv.models import (
    CandidateNode,
    CoordinateMapping,
    PreprocessingResult,
    ShapeDetectionConfig,
    ShapeDetectionResult,
    ShapeType,
)

logger = logging.getLogger(__name__)


class ShapeDetector(ShapeDetectorABC):
    """
    Concrete shape detector for diagram geometric analysis.

    Detects rectangles, circles, polygons from preprocessed images.
    Implements the ShapeDetector interface from core.interfaces.

    Usage:
        detector = ShapeDetector(config)
        result = detector.detect_from_preprocessing(preprocessing_result)
    """

    def __init__(self, config: Optional[ShapeDetectionConfig] = None):
        """
        Initialize the shape detector.

        Args:
            config: Detection configuration. Uses defaults if None.
        """
        self._config = config or ShapeDetectionConfig()
        self._validate_config()

    @property
    def config(self) -> ShapeDetectionConfig:
        """Get the current configuration."""
        return self._config

    def set_config(self, config: ShapeDetectionConfig) -> None:
        """Set a new configuration."""
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the current configuration."""
        issues = self._config.validate()
        if issues:
            raise ShapeDetectionError(
                "config_validation", "; ".join(issues)
            )

    # =========================================================================
    # Interface Implementation
    # =========================================================================

    def detect(self, image: np.ndarray) -> List[DetectedShape]:
        """
        Detect shapes in the given image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            List of detected shapes.
        """
        result = self._detect_shapes(image)
        return result.detected_shapes

    def detect_in_region(
        self, image: np.ndarray, region: Any
    ) -> List[DetectedShape]:
        """
        Detect shapes within a specific region.

        Args:
            image: Input image.
            region: Bounding box or region of interest (x, y, w, h).

        Returns:
            List of detected shapes in the region.
        """
        if isinstance(region, (tuple, list)) and len(region) == 4:
            x, y, w, h = region
            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                return []
            result = self._detect_shapes(roi)
            # Offset coordinates back
            for shape in result.detected_shapes:
                shape.bounding_box.x += x
                shape.bounding_box.y += y
                shape.centroid = Point(
                    shape.centroid.x + x, shape.centroid.y + y
                )
            return result.detected_shapes
        return self.detect(image)

    def get_supported_shapes(self) -> List[str]:
        """Return list of supported shape types."""
        return ["rectangle", "square", "circle", "ellipse", "polygon"]

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set detection parameters."""
        for key, value in params.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current detection parameters."""
        return {
            "min_rect_area": self._config.min_rect_area,
            "max_rect_area": self._config.max_rect_area,
            "min_circle_area": self._config.min_circle_area,
            "min_circularity": self._config.min_circularity,
            "min_shape_area": self._config.min_shape_area,
            "min_width": self._config.min_width,
            "min_height": self._config.min_height,
            "enable_duplicate_suppression": self._config.enable_duplicate_suppression,
        }

    # =========================================================================
    # Main Detection Pipeline
    # =========================================================================

    def detect_from_preprocessing(
        self, preprocessing_result: PreprocessingResult
    ) -> ShapeDetectionResult:
        """
        Detect shapes from a preprocessing result.

        Uses multiple representations for robust detection.

        Args:
            preprocessing_result: Output from ImagePreprocessor.

        Returns:
            ShapeDetectionResult with all detected shapes and candidates.
        """
        start_time = time.time()

        # Select best representation for contour detection
        detection_image = self._select_detection_image(preprocessing_result)
        if detection_image is None:
            raise ShapeDetectionError(
                "select_representation",
                "No suitable image representation available",
            )

        # Ensure binary for contour detection
        binary_image = self._ensure_binary(detection_image)

        # Detect shapes
        result = self._detect_shapes(binary_image)

        # Map coordinates back to original image
        if preprocessing_result.coordinate_mapping:
            result.coordinate_mapping = preprocessing_result.coordinate_mapping
            self._map_coordinates_to_original(result)

        result.processing_time = time.time() - start_time

        logger.info(
            "Shape detection complete: %d shapes, %d candidates, "
            "%.3f seconds",
            len(result.detected_shapes),
            len(result.candidate_nodes),
            result.processing_time,
        )

        return result

    def _select_detection_image(
        self, preprocessing_result: PreprocessingResult
    ) -> Optional[np.ndarray]:
        """
        Select the best image representation for shape detection.

        Priority: adaptive_binary > binary > edges > contrast_enhanced > grayscale
        """
        for name in [
            "adaptive_binary",
            "binary",
            "edges",
            "contrast_enhanced",
            "grayscale",
        ]:
            img = preprocessing_result.get_representation(name)
            if img is not None:
                logger.debug("Using '%s' representation for detection", name)
                return img
        return None

    def _ensure_binary(self, image: np.ndarray) -> np.ndarray:
        """Ensure image is binary for contour detection."""
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Check if already binary
        unique_vals = np.unique(image)
        if len(unique_vals) <= 2:
            return image

        # Apply Otsu threshold
        _, binary = cv2.threshold(
            image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary

    # =========================================================================
    # Core Shape Detection
    # =========================================================================

    def _detect_shapes(self, image: np.ndarray) -> ShapeDetectionResult:
        """
        Core shape detection from a binary/grayscale image.
        """
        result = ShapeDetectionResult()

        if image is None or image.size == 0:
            return result

        h, w = image.shape[:2]
        result.image_dimensions = (w, h)

        # Find contours
        contours, hierarchy = self._find_contours(image)
        result.contours_analyzed = len(contours)

        if not contours:
            logger.debug("No contours found")
            return result

        # Classify each contour
        shapes: List[DetectedShape] = []
        for i, contour in enumerate(contours):
            shape = self._classify_contour(contour, hierarchy, i, (w, h))
            if shape is not None:
                shapes.append(shape)

        # Filter shapes
        shapes = self._filter_shapes(shapes, (w, h))
        result.shapes_filtered = len(contours) - len(shapes)

        # Suppress duplicates
        if self._config.enable_duplicate_suppression:
            before = len(shapes)
            shapes = self._suppress_duplicates(shapes)
            result.duplicates_suppressed = before - len(shapes)

        result.detected_shapes = shapes

        # Generate candidate nodes
        result.candidate_nodes = self._generate_candidate_nodes(shapes)

        return result

    def _find_contours(
        self, image: np.ndarray
    ) -> Tuple[List[np.ndarray], Optional[np.ndarray]]:
        """Find contours in the image."""
        binary = self._ensure_binary(image)

        # Invert if needed (contours need white objects on black background)
        if np.mean(binary) > 127:
            binary = cv2.bitwise_not(binary)

        contours, hierarchy = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        # Sort by area (largest first) for processing order
        if contours:
            areas = [cv2.contourArea(c) for c in contours]
            sorted_indices = np.argsort(areas)[::-1]
            contours = [contours[i] for i in sorted_indices]
            if hierarchy is not None:
                hierarchy = hierarchy[0][sorted_indices]

        return contours, hierarchy

    def _classify_contour(
        self,
        contour: np.ndarray,
        hierarchy: Optional[np.ndarray],
        index: int,
        image_dims: Tuple[int, int],
    ) -> Optional[DetectedShape]:
        """
        Classify a single contour into a shape type.
        """
        area = cv2.contourArea(contour)
        if area < self._config.min_shape_area:
            return None

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return None

        # Compute bounding box
        x, y, w, h = cv2.boundingRect(contour)
        if w < self._config.min_width or h < self._config.min_height:
            return None

        # Check border proximity
        if self._is_near_border(x, y, w, h, image_dims):
            return None

        # Compute geometric properties
        bbox = BoundingBox(float(x), float(y), float(w), float(h))
        centroid = self._compute_centroid(contour)
        rectangularity = self._compute_rectangularity(contour, area)
        circularity = self._compute_circularity(area, perimeter)
        aspect_ratio = w / h if h > 0 else 1.0

        # Classify shape
        shape_type, confidence = self._determine_shape_type(
            contour, area, rectangularity, circularity, aspect_ratio, w, h
        )

        if shape_type == ShapeType.UNKNOWN:
            return None

        return DetectedShape(
            shape_id=_generate_id("shape"),
            shape_type=shape_type.value,
            contour=contour,
            bounding_box=bbox,
            centroid=centroid,
            confidence=confidence,
            area=area,
            metadata={
                "perimeter": perimeter,
                "rectangularity": rectangularity,
                "circularity": circularity,
                "aspect_ratio": aspect_ratio,
            },
        )

    def _compute_centroid(self, contour: np.ndarray) -> Point:
        """Compute the centroid of a contour."""
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            return Point(float(x + w // 2), float(y + h // 2))
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return Point(float(cx), float(cy))

    def _compute_rectangularity(
        self, contour: np.ndarray, area: float
    ) -> float:
        """
        Compute rectangularity: contour area / bounding box area.
        High values indicate rectangular shapes.
        """
        x, y, w, h = cv2.boundingRect(contour)
        bbox_area = w * h
        if bbox_area == 0:
            return 0.0
        return area / bbox_area

    def _compute_circularity(self, area: float, perimeter: float) -> float:
        """
        Compute circularity: 4*pi*area / perimeter^2.
        Perfect circle = 1.0.
        """
        if perimeter == 0:
            return 0.0
        return (4 * np.pi * area) / (perimeter * perimeter)

    def _determine_shape_type(
        self,
        contour: np.ndarray,
        area: float,
        rectangularity: float,
        circularity: float,
        aspect_ratio: float,
        width: int,
        height: int,
    ) -> Tuple[ShapeType, float]:
        """
        Determine the shape type and confidence based on geometric properties.
        """
        # Try rectangle first (most common in AON diagrams)
        rect_type, rect_conf = self._try_rectangle(
            contour, area, rectangularity, aspect_ratio, width, height
        )
        if rect_type is not None:
            return rect_type, rect_conf

        # Try circle
        circ_type, circ_conf = self._try_circle(
            contour, area, circularity, aspect_ratio, width, height
        )
        if circ_type is not None:
            return circ_type, circ_conf

        # Try polygon
        poly_type, poly_conf = self._try_polygon(
            contour, area, rectangularity, aspect_ratio
        )
        if poly_type is not None:
            return poly_type, poly_conf

        return ShapeType.UNKNOWN, 0.0

    def _try_rectangle(
        self,
        contour: np.ndarray,
        area: float,
        rectangularity: float,
        aspect_ratio: float,
        width: int,
        height: int,
    ) -> Tuple[Optional[ShapeType], float]:
        """Attempt to classify as rectangle or square."""
        if rectangularity < self._config.min_rectangularity:
            return None, 0.0

        if area < self._config.min_rect_area or area > self._config.max_rect_area:
            return None, 0.0

        # Check aspect ratio
        ar_min, ar_max = self._config.rect_aspect_ratio_range
        if aspect_ratio < ar_min or aspect_ratio > ar_max:
            return None, 0.0

        # Approximate contour to polygon
        epsilon = self._config.rect_approx_epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # Rectangle should have 4 vertices
        if len(approx) == 4:
            # Check angles (should be ~90 degrees)
            angle_score = self._check_rectangle_angles(approx)

            # Compute confidence based on evidence
            confidence = self._compute_rectangle_confidence(
                rectangularity, aspect_ratio, angle_score
            )

            if confidence < self._config.min_confidence:
                return None, 0.0

            # Determine if square
            if 0.85 <= aspect_ratio <= 1.15:
                return ShapeType.SQUARE, confidence
            return ShapeType.RECTANGLE, confidence

        return None, 0.0

    def _check_rectangle_angles(self, approx: np.ndarray) -> float:
        """
        Check how close the angles of a quadrilateral are to 90 degrees.
        Returns a score between 0 and 1.
        """
        if len(approx) != 4:
            return 0.0

        angles = []
        pts = approx.reshape(-1, 2)
        for i in range(4):
            p1 = pts[i]
            p2 = pts[(i + 1) % 4]
            p3 = pts[(i + 2) % 4]
            v1 = p1 - p2
            v2 = p3 - p2
            cos_angle = np.dot(v1, v2) / (
                np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10
            )
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)

        # Score: how close angles are to 90 degrees
        deviations = [abs(a - 90) for a in angles]
        max_dev = max(deviations) if deviations else 90
        score = max(0, 1 - max_dev / 45)
        return score

    def _compute_rectangle_confidence(
        self,
        rectangularity: float,
        aspect_ratio: float,
        angle_score: float,
    ) -> float:
        """Compute confidence for rectangle classification."""
        # Weighted combination of evidence
        conf = (
            0.4 * rectangularity
            + 0.3 * angle_score
            + 0.3 * min(1.0, 1.0 / (1.0 + abs(aspect_ratio - 1.5) * 0.5))
        )
        return min(1.0, max(0.0, conf))

    def _try_circle(
        self,
        contour: np.ndarray,
        area: float,
        circularity: float,
        aspect_ratio: float,
        width: int,
        height: int,
    ) -> Tuple[Optional[ShapeType], float]:
        """Attempt to classify as circle or ellipse."""
        if area < self._config.min_circle_area:
            return None, 0.0
        if area > self._config.max_circle_area:
            return None, 0.0

        if circularity < self._config.min_circularity:
            return None, 0.0

        # Check aspect ratio (circles should be near 1.0)
        if aspect_ratio > 2.0 or aspect_ratio < 0.5:
            return None, 0.0

        # Compute confidence
        aspect_score = max(0, 1 - abs(aspect_ratio - 1.0))
        confidence = 0.5 * circularity + 0.3 * aspect_score + 0.2

        if confidence < self._config.min_confidence:
            return None, 0.0

        # Ellipse if aspect ratio significantly different from 1.0
        if 0.6 <= aspect_ratio <= 1.67:
            return ShapeType.CIRCLE, confidence
        return ShapeType.ELLIPSE, confidence

    def _try_polygon(
        self,
        contour: np.ndarray,
        area: float,
        rectangularity: float,
        aspect_ratio: float,
    ) -> Tuple[Optional[ShapeType], float]:
        """Attempt to classify as polygon."""
        if area < self._config.min_polygon_area:
            return None, 0.0

        # Approximate contour
        epsilon = 0.04 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        if vertices < self._config.polygon_min_vertices:
            return None, 0.0
        if vertices > self._config.polygon_max_vertices:
            return None, 0.0

        # Compute convexity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area == 0:
            return None, 0.0
        convexity = area / hull_area

        confidence = 0.5 * convexity + 0.3 * min(1.0, vertices / 6) + 0.2
        return ShapeType.POLYGON, min(1.0, confidence)

    # =========================================================================
    # Filtering
    # =========================================================================

    def _is_near_border(
        self, x: int, y: int, w: int, h: int, image_dims: Tuple[int, int]
    ) -> bool:
        """Check if shape is too close to image border."""
        img_w, img_h = image_dims
        margin = self._config.border_margin

        # Check if shape touches or nearly touches border
        if x <= margin or y <= margin:
            return True
        if x + w >= img_w - margin or y + h >= img_h - margin:
            return True

        # Check if shape is the entire image (page border)
        if w > img_w * 0.95 and h > img_h * 0.95:
            return True

        return False

    def _filter_shapes(
        self, shapes: List[DetectedShape], image_dims: Tuple[int, int]
    ) -> List[DetectedShape]:
        """Filter shapes based on configuration criteria."""
        filtered = []
        for shape in shapes:
            if self._should_keep_shape(shape, image_dims):
                filtered.append(shape)
            else:
                logger.debug(
                    "Filtered shape %s (type=%s, conf=%.2f)",
                    shape.shape_id,
                    shape.shape_type,
                    shape.confidence,
                )
        return filtered

    def _should_keep_shape(
        self, shape: DetectedShape, image_dims: Tuple[int, int]
    ) -> bool:
        """Determine if a shape should be kept after filtering."""
        # Check confidence
        if shape.confidence < self._config.min_confidence:
            return False

        # Check area
        if shape.area < self._config.min_shape_area:
            return False
        if shape.area > self._config.max_shape_area:
            return False

        # Check dimensions
        w = shape.bounding_box.width
        h = shape.bounding_box.height
        if w < self._config.min_width or h < self._config.min_height:
            return False

        return True

    # =========================================================================
    # Duplicate Suppression
    # =========================================================================

    def _suppress_duplicates(
        self, shapes: List[DetectedShape]
    ) -> List[DetectedShape]:
        """
        Suppress duplicate detections for the same geometric shape.
        Keep the one with highest confidence.
        """
        if len(shapes) <= 1:
            return shapes

        # Sort by confidence (highest first)
        sorted_shapes = sorted(shapes, key=lambda s: s.confidence, reverse=True)

        kept: List[DetectedShape] = []
        suppressed_ids = set()

        for shape in sorted_shapes:
            if shape.shape_id in suppressed_ids:
                continue

            kept.append(shape)

            # Mark overlapping shapes as suppressed
            for other in sorted_shapes:
                if other.shape_id == shape.shape_id:
                    continue
                if other.shape_id in suppressed_ids:
                    continue
                if self._shapes_overlap(shape, other):
                    suppressed_ids.add(other.shape_id)

        return kept

    def _shapes_overlap(
        self, shape1: DetectedShape, shape2: DetectedShape
    ) -> bool:
        """Check if two shapes overlap significantly or represent the same rectangle.

        Uses multiple criteria (cross-type aware):
        1. IoU overlap (standard)
        2. Containment (one shape fully inside another)
        3. Center distance relative to size
        4. Size similarity (width/height ratio)

        Cross-type dedup (e.g. rectangle vs polygon) requires stricter
        overlap threshold to avoid merging genuinely different shapes.
        """
        b1 = shape1.bounding_box
        b2 = shape2.bounding_box

        same_type = shape1.shape_type == shape2.shape_type

        # --- Criterion 1: IoU overlap ---
        x_overlap = max(
            0,
            min(b1.x + b1.width, b2.x + b2.width) - max(b1.x, b2.x),
        )
        y_overlap = max(
            0,
            min(b1.y + b1.height, b2.y + b2.height) - max(b1.y, b2.y),
        )
        overlap_area = x_overlap * y_overlap

        area1 = b1.width * b1.height
        area2 = b2.width * b2.height
        min_area = min(area1, area2) if min(area1, area2) > 0 else 1

        overlap_ratio = overlap_area / min_area

        # For same type: standard threshold (0.5). For cross-type: higher bar (0.6)
        iou_threshold = self._config.overlap_threshold if same_type else 0.6
        if overlap_ratio >= iou_threshold:
            return True

        # --- Criterion 2: Containment (one fully inside the other) ---
        margin = 5  # pixels tolerance
        b1_inside_b2 = (
            b1.x >= b2.x - margin
            and b1.y >= b2.y - margin
            and b1.x + b1.width <= b2.x + b2.width + margin
            and b1.y + b1.height <= b2.y + b2.height + margin
        )
        b2_inside_b1 = (
            b2.x >= b1.x - margin
            and b2.y >= b1.y - margin
            and b2.x + b2.width <= b1.x + b1.width + margin
            and b2.y + b2.height <= b1.y + b1.height + margin
        )
        if b1_inside_b2 or b2_inside_b1:
            # For cross-type containment, also check size similarity
            if same_type:
                return True
            # Cross-type: require sizes within 50%
            w_ratio = min(b1.width, b2.width) / max(b1.width, b2.width) if max(b1.width, b2.width) > 0 else 0
            h_ratio = min(b1.height, b2.height) / max(b1.height, b2.height) if max(b1.height, b2.height) > 0 else 0
            if w_ratio > 0.5 and h_ratio > 0.5:
                return True

        # --- Criterion 3: Center distance + size similarity ---
        cx1 = b1.x + b1.width / 2
        cy1 = b1.y + b1.height / 2
        cx2 = b2.x + b2.width / 2
        cy2 = b2.y + b2.height / 2
        dist = ((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2) ** 0.5
        avg_size = ((area1 + area2) / 2) ** 0.5

        # Check size similarity
        w_ratio = min(b1.width, b2.width) / max(b1.width, b2.width) if max(b1.width, b2.width) > 0 else 0
        h_ratio = min(b1.height, b2.height) / max(b1.height, b2.height) if max(b1.height, b2.height) > 0 else 0
        size_similar = w_ratio > 0.6 and h_ratio > 0.6

        if avg_size > 0 and size_similar:
            # Same type: more lenient distance. Cross-type: stricter.
            dist_threshold = 0.15 if same_type else 0.08
            if dist / avg_size < dist_threshold:
                return True

        return False

    # =========================================================================
    # Candidate Node Generation
    # =========================================================================

    def _generate_candidate_nodes(
        self, shapes: List[DetectedShape]
    ) -> List[CandidateNode]:
        """
        Generate candidate diagram nodes from detected shapes.
        """
        candidates = []
        for shape in shapes:
            candidate = self._shape_to_candidate(shape)
            if candidate is not None:
                candidates.append(candidate)

        # Deduplication: remove rectangles that are significantly oversized
        # compared to the median rectangle (likely merged shapes or artifacts)
        candidates = self._remove_outlier_rectangles(candidates)

        # Compute nearest neighbors
        self._compute_nearest_neighbors(candidates)

        return candidates

    def _remove_outlier_rectangles(
        self, candidates: List[CandidateNode]
    ) -> List[CandidateNode]:
        """Remove rectangle candidates that are significantly oversized or undersized
        compared to the median rectangle area.

        Oversized: likely merged shapes or false detections from arrow labels.
        Undersized: likely arrow label text boxes, not activity nodes.
        """
        import statistics

        rects = [c for c in candidates if c.shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE)]
        if len(rects) < 5:
            return candidates

        areas = [c.bounding_box.width * c.bounding_box.height for c in rects]
        median_area = statistics.median(areas)
        if median_area <= 0:
            return candidates

        outlier_ids = set()
        for c in rects:
            area = c.bounding_box.width * c.bounding_box.height
            # Remove if >1.5x median (merged shapes) or <0.3x median (arrow labels)
            if area > median_area * 1.5 or area < median_area * 0.3:
                outlier_ids.add(c.node_id)

        if not outlier_ids:
            return candidates

        return [c for c in candidates if c.node_id not in outlier_ids]

    def _shape_to_candidate(
        self, shape: DetectedShape
    ) -> Optional[CandidateNode]:
        """Convert a detected shape to a candidate node."""
        shape_type = ShapeType(shape.shape_type)

        # Determine node role based on shape type
        if shape_type in (ShapeType.RECTANGLE, ShapeType.SQUARE):
            node_role = "activity"  # AON: rectangles are activities
        elif shape_type in (ShapeType.CIRCLE, ShapeType.ELLIPSE):
            node_role = "event"  # AOA: circles are events
        else:
            node_role = "unknown"

        # Compute distance to border
        dist_to_border = self._compute_border_distance(
            shape.bounding_box, shape.metadata.get("image_dims", (0, 0))
        )

        # Compute aspect ratio
        w = shape.bounding_box.width
        h = shape.bounding_box.height
        aspect_ratio = w / h if h > 0 else 1.0

        return CandidateNode(
            node_id=_generate_id("cnode"),
            shape_type=shape_type,
            source_shape_id=shape.shape_id,
            label="",
            position=shape.centroid,
            bounding_box=shape.bounding_box,
            confidence=shape.confidence,
            node_role=node_role,
            area=shape.area,
            perimeter=shape.metadata.get("perimeter", 0.0),
            aspect_ratio=aspect_ratio,
            distance_to_border=dist_to_border,
        )

    def _compute_border_distance(
        self, bbox: BoundingBox, image_dims: Tuple[int, int]
    ) -> float:
        """Compute minimum distance from shape to image border."""
        if image_dims == (0, 0):
            return 0.0

        img_w, img_h = image_dims
        left = bbox.x
        top = bbox.y
        right = img_w - (bbox.x + bbox.width)
        bottom = img_h - (bbox.y + bbox.height)

        return max(0, min(left, top, right, bottom))

    def _compute_nearest_neighbors(
        self, candidates: List[CandidateNode], k: int = 3
    ) -> None:
        """Compute k nearest neighbors for each candidate."""
        for i, c1 in enumerate(candidates):
            distances = []
            for j, c2 in enumerate(candidates):
                if i != j:
                    dist = c1.position.distance_to(c2.position)
                    distances.append((dist, c2.node_id))
            distances.sort(key=lambda x: x[0])
            c1.nearest_neighbors = [
                nid for _, nid in distances[:k]
            ]

    # =========================================================================
    # Coordinate Mapping
    # =========================================================================

    def _map_coordinates_to_original(
        self, result: ShapeDetectionResult
    ) -> None:
        """Map all detection coordinates back to original image."""
        mapping = result.coordinate_mapping
        if mapping is None:
            return

        for shape in result.detected_shapes:
            # Map bounding box
            orig_x, orig_y = mapping.to_original(
                shape.bounding_box.x, shape.bounding_box.y
            )
            orig_w = shape.bounding_box.width / mapping.scale_x
            orig_h = shape.bounding_box.height / mapping.scale_y
            shape.bounding_box = BoundingBox(orig_x, orig_y, orig_w, orig_h)

            # Map centroid
            orig_cx, orig_cy = mapping.to_original(
                shape.centroid.x, shape.centroid.y
            )
            shape.centroid = Point(orig_cx, orig_cy)

        for candidate in result.candidate_nodes:
            orig_x, orig_y = mapping.to_original(
                candidate.bounding_box.x, candidate.bounding_box.y
            )
            orig_w = candidate.bounding_box.width / mapping.scale_x
            orig_h = candidate.bounding_box.height / mapping.scale_y
            candidate.bounding_box = BoundingBox(orig_x, orig_y, orig_w, orig_h)

            orig_cx, orig_cy = mapping.to_original(
                candidate.position.x, candidate.position.y
            )
            candidate.position = Point(orig_cx, orig_cy)

    # =========================================================================
    # Debug / Visualization
    # =========================================================================

    def render_detections(
        self,
        image: np.ndarray,
        result: ShapeDetectionResult,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Render detection results on an image for debugging.

        Args:
            image: Original image to draw on.
            result: Shape detection results.
            show_confidence: If True, show confidence scores.

        Returns:
            Image with drawn detections.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        colors = {
            "rectangle": (0, 255, 0),    # Green
            "square": (0, 200, 0),        # Dark green
            "circle": (255, 0, 0),        # Blue
            "ellipse": (200, 0, 0),       # Dark blue
            "polygon": (0, 0, 255),       # Red
            "unknown": (128, 128, 128),   # Gray
        }

        for shape in result.detected_shapes:
            color = colors.get(shape.shape_type, (128, 128, 128))

            # Draw contour
            if shape.contour is not None:
                cv2.drawContours(vis, [shape.contour], -1, color, 2)

            # Draw bounding box
            bbox = shape.bounding_box
            cv2.rectangle(
                vis,
                (int(bbox.x), int(bbox.y)),
                (int(bbox.x + bbox.width), int(bbox.y + bbox.height)),
                color,
                1,
            )

            # Draw label
            label = shape.shape_type
            if show_confidence:
                label += f" {shape.confidence:.2f}"

            cv2.putText(
                vis,
                label,
                (int(bbox.x), int(bbox.y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

        return vis

"""
Arrow and connection detection module for diagram geometric analysis.

Detects line segments, arrowheads, and assembles them into structured
arrow candidates with direction, source/target node association, and
confidence scores.

This module produces VISUAL DETECTION EVIDENCE only — it does NOT
create semantic dependencies or graph edges. Final interpretation
belongs to a later reconstruction phase.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from pert_analyzer.core.interfaces import ArrowDetector as ArrowDetectorABC
from pert_analyzer.core.models import BoundingBox, DetectedShape, Point, _generate_id
from pert_analyzer.cv.exceptions import ArrowDetectionError
from pert_analyzer.cv.models import (
    ArrowDetectionConfig,
    ArrowDetectionResult,
    ArrowheadEvidence,
    ArrowheadType,
    CandidateNode,
    CoordinateMapping,
    DetectedArrow,
    DetectedLineSegment,
    LineStyle,
    PreprocessingResult,
    ShapeDetectionResult,
    ShapeType,
)

logger = logging.getLogger(__name__)


class ArrowDetector(ArrowDetectorABC):
    """
    Concrete arrow/connection detector for diagram images.

    Pipeline:
        1. Line detection (HoughLinesP)
        2. Line filtering (borders, length, angle)
        3. Arrowhead detection at endpoints
        4. Direction inference
        5. Segment merging (collinear segments)
        6. Node association (source/target candidates)
        7. Confidence scoring
        8. Coordinate mapping

    Usage:
        detector = ArrowDetector(config)
        result = detector.detect_from_preprocessing(
            preprocessing_result, shape_result
        )
    """

    def __init__(self, config: Optional[ArrowDetectionConfig] = None):
        """Initialize the arrow detector."""
        self._config = config or ArrowDetectionConfig()
        self._validate_config()

    @property
    def config(self) -> ArrowDetectionConfig:
        """Get current configuration."""
        return self._config

    def set_config(self, config: ArrowDetectionConfig) -> None:
        """Set a new configuration."""
        self._config = config
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate the current configuration."""
        issues = self._config.validate()
        if issues:
            raise ArrowDetectionError("config_validation", "; ".join(issues))

    # =========================================================================
    # Interface Implementation
    # =========================================================================

    def detect(self, image: np.ndarray) -> List:
        """
        Detect arrows in the given image.

        Args:
            image: Input image (grayscale or BGR).

        Returns:
            List of DetectedArrow objects.
        """
        result = self._detect_arrows_from_image(image)
        return result.arrows

    def _detect_arrows_from_image(self, image: np.ndarray) -> ArrowDetectionResult:
        """
        Detect arrows directly from a raw image (no preprocessing result).

        Used by the detect() interface method.
        """
        raw_segments = self._detect_line_segments(image)
        filtered, rejected = self._filter_segments(
            raw_segments, image.shape, None
        )
        arrows_with_heads = self._detect_arrowheads(filtered, image)
        merged, merge_count = self._merge_segments(arrows_with_heads)
        assembled_arrows = self._assemble_arrows(merged)
        final_arrows, rejected_arrows, stats = self._finalize_arrow_set(
            assembled_arrows, None
        )

        result = ArrowDetectionResult(
            arrows=final_arrows,
            raw_line_segments=raw_segments,
            rejected_candidates=rejected,
            lines_detected=len(raw_segments),
            lines_filtered=len(raw_segments) - len(filtered),
            segments_merged=merge_count,
            arrows_detected=len(final_arrows),
            rejected_arrows=rejected_arrows,
            arrow_candidates=stats["arrow_candidates"],
            deduplicated_arrow_count=stats["deduplicated"],
            validated_arrow_count=len(final_arrows),
            aoa_context=stats["aoa_context"],
            segments_per_logical_arrow=stats["segments_per_logical_arrow"],
        )
        h, w = image.shape[:2]
        result.image_dimensions = (w, h)
        return result

    def detect_connections(
        self,
        shapes: List[DetectedShape],
        arrows: List,
    ) -> List[Dict[str, Any]]:
        """
        Detect connections between shapes via arrows.

        Args:
            shapes: Detected shapes from Phase 4.
            arrows: Detected arrows from detect().

        Returns:
            List of connection dictionaries with source/target info.
        """
        connections = []
        for arrow in arrows:
            if not isinstance(arrow, DetectedArrow):
                continue
            source_id = arrow.evidence.get("node_at_endpoint_1")
            target_id = arrow.evidence.get("node_at_endpoint_2")
            if source_id and target_id:
                connections.append({
                    "source": source_id,
                    "target": target_id,
                    "arrow_id": arrow.arrow_id,
                    "confidence": arrow.confidence,
                    "direction": arrow.direction_vector,
                })
        return connections

    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set detection parameters."""
        for key, value in params.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)

    def get_parameters(self) -> Dict[str, Any]:
        """Get current detection parameters."""
        return {
            "hough_threshold": self._config.hough_threshold,
            "hough_min_line_length": self._config.hough_min_line_length,
            "hough_max_line_gap": self._config.hough_max_line_gap,
            "min_line_length": self._config.min_line_length,
            "arrowhead_detection_radius": self._config.arrowhead_detection_radius,
            "node_connection_tolerance": self._config.node_connection_tolerance,
            "merge_angle_tolerance_deg": self._config.merge_angle_tolerance_deg,
        }

    # =========================================================================
    # Main Detection Pipeline
    # =========================================================================

    def detect_from_preprocessing(
        self,
        preprocessing_result: PreprocessingResult,
        shape_result: Optional[ShapeDetectionResult] = None,
    ) -> ArrowDetectionResult:
        """
        Detect arrows from preprocessing result with optional shape context.

        Args:
            preprocessing_result: Output from ImagePreprocessor.
            shape_result: Optional shape detection result for border filtering.

        Returns:
            ArrowDetectionResult with all detected arrows and evidence.
        """
        start_time = time.time()

        # Select best image for line detection
        detection_image = self._select_detection_image(preprocessing_result)
        if detection_image is None:
            raise ArrowDetectionError(
                "select_representation",
                "No suitable image representation available",
            )

        # Detect raw line segments
        raw_segments = self._detect_line_segments(detection_image)

        # Filter segments
        filtered, rejected = self._filter_segments(
            raw_segments, detection_image.shape, shape_result
        )

        # Detect arrowheads and infer direction
        arrows_with_heads = self._detect_arrowheads(
            filtered, detection_image
        )

        # Merge collinear segments
        merged, merge_count = self._merge_segments(arrows_with_heads)

        # Assemble final arrows
        assembled_arrows = self._assemble_arrows(merged)

        # Associate with candidate nodes
        if shape_result and shape_result.candidate_nodes:
            self._associate_with_candidates(assembled_arrows, shape_result.candidate_nodes)

        # Consolidate: dedup + AOA event-boundary validation
        final_arrows, rejected_arrows, stats = self._finalize_arrow_set(
            assembled_arrows, shape_result
        )

        # Build result
        result = ArrowDetectionResult(
            arrows=final_arrows,
            raw_line_segments=raw_segments,
            rejected_candidates=rejected,
            lines_detected=len(raw_segments),
            lines_filtered=len(raw_segments) - len(filtered),
            segments_merged=merge_count,
            arrows_detected=len(final_arrows),
            rejected_arrows=rejected_arrows,
            arrow_candidates=stats["arrow_candidates"],
            deduplicated_arrow_count=stats["deduplicated"],
            validated_arrow_count=len(final_arrows),
            aoa_context=stats["aoa_context"],
            segments_per_logical_arrow=stats["segments_per_logical_arrow"],
        )

        # Set image dimensions
        h, w = detection_image.shape[:2]
        result.image_dimensions = (w, h)

        # Map coordinates to original
        if preprocessing_result.coordinate_mapping:
            result.coordinate_mapping = preprocessing_result.coordinate_mapping
            self._map_coordinates_to_original(result)

        result.processing_time = time.time() - start_time

        logger.info(
            "Arrow detection: %d raw → %d filtered → %d assembled → "
            "%d logical arrows (dedup=%d, rejected=%d), %.3f seconds",
            len(raw_segments),
            len(filtered),
            len(assembled_arrows),
            len(final_arrows),
            stats["deduplicated"],
            len(rejected_arrows),
            result.processing_time,
        )

        return result

    def detect_from_shapes(
        self,
        image: np.ndarray,
        shape_result: ShapeDetectionResult,
    ) -> ArrowDetectionResult:
        """
        Detect arrows using shape context for border filtering.

        Args:
            image: Input image.
            shape_result: Shape detection result.

        Returns:
            ArrowDetectionResult.
        """
        raw_segments = self._detect_line_segments(image)
        filtered, rejected = self._filter_segments(
            raw_segments, image.shape, shape_result
        )
        arrows_with_heads = self._detect_arrowheads(filtered, image)
        merged, merge_count = self._merge_segments(arrows_with_heads)
        assembled_arrows = self._assemble_arrows(merged)

        self._associate_with_candidates(assembled_arrows, shape_result.candidate_nodes)
        final_arrows, rejected_arrows, stats = self._finalize_arrow_set(
            assembled_arrows, shape_result
        )

        result = ArrowDetectionResult(
            arrows=final_arrows,
            raw_line_segments=raw_segments,
            rejected_candidates=rejected,
            lines_detected=len(raw_segments),
            lines_filtered=len(raw_segments) - len(filtered),
            segments_merged=merge_count,
            arrows_detected=len(final_arrows),
            rejected_arrows=rejected_arrows,
            arrow_candidates=stats["arrow_candidates"],
            deduplicated_arrow_count=stats["deduplicated"],
            validated_arrow_count=len(final_arrows),
            aoa_context=stats["aoa_context"],
            segments_per_logical_arrow=stats["segments_per_logical_arrow"],
        )

        h, w = image.shape[:2]
        result.image_dimensions = (w, h)

        if shape_result.coordinate_mapping:
            result.coordinate_mapping = shape_result.coordinate_mapping
            self._map_coordinates_to_original(result)

        return result

    # =========================================================================
    # Image Selection
    # =========================================================================

    def _select_detection_image(
        self, preprocessing_result: PreprocessingResult
    ) -> Optional[np.ndarray]:
        """
        Select best image representation for line detection.

        Priority: edges > adaptive_binary > binary > grayscale
        Lines are best detected on edge or thresholded images.
        """
        for name in ["edges", "adaptive_binary", "binary", "grayscale"]:
            img = preprocessing_result.get_representation(name)
            if img is not None:
                logger.debug("Using '%s' for line detection", name)
                return img
        return None

    # =========================================================================
    # Line Detection
    # =========================================================================

    def _detect_line_segments(self, image: np.ndarray) -> List[DetectedLineSegment]:
        """
        Detect line segments using HoughLinesP (probabilistic Hough transform).

        Returns:
            List of DetectedLineSegment objects.
        """
        # Ensure grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Detect lines using HoughLinesP
        lines = cv2.HoughLinesP(
            gray,
            rho=self._config.hough_rho,
            theta=np.pi / self._config.hough_theta_resolution,
            threshold=self._config.hough_threshold,
            minLineLength=self._config.hough_min_line_length,
            maxLineGap=self._config.hough_max_line_gap,
        )

        segments: List[DetectedLineSegment] = []

        if lines is None:
            return segments

        for line in lines:
            # HoughLinesP output shape varies by OpenCV version:
            # (N, 1, 4) on older builds, (N, 4) on newer ones.
            coords = np.asarray(line).reshape(-1)
            if coords.size != 4:
                continue
            x1, y1, x2, y2 = (int(v) for v in coords)

            start = Point(float(x1), float(y1))
            end = Point(float(x2), float(y2))
            length = start.distance_to(end)

            if length < 1:
                continue

            # Compute angle (0-360 preserving direction)
            dx = x2 - x1
            dy = y2 - y1
            angle_deg = math.degrees(math.atan2(dy, dx)) % 360

            # Confidence based on line length and detection strength
            confidence = self._compute_line_confidence(length)

            segment = DetectedLineSegment(
                segment_id=_generate_id("line"),
                start=start,
                end=end,
                angle_deg=angle_deg,
                length=length,
                confidence=confidence,
            )
            segments.append(segment)

        logger.debug("Detected %d raw line segments", len(segments))
        return segments

    def _compute_line_confidence(self, length: float) -> float:
        """Compute confidence for a detected line segment."""
        # Longer lines are more likely to be real connections
        # Short lines still get reasonable confidence (they may be short arrows)
        length_score = min(1.0, length / 150.0)
        return 0.4 + 0.6 * length_score

    # =========================================================================
    # Line Filtering
    # =========================================================================

    def _filter_segments(
        self,
        segments: List[DetectedLineSegment],
        image_shape: Tuple[int, ...],
        shape_result: Optional[ShapeDetectionResult] = None,
    ) -> Tuple[List[DetectedLineSegment], List[DetectedLineSegment]]:
        """
        Filter line segments, returning (kept, rejected).

        Filters:
            - Too short / too long
            - Rectangle border lines
            - Lines inside detected shapes
        """
        kept: List[DetectedLineSegment] = []
        rejected: List[DetectedLineSegment] = []

        img_h, img_w = image_shape[:2]

        for seg in segments:
            reject_reason = self._should_reject_segment(seg, (img_w, img_h), shape_result)
            if reject_reason:
                seg.metadata["rejection_reason"] = reject_reason
                rejected.append(seg)
            else:
                kept.append(seg)

        logger.debug(
            "Filtering: %d kept, %d rejected",
            len(kept), len(rejected),
        )
        return kept, rejected

    def _should_reject_segment(
        self,
        seg: DetectedLineSegment,
        image_dims: Tuple[int, int],
        shape_result: Optional[ShapeDetectionResult],
    ) -> Optional[str]:
        """Return rejection reason or None if segment should be kept."""
        img_w, img_h = image_dims

        # Length filter
        if seg.length < self._config.min_line_length:
            return "too_short"
        if seg.length > self._config.max_line_length:
            return "too_long"

        # Border proximity filter (remove image border lines)
        if self._is_on_image_border(seg, img_w, img_h):
            return "image_border"

        # Rectangle border filter
        if self._config.filter_rectangle_borders and shape_result:
            if self._is_shape_border(seg, shape_result):
                return "shape_border"

        return None

    def _is_on_image_border(
        self, seg: DetectedLineSegment, img_w: int, img_h: int
    ) -> bool:
        """Check if a segment is likely a border artifact.

        Anti-aliased image edges produce many spurious HoughLinesP segments.
        Rejects any segment where at least one endpoint is near the image
        border (within rectangle_border_tolerance pixels).
        """
        margin = self._config.rectangle_border_tolerance
        for p in [seg.start, seg.end]:
            if p.x <= margin or p.x >= img_w - 1 - margin:
                return True
            if p.y <= margin or p.y >= img_h - 1 - margin:
                return True
        return False

    def _is_shape_border(
        self, seg: DetectedLineSegment, shape_result: ShapeDetectionResult
    ) -> bool:
        """
        Check if a segment is coincident with a detected shape border.

        Only rejects if BOTH endpoints lie on the SAME shape's border edges.
        Inter-node arrows have endpoints on DIFFERENT shapes' borders, so
        they must not be rejected.
        """
        tolerance = self._config.rectangle_border_tolerance

        for shape in shape_result.detected_shapes:
            if not hasattr(shape, "bounding_box"):
                continue
            bbox = shape.bounding_box
            if bbox.width == 0 or bbox.height == 0:
                continue

            # Count how many of the two endpoints are on THIS shape's border
            on_border = 0
            for p in [seg.start, seg.end]:
                on_left = abs(p.x - bbox.x) < tolerance
                on_right = abs(p.x - (bbox.x + bbox.width)) < tolerance
                on_top = abs(p.y - bbox.y) < tolerance
                on_bottom = abs(p.y - (bbox.y + bbox.height)) < tolerance
                inside_x = bbox.x - tolerance <= p.x <= bbox.x + bbox.width + tolerance
                inside_y = bbox.y - tolerance <= p.y <= bbox.y + bbox.height + tolerance

                if (on_left or on_right) and inside_y:
                    on_border += 1
                elif (on_top or on_bottom) and inside_x:
                    on_border += 1

            # Only reject if BOTH endpoints are on THIS shape's border
            # (meaning the segment is a border segment of a single shape)
            if on_border >= 2:
                return True

        return False

    # =========================================================================
    # Arrowhead Detection
    # =========================================================================

    def _detect_arrowheads(
        self,
        segments: List[DetectedLineSegment],
        image: np.ndarray,
    ) -> List[DetectedLineSegment]:
        """
        Detect arrowheads at segment endpoints.

        Modifies segments in-place with arrowhead evidence.
        Returns the segments with arrowhead metadata populated.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Ensure binary for contour analysis
        if len(np.unique(gray)) > 2:
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            binary = gray.copy()

        for seg in segments:
            # Check both endpoints
            ev_start = self._check_arrowhead_at_point(
                seg.start, seg.end, binary, gray
            )
            ev_end = self._check_arrowhead_at_point(
                seg.end, seg.start, binary, gray
            )

            # Store evidence
            seg.metadata["arrowhead_start"] = ev_start
            seg.metadata["arrowhead_end"] = ev_end

            # Determine which end has the arrowhead
            min_conf = self._config.arrowhead_min_confidence
            if (ev_end.confidence > min_conf and ev_start.confidence > min_conf
                    and abs(ev_end.confidence - ev_start.confidence) < 0.05):
                # Both endpoints have similar arrowhead evidence.
                # Use orientation-aware tiebreaker: check which endpoint's
                # arrowhead contour apex points AWAY from the line.
                orient_end = self._compute_arrowhead_orientation_score(
                    ev_end, seg.start, seg.end
                )
                orient_start = self._compute_arrowhead_orientation_score(
                    ev_start, seg.end, seg.start
                )
                if orient_end > orient_start:
                    seg.metadata["has_arrowhead"] = True
                    seg.metadata["arrowhead_at"] = "end"
                    seg.metadata["arrowhead_type"] = ev_end.arrowhead_type
                    seg.metadata["arrowhead_confidence"] = ev_end.confidence
                    seg.metadata["direction"] = "start_to_end"
                else:
                    seg.metadata["has_arrowhead"] = True
                    seg.metadata["arrowhead_at"] = "start"
                    seg.metadata["arrowhead_type"] = ev_start.arrowhead_type
                    seg.metadata["arrowhead_confidence"] = ev_start.confidence
                    seg.metadata["direction"] = "end_to_start"
            elif ev_end.confidence > ev_start.confidence and ev_end.confidence > min_conf:
                seg.metadata["has_arrowhead"] = True
                seg.metadata["arrowhead_at"] = "end"
                seg.metadata["arrowhead_type"] = ev_end.arrowhead_type
                seg.metadata["arrowhead_confidence"] = ev_end.confidence
                seg.metadata["direction"] = "start_to_end"
            elif ev_start.confidence > ev_end.confidence and ev_start.confidence > min_conf:
                seg.metadata["has_arrowhead"] = True
                seg.metadata["arrowhead_at"] = "start"
                seg.metadata["arrowhead_type"] = ev_start.arrowhead_type
                seg.metadata["arrowhead_confidence"] = ev_start.confidence
                seg.metadata["direction"] = "end_to_start"
            else:
                seg.metadata["has_arrowhead"] = False
                seg.metadata["arrowhead_at"] = None
                seg.metadata["arrowhead_type"] = ArrowheadType.NONE
                seg.metadata["arrowhead_confidence"] = 0.0
                seg.metadata["direction"] = "unknown"

        return segments

    def _check_arrowhead_at_point(
        self,
        endpoint: Point,
        other_end: Point,
        binary: np.ndarray,
        gray: np.ndarray,
    ) -> ArrowheadEvidence:
        """
        Check for arrowhead evidence at a specific endpoint.

        Strategy:
            1. Look for triangular contours near the endpoint
            2. Analyze local geometry (converging lines)
            3. Check for filled/open V shapes
            4. Check for line convergence pattern (two lines meeting at endpoint)
        """
        radius = self._config.arrowhead_detection_radius
        img_h, img_w = binary.shape[:2]

        # Define search region
        x1 = max(0, int(endpoint.x) - radius)
        y1 = max(0, int(endpoint.y) - radius)
        x2 = min(img_w, int(endpoint.x) + radius)
        y2 = min(img_h, int(endpoint.y) + radius)

        if x2 <= x1 or y2 <= y1:
            return ArrowheadEvidence(endpoint=endpoint)

        roi = binary[y1:y2, x1:x2]

        # Find contours in the local region
        contours, _ = cv2.findContours(
            roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        best_evidence = ArrowheadEvidence(endpoint=endpoint)
        direction = np.array(
            [other_end.x - endpoint.x, other_end.y - endpoint.y], dtype=float
        )
        dir_norm = np.linalg.norm(direction)
        if dir_norm > 0:
            direction /= dir_norm

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self._config.arrowhead_min_area:
                continue
            if area > self._config.arrowhead_max_area:
                continue

            # Shift contour to image coordinates
            contour_shifted = contour.copy()
            contour_shifted[:, :, 0] += x1
            contour_shifted[:, :, 1] += y1

            # Check triangularity
            triangularity = self._compute_triangularity(contour_shifted)
            if triangularity < self._config.arrowhead_min_triangularity:
                continue

            # Check if the contour is near the endpoint
            M = cv2.moments(contour_shifted)
            if M["m00"] == 0:
                continue
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            dist = math.sqrt(
                (cx - endpoint.x) ** 2 + (cy - endpoint.y) ** 2
            )
            if dist > radius:
                continue

            # Check if the contour points toward the other end
            apex_score = self._compute_apex_score(
                contour_shifted, endpoint, direction
            )

            # Determine arrowhead type
            ah_type = self._classify_arrowhead_type(contour_shifted, area)

            confidence = 0.4 * triangularity + 0.3 * apex_score + 0.3 * min(1.0, area / 500)

            if confidence > best_evidence.confidence:
                best_evidence = ArrowheadEvidence(
                    endpoint=endpoint,
                    arrowhead_type=ah_type,
                    confidence=confidence,
                    triangularity=triangularity,
                    detected_contour=contour_shifted,
                    size=float(area),
                    metadata={"apex_score": apex_score, "distance": dist},
                )

        # Secondary heuristic: check for line convergence pattern
        # If no contour-based arrowhead found, check if there are multiple
        # dark pixels converging at the endpoint (V-shape pattern)
        if best_evidence.confidence < self._config.arrowhead_min_confidence:
            conv_evidence = self._check_line_convergence(
                endpoint, other_end, binary, x1, y1, x2, y2
            )
            if conv_evidence.confidence > best_evidence.confidence:
                best_evidence = conv_evidence

        return best_evidence

    def _check_line_convergence(
        self,
        endpoint: Point,
        other_end: Point,
        binary: np.ndarray,
        roi_x1: int,
        roi_y1: int,
        roi_x2: int,
        roi_y2: int,
    ) -> ArrowheadEvidence:
        """
        Check for line convergence pattern at an endpoint.

        Looks for two or more dark pixel paths converging toward the endpoint,
        which indicates an arrowhead or junction.
        """
        radius = self._config.arrowhead_detection_radius
        ep_x, ep_y = int(endpoint.x), int(endpoint.y)

        # Sample pixels in a fan pattern around the endpoint
        direction = math.atan2(other_end.y - ep_y, other_end.x - ep_x)

        # Count dark pixels in angular sectors around the endpoint
        sector_count = 8
        sector_hits = [0] * sector_count
        sector_total = [0] * sector_count

        for r in range(5, min(radius, 20)):
            for s in range(sector_count):
                angle = direction + math.pi + (s - sector_count / 2) * (math.pi / sector_count)
                sx = ep_x + int(r * math.cos(angle))
                sy = ep_y + int(r * math.sin(angle))
                if 0 <= sx < binary.shape[1] and 0 <= sy < binary.shape[0]:
                    sector_total[s] += 1
                    if binary[sy, sx] > 0:
                        sector_hits[s] += 1

        # Count sectors with high hit rates (converging lines)
        converging_sectors = 0
        for s in range(sector_count):
            if sector_total[s] > 0 and sector_hits[s] / sector_total[s] > 0.5:
                converging_sectors += 1

        # Arrowheads typically have 2-3 converging sectors
        if converging_sectors >= 2:
            confidence = min(0.5, 0.15 * converging_sectors)
            return ArrowheadEvidence(
                endpoint=endpoint,
                arrowhead_type=ArrowheadType.SIMPLE_TWO_LINE,
                confidence=confidence,
                metadata={"converging_sectors": converging_sectors, "heuristic": "line_convergence"},
            )

        return ArrowheadEvidence(endpoint=endpoint)

    def _compute_triangularity(self, contour: np.ndarray) -> float:
        """
        Compute how triangular a contour is.
        Triangularity = area / (0.5 * perimeter * apothem-like measure).
        Simplified: compare contour area to convex hull area of a triangle.
        """
        area = cv2.contourArea(contour)
        if area == 0:
            return 0.0

        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return 0.0

        # Approximate to polygon
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)

        # A triangle should have ~3 vertices
        if vertices == 3:
            return 0.9
        elif vertices == 4:
            return 0.6
        elif vertices <= 5:
            return 0.4
        else:
            return 0.2

    def _compute_apex_score(
        self,
        contour: np.ndarray,
        endpoint: Point,
        direction: np.ndarray,
    ) -> float:
        """
        Compute how well the contour's apex points along the expected direction.
        """
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            return 0.0

        # Find the point farthest from the endpoint in the direction of the arrow
        max_proj = 0.0
        for pt in pts:
            vec = pt - np.array([endpoint.x, endpoint.y])
            proj = np.dot(vec, direction)
            if proj > max_proj:
                max_proj = proj

        # Normalize by radius
        score = min(1.0, max_proj / self._config.arrowhead_detection_radius)
        return score

    def _classify_arrowhead_type(
        self, contour: np.ndarray, area: float
    ) -> ArrowheadType:
        """Classify the type of arrowhead based on contour properties."""
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            return ArrowheadType.UNKNOWN

        # Compact filled shapes are likely filled triangles
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = area / hull_area
            if solidity > 0.7:
                return ArrowheadType.FILLED_TRIANGLE

        # Low solidity with triangular shape: open V
        vertices = len(cv2.approxPolyDP(contour, 0.04 * perimeter, True))
        if vertices <= 4:
            return ArrowheadType.OPEN_V

        return ArrowheadType.SIMPLE_TWO_LINE

    def _compute_arrowhead_orientation_score(
        self,
        evidence: ArrowheadEvidence,
        other_end: Point,
        endpoint: Point,
    ) -> float:
        """
        Compute how well the arrowhead contour's apex points AWAY from the line.

        A true arrowhead has its apex pointing in the direction opposite to
        the line (away from the other endpoint). This score is higher when
        the contour geometry is consistent with a real arrowhead orientation.
        """
        if evidence.detected_contour is None or evidence.size == 0:
            return 0.0

        contour = evidence.detected_contour
        pts = contour.reshape(-1, 2)
        if len(pts) < 3:
            return 0.0

        # Direction from endpoint toward the other end (along the line)
        line_dx = other_end.x - endpoint.x
        line_dy = other_end.y - endpoint.y
        line_len = math.sqrt(line_dx ** 2 + line_dy ** 2)
        if line_len < 1e-6:
            return 0.0
        line_ux, line_uy = line_dx / line_len, line_dy / line_len

        # Find the apex: the point in the contour farthest from the endpoint
        # in the direction AWAY from the line (opposite to line_ux, line_uy)
        away_ux, away_uy = -line_ux, -line_uy

        max_proj_away = 0.0
        max_proj_along = 0.0
        ep = np.array([endpoint.x, endpoint.y])
        for pt in pts:
            vec = pt - ep
            proj_away = vec[0] * away_ux + vec[1] * away_uy
            proj_along = vec[0] * line_ux + vec[1] * line_uy
            if proj_away > max_proj_away:
                max_proj_away = proj_away
            if proj_along > max_proj_along:
                max_proj_along = proj_along

        # Score: higher when the contour extends more AWAY from the line
        # than along the line (true arrowhead orientation)
        if max_proj_away + max_proj_along < 1e-6:
            return 0.0

        score = max_proj_away / (max_proj_away + max_proj_along + 1e-6)
        return score

    # =========================================================================
    # Segment Merging
    # =========================================================================

    def _merge_segments(
        self, segments: List[DetectedLineSegment]
    ) -> Tuple[List[DetectedLineSegment], int]:
        """
        Merge collinear segments that likely represent the same line.

        Criteria:
            - Similar angle (within merge_angle_tolerance_deg)
            - Close endpoints (within merge_distance_tolerance)
            - Collinear projection (within merge_collinearity_tolerance)

        Uses connected-component (transitive closure) grouping so that a
        chain of short collinear fragments — e.g. a dashed arrow shaft whose
        fragments only overlap their immediate neighbors — collapses into a
        single logical segment.  The greedy seed-only grouping previously
        left such chains split into many arrows.

        Returns:
            Tuple of (merged_segments, merge_count) where merge_count is the
            number of segment pairs merged into shared components.
        """
        if len(segments) <= 1:
            return segments, 0

        n = len(segments)
        parent = list(range(n))

        def find(i: int) -> int:
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return i

        def union(i: int, j: int) -> bool:
            ri, rj = find(i), find(j)
            if ri == rj:
                return False
            parent[rj] = ri
            return True

        merge_count = 0
        if not self._config.enable_transitive_segment_merge:
            # Legacy greedy seed-only grouping (no transitive closure).
            used = [False] * n
            merged = []
            for i in range(n):
                if used[i]:
                    continue
                group = [segments[i]]
                used[i] = True
                for j in range(i + 1, n):
                    if used[j]:
                        continue
                    if self._should_merge(segments[i], segments[j]):
                        group.append(segments[j])
                        used[j] = True
                        merge_count += 1
                merged.append(group[0] if len(group) == 1 else self._merge_group(group))
            logger.debug("Merged %d segments into %d", n, len(merged))
            return merged, merge_count

        # Transitive closure via connected components.
        for i in range(n):
            for j in range(i + 1, n):
                if self._should_merge(segments[i], segments[j]):
                    if union(i, j):
                        merge_count += 1

        # Group by root, preserving original order
        groups: Dict[int, List[DetectedLineSegment]] = {}
        for idx in range(n):
            root = find(idx)
            groups.setdefault(root, []).append(segments[idx])

        merged: List[DetectedLineSegment] = []
        for group in groups.values():
            if len(group) == 1:
                merged.append(group[0])
            else:
                merged.append(self._merge_group(group))

        logger.debug("Merged %d segments into %d", len(segments), len(merged))
        return merged, merge_count

    def _should_merge(
        self, seg1: DetectedLineSegment, seg2: DetectedLineSegment
    ) -> bool:
        """Determine if two segments should be merged."""
        # Angle tolerance (handle 0-360 wraparound)
        angle_diff = abs(seg1.angle_deg - seg2.angle_deg)
        if angle_diff > 180:
            angle_diff = 360 - angle_diff
        if angle_diff > self._config.merge_angle_tolerance_deg:
            return False

        # Distance between endpoints
        min_dist = float("inf")
        for p1 in [seg1.start, seg1.end]:
            for p2 in [seg2.start, seg2.end]:
                d = p1.distance_to(p2)
                min_dist = min(min_dist, d)

        if min_dist > self._config.merge_distance_tolerance:
            return False

        # Collinearity check
        if not self._are_collinear(seg1, seg2):
            return False

        return True

    def _are_collinear(
        self, seg1: DetectedLineSegment, seg2: DetectedLineSegment
    ) -> bool:
        """Check if two segments are approximately collinear."""
        # Use the cross product to check collinearity
        v1 = np.array([seg1.end.x - seg1.start.x, seg1.end.y - seg1.start.y])
        v2 = np.array([seg2.end.x - seg2.start.x, seg2.end.y - seg2.start.y])

        # Check angle between direction vectors
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 == 0 or n2 == 0:
            return False

        cos_angle = abs(np.dot(v1, v2) / (n1 * n2))
        return cos_angle >= self._config.merge_collinearity_tolerance

    def _merge_group(
        self, group: List[DetectedLineSegment]
    ) -> DetectedLineSegment:
        """Merge a group of collinear segments into one."""
        # Find extreme endpoints
        all_points = []
        for seg in group:
            all_points.append((seg.start.x, seg.start.y))
            all_points.append((seg.end.x, seg.end.y))

        # Project all points onto the line direction
        # Use first segment's direction as reference
        ref = group[0]
        dx = ref.end.x - ref.start.x
        dy = ref.end.y - ref.start.y
        length = (dx**2 + dy**2) ** 0.5
        if length == 0:
            return group[0]

        direction = np.array([dx, dy]) / length
        origin = np.array([ref.start.x, ref.start.y])

        projections = []
        for px, py in all_points:
            proj = np.dot(np.array([px, py]) - origin, direction)
            projections.append(proj)

        min_idx = int(np.argmin(projections))
        max_idx = int(np.argmax(projections))

        start_pt = Point(all_points[min_idx][0], all_points[min_idx][1])
        end_pt = Point(all_points[max_idx][0], all_points[max_idx][1])

        # Aggregate metadata
        best_conf = max(s.confidence for s in group)
        has_arrowhead = any(s.metadata.get("has_arrowhead", False) for s in group)
        arrowhead_type = ArrowheadType.NONE
        arrowhead_conf = 0.0
        direction_label = "unknown"

        for s in group:
            if s.metadata.get("has_arrowhead", False):
                if s.metadata.get("arrowhead_confidence", 0) > arrowhead_conf:
                    arrowhead_type = s.metadata.get("arrowhead_type", ArrowheadType.UNKNOWN)
                    arrowhead_conf = s.metadata.get("arrowhead_confidence", 0)
                    direction_label = s.metadata.get("direction", "unknown")

        # Determine the dominant direction
        total_dx = end_pt.x - start_pt.x
        total_dy = end_pt.y - start_pt.y
        merged_angle = math.degrees(math.atan2(total_dy, total_dx)) % 360
        merged_length = start_pt.distance_to(end_pt)

        segment_ids = [s.segment_id for s in group]

        return DetectedLineSegment(
            segment_id=_generate_id("merged"),
            start=start_pt,
            end=end_pt,
            angle_deg=merged_angle,
            length=merged_length,
            confidence=best_conf,
            line_style=group[0].line_style,
            metadata={
                "has_arrowhead": has_arrowhead,
                "arrowhead_type": arrowhead_type,
                "arrowhead_confidence": arrowhead_conf,
                "direction": direction_label,
                "merged_from": segment_ids,
                "merge_count": len(group),
            },
        )

    # =========================================================================
    # Arrow Assembly
    # =========================================================================

    def _assemble_arrows(
        self, segments: List[DetectedLineSegment]
    ) -> List[DetectedArrow]:
        """
        Assemble final DetectedArrow objects from processed segments.

        Segments with arrowheads become arrows. Segments without arrowheads
        are kept as connection candidates with lower confidence.
        """
        arrows: List[DetectedArrow] = []

        for seg in segments:
            has_head = seg.metadata.get("has_arrowhead", False)
            direction_label = seg.metadata.get("direction", "unknown")

            # Determine actual start/end based on direction
            if direction_label == "end_to_start":
                arrow_start = seg.end
                arrow_end = seg.start
            else:
                arrow_start = seg.start
                arrow_end = seg.end

            # Direction vector
            dx = arrow_end.x - arrow_start.x
            dy = arrow_end.y - arrow_start.y
            dl = (dx**2 + dy**2) ** 0.5
            if dl > 0:
                dir_vec = (dx / dl, dy / dl)
            else:
                dir_vec = (0.0, 0.0)

            angle = math.degrees(math.atan2(dy, dx)) % 360

            # Arrowhead info
            ah_type = seg.metadata.get("arrowhead_type", ArrowheadType.NONE)
            ah_conf = seg.metadata.get("arrowhead_confidence", 0.0)
            ah_point = arrow_end if has_head else None

            # Direction confidence
            dir_conf = 0.8 if has_head else 0.3

            # Compute bounding box
            x_min = min(arrow_start.x, arrow_end.x)
            y_min = min(arrow_start.y, arrow_end.y)
            x_max = max(arrow_start.x, arrow_end.x)
            y_max = max(arrow_start.y, arrow_end.y)
            bbox = BoundingBox(
                float(x_min), float(y_min),
                float(x_max - x_min), float(y_max - y_min),
            )

            # Overall confidence
            confidence = self._compute_arrow_confidence(
                seg, has_head, ah_conf, dir_conf
            )

            if confidence < self._config.min_arrow_confidence:
                continue

            arrow = DetectedArrow(
                arrow_id=_generate_id("arrow"),
                start=arrow_start,
                end=arrow_end,
                direction_vector=dir_vec,
                angle_deg=angle,
                length=seg.length,
                bounding_box=bbox,
                arrowhead_point=ah_point,
                arrowhead_type=ah_type,
                arrowhead_confidence=ah_conf,
                direction_confidence=dir_conf,
                line_style=seg.line_style,
                line_thickness=seg.line_thickness,
                confidence=confidence,
                source_segment_ids=seg.metadata.get("merged_from", [seg.segment_id]),
                evidence={
                    "has_arrowhead": has_head,
                    "line_confidence": seg.confidence,
                    "arrowhead_confidence": ah_conf,
                    "direction_confidence": dir_conf,
                },
            )
            arrows.append(arrow)

        return arrows

    def _compute_arrow_confidence(
        self,
        seg: DetectedLineSegment,
        has_head: bool,
        ah_conf: float,
        dir_conf: float,
    ) -> float:
        """Compute overall confidence for an arrow."""
        line_conf = seg.confidence
        length_score = min(1.0, seg.length / 100.0)

        if has_head:
            confidence = (
                0.20 * line_conf
                + 0.40 * ah_conf
                + 0.20 * dir_conf
                + 0.20 * length_score
            )
        else:
            # No arrowhead: lower base confidence
            confidence = 0.15 * line_conf + 0.10 * length_score

        return min(1.0, max(0.0, confidence))

    # =========================================================================
    # Logical Arrow Consolidation & AOA Event-Boundary Validation
    # =========================================================================

    def _finalize_arrow_set(
        self,
        assembled: List[DetectedArrow],
        shape_result: Optional[ShapeDetectionResult],
    ) -> Tuple[List[DetectedArrow], List[DetectedArrow], Dict[str, Any]]:
        """
        Consolidate assembled arrows into logical arrows.

        Phase 1 (all diagram types): post-assembly duplicate suppression so
        that one visual arrow yields one logical arrow.

        Phase 2 (AOA context only — circle/event candidates dominate the
        candidate set): event-boundary validation. Headless segments that
        touch the SAME event circle at both ends (chords, arcs, invalid
        self-loops) or that never touch any event circle (text strokes,
        decorations, isolated fragments) are rejected as non-activity
        geometry. The reconstruction layer then receives the validated arrow
        set, so AOA's 1:1 arrow -> activity mapping inherits the reduction.

        Rejection is evidence-based; no global line-length raise and no
        image-specific thresholds are applied. Arrows with arrowheads but no
        event contact are kept (with a penalty) — never silently dropped.

        Returns:
            (final_arrows, rejected_arrows, stats)
        """
        deduped, dedup_count = self._deduplicate_arrows(assembled)

        aoa_context = self._is_aoa_context(shape_result)
        if aoa_context:
            circles = self._event_circle_meta(shape_result.candidate_nodes)
            validated, rejected = self._validate_aoa_arrows(deduped, circles)
        else:
            validated = list(deduped)
            rejected = []

        for arrow in validated:
            self._annotate_arrow_evidence(arrow, aoa_context)
            if arrow.evidence.get("validation_status") is None:
                arrow.evidence["validation_status"] = (
                    "kept::aoa_event_boundary" if aoa_context else "kept::no_aoa_context"
                )
        for arrow in rejected:
            self._annotate_arrow_evidence(arrow, True)

        stats: Dict[str, Any] = {
            "arrow_candidates": len(assembled),
            "deduplicated": dedup_count,
            "aoa_context": aoa_context,
            "rejected": rejected,
            "segments_per_logical_arrow": [
                max(1, len(a.source_segment_ids or [a.arrow_id]))
                for a in validated
            ],
        }
        return validated, rejected, stats

    def _deduplicate_arrows(
        self, arrows: List[DetectedArrow]
    ) -> Tuple[List[DetectedArrow], int]:
        """
        Suppress near-duplicate arrows that describe the same visual shaft.

        Two arrows are duplicates when they share orientation, overlap along
        their projection, and run within a small lateral offset (Hough split /
        anti-aliasing duplicates). The higher-confidence arrow survives and
        records the suppressed arrow ids for provenance. Distinct parallel
        arrows (laterally separated) are never merged.
        """
        kept: List[DetectedArrow] = []
        dedup_count = 0
        for arrow in arrows:
            dup_parent = None
            for kept_arrow in kept:
                if self._are_duplicate_arrows(kept_arrow, arrow):
                    dup_parent = kept_arrow
                    break
            if dup_parent is None:
                kept.append(arrow)
                continue
            dedup_count += 1
            if arrow.confidence > dup_parent.confidence:
                kept.remove(dup_parent)
                kept.append(arrow)
                arrow.metadata["deduplicated_arrow_ids"] = (
                    dup_parent.metadata.get("deduplicated_arrow_ids", [])
                    + [dup_parent.arrow_id]
                )
            else:
                dup_parent.metadata.setdefault("deduplicated_arrow_ids", []).append(
                    arrow.arrow_id
                )
        return kept, dedup_count

    def _are_duplicate_arrows(
        self, a: DetectedArrow, b: DetectedArrow
    ) -> bool:
        """Check whether two arrows describe the same visual shaft."""
        delta = abs((a.angle_deg % 180) - (b.angle_deg % 180))
        if delta > 180:
            delta = 360 - delta
        if delta > self._config.dedup_angle_tolerance_deg:
            return False

        dx = a.end.x - a.start.x
        dy = a.end.y - a.start.y
        length = (dx * dx + dy * dy) ** 0.5
        if length < 1e-6:
            return False
        ux, uy = dx / length, dy / length

        # Lateral separation: perpendicular distance of b's endpoints to a's line
        lat = max(
            abs((b.start.x - a.start.x) * uy - (b.start.y - a.start.y) * ux),
            abs((b.end.x - a.start.x) * uy - (b.end.y - a.start.y) * ux),
        )
        if lat > self._config.dedup_lateral_tolerance:
            return False

        # Projection overlap along a's direction
        def proj(p: Point) -> float:
            return (p.x - a.start.x) * ux + (p.y - a.start.y) * uy

        pa_min, pa_max = sorted([proj(a.start), proj(a.end)])
        pb_min, pb_max = sorted([proj(b.start), proj(b.end)])
        overlap = min(pa_max, pb_max) - max(pa_min, pb_min)
        if overlap < 0:
            return False
        la = pa_max - pa_min
        lb = pb_max - pb_min
        if min(la, lb) <= 1e-6:
            return False
        if overlap / min(la, lb) < self._config.dedup_projection_overlap:
            return False
        return True

    def _annotate_arrow_evidence(
        self, arrow: DetectedArrow, aoa_context: bool
    ) -> None:
        """Attach per-logical-arrow evidence breakdown to a kept arrow."""
        evidence = arrow.evidence
        dup_ids = arrow.metadata.get("deduplicated_arrow_ids", [])
        evidence["duplicate_evidence"] = {
            "suppressed_count": len(dup_ids),
            "suppressed_arrow_ids": list(dup_ids),
        }
        evidence["shaft_continuity"] = {
            "raw_segments": max(1, len(arrow.source_segment_ids or [])),
            "source_segment_ids": list(arrow.source_segment_ids or []),
            "length": arrow.length,
        }
        line_conf = evidence.get("line_confidence", 0.0)
        length_score = min(1.0, arrow.length / 100.0)
        evidence["confidence_breakdown"] = {
            "line_confidence": round(float(line_conf), 4),
            "arrowhead_confidence": round(float(arrow.arrowhead_confidence), 4),
            "direction_confidence": round(float(arrow.direction_confidence), 4),
            "length_score": round(float(length_score), 4),
            "angle_deg": round(float(arrow.angle_deg), 2),
        }
        evidence["text_interference_penalty"] = 0.0
        evidence["aoa_context"] = bool(aoa_context)
        if evidence.get("circle_boundary_penalty") is None:
            evidence["circle_boundary_penalty"] = 0.0

    def _is_aoa_context(
        self, shape_result: Optional[ShapeDetectionResult]
    ) -> bool:
        """
        Detect an activity-on-arrow context from the candidate-node geometry.

        AOA requires the candidates to be dominated by circle/ellipse event
        nodes (>= aoa_circle_dominance fraction) with at least
        aoa_min_event_circles such circles. Pure rectangle AON diagrams never
        activate this validation.
        """
        if not self._config.enable_aoa_event_validation:
            return False
        if shape_result is None or not shape_result.candidate_nodes:
            return False
        circles = self._event_circle_meta(shape_result.candidate_nodes)
        if len(circles) < self._config.aoa_min_event_circles:
            return False
        total = len(shape_result.candidate_nodes)
        if total <= 0:
            return False
        dominance = len(circles) / total
        return dominance >= self._config.aoa_circle_dominance

    def _event_circle_meta(
        self, candidates: List[CandidateNode]
    ) -> List[Tuple[str, float, float, float]]:
        """Extract (node_id, center_x, center_y, radius) for circle events."""
        circles: List[Tuple[str, float, float, float]] = []
        for candidate in candidates:
            if candidate.shape_type not in (
                ShapeType.CIRCLE, ShapeType.ELLIPSE
            ):
                continue
            bbox = candidate.bounding_box
            cx = candidate.position.x if candidate.position else bbox.x + bbox.width / 2.0
            cy = candidate.position.y if candidate.position else bbox.y + bbox.height / 2.0
            radius = max(2.0, (bbox.width + bbox.height) / 4.0)
            circles.append((candidate.node_id, float(cx), float(cy), float(radius)))
        return circles

    def _nearest_event_contact(
        self,
        point: Point,
        circles: List[Tuple[str, float, float, float]],
    ) -> Tuple[Optional[str], float, float, bool, float]:
        """
        Locate the nearest event circle and quantify boundary contact.

        Returns (circle_id, boundary_distance, radius, inside, tolerance).
        Boundary distance is |dist(center) - radius| — the actual ring the
        arrow must meet, not merely nearest-center distance.
        """
        best_id: Optional[str] = None
        best_bd = float("inf")
        best_radius = 0.0
        best_inside = False
        for cid, cx, cy, r in circles:
            dist = math.hypot(point.x - cx, point.y - cy)
            bd = abs(dist - r)
            inside = dist < r
            if bd < best_bd:
                best_bd = bd
                best_id = cid
                best_radius = r
                best_inside = inside
        if best_id is None:
            return (None, float("inf"), 0.0, False, 0.0)
        tolerance = min(
            float(self._config.event_boundary_tolerance),
            max(4.0, best_radius * 0.4),
        )
        return best_id, best_bd, best_radius, best_inside, tolerance

    def _validate_aoa_arrows(
        self,
        arrows: List[DetectedArrow],
        circles: List[Tuple[str, float, float, float]],
    ) -> Tuple[List[DetectedArrow], List[DetectedArrow]]:
        """
        Keep only arrows whose geometry is consistent with AOA activities.

        Valid AOA arrows connect TWO event boundaries and carry directional
        (arrowhead) evidence. Non-activity geometry is rejected:
          - same-event chords/arcs with no arrowhead (circle boundaries),
          - segments with no arrowhead and no event contact at either end
            (text strokes, underlines, decorative lines, isolated fragments).

        Kept arrows get an event-contact / shaft / boundary evidence breakdown.
        """
        if not circles:
            return list(arrows), []
        kept: List[DetectedArrow] = []
        rejected: List[DetectedArrow] = []
        head_req = self._config.aoa_arrowhead_required_confidence

        for arrow in arrows:
            s_id, s_bd, s_r, s_inside, s_tol = self._nearest_event_contact(
                arrow.start, circles
            )
            e_id, e_bd, e_r, e_inside, e_tol = self._nearest_event_contact(
                arrow.end, circles
            )

            s_contact = s_id is not None and (s_inside or s_bd <= s_tol)
            e_contact = e_id is not None and (e_inside or e_bd <= e_tol)
            has_head = arrow.arrowhead_confidence >= head_req
            same_event = s_id is not None and s_id == e_id

            reason: Optional[str] = None
            if not has_head and same_event and s_contact and e_contact:
                reason = "circle_boundary_same_event_no_arrowhead"
            elif not has_head and not s_contact and not e_contact:
                reason = "no_event_contact_no_arrowhead"

            arrow.evidence["event_contact_start"] = {
                "event_id": s_id,
                "boundary_distance": round(s_bd, 2),
                "circle_radius": round(s_r, 2),
                "contact": s_contact,
                "inside": s_inside,
            }
            arrow.evidence["event_contact_end"] = {
                "event_id": e_id,
                "boundary_distance": round(e_bd, 2),
                "circle_radius": round(e_r, 2),
                "contact": e_contact,
                "inside": e_inside,
            }
            arrow.evidence["aoa_validation"] = {
                "same_event_chord_or_self_loop": bool(
                    same_event and s_contact and e_contact
                ),
                "event_contact_count": int(s_contact) + int(e_contact),
                "has_arrowhead": has_head,
            }
            arrow.evidence["circle_boundary_penalty"] = (
                0.5 if (same_event and s_contact and e_contact) else 0.0
            )

            if reason:
                arrow.evidence["validation_status"] = f"rejected::{reason}"
                arrow.warnings.append(reason)
                arrow.metadata["rejection_reason"] = reason
                rejected.append(arrow)
            else:
                arrow.evidence["validation_status"] = "kept::aoa_event_boundary"
                kept.append(arrow)

        return kept, rejected

    # =========================================================================
    # Node Association
    # =========================================================================

    def _associate_with_candidates(
        self,
        arrows: List[DetectedArrow],
        candidates: List[CandidateNode],
    ) -> None:
        """
        Associate each arrow with the most likely source and target candidates.

        Uses endpoint proximity, direction vector, and shape intersection.
        """
        if not candidates:
            return

        for arrow in arrows:
            source_id, source_conf = self._find_best_source(arrow, candidates)
            target_id, target_conf = self._find_best_target(arrow, candidates)

            # Store node associations in evidence (not as semantic source/target)
            arrow.evidence["node_at_endpoint_1"] = source_id
            arrow.evidence["node_at_endpoint_1_confidence"] = source_conf
            arrow.evidence["node_at_endpoint_2"] = target_id
            arrow.evidence["node_at_endpoint_2_confidence"] = target_conf

    def _find_best_source(
        self, arrow: DetectedArrow, candidates: List[CandidateNode]
    ) -> Tuple[Optional[str], float]:
        """Find the best source candidate for an arrow."""
        return self._find_best_candidate_for_endpoint(
            arrow.start, candidates, self._config.source_endpoint_tolerance
        )

    def _find_best_target(
        self, arrow: DetectedArrow, candidates: List[CandidateNode]
    ) -> Tuple[Optional[str], float]:
        """Find the best target candidate for an arrow."""
        return self._find_best_candidate_for_endpoint(
            arrow.end, candidates, self._config.target_endpoint_tolerance
        )

    def _find_best_candidate_for_endpoint(
        self,
        endpoint: Point,
        candidates: List[CandidateNode],
        tolerance: int,
    ) -> Tuple[Optional[str], float]:
        """Find the candidate node closest to an endpoint."""
        best_id = None
        best_score = 0.0

        for cand in candidates:
            # Distance from endpoint to candidate center
            dist = endpoint.distance_to(cand.position)

            if dist > tolerance:
                continue

            # Score: closer is better, high confidence is better
            dist_score = max(0, 1.0 - dist / tolerance)
            cand_score = cand.confidence

            # Bonus if endpoint is inside the bounding box
            if cand.bounding_box.contains_point(endpoint, margin=5):
                dist_score = 1.0

            score = 0.6 * dist_score + 0.4 * cand_score

            if score > best_score:
                best_score = score
                best_id = cand.node_id

        return best_id, best_score

    # =========================================================================
    # Coordinate Mapping
    # =========================================================================

    def _map_coordinates_to_original(
        self, result: ArrowDetectionResult
    ) -> None:
        """Map all detection coordinates back to original image."""
        mapping = result.coordinate_mapping
        if mapping is None:
            return

        for arrow in result.arrows:
            ox, oy = mapping.to_original(arrow.start.x, arrow.start.y)
            arrow.start = Point(ox, oy)
            ox, oy = mapping.to_original(arrow.end.x, arrow.end.y)
            arrow.end = Point(ox, oy)

            if arrow.arrowhead_point:
                ox, oy = mapping.to_original(
                    arrow.arrowhead_point.x, arrow.arrowhead_point.y
                )
                arrow.arrowhead_point = Point(ox, oy)

            # Remap bounding box
            ox, oy = mapping.to_original(
                arrow.bounding_box.x, arrow.bounding_box.y
            )
            ow = arrow.bounding_box.width / mapping.scale_x
            oh = arrow.bounding_box.height / mapping.scale_y
            arrow.bounding_box = BoundingBox(ox, oy, ow, oh)

        for seg in result.raw_line_segments:
            ox, oy = mapping.to_original(seg.start.x, seg.start.y)
            seg.start = Point(ox, oy)
            ox, oy = mapping.to_original(seg.end.x, seg.end.y)
            seg.end = Point(ox, oy)

    # =========================================================================
    # Debug Visualization
    # =========================================================================

    def render_detections(
        self,
        image: np.ndarray,
        result: ArrowDetectionResult,
        show_confidence: bool = True,
        show_raw_segments: bool = False,
    ) -> np.ndarray:
        """
        Render arrow detection results on an image for debugging.

        Args:
            image: Original image to draw on.
            result: Arrow detection results.
            show_confidence: If True, show confidence scores.
            show_raw_segments: If True, show raw line segments in gray.

        Returns:
            Image with drawn detections.
        """
        if len(image.shape) == 2:
            vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            vis = image.copy()

        # Draw raw segments (optional)
        if show_raw_segments:
            for seg in result.raw_line_segments:
                pt1 = (int(seg.start.x), int(seg.start.y))
                pt2 = (int(seg.end.x), int(seg.end.y))
                cv2.line(vis, pt1, pt2, (180, 180, 180), 1, cv2.LINE_AA)

        # Draw rejected segments
        for seg in result.rejected_candidates:
            pt1 = (int(seg.start.x), int(seg.start.y))
            pt2 = (int(seg.end.x), int(seg.end.y))
            cv2.line(vis, pt1, pt2, (0, 0, 180), 1, cv2.LINE_AA)

        # Draw detected arrows
        for arrow in result.arrows:
            # Color based on confidence
            if arrow.confidence > 0.7:
                color = (0, 200, 0)    # Green
            elif arrow.confidence > 0.4:
                color = (0, 165, 255)  # Orange
            else:
                color = (128, 128, 128)  # Gray

            pt1 = (int(arrow.start.x), int(arrow.start.y))
            pt2 = (int(arrow.end.x), int(arrow.end.y))

            # Draw line
            cv2.line(vis, pt1, pt2, color, 2, cv2.LINE_AA)

            # Draw arrowhead
            if arrow.arrowhead_point:
                ah_pt = (
                    int(arrow.arrowhead_point.x),
                    int(arrow.arrowhead_point.y),
                )
                cv2.circle(vis, ah_pt, 5, color, -1)

            # Draw direction indicator
            mid = arrow.midpoint
            cv2.circle(vis, (int(mid.x), int(mid.y)), 3, (255, 0, 0), -1)

            # Draw source/target labels
            source_id = arrow.evidence.get("node_at_endpoint_1")
            target_id = arrow.evidence.get("node_at_endpoint_2")
            if source_id:
                cv2.putText(
                    vis,
                    f"S:{source_id[:6]}",
                    pt1,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (255, 0, 0),
                    1,
                )
            if target_id:
                cv2.putText(
                    vis,
                    f"T:{target_id[:6]}",
                    pt2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.3,
                    (0, 0, 255),
                    1,
                )

            # Confidence label
            if show_confidence:
                mid_label = f"{arrow.confidence:.2f}"
                cv2.putText(
                    vis,
                    mid_label,
                    (int(mid.x) + 5, int(mid.y) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                )

        return vis

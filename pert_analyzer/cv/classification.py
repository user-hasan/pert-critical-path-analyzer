"""
Diagram type classification module.

Classifies network diagrams as AON (Activity-on-Node) or AOA
(Activity-on-Arrow) based on shape distribution and geometric evidence.

This is an initial classifier that produces probabilistic results
with confidence scores. It does NOT claim certainty.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pert_analyzer.core.interfaces import DiagramClassifier as DiagramClassifierABC
from pert_analyzer.core.models import Arrow, DetectedShape, DiagramType, OCRResult
from pert_analyzer.cv.exceptions import ClassificationError
from pert_analyzer.cv.models import (
    CandidateNode,
    DiagramClassificationResult,
    ShapeDetectionResult,
    ShapeType,
)

logger = logging.getLogger(__name__)


class DiagramClassifier(DiagramClassifierABC):
    """
    Classifies network diagrams as AON or AOA based on shape evidence.

    Uses shape counts, geometric properties, and spatial distribution
    to produce a probabilistic classification with confidence.

    Usage:
        classifier = DiagramClassifier()
        result = classifier.classify_from_detection(shape_result)
    """

    def __init__(self):
        """Initialize the classifier."""
        self._last_confidence: float = 0.0
        self._last_result: Optional[DiagramClassificationResult] = None

    # =========================================================================
    # Interface Implementation
    # =========================================================================

    def classify(
        self,
        shapes: List[DetectedShape],
        arrows: List[Arrow],
        ocr_results: Optional[List[OCRResult]] = None,
    ) -> Tuple[str, float]:
        """
        Classify the diagram type.

        Args:
            shapes: Detected shapes in the image.
            arrows: Detected arrows (currently unused, available for Phase 5).
            ocr_results: OCR results (currently unused, available for Phase 6).

        Returns:
            Tuple of (diagram_type, confidence).
        """
        # Build a minimal detection result
        detection_result = ShapeDetectionResult(
            detected_shapes=shapes,
        )

        result = self.classify_from_detection(detection_result)
        return result.diagram_type, result.confidence

    def get_supported_types(self) -> List[str]:
        """Return list of supported diagram types."""
        return ["AON", "AOA", "UNKNOWN"]

    def get_confidence(self) -> float:
        """Return the confidence of the last classification."""
        return self._last_confidence

    # =========================================================================
    # Enhanced Classification
    # =========================================================================

    def classify_from_detection(
        self, detection_result: ShapeDetectionResult
    ) -> DiagramClassificationResult:
        """
        Classify diagram type from shape detection results.

        Analyzes shape distribution to determine if the diagram is
        more likely AON or AOA.

        Args:
            detection_result: Result from shape detection.

        Returns:
            DiagramClassificationResult with type, confidence, and evidence.
        """
        shapes = detection_result.detected_shapes
        candidates = detection_result.candidate_nodes

        result = DiagramClassificationResult()

        if not shapes:
            result.diagram_type = "UNKNOWN"
            result.confidence = 0.0
            result.warnings.append("No shapes detected for classification")
            self._last_result = result
            self._last_confidence = 0.0
            return result

        # Count shape types
        rect_count, circle_count, poly_count = self._count_shapes(shapes)
        result.rectangle_count = rect_count
        result.circle_count = circle_count
        result.polygon_count = poly_count

        # Compute size statistics
        size_stats = self._compute_size_stats(candidates)

        # Compute evidence scores
        aon_evidence = self._compute_aon_evidence(
            rect_count, circle_count, poly_count, size_stats
        )
        aoa_evidence = self._compute_aoa_evidence(
            rect_count, circle_count, poly_count, size_stats
        )

        # Normalize scores
        total = aon_evidence + aoa_evidence
        if total > 0:
            aon_score = aon_evidence / total
            aoa_score = aoa_evidence / total
        else:
            aon_score = 0.0
            aoa_score = 0.0

        # Determine classification
        confidence_threshold = 0.3  # Minimum confidence for non-UNKNOWN
        min_shapes_for_classification = 2

        total_shapes = rect_count + circle_count + poly_count

        if total_shapes < min_shapes_for_classification:
            result.diagram_type = "UNKNOWN"
            result.confidence = 0.0
            result.warnings.append(
                f"Insufficient shapes for classification "
                f"({total_shapes} found, need {min_shapes_for_classification})"
            )
        elif aon_score > aoa_score and aon_score > confidence_threshold:
            result.diagram_type = "AON"
            result.confidence = aon_score
            result.alternative_type = "AOA"
            result.alternative_confidence = aoa_score
        elif aoa_score > aon_score and aoa_score > confidence_threshold:
            result.diagram_type = "AOA"
            result.confidence = aoa_score
            result.alternative_type = "AON"
            result.alternative_confidence = aon_score
        else:
            result.diagram_type = "UNKNOWN"
            result.confidence = max(aon_score, aoa_score)
            result.warnings.append(
                "Classification confidence below threshold; "
                "diagram type ambiguous"
            )

        # Store evidence
        result.evidence = {
            "rectangle_count": rect_count,
            "circle_count": circle_count,
            "polygon_count": poly_count,
            "total_shapes": total_shapes,
            "aon_score": aon_score,
            "aoa_score": aoa_score,
            "aon_evidence_raw": aon_evidence,
            "aoa_evidence_raw": aoa_evidence,
            "size_stats": size_stats,
        }

        self._last_result = result
        self._last_confidence = result.confidence

        logger.info(
            "Classification: %s (conf=%.2f) — %d rects, %d circles, %d polygons",
            result.diagram_type,
            result.confidence,
            rect_count,
            circle_count,
            poly_count,
        )

        return result

    # =========================================================================
    # Evidence Computation
    # =========================================================================

    def _count_shapes(
        self, shapes: List[DetectedShape]
    ) -> Tuple[int, int, int]:
        """Count rectangles, circles, and polygons."""
        rect_count = 0
        circle_count = 0
        poly_count = 0

        for shape in shapes:
            st = shape.shape_type if hasattr(shape, "shape_type") else "unknown"
            if st in ("rectangle", "square"):
                rect_count += 1
            elif st in ("circle", "ellipse"):
                circle_count += 1
            elif st == "polygon":
                poly_count += 1

        return rect_count, circle_count, poly_count

    def _compute_size_stats(
        self, candidates: List[CandidateNode]
    ) -> Dict[str, float]:
        """Compute size statistics for candidate nodes."""
        if not candidates:
            return {"mean_area": 0, "std_area": 0, "size_variance": 0}

        areas = [c.area for c in candidates if c.area > 0]
        if not areas:
            return {"mean_area": 0, "std_area": 0, "size_variance": 0}

        mean_area = float(np.mean(areas))
        std_area = float(np.std(areas))
        cv = std_area / mean_area if mean_area > 0 else 0

        return {
            "mean_area": mean_area,
            "std_area": std_area,
            "size_variance": cv,
        }

    def _compute_aon_evidence(
        self,
        rect_count: int,
        circle_count: int,
        poly_count: int,
        size_stats: Dict[str, float],
    ) -> float:
        """
        Compute evidence score for AON classification.

        AON diagrams typically have:
        - Many rectangular activity nodes
        - Few or no circular event nodes
        - Consistent rectangle sizes
        """
        evidence = 0.0

        # Rectangle dominance
        total = rect_count + circle_count + poly_count
        if total > 0:
            rect_ratio = rect_count / total
            evidence += rect_ratio * 0.4

        # Absolute rectangle count
        if rect_count >= 3:
            evidence += 0.3
        elif rect_count >= 2:
            evidence += 0.2
        elif rect_count >= 1:
            evidence += 0.1

        # Circle penalty (AOA uses circles)
        if circle_count > rect_count:
            evidence -= 0.2
        elif circle_count > 0:
            evidence -= 0.1

        # Size consistency bonus
        if size_stats["size_variance"] < 0.3 and rect_count >= 2:
            evidence += 0.15

        return max(0.0, evidence)

    def _compute_aoa_evidence(
        self,
        rect_count: int,
        circle_count: int,
        poly_count: int,
        size_stats: Dict[str, float],
    ) -> float:
        """
        Compute evidence score for AOA classification.

        AOA diagrams typically have:
        - Many circular event nodes
        - Few or no rectangular nodes
        - Activities appear on arrows (not as shapes)
        """
        evidence = 0.0

        # Circle dominance
        total = rect_count + circle_count + poly_count
        if total > 0:
            circle_ratio = circle_count / total
            evidence += circle_ratio * 0.4

        # Absolute circle count
        if circle_count >= 3:
            evidence += 0.3
        elif circle_count >= 2:
            evidence += 0.2
        elif circle_count >= 1:
            evidence += 0.1

        # Rectangle penalty (AON uses rectangles)
        if rect_count > circle_count:
            evidence -= 0.2
        elif rect_count > 0:
            evidence -= 0.1

        # Size consistency for circles
        if size_stats["size_variance"] < 0.3 and circle_count >= 2:
            evidence += 0.15

        return max(0.0, evidence)

    # =========================================================================
    # Utility
    # =========================================================================

    def get_last_result(
        self,
    ) -> Optional[DiagramClassificationResult]:
        """Get the last classification result."""
        return self._last_result

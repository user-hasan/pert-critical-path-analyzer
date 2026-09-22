"""
Debug rendering utilities for OCR results.

Draws OCR bounding boxes, recognized text, confidence scores,
shape associations, arrow associations, and ambiguity indicators
on images for visual debugging.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    OCRTextRegion,
    OCRProcessingResult,
    TextAssociationResult,
    TextType,
)


# Color palette for different text types
TEXT_TYPE_COLORS: Dict[TextType, Tuple[int, int, int]] = {
    TextType.UNKNOWN: (128, 128, 128),          # Gray
    TextType.ACTIVITY_ID_CANDIDATE: (0, 200, 0),  # Green
    TextType.TEXT_LABEL_CANDIDATE: (0, 150, 255),  # Blue
    TextType.NUMERIC_CANDIDATE: (255, 100, 0),   # Orange
    TextType.POSSIBLE_PERT_VALUE: (255, 0, 255),  # Magenta
    TextType.DURATION_CANDIDATE: (0, 200, 200),   # Cyan
    TextType.EVENT_LABEL: (200, 200, 0),          # Yellow
}


class OCRDebugRenderer:
    """
    Renders OCR results on images for visual debugging.

    Draws bounding boxes, text labels, confidence scores,
    association lines, and ambiguity indicators.
    """

    def __init__(self, font_scale: float = 0.4, thickness: int = 1):
        """
        Initialize renderer.

        Args:
            font_scale: Font size for text rendering.
            thickness: Line thickness for drawings.
        """
        self.font_scale = font_scale
        self.thickness = thickness

    def render_regions(
        self,
        image: np.ndarray,
        regions: List[OCRTextRegion],
        show_confidence: bool = True,
        show_text_type: bool = True,
    ) -> np.ndarray:
        """
        Render OCR text regions on an image.

        Args:
            image: Input image (will be copied).
            regions: OCR text regions to render.
            show_confidence: Whether to show confidence scores.
            show_text_type: Whether to show text type labels.

        Returns:
            Image with rendered OCR regions.
        """
        import cv2

        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        for region in regions:
            color = TEXT_TYPE_COLORS.get(
                region.text_type, TEXT_TYPE_COLORS[TextType.UNKNOWN]
            )
            bbox = region.bounding_box

            # Draw bounding box
            x1 = int(bbox.x)
            y1 = int(bbox.y)
            x2 = int(bbox.x + bbox.width)
            y2 = int(bbox.y + bbox.height)
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)

            # Draw text label
            label_parts = [region.text[:20]]  # Truncate long text
            if show_text_type and region.text_type != TextType.UNKNOWN:
                label_parts.append(f"[{region.text_type.value[:8]}]")
            if show_confidence:
                label_parts.append(f"{region.confidence:.2f}")

            label = " ".join(label_parts)
            text_y = max(y1 - 5, 15)
            cv2.putText(
                output, label, (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                color, self.thickness, cv2.LINE_AA,
            )

        return output

    def render_associations(
        self,
        image: np.ndarray,
        regions: List[OCRTextRegion],
        association_results: List[TextAssociationResult],
        candidate_nodes: Optional[List[Any]] = None,
        detected_arrows: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Render text associations on an image.

        Args:
            image: Input image (will be copied).
            regions: OCR text regions.
            association_results: Association results for each region.
            candidate_nodes: Candidate nodes (for drawing shapes).
            detected_arrows: Detected arrows (for drawing arrows).

        Returns:
            Image with rendered associations.
        """
        import cv2

        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        # Build lookup for association results
        assoc_lookup = {a.text_region_id: a for a in association_results}

        for region in regions:
            assoc = assoc_lookup.get(region.region_id)
            if not assoc:
                continue

            text_center = region.bounding_box.center
            color = TEXT_TYPE_COLORS.get(
                region.text_type, TEXT_TYPE_COLORS[TextType.UNKNOWN]
            )

            if assoc.best_association:
                target_id = assoc.best_association.candidate_target_id
                target_type = assoc.best_association.target_type

                # Find target center
                target_center = None
                if target_type == AssociationTargetType.NODE and candidate_nodes:
                    for node in candidate_nodes:
                        if hasattr(node, "node_id") and node.node_id == target_id:
                            target_center = node.bounding_box.center
                            break
                elif target_type == AssociationTargetType.ARROW and detected_arrows:
                    for arrow in detected_arrows:
                        if hasattr(arrow, "arrow_id") and arrow.arrow_id == target_id:
                            target_center = arrow.midpoint
                            break

                if target_center:
                    # Draw association line
                    cv2.line(
                        output,
                        (int(text_center.x), int(text_center.y)),
                        (int(target_center.x), int(target_center.y)),
                        color, 1, cv2.LINE_AA,
                    )

                    # Draw arrowhead on association line
                    self._draw_arrowhead(
                        output, text_center, target_center, color
                    )

            # Draw ambiguity indicator
            if assoc.is_ambiguous:
                self._draw_ambiguity_marker(output, region.bounding_box, color)

            # Draw confidence score
            score = assoc.best_association.association_score if assoc.best_association else 0.0
            score_text = f"S:{score:.2f}"
            cv2.putText(
                output, score_text,
                (int(region.bounding_box.x), int(region.bounding_box.y + region.bounding_box.height + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA,
            )

        return output

    def render_full(
        self,
        image: np.ndarray,
        ocr_result: OCRProcessingResult,
        candidate_nodes: Optional[List[Any]] = None,
        detected_arrows: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Render complete OCR results on an image.

        Args:
            image: Input image.
            ocr_result: Complete OCR processing result.
            candidate_nodes: Candidate nodes.
            detected_arrows: Detected arrows.

        Returns:
            Fully rendered debug image.
        """
        output = image.copy()

        # Render regions
        output = self.render_regions(
            output, ocr_result.regions,
            show_confidence=True, show_text_type=True,
        )

        # Render associations
        if ocr_result.association_results:
            output = self.render_associations(
                output, ocr_result.regions, ocr_result.association_results,
                candidate_nodes, detected_arrows,
            )

        # Render groups
        if ocr_result.groups:
            output = self._render_groups(output, ocr_result.groups)

        return output

    def _draw_arrowhead(
        self,
        image: np.ndarray,
        from_point: Point,
        to_point: Point,
        color: Tuple[int, int, int],
        size: int = 8,
    ) -> None:
        """Draw a small arrowhead on the association line."""
        import cv2
        import math

        dx = to_point.x - from_point.x
        dy = to_point.y - from_point.y
        length = math.sqrt(dx * dx + dy * dy)
        if length == 0:
            return

        # Unit vector
        ux, uy = dx / length, dy / length

        # Arrowhead at 70% along the line
        tip_x = from_point.x + 0.7 * dx
        tip_y = from_point.y + 0.7 * dy

        # Perpendicular
        px, py = -uy, ux

        # Arrowhead points
        p1 = (int(tip_x + size * ux + size * 0.4 * px),
               int(tip_y + size * uy + size * 0.4 * py))
        p2 = (int(tip_x + size * ux - size * 0.4 * px),
               int(tip_y + size * uy - size * 0.4 * py))
        tip = (int(tip_x), int(tip_y))

        cv2.fillPoly(image, [np.array([p1, tip, p2])], color)

    def _draw_ambiguity_marker(
        self,
        image: np.ndarray,
        bbox: BoundingBox,
        color: Tuple[int, int, int],
    ) -> None:
        """Draw an ambiguity marker (double border) around a text region."""
        import cv2

        x1 = int(bbox.x) - 2
        y1 = int(bbox.y) - 2
        x2 = int(bbox.x + bbox.width) + 2
        y2 = int(bbox.y + bbox.height) + 2
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def _render_groups(
        self,
        image: np.ndarray,
        groups: List[Any],
    ) -> np.ndarray:
        """Render text groups on the image."""
        import cv2

        output = image
        for group in groups:
            if group.combined_bounding_box:
                bbox = group.combined_bounding_box
                x1 = int(bbox.x) - 3
                y1 = int(bbox.y) - 3
                x2 = int(bbox.x + bbox.width) + 3
                y2 = int(bbox.y + bbox.height) + 3
                cv2.rectangle(
                    output, (x1, y1), (x2, y2),
                    (200, 200, 200), 1, cv2.LINE_AA,
                )
                # Draw group ID
                cv2.putText(
                    output, f"G:{group.group_id[:6]}",
                    (x1, y2 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                    (200, 200, 200), 1, cv2.LINE_AA,
                )
        return output

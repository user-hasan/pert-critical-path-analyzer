"""
Spatial association engine for text-to-shape and text-to-arrow mapping.

Associates OCR text regions with detected shapes (nodes) and arrows
based on geometric features: bounding box containment, overlap ratio,
center distance, relative position, and proximity.

Produces structured association evidence, NOT final semantic dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    OCRTextRegion,
    TextAssociation,
    TextAssociationResult,
)
from pert_analyzer.cv.models import CandidateNode, DetectedArrow


@dataclass
class AssociationConfig:
    """Configuration for spatial association."""

    # Containment
    containment_threshold: float = 0.5  # Min overlap ratio for containment

    # Distance thresholds
    max_center_distance: float = 200.0
    max_arrow_perpendicular_distance: float = 50.0
    max_arrow_projection_distance: float = 100.0

    # Scoring weights
    containment_weight: float = 0.4
    distance_weight: float = 0.3
    overlap_weight: float = 0.2
    position_weight: float = 0.1

    # Ambiguity
    ambiguity_score_threshold: float = 0.15  # Max difference between top-2 candidates

    # Minimum scores
    min_association_score: float = 0.1


class SpatialAssociator:
    """
    Associates OCR text regions with detected shapes and arrows.

    Uses geometric features to compute association scores and produce
    structured evidence for diagram reconstruction.
    """

    def __init__(self, config: Optional[AssociationConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or AssociationConfig()

    def associate_text_to_shapes(
        self,
        text_regions: List[OCRTextRegion],
        candidate_nodes: List[CandidateNode],
    ) -> List[TextAssociationResult]:
        """
        Associate text regions with candidate nodes (shapes).

        Args:
            text_regions: OCR text regions to associate.
            candidate_nodes: Candidate diagram nodes (from shape detection).

        Returns:
            List of TextAssociationResult, one per text region.
        """
        results = []

        for region in text_regions:
            associations = []
            for node in candidate_nodes:
                score, evidence = self._compute_shape_association(region, node)
                if score >= self.config.min_association_score:
                    assoc = TextAssociation(
                        text_region_id=region.region_id,
                        candidate_target_id=node.node_id,
                        target_type=AssociationTargetType.NODE,
                        association_score=score,
                        evidence=evidence,
                        reasons=self._build_reasons(evidence, "shape"),
                    )
                    associations.append(assoc)

            # Sort by score descending
            associations.sort(key=lambda a: a.association_score, reverse=True)

            # Mark best candidate
            if associations:
                associations[0].is_best_candidate = True

            # Check ambiguity
            is_ambiguous = self._check_ambiguity(associations)

            # Build result
            best = associations[0] if associations else None
            result = TextAssociationResult(
                text_region_id=region.region_id,
                associations=associations,
                best_association=best,
                is_ambiguous=is_ambiguous,
                has_no_match=len(associations) == 0,
            )
            results.append(result)

        return results

    def associate_text_to_arrows(
        self,
        text_regions: List[OCRTextRegion],
        detected_arrows: List[DetectedArrow],
    ) -> List[TextAssociationResult]:
        """
        Associate text regions with detected arrows.

        Args:
            text_regions: OCR text regions to associate.
            detected_arrows: Detected arrows from arrow detection.

        Returns:
            List of TextAssociationResult, one per text region.
        """
        results = []

        for region in text_regions:
            associations = []
            for arrow in detected_arrows:
                score, evidence = self._compute_arrow_association(region, arrow)
                if score >= self.config.min_association_score:
                    assoc = TextAssociation(
                        text_region_id=region.region_id,
                        candidate_target_id=arrow.arrow_id,
                        target_type=AssociationTargetType.ARROW,
                        association_score=score,
                        evidence=evidence,
                        reasons=self._build_reasons(evidence, "arrow"),
                    )
                    associations.append(assoc)

            associations.sort(key=lambda a: a.association_score, reverse=True)
            if associations:
                associations[0].is_best_candidate = True

            is_ambiguous = self._check_ambiguity(associations)
            best = associations[0] if associations else None

            result = TextAssociationResult(
                text_region_id=region.region_id,
                associations=associations,
                best_association=best,
                is_ambiguous=is_ambiguous,
                has_no_match=len(associations) == 0,
            )
            results.append(result)

        return results

    def associate_all(
        self,
        text_regions: List[OCRTextRegion],
        candidate_nodes: List[CandidateNode],
        detected_arrows: List[DetectedArrow],
    ) -> List[TextAssociationResult]:
        """
        Associate text regions with both shapes and arrows.

        Combines shape and arrow associations, picking the best overall
        match for each text region.

        Args:
            text_regions: OCR text regions to associate.
            candidate_nodes: Candidate diagram nodes.
            detected_arrows: Detected arrows.

        Returns:
            List of TextAssociationResult, one per text region.
        """
        shape_results = self.associate_text_to_shapes(text_regions, candidate_nodes)
        arrow_results = self.associate_text_to_arrows(text_regions, detected_arrows)

        combined = []
        for s_res, a_res in zip(shape_results, arrow_results):
            all_associations = s_res.associations + a_res.associations
            all_associations.sort(key=lambda a: a.association_score, reverse=True)

            if all_associations:
                all_associations[0].is_best_candidate = True

            is_ambiguous = self._check_ambiguity(all_associations)
            best = all_associations[0] if all_associations else None

            combined.append(TextAssociationResult(
                text_region_id=s_res.text_region_id,
                associations=all_associations,
                best_association=best,
                is_ambiguous=is_ambiguous,
                has_no_match=len(all_associations) == 0,
            ))

        return combined

    def _compute_shape_association(
        self,
        region: OCRTextRegion,
        node: CandidateNode,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute association score between a text region and a shape.

        Returns:
            Tuple of (score, evidence_dict).
        """
        evidence: Dict[str, Any] = {}
        text_bbox = region.bounding_box
        shape_bbox = node.bounding_box

        # 1. Containment check: is text center inside shape?
        text_center = text_bbox.center
        is_contained = shape_bbox.contains_point(text_center)
        evidence["is_contained"] = is_contained

        # 2. Overlap ratio
        overlap_ratio = self._compute_overlap_ratio(text_bbox, shape_bbox)
        evidence["overlap_ratio"] = overlap_ratio

        # 3. Center distance
        shape_center = shape_bbox.center
        distance = text_center.distance_to(shape_center)
        evidence["center_distance"] = distance

        # 4. Relative position
        rel_x = (text_center.x - shape_bbox.x) / max(shape_bbox.width, 1)
        rel_y = (text_center.y - shape_bbox.y) / max(shape_bbox.height, 1)
        evidence["relative_position"] = (rel_x, rel_y)

        # Compute weighted score
        score = 0.0

        if is_contained:
            score += self.config.containment_weight
            evidence["containment_bonus"] = self.config.containment_weight

        # Overlap contribution
        score += overlap_ratio * self.config.overlap_weight

        # Distance contribution (inverse — closer is better)
        if distance < self.config.max_center_distance:
            distance_score = 1.0 - (distance / self.config.max_center_distance)
            score += distance_score * self.config.distance_weight

        # Position contribution (prefer centered text)
        center_offset = abs(rel_x - 0.5) + abs(rel_y - 0.5)
        position_score = max(0, 1.0 - center_offset)
        score += position_score * self.config.position_weight

        evidence["final_score"] = score
        return (score, evidence)

    def _compute_arrow_association(
        self,
        region: OCRTextRegion,
        arrow: DetectedArrow,
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute association score between a text region and an arrow.

        Returns:
            Tuple of (score, evidence_dict).
        """
        evidence: Dict[str, Any] = {}
        text_center = region.bounding_box.center

        # 1. Distance to arrow midpoint
        arrow_mid = arrow.midpoint
        mid_distance = text_center.distance_to(arrow_mid)
        evidence["midpoint_distance"] = mid_distance

        # 2. Perpendicular distance to arrow line
        perp_dist = self._point_to_line_distance(
            text_center, arrow.start, arrow.end
        )
        evidence["perpendicular_distance"] = perp_dist

        # 3. Projection onto arrow segment
        proj_dist = self._point_to_segment_distance(
            text_center, arrow.start, arrow.end
        )
        evidence["segment_distance"] = proj_dist

        # 4. Arrow length (longer arrows more likely to have labels)
        evidence["arrow_length"] = arrow.length

        # Compute score
        score = 0.0

        # Midpoint proximity
        if mid_distance < self.config.max_arrow_perpendicular_distance * 2:
            mid_score = 1.0 - (
                mid_distance / (self.config.max_arrow_perpendicular_distance * 2)
            )
            score += mid_score * 0.4

        # Perpendicular proximity
        if perp_dist < self.config.max_arrow_perpendicular_distance:
            perp_score = 1.0 - (
                perp_dist / self.config.max_arrow_perpendicular_distance
            )
            score += perp_score * 0.35

        # Segment proximity
        if proj_dist < self.config.max_arrow_projection_distance:
            proj_score = 1.0 - (
                proj_dist / self.config.max_arrow_projection_distance
            )
            score += proj_score * 0.25

        evidence["final_score"] = score
        return (score, evidence)

    def _compute_overlap_ratio(
        self, box_a: BoundingBox, box_b: BoundingBox
    ) -> float:
        """Compute overlap ratio of box_a relative to box_b."""
        x_overlap = max(
            0,
            min(box_a.x + box_a.width, box_b.x + box_b.width)
            - max(box_a.x, box_b.x),
        )
        y_overlap = max(
            0,
            min(box_a.y + box_a.height, box_b.y + box_b.height)
            - max(box_a.y, box_b.y),
        )
        overlap_area = x_overlap * y_overlap
        if box_a.area == 0:
            return 0.0
        return overlap_area / box_a.area

    def _point_to_line_distance(
        self, point: Point, line_start: Point, line_end: Point
    ) -> float:
        """Compute perpendicular distance from point to infinite line."""
        dx = line_end.x - line_start.x
        dy = line_end.y - line_start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return point.distance_to(line_start)
        t = max(
            0,
            min(
                1,
                ((point.x - line_start.x) * dx + (point.y - line_start.y) * dy)
                / length_sq,
            ),
        )
        proj_x = line_start.x + t * dx
        proj_y = line_start.y + t * dy
        return point.distance_to(Point(proj_x, proj_y))

    def _point_to_segment_distance(
        self, point: Point, seg_start: Point, seg_end: Point
    ) -> float:
        """Compute distance from point to line segment."""
        dx = seg_end.x - seg_start.x
        dy = seg_end.y - seg_start.y
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            return point.distance_to(seg_start)
        t = max(
            0,
            min(
                1,
                ((point.x - seg_start.x) * dx + (point.y - seg_start.y) * dy)
                / length_sq,
            ),
        )
        proj_x = seg_start.x + t * dx
        proj_y = seg_start.y + t * dy
        return point.distance_to(Point(proj_x, proj_y))

    def _check_ambiguity(self, associations: List[TextAssociation]) -> bool:
        """Check if associations are ambiguous (close scores)."""
        if len(associations) < 2:
            return False
        top_scores = sorted(
            [a.association_score for a in associations], reverse=True
        )
        diff = top_scores[0] - top_scores[1]
        return diff < self.config.ambiguity_score_threshold

    def _build_reasons(
        self, evidence: Dict[str, Any], target_type: str
    ) -> List[str]:
        """Build human-readable reasons from evidence."""
        reasons = []
        if target_type == "shape":
            if evidence.get("is_contained"):
                reasons.append("Text center is inside shape bounding box")
            overlap = evidence.get("overlap_ratio", 0)
            if overlap > 0.5:
                reasons.append(f"High overlap ratio: {overlap:.2f}")
            dist = evidence.get("center_distance", float("inf"))
            if dist < 50:
                reasons.append(f"Close to shape center: {dist:.1f}px")
        elif target_type == "arrow":
            perp = evidence.get("perpendicular_distance", float("inf"))
            if perp < 30:
                reasons.append(f"Near arrow line: {perp:.1f}px perpendicular")
            mid = evidence.get("midpoint_distance", float("inf"))
            if mid < 50:
                reasons.append(f"Near arrow midpoint: {mid:.1f}px")
        return reasons

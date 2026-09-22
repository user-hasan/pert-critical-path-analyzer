"""
Geometry-aware arrow-to-node association for AON diagrams.

Replaces proximity-dominated endpoint matching with boundary intersection,
direction-aware scoring, and self-association rejection.

For AON diagrams, arrows terminate at NODE BOUNDARIES, not at node centers.
This module constructs rays from arrow endpoints and finds boundary intersections
to correctly identify source and target nodes.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType

logger = logging.getLogger(__name__)


@dataclass
class BoundaryIntersection:
    """A point where a ray/segment intersects a rectangle boundary."""

    point: Point
    edge_name: str  # "top", "bottom", "left", "right"
    distance_from_origin: float  # Distance from ray origin to intersection
    parameter: float  # Parametric value along ray (0-1 for segment)


@dataclass
class CandidateScore:
    """Score for a candidate node as source or target of an arrow."""

    candidate_id: str
    score: float
    boundary_intersection: float  # 0-1: how well the arrow hits the boundary
    endpoint_distance: float  # 0-1: distance from endpoint to boundary (closer=better)
    direction_alignment: float  # 0-1: does arrow direction point toward/away from node
    angle_consistency: float  # 0-1: angular alignment
    shape_confidence: float  # 0-1: node detection confidence
    arrow_confidence: float  # 0-1: arrow detection confidence
    intersection_point: Optional[Point] = None
    evidence: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArrowAssociationResult:
    """Result of associating a single arrow with source/target nodes."""

    arrow_id: str
    source_candidates: List[CandidateScore] = field(default_factory=list)
    target_candidates: List[CandidateScore] = field(default_factory=list)
    best_source: Optional[CandidateScore] = None
    best_target: Optional[CandidateScore] = None
    is_self_association: bool = False
    is_ambiguous_source: bool = False
    is_ambiguous_target: bool = False
    rejected: bool = False
    rejection_reason: str = ""


class GeometryAssociator:
    """
    Geometry-aware arrow-to-node association for AON diagrams.

    Uses boundary intersection, direction alignment, and scoring
    instead of simple endpoint-to-center proximity.
    """

    def __init__(
        self,
        boundary_tolerance: float = 15.0,
        ray_extension: float = 30.0,
        direction_tolerance_deg: float = 45.0,
        min_score: float = 0.15,
        ambiguity_threshold: float = 0.15,
        max_candidates: int = 5,
    ):
        """
        Initialize the geometry associator.

        Args:
            boundary_tolerance: Max distance from endpoint to boundary for a hit
            ray_extension: How far to extend rays backward from endpoints
            direction_tolerance_deg: Max angular deviation for direction alignment
            min_score: Minimum score to accept a candidate
            ambiguity_threshold: Max score difference between top-2 to flag ambiguity
            max_candidates: Max candidates to return per endpoint
        """
        self.boundary_tolerance = boundary_tolerance
        self.ray_extension = ray_extension
        self.direction_tolerance_deg = direction_tolerance_deg
        self.min_score = min_score
        self.ambiguity_threshold = ambiguity_threshold
        self.max_candidates = max_candidates

    # =========================================================================
    # Public API
    # =========================================================================

    def associate_arrows(
        self,
        arrows: List[DetectedArrow],
        candidates: List[CandidateNode],
    ) -> List[ArrowAssociationResult]:
        """
        Associate all arrows with source/target nodes using geometry.

        Args:
            arrows: Detected arrows with start/end/arrowhead/direction
            candidates: Candidate nodes (rectangles for AON)

        Returns:
            List of ArrowAssociationResult with scored candidates
        """
        results = []
        for arrow in arrows:
            result = self.associate_single_arrow(arrow, candidates)
            results.append(result)
        return results

    def associate_single_arrow(
        self,
        arrow: DetectedArrow,
        candidates: List[CandidateNode],
    ) -> ArrowAssociationResult:
        """
        Associate a single arrow with source/target nodes.

        Uses:
        - Arrowhead point for target detection
        - Arrow tail for source detection
        - Direction vector for angular alignment
        - Boundary intersection for scoring
        """
        result = ArrowAssociationResult(arrow_id=arrow.arrow_id)

        if not candidates:
            result.rejected = True
            result.rejection_reason = "no_candidates"
            return result

        # Determine arrow direction
        direction = self._get_arrow_direction(arrow)
        if direction is None:
            result.rejected = True
            result.rejection_reason = "no_direction"
            return result

        dx, dy = direction
        arrow_length = math.sqrt(dx * dx + dy * dy)
        if arrow_length < 1e-6:
            result.rejected = True
            result.rejection_reason = "zero_length_arrow"
            return result

        # Normalize direction
        ndx, ndy = dx / arrow_length, dy / arrow_length

        # Get target point (arrowhead preferred, else end)
        target_point = arrow.arrowhead_point or arrow.end
        target_method = "arrowhead" if arrow.arrowhead_point else "end"

        # Get source point (tail)
        source_point = arrow.start
        source_method = "tail"

        # Score all candidates for target (ahead of arrowhead)
        result.target_candidates = self._score_candidates(
            candidates=candidates,
            reference_point=target_point,
            direction=(ndx, ndy),
            is_target=True,
            arrow=arrow,
        )

        # Score all candidates for source (behind tail)
        result.source_candidates = self._score_candidates(
            candidates=candidates,
            reference_point=source_point,
            direction=(ndx, ndy),
            is_target=False,
            arrow=arrow,
        )

        # Select best source and target
        result.best_source = self._select_best(result.source_candidates, is_target=False)
        result.best_target = self._select_best(result.target_candidates, is_target=True)

        # Check for self-association
        if (result.best_source and result.best_target
                and result.best_source.candidate_id == result.best_target.candidate_id):
            result.is_self_association = True
            # Try to find alternative
            self._handle_self_association(result, arrow, candidates)

        # Check for ambiguity
        result.is_ambiguous_source = self._check_ambiguity(result.source_candidates)
        result.is_ambiguous_target = self._check_ambiguity(result.target_candidates)

        # Reject if no valid candidates
        if not result.best_source:
            result.rejected = True
            result.rejection_reason = "no_valid_source"
        elif not result.best_target:
            result.rejected = True
            result.rejection_reason = "no_valid_target"

        # Store metadata
        result.source_candidates = result.source_candidates[:self.max_candidates]
        result.target_candidates = result.target_candidates[:self.max_candidates]

        return result

    # =========================================================================
    # Direction
    # =========================================================================

    def _get_arrow_direction(self, arrow: DetectedArrow) -> Optional[Tuple[float, float]]:
        """Get arrow direction vector (tail to head)."""
        # Prefer direction_vector if set
        if arrow.direction_vector and arrow.direction_vector != (0.0, 0.0):
            return arrow.direction_vector

        # Use arrowhead if available
        if arrow.arrowhead_point:
            dx = arrow.arrowhead_point.x - arrow.start.x
            dy = arrow.arrowhead_point.y - arrow.start.y
            return (dx, dy)

        # Fall back to start -> end
        dx = arrow.end.x - arrow.start.x
        dy = arrow.end.y - arrow.start.y
        return (dx, dy)

    # =========================================================================
    # Candidate Scoring
    # =========================================================================

    def _score_candidates(
        self,
        candidates: List[CandidateNode],
        reference_point: Point,
        direction: Tuple[float, float],
        is_target: bool,
        arrow: DetectedArrow,
    ) -> List[CandidateScore]:
        """Score all candidates for source or target position."""
        scored = []
        ndx, ndy = direction

        for cand in candidates:
            score = self._score_single_candidate(
                candidate=cand,
                reference_point=reference_point,
                direction=(ndx, ndy),
                is_target=is_target,
                arrow=arrow,
            )
            if score.score >= self.min_score:
                scored.append(score)

        # Sort by score descending
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[:self.max_candidates * 2]

    def _score_single_candidate(
        self,
        candidate: CandidateNode,
        reference_point: Point,
        direction: Tuple[float, float],
        is_target: bool,
        arrow: DetectedArrow,
    ) -> CandidateScore:
        """Score a single candidate node."""
        bb = candidate.bounding_box
        ndx, ndy = direction

        # Direct distance from reference point to candidate center
        dist_to_center = reference_point.distance_to(candidate.position)

        # If very close to center, this is almost certainly the correct node
        center_proximity = 0.0
        if dist_to_center < 30:
            center_proximity = max(0, 1.0 - dist_to_center / 30.0)

        # If endpoint is inside the bounding box, strong signal
        inside_bonus = 0.0
        if bb.contains_point(reference_point, margin=10):
            inside_bonus = 0.8

        # 1. Boundary intersection test
        intersection_score, intersection_point, edge_distance = (
            self._boundary_intersection_score(reference_point, bb, is_target, ndx, ndy)
        )

        # 2. Endpoint distance to boundary
        endpoint_distance = self._endpoint_distance_score(reference_point, bb)

        # 3. Direction alignment
        direction_score = self._direction_alignment_score(
            reference_point, bb, ndx, ndy, is_target
        )

        # 4. Angle consistency
        angle_score = self._angle_consistency_score(
            reference_point, bb, ndx, ndy, is_target
        )

        # 5. Shape confidence
        shape_conf = candidate.confidence

        # 6. Arrow confidence
        arrow_conf = arrow.confidence

        # Proximity-dominant scoring: if very close to center, override other scores
        if center_proximity > 0.9:
            # Very close to center - almost certainly correct
            score = 0.90 + 0.10 * direction_score
        elif inside_bonus > 0:
            # Inside bounding box - strong signal, direction irrelevant
            # Arrow starts at boundary of this node; direction toward center is irrelevant
            score = (
                0.35 * inside_bonus
                + 0.25 * intersection_score
                + 0.15 * endpoint_distance
                + 0.05 * direction_score  # Reduced: direction doesn't matter when inside
                + 0.10 * angle_score
                + 0.05 * shape_conf
                + 0.05 * arrow_conf
            )
        else:
            # Normal scoring
            score = (
                0.35 * intersection_score
                + 0.25 * endpoint_distance
                + 0.20 * direction_score
                + 0.10 * angle_score
                + 0.05 * shape_conf
                + 0.05 * arrow_conf
            )

        evidence = {
            "boundary_intersection": intersection_score,
            "endpoint_distance": endpoint_distance,
            "direction_alignment": direction_score,
            "angle_consistency": angle_score,
            "shape_confidence": shape_conf,
            "arrow_confidence": arrow_conf,
            "edge_distance": edge_distance,
            "is_target": is_target,
            "dist_to_center": dist_to_center,
            "center_proximity": center_proximity,
            "inside_bonus": inside_bonus,
        }

        return CandidateScore(
            candidate_id=candidate.node_id,
            score=score,
            boundary_intersection=intersection_score,
            endpoint_distance=endpoint_distance,
            direction_alignment=direction_score,
            angle_consistency=angle_score,
            shape_confidence=shape_conf,
            arrow_confidence=arrow_conf,
            intersection_point=intersection_point,
            evidence=evidence,
        )

    # =========================================================================
    # Boundary Intersection
    # =========================================================================

    def _boundary_intersection_score(
        self,
        point: Point,
        bb: BoundingBox,
        is_target: bool,
        ndx: float,
        ndy: float,
    ) -> Tuple[float, Optional[Point], float]:
        """
        Score how well a ray from point intersects the rectangle boundary.

        For target: ray goes forward along arrow direction
        For source: ray goes backward (reverse direction)

        Returns (score, intersection_point, distance_to_boundary)
        """
        # Construct ray direction
        if is_target:
            rdx, rdy = ndx, ndy  # Forward
        else:
            rdx, rdy = -ndx, -ndy  # Backward

        # Find intersection with rectangle edges
        intersections = self._ray_rectangle_intersections(point, rdx, rdy, bb)

        if not intersections:
            # Check if point is inside or very close to boundary
            dist_to_boundary = self._distance_to_rectangle_boundary(point, bb)
            if dist_to_boundary <= self.boundary_tolerance:
                score = max(0, 1.0 - dist_to_boundary / self.boundary_tolerance)
                return score, None, dist_to_boundary
            return 0.0, None, float("inf")

        # Find closest intersection in the correct direction
        best_intersection = None
        best_dist = float("inf")

        for ix, iy, edge, param in intersections:
            if param >= 0:  # Intersection is in the correct direction
                dist = math.sqrt((ix - point.x) ** 2 + (iy - point.y) ** 2)
                if dist < best_dist:
                    best_dist = dist
                    best_intersection = (ix, iy, edge)

        if best_intersection is None:
            return 0.0, None, float("inf")

        ix, iy, edge = best_intersection
        intersection_point = Point(ix, iy)

        # Score: closer intersection is better (up to ray_extension)
        if best_dist <= self.ray_extension:
            dist_score = 1.0 - (best_dist / self.ray_extension) * 0.3
        else:
            dist_score = max(0, 1.0 - best_dist / (self.ray_extension * 3))

        # Bonus for hitting the boundary directly
        boundary_bonus = 0.2

        score = min(1.0, dist_score + boundary_bonus)
        return score, intersection_point, best_dist

    def _ray_rectangle_intersections(
        self,
        origin: Point,
        rdx: float,
        rdy: float,
        bb: BoundingBox,
    ) -> List[Tuple[float, float, str, float]]:
        """
        Find intersections of a ray with rectangle edges.

        Ray: origin + t * (rdx, rdy) for t >= 0

        Returns list of (x, y, edge_name, parameter_t)
        """
        intersections = []

        # Rectangle edges: x = left, x = right, y = top, y = bottom
        edges = [
            (bb.x, "left"),  # x = left
            (bb.x + bb.width, "right"),  # x = right
            (bb.y, "top"),  # y = top
            (bb.y + bb.height, "bottom"),  # y = bottom
        ]

        for edge_val, edge_name in edges:
            if edge_name in ("left", "right"):
                # Vertical edge: x = edge_val
                if abs(rdx) < 1e-10:
                    continue  # Ray is parallel to vertical edge
                t = (edge_val - origin.x) / rdx
                if t < 0:
                    continue  # Intersection behind origin
                y = origin.y + t * rdy
                if bb.y - 1 <= y <= bb.y + bb.height + 1:
                    intersections.append((edge_val, y, edge_name, t))
            else:
                # Horizontal edge: y = edge_val
                if abs(rdy) < 1e-10:
                    continue  # Ray is parallel to horizontal edge
                t = (edge_val - origin.y) / rdy
                if t < 0:
                    continue  # Intersection behind origin
                x = origin.x + t * rdx
                if bb.x - 1 <= x <= bb.x + bb.width + 1:
                    intersections.append((x, edge_val, edge_name, t))

        return intersections

    def _distance_to_rectangle_boundary(self, point: Point, bb: BoundingBox) -> float:
        """Distance from point to nearest rectangle boundary edge."""
        # Clamp to rectangle
        cx = max(bb.x, min(point.x, bb.x + bb.width))
        cy = max(bb.y, min(point.y, bb.y + bb.height))

        # Distance to nearest edge
        dist_left = abs(point.x - bb.x)
        dist_right = abs(point.x - (bb.x + bb.width))
        dist_top = abs(point.y - bb.y)
        dist_bottom = abs(point.y - (bb.y + bb.height))

        # If inside, distance is min distance to any edge
        if bb.x <= point.x <= bb.x + bb.width and bb.y <= point.y <= bb.y + bb.height:
            return min(dist_left, dist_right, dist_top, dist_bottom)

        # If outside, distance to nearest corner or edge
        dx = max(bb.x - point.x, 0, point.x - (bb.x + bb.width))
        dy = max(bb.y - point.y, 0, point.y - (bb.y + bb.height))
        return math.sqrt(dx * dx + dy * dy)

    # =========================================================================
    # Endpoint Distance
    # =========================================================================

    def _endpoint_distance_score(self, point: Point, bb: BoundingBox) -> float:
        """Score based on distance from endpoint to rectangle boundary."""
        dist = self._distance_to_rectangle_boundary(point, bb)
        if dist <= self.boundary_tolerance:
            return 1.0 - (dist / self.boundary_tolerance) * 0.5
        elif dist <= self.boundary_tolerance * 3:
            return max(0, 0.5 - (dist - self.boundary_tolerance) / (self.boundary_tolerance * 2) * 0.5)
        return 0.0

    # =========================================================================
    # Direction Alignment
    # =========================================================================

    def _direction_alignment_score(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_target: bool,
    ) -> float:
        """Score whether arrow direction points toward/away from the node."""
        center = bb.center

        # Vector from reference point to node center
        to_center_x = center.x - point.x
        to_center_y = center.y - point.y
        to_center_len = math.sqrt(to_center_x ** 2 + to_center_y ** 2)

        if to_center_len < 1e-6:
            return 0.5  # Point is at center

        # Normalize
        tcx, tcy = to_center_x / to_center_len, to_center_y / to_center_len

        # For target: arrow direction should point toward node
        # For source: arrow direction should point away from node (reverse)
        if is_target:
            dot = ndx * tcx + ndy * tcy
        else:
            dot = -(ndx * tcx + ndy * tcy)

        # Convert to 0-1 score
        # dot = 1 means perfect alignment, -1 means opposite
        score = (dot + 1) / 2
        return score

    # =========================================================================
    # Angle Consistency
    # =========================================================================

    def _angle_consistency_score(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_target: bool,
    ) -> float:
        """Score angular consistency between arrow and node position."""
        center = bb.center

        # Angle from point to center
        angle_to_center = math.atan2(center.y - point.y, center.x - point.x)

        # Arrow angle
        arrow_angle = math.atan2(ndy, ndx)

        # For target: angles should be similar
        # For source: angles should be opposite
        if is_target:
            angle_diff = abs(angle_to_center - arrow_angle)
        else:
            angle_diff = abs(angle_to_center - (arrow_angle + math.pi))

        # Normalize to 0-pi
        angle_diff = angle_diff % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        # Convert to score
        tolerance_rad = math.radians(self.direction_tolerance_deg)
        if angle_diff <= tolerance_rad:
            return 1.0 - (angle_diff / tolerance_rad) * 0.5
        else:
            return max(0, 0.5 - (angle_diff - tolerance_rad) / (math.pi - tolerance_rad) * 0.5)

    # =========================================================================
    # Selection
    # =========================================================================

    def _select_best(
        self,
        candidates: List[CandidateScore],
        is_target: bool,
    ) -> Optional[CandidateScore]:
        """Select the best candidate from scored list."""
        if not candidates:
            return None

        # Filter out candidates that are geometrically on the wrong side
        valid = []
        for c in candidates:
            if c.score >= self.min_score:
                valid.append(c)

        if not valid:
            return None

        return valid[0]

    def _check_ambiguity(self, candidates: List[CandidateScore]) -> bool:
        """Check if top-2 candidates are too close in score."""
        if len(candidates) < 2:
            return False
        top2_diff = candidates[0].score - candidates[1].score
        return top2_diff < self.ambiguity_threshold

    # =========================================================================
    # Self-Association Handling
    # =========================================================================

    def _handle_self_association(
        self,
        result: ArrowAssociationResult,
        arrow: DetectedArrow,
        candidates: List[CandidateNode],
    ) -> None:
        """
        Handle self-association by trying to find an alternative.

        If source == target, the arrow is suspicious.
        Try to find the second-best candidate for either source or target.
        """
        # Try second-best target
        if len(result.target_candidates) >= 2:
            second_target = result.target_candidates[1]
            if second_target.candidate_id != result.best_source.candidate_id:
                result.best_target = second_target
                result.is_self_association = False
                return

        # Try second-best source
        if len(result.source_candidates) >= 2:
            second_source = result.source_candidates[1]
            if second_source.candidate_id != result.best_target.candidate_id:
                result.best_source = second_source
                result.is_self_association = False
                return

        # No alternative found - mark as rejected
        result.rejected = True
        result.rejection_reason = "self_association_no_alternative"

"""
Independent direction-evidence layer for arrow direction resolution.

Given a FIXED node pair A <-> B, this module produces evidence for:
  A → B
  B → A

Using geometry only:
  - boundary intersection at source/target
  - boundary normal reasoning
  - endpoint contact classification (which side)
  - segment continuity (leaves source, enters target)
  - angular consistency
  - crossing penalty

This module does NOT change the node pair. It only resolves direction.
It does NOT rely on Hough endpoint ordering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point


class ContactSide(Enum):
    """Which side of a rectangle an endpoint contacts."""
    LEFT = "left"
    RIGHT = "right"
    TOP = "top"
    BOTTOM = "bottom"
    CORNER = "corner"
    INSIDE = "inside"
    FAR = "far"


@dataclass
class EndpointContact:
    """Classification of how an endpoint contacts a rectangle."""
    side: ContactSide = ContactSide.FAR
    distance: float = 0.0
    normal: Tuple[float, float] = (0.0, 0.0)
    confidence: float = 0.0
    boundary_point: Optional[Point] = None


@dataclass
class DirectionEvidenceResult:
    """Complete evidence for one orientation (A→B or B→A)."""
    source_contact: EndpointContact = field(default_factory=EndpointContact)
    target_contact: EndpointContact = field(default_factory=EndpointContact)
    source_exits_boundary: float = 0.0
    target_enters_boundary: float = 0.0
    normal_consistency: float = 0.0
    continuity_score: float = 0.0
    angular_consistency: float = 0.0
    crossing_penalty: float = 0.0
    boundary_path_score: float = 0.0
    total_score: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)


class DirectionEvidenceLayer:
    """
    Independent direction-evidence layer.

    Receives a fixed node pair and a geometric segment,
    and scores both orientations using boundary geometry.
    """

    def __init__(
        self,
        min_boundary_distance: float = 80.0,
        max_direction_tolerance_deg: float = 40.0,
    ):
        self.min_boundary_distance = min_boundary_distance
        self.max_direction_tolerance_deg = max_direction_tolerance_deg

    def score_orientation(
        self,
        source_bb: BoundingBox,
        target_bb: BoundingBox,
        segment_endpoint_1: Point,
        segment_endpoint_2: Point,
        geometric_axis: Tuple[float, float],
        arrowhead_candidates: Optional[List[Point]] = None,
        arrowhead_confidence: float = 0.0,
        all_candidates: Optional[List[BoundingBox]] = None,
    ) -> DirectionEvidenceResult:
        """
        Score one orientation: source → target.

        The segment has normalized endpoints (endpoint_1, endpoint_2).
        For this scoring, we need to determine which endpoint is near
        the source and which is near the target, based on boundary geometry.

        Args:
            source_bb: Bounding box of the source node.
            target_bb: Bounding box of the target node.
            segment_endpoint_1: First endpoint (normalized).
            segment_endpoint_2: Second endpoint (normalized).
            geometric_axis: Unit direction from endpoint_1 to endpoint_2.
            arrowhead_candidates: List of detected arrowhead points.
            arrowhead_confidence: Confidence of arrowhead detection.
            all_candidates: All node bounding boxes (for crossing check).

        Returns:
            DirectionEvidenceResult with all evidence factors.
        """
        result = DirectionEvidenceResult()

        # Classify endpoint contacts for both source and target
        result.source_contact = self._classify_contact(
            segment_endpoint_1, source_bb
        )
        result.target_contact = self._classify_contact(
            segment_endpoint_2, target_bb
        )

        # Also check the other assignment (endpoint_1 near target, endpoint_2 near source)
        source_contact_alt = self._classify_contact(
            segment_endpoint_1, target_bb
        )
        target_contact_alt = self._classify_contact(
            segment_endpoint_2, source_bb
        )

        # Use the assignment that makes more geometric sense
        # (endpoint closer to source → source contact)
        if (source_contact_alt.distance + target_contact_alt.distance <
                result.source_contact.distance + result.target_contact.distance):
            # Swapped assignment is better
            result.source_contact = target_contact_alt
            result.target_contact = source_contact_alt

        # Boundary projection for inside endpoints
        # When an endpoint is INSIDE a rectangle, project along the geometric
        # axis to find the boundary intersection. This gives us a proper
        # boundary contact point with a normal.
        if result.source_contact.side == ContactSide.INSIDE:
            projected = self._project_to_boundary(
                segment_endpoint_1, geometric_axis, source_bb, forward=True
            )
            if projected:
                result.source_contact = projected

        if result.target_contact.side == ContactSide.INSIDE:
            projected = self._project_to_boundary(
                segment_endpoint_2, geometric_axis, target_bb, forward=False
            )
            if projected:
                result.target_contact = projected

        # Between-rectangle projection: when endpoint is FAR from target,
        # project along the geometric axis toward the target to find the
        # boundary it would enter.
        if (result.target_contact.side == ContactSide.FAR or
                result.target_contact.distance > self.min_boundary_distance * 0.5):
            # Project toward target (along geometric axis direction)
            projected = self._project_to_boundary(
                segment_endpoint_2, geometric_axis, target_bb, forward=True
            )
            if projected:
                result.target_contact = projected

        # Similarly for source: when endpoint is FAR from source,
        # project backward to find where it exited.
        if (result.source_contact.side == ContactSide.FAR or
                result.source_contact.distance > self.min_boundary_distance * 0.5):
            # Project away from source (opposite to geometric axis)
            projected = self._project_to_boundary(
                segment_endpoint_1, geometric_axis, source_bb, forward=False
            )
            if projected:
                result.source_contact = projected

        # 1. Boundary normal reasoning
        result.normal_consistency = self._score_normal_consistency(
            result.source_contact, result.target_contact, geometric_axis
        )

        # 2. Source exits boundary
        result.source_exits_boundary = self._score_source_exits(
            result.source_contact, geometric_axis
        )

        # 3. Target enters boundary
        result.target_enters_boundary = self._score_target_enters(
            result.target_contact, geometric_axis
        )

        # 4. Segment continuity
        result.continuity_score = self._score_segment_continuity(
            segment_endpoint_1, segment_endpoint_2,
            source_bb, target_bb,
            result.source_contact, result.target_contact
        )

        # 5. Angular consistency
        result.angular_consistency = self._score_angular_consistency(
            source_bb, target_bb, geometric_axis
        )

        # 6. Boundary path score (combined source→segment→target)
        result.boundary_path_score = self._score_boundary_path(
            result.source_contact, result.target_contact,
            result.source_exits_boundary, result.target_enters_boundary,
            geometric_axis
        )

        # 7. Crossing penalty
        result.crossing_penalty = self._score_crossing(
            segment_endpoint_1, geometric_axis, target_bb, all_candidates or []
        )

        # 8. Arrowhead evidence
        arrowhead_score = 0.0
        if arrowhead_candidates and arrowhead_confidence > 0:
            arrowhead_score = self._score_arrowhead(
                arrowhead_candidates, arrowhead_confidence, target_bb
            )

        # Combined score
        # When arrowhead is available, it's the strongest signal
        # When not, boundary geometry dominates
        if arrowhead_confidence > 0.3:
            result.total_score = (
                0.30 * arrowhead_score
                + 0.15 * result.source_exits_boundary
                + 0.15 * result.target_enters_boundary
                + 0.15 * result.normal_consistency
                + 0.10 * result.boundary_path_score
                + 0.10 * result.continuity_score
                + 0.05 * result.angular_consistency
                - 0.10 * result.crossing_penalty
            )
        else:
            # No reliable arrowhead — boundary geometry dominates
            result.total_score = (
                0.25 * result.source_exits_boundary
                + 0.25 * result.target_enters_boundary
                + 0.20 * result.normal_consistency
                + 0.15 * result.boundary_path_score
                + 0.10 * result.continuity_score
                + 0.05 * result.angular_consistency
                - 0.10 * result.crossing_penalty
            )

        result.total_score = max(0.0, min(1.0, result.total_score))

        result.evidence = {
            "source_side": result.source_contact.side.value,
            "source_distance": result.source_contact.distance,
            "source_normal": result.source_contact.normal,
            "target_side": result.target_contact.side.value,
            "target_distance": result.target_contact.distance,
            "target_normal": result.target_contact.normal,
            "source_exits": result.source_exits_boundary,
            "target_enters": result.target_enters_boundary,
            "normal_consistency": result.normal_consistency,
            "continuity": result.continuity_score,
            "angular": result.angular_consistency,
            "boundary_path": result.boundary_path_score,
            "crossing_penalty": result.crossing_penalty,
            "arrowhead": arrowhead_score,
            "total": result.total_score,
        }

        return result

    # =========================================================================
    # Endpoint Contact Classification
    # =========================================================================

    def _classify_contact(
        self, point: Point, bb: BoundingBox
    ) -> EndpointContact:
        """Classify which side of a rectangle a point contacts."""
        dist = self._distance_to_boundary(point, bb)

        if dist > self.min_boundary_distance:
            return EndpointContact(
                side=ContactSide.FAR,
                distance=dist,
                normal=(0.0, 0.0),
                confidence=0.0,
            )

        # Determine which side
        cx = bb.x + bb.width / 2
        cy = bb.y + bb.height / 2

        # Relative position within rectangle
        rx = (point.x - bb.x) / bb.width if bb.width > 0 else 0.5
        ry = (point.y - bb.y) / bb.height if bb.height > 0 else 0.5

        # Check if inside
        if 0.05 <= rx <= 0.95 and 0.05 <= ry <= 0.95:
            return EndpointContact(
                side=ContactSide.INSIDE,
                distance=dist,
                normal=(0.0, 0.0),
                confidence=0.3,
            )

        # Determine dominant side
        dx_left = abs(point.x - bb.x)
        dx_right = abs(point.x - (bb.x + bb.width))
        dy_top = abs(point.y - bb.y)
        dy_bottom = abs(point.y - (bb.y + bb.height))

        min_dx = min(dx_left, dx_right)
        min_dy = min(dy_top, dy_bottom)

        # Corner check
        corner_threshold = min(bb.width, bb.height) * 0.2
        if min_dx < corner_threshold and min_dy < corner_threshold:
            # Determine which corner
            if point.x < cx and point.y < cy:
                normal = (-0.707, -0.707)
            elif point.x >= cx and point.y < cy:
                normal = (0.707, -0.707)
            elif point.x < cx and point.y >= cy:
                normal = (-0.707, 0.707)
            else:
                normal = (0.707, 0.707)
            return EndpointContact(
                side=ContactSide.CORNER,
                distance=dist,
                normal=normal,
                confidence=max(0.0, 1.0 - dist / 30.0),
            )

        # Edge classification
        if min_dx < min_dy:
            if dx_left < dx_right:
                side = ContactSide.LEFT
                normal = (-1.0, 0.0)
            else:
                side = ContactSide.RIGHT
                normal = (1.0, 0.0)
        else:
            if dy_top < dy_bottom:
                side = ContactSide.TOP
                normal = (0.0, -1.0)
            else:
                side = ContactSide.BOTTOM
                normal = (0.0, 1.0)

        confidence = max(0.0, 1.0 - dist / self.min_boundary_distance)

        return EndpointContact(
            side=side,
            distance=dist,
            normal=normal,
            confidence=confidence,
        )

    # =========================================================================
    # Boundary Normal Reasoning
    # =========================================================================

    def _score_normal_consistency(
        self,
        source_contact: EndpointContact,
        target_contact: EndpointContact,
        geometric_axis: Tuple[float, float],
    ) -> float:
        """
        Score whether boundary normals are consistent with the direction.

        Source should have outward normal aligned with direction.
        Target should have inward normal aligned with direction.
        """
        if (source_contact.side == ContactSide.FAR or
                target_contact.side == ContactSide.FAR):
            return 0.0

        # Source outward normal should agree with geometric axis
        src_dot = (source_contact.normal[0] * geometric_axis[0] +
                   source_contact.normal[1] * geometric_axis[1])

        # Target inward normal should agree with geometric axis
        # (inward = negated outward)
        tgt_inward = (-target_contact.normal[0], -target_contact.normal[1])
        tgt_dot = (tgt_inward[0] * geometric_axis[0] +
                   tgt_inward[1] * geometric_axis[1])

        # Both should be positive for a valid direction
        src_score = max(0.0, src_dot) * source_contact.confidence
        tgt_score = max(0.0, tgt_dot) * target_contact.confidence

        return (src_score + tgt_score) / 2.0

    # =========================================================================
    # Source Exits / Target Enters
    # =========================================================================

    def _score_source_exits(
        self,
        source_contact: EndpointContact,
        geometric_axis: Tuple[float, float],
    ) -> float:
        """
        Score whether the segment exits the source boundary.

        If source contact is on the right side, the segment should go right (+x).
        If source contact is on the left side, the segment should go left (-x).
        """
        if source_contact.side in (ContactSide.FAR, ContactSide.INSIDE):
            return 0.0

        # Normal points outward from the boundary
        # Direction should agree with the outward normal
        dot = (source_contact.normal[0] * geometric_axis[0] +
               source_contact.normal[1] * geometric_axis[1])

        if dot > 0:
            return min(1.0, dot * 1.5) * source_contact.confidence
        return 0.0

    def _score_target_enters(
        self,
        target_contact: EndpointContact,
        geometric_axis: Tuple[float, float],
    ) -> float:
        """
        Score whether the segment enters the target boundary.

        If target contact is on the left side, the segment should arrive from
        the left (i.e., direction points right, +x). So the direction should
        OPPOSE the outward normal (enter from outside).
        """
        if target_contact.side in (ContactSide.FAR, ContactSide.INSIDE):
            return 0.0

        # Direction should oppose the outward normal (enter from outside)
        dot = (target_contact.normal[0] * geometric_axis[0] +
               target_contact.normal[1] * geometric_axis[1])

        if dot < 0:
            return min(1.0, abs(dot) * 1.5) * target_contact.confidence
        return 0.0

    # =========================================================================
    # Segment Continuity
    # =========================================================================

    def _score_segment_continuity(
        self,
        endpoint_1: Point,
        endpoint_2: Point,
        source_bb: BoundingBox,
        target_bb: BoundingBox,
        source_contact: EndpointContact,
        target_contact: EndpointContact,
    ) -> float:
        """
        Score whether the segment forms a continuous path from source to target.

        The segment should:
        1. Start near the source boundary
        2. End near the target boundary
        3. Not pass through other nodes
        """
        if (source_contact.side == ContactSide.FAR or
                target_contact.side == ContactSide.FAR):
            return 0.0

        # Distance from source boundary to nearest endpoint
        src_dist = min(
            self._distance_to_boundary(endpoint_1, source_bb),
            self._distance_to_boundary(endpoint_2, source_bb),
        )
        # Distance from target boundary to nearest endpoint
        tgt_dist = min(
            self._distance_to_boundary(endpoint_1, target_bb),
            self._distance_to_boundary(endpoint_2, target_bb),
        )

        # Both endpoints should be near their respective boundaries
        src_score = max(0.0, 1.0 - src_dist / self.min_boundary_distance)
        tgt_score = max(0.0, 1.0 - tgt_dist / self.min_boundary_distance)

        return (src_score * source_contact.confidence +
                tgt_score * target_contact.confidence) / 2.0

    # =========================================================================
    # Angular Consistency
    # =========================================================================

    def _score_angular_consistency(
        self,
        source_bb: BoundingBox,
        target_bb: BoundingBox,
        geometric_axis: Tuple[float, float],
    ) -> float:
        """Score angular consistency between segment and node centers."""
        src_center = source_bb.center
        tgt_center = target_bb.center

        angle_to_target = math.atan2(
            tgt_center.y - src_center.y,
            tgt_center.x - src_center.x,
        )
        axis_angle = math.atan2(geometric_axis[1], geometric_axis[0])

        angle_diff = abs(angle_to_target - axis_angle)
        angle_diff = angle_diff % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        tolerance = math.radians(self.max_direction_tolerance_deg)
        if angle_diff > tolerance:
            return 0.0
        return 1.0 - (angle_diff / tolerance) * 0.5

    # =========================================================================
    # Boundary Path Score
    # =========================================================================

    def _score_boundary_path(
        self,
        source_contact: EndpointContact,
        target_contact: EndpointContact,
        source_exits: float,
        target_enters: float,
        geometric_axis: Tuple[float, float],
    ) -> float:
        """
        Combined score for the full boundary path: source→segment→target.

        A strong direction has:
        - Source exits cleanly
        - Target enters cleanly
        - Normal consistency along the path
        """
        if (source_contact.side == ContactSide.FAR or
                target_contact.side == ContactSide.FAR):
            return 0.0

        # Path completeness: both exits and enters should be strong
        path_completeness = (source_exits + target_enters) / 2.0

        # Side consistency: source and target should be on opposite sides
        # relative to the direction
        side_bonus = 0.0
        if (source_contact.side == ContactSide.RIGHT and
                target_contact.side == ContactSide.LEFT and
                geometric_axis[0] < 0):
            side_bonus = 0.3  # Right→Left matches
        elif (source_contact.side == ContactSide.LEFT and
                target_contact.side == ContactSide.RIGHT and
                geometric_axis[0] > 0):
            side_bonus = 0.3  # Left→Right matches
        elif (source_contact.side == ContactSide.BOTTOM and
                target_contact.side == ContactSide.TOP and
                geometric_axis[1] < 0):
            side_bonus = 0.3  # Bottom→Top matches
        elif (source_contact.side == ContactSide.TOP and
                target_contact.side == ContactSide.BOTTOM and
                geometric_axis[1] > 0):
            side_bonus = 0.3  # Top→Bottom matches

        return min(1.0, path_completeness + side_bonus)

    # =========================================================================
    # Crossing Penalty
    # =========================================================================

    def _score_crossing(
        self,
        source_point: Point,
        geometric_axis: Tuple[float, float],
        target_bb: BoundingBox,
        all_candidates: List[BoundingBox],
    ) -> float:
        """Penalize if the segment crosses unrelated node interiors."""
        penalty = 0.0
        for bb in all_candidates:
            if bb == target_bb:
                continue
            intersections = self._ray_rectangle_intersections(
                source_point, geometric_axis[0], geometric_axis[1], bb
            )
            for _, _, _, t in intersections:
                if t > 0:
                    penalty += 0.3
                    break
        return min(1.0, penalty)

    # =========================================================================
    # Arrowhead Evidence
    # =========================================================================

    def _score_arrowhead(
        self,
        arrowhead_candidates: List[Point],
        confidence: float,
        target_bb: BoundingBox,
    ) -> float:
        """Score arrowhead evidence near the target."""
        best = 0.0
        for ah in arrowhead_candidates:
            dist = self._distance_to_boundary(ah, target_bb)
            if dist < 10:
                best = max(best, 1.0)
            elif dist < 30:
                best = max(best, 0.7)
            elif dist < 60:
                best = max(best, 0.4)
            else:
                best = max(best, 0.1)
        return best * confidence

    # =========================================================================
    # Boundary Projection
    # =========================================================================

    def _project_to_boundary(
        self,
        point: Point,
        geometric_axis: Tuple[float, float],
        bb: BoundingBox,
        forward: bool = True,
    ) -> Optional[EndpointContact]:
        """
        Project an inside point to the rectangle boundary along the geometric axis.

        When an arrow endpoint is inside a source/target rectangle (common when
        the detector returns mid-segment points), we project along the segment
        direction to find where it intersects the boundary.

        Args:
            point: The inside point.
            geometric_axis: Direction to project (unit vector).
            bb: The rectangle bounding box.
            forward: If True, project in the direction of geometric_axis.
                     If False, project opposite.

        Returns:
            EndpointContact with the projected boundary point and normal,
            or None if no intersection found.
        """
        ndx = geometric_axis[0] if forward else -geometric_axis[0]
        ndy = geometric_axis[1] if forward else -geometric_axis[1]

        intersections = self._ray_rectangle_intersections(
            point, ndx, ndy, bb
        )

        if not intersections:
            return None

        # Take the nearest intersection
        ix, iy, edge, t = min(intersections, key=lambda x: x[3])

        # Determine side and normal from edge
        side_map = {
            "top": (ContactSide.TOP, (0.0, -1.0)),
            "bottom": (ContactSide.BOTTOM, (0.0, 1.0)),
            "left": (ContactSide.LEFT, (-1.0, 0.0)),
            "right": (ContactSide.RIGHT, (1.0, 0.0)),
        }
        side, normal = side_map.get(edge, (ContactSide.CORNER, (0.0, 0.0)))

        # For projected points, use fixed confidence since the projection
        # itself is geometrically valid. The real evaluation comes from
        # normal consistency and boundary path scores.
        confidence = 0.7

        return EndpointContact(
            side=side,
            distance=0.0,  # On the boundary
            normal=normal,
            confidence=confidence,
            boundary_point=Point(ix, iy),
        )

    # =========================================================================
    # Geometry Helpers
    # =========================================================================

    def _distance_to_boundary(self, point: Point, bb: BoundingBox) -> float:
        """Distance from point to nearest rectangle boundary."""
        cx = max(bb.x, min(point.x, bb.x + bb.width))
        cy = max(bb.y, min(point.y, bb.y + bb.height))
        dx = point.x - cx
        dy = point.y - cy
        return math.sqrt(dx ** 2 + dy ** 2)

    def _ray_rectangle_intersections(
        self,
        origin: Point,
        ndx: float,
        ndy: float,
        bb: BoundingBox,
    ) -> List[Tuple[float, float, str, float]]:
        """Find ray-rectangle intersections."""
        results = []
        edges = [
            (bb.x, bb.y, bb.x + bb.width, bb.y, "top"),
            (bb.x + bb.width, bb.y, bb.x + bb.width, bb.y + bb.height, "right"),
            (bb.x, bb.y + bb.height, bb.x + bb.width, bb.y + bb.height, "bottom"),
            (bb.x, bb.y, bb.x, bb.y + bb.height, "left"),
        ]

        for x1, y1, x2, y2, edge in edges:
            dx = x2 - x1
            dy = y2 - y1
            denom = ndx * dy - ndy * dx
            if abs(denom) < 1e-10:
                continue
            t = ((x1 - origin.x) * dy - (y1 - origin.y) * dx) / denom
            u = ((x1 - origin.x) * ndy - (y1 - origin.y) * ndx) / denom
            if u >= 0 and u <= 1 and t > 0:
                ix = origin.x + ndx * t
                iy = origin.y + ndy * t
                results.append((ix, iy, edge, t))

        return results

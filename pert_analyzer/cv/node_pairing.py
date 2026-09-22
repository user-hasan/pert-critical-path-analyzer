"""
Node pairing resolution for AON arrow-to-node association.

Separates arrow geometry from node identification. For each detected
arrow, the pairing resolver determines WHICH TWO NODES the arrow
connects — independently of arrow direction.

Pipeline:
  DetectedArrow (deduplicated)
  → find all nearby node candidates for each endpoint
  → generate candidate pairs
  → score each pair on boundary geometry, continuity, crossing
  → select best pair (or REVIEW/REJECT)
  → NodePairResult with selected pair, alternatives, evidence

The pairing stage MUST NOT decide direction. Direction is resolved
later by ArrowDirectionResolver for the selected pair only.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType

logger = logging.getLogger(__name__)


class PairingStatus(Enum):
    """Status of the node pairing resolution."""
    PAIR_CONFIRMED = "pair_confirmed"
    PAIR_CONTEXTUALLY_RESOLVED = "pair_contextually_resolved"
    PAIR_REVIEW_REQUIRED = "pair_review_required"
    PAIR_REJECTED = "pair_rejected"


@dataclass
class PairingEvidence:
    """Evidence vector for a node pair candidate."""
    boundary_contact_a: float = 0.0
    boundary_contact_b: float = 0.0
    endpoint_distance_a: float = 0.0
    endpoint_distance_b: float = 0.0
    ray_intersection_a: float = 0.0
    ray_intersection_b: float = 0.0
    line_continuity: float = 0.0
    angle_consistency: float = 0.0
    crossing_penalty: float = 0.0
    competing_pair_penalty: float = 0.0
    node_proximity: float = 0.0
    unanchored_fragment: float = 0.0
    arrow_opposition_penalty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "boundary_contact_a": self.boundary_contact_a,
            "boundary_contact_b": self.boundary_contact_b,
            "endpoint_distance_a": self.endpoint_distance_a,
            "endpoint_distance_b": self.endpoint_distance_b,
            "ray_intersection_a": self.ray_intersection_a,
            "ray_intersection_b": self.ray_intersection_b,
            "line_continuity": self.line_continuity,
            "angle_consistency": self.angle_consistency,
            "crossing_penalty": self.crossing_penalty,
            "competing_pair_penalty": self.competing_pair_penalty,
            "node_proximity": self.node_proximity,
            "unanchored_fragment": self.unanchored_fragment,
            "arrow_opposition_penalty": self.arrow_opposition_penalty,
        }


@dataclass
class NodePairCandidate:
    """A candidate pair of nodes that an arrow might connect."""
    arrow_id: str = ""
    node_a_id: str = ""
    node_b_id: str = ""
    endpoint_1: Point = field(default_factory=lambda: Point(0, 0))
    endpoint_2: Point = field(default_factory=lambda: Point(0, 0))
    score: float = 0.0
    evidence: PairingEvidence = field(default_factory=PairingEvidence)
    status: PairingStatus = PairingStatus.PAIR_REJECTED
    metadata: Dict[str, Any] = field(default_factory=dict)
    ocr_used: bool = False


@dataclass
class NodePairResult:
    """Result of node pairing for a single arrow."""
    arrow_id: str = ""
    selected: Optional[NodePairCandidate] = None
    alternatives: List[NodePairCandidate] = field(default_factory=list)
    status: PairingStatus = PairingStatus.PAIR_REJECTED
    confidence: float = 0.0
    rejection_reasons: List[str] = field(default_factory=list)
    all_candidates: List[NodePairCandidate] = field(default_factory=list)


class NodePairingResolver:
    """
    Resolves which two nodes an arrow connects.

    For each deduplicated arrow:
    1. Find plausible node candidates near each endpoint.
    2. Generate all candidate pairs from these candidates.
    3. Score each pair on boundary geometry, continuity, crossing.
    4. Select the best pair or mark as REVIEW/REJECT.
    """

    def __init__(
        self,
        max_endpoint_distance: float = 120.0,
        confirmed_margin: float = 0.15,
        min_boundary_distance: float = 80.0,
        max_candidates_per_endpoint: int = 4,
    ):
        self.max_endpoint_distance = max_endpoint_distance
        self.confirmed_margin = confirmed_margin
        self.min_boundary_distance = min_boundary_distance
        self.max_candidates_per_endpoint = max_candidates_per_endpoint

    def resolve_pairing(
        self,
        arrow: DetectedArrow,
        candidates: List[CandidateNode],
    ) -> NodePairResult:
        """
        Resolve which two nodes an arrow connects.

        Args:
            arrow: The detected arrow.
            candidates: Rectangle candidate nodes.

        Returns:
            NodePairResult with selected pair, alternatives, evidence.
        """
        result = NodePairResult(arrow_id=arrow.arrow_id)

        # Step 1: Find node candidates near each endpoint
        near_a = self._find_nearby_nodes(arrow.start, candidates)
        near_b = self._find_nearby_nodes(arrow.end, candidates)

        if len(near_a) == 0:
            result.rejection_reasons.append("no_node_near_endpoint_a")
            return result
        if len(near_b) == 0:
            result.rejection_reasons.append("no_node_near_endpoint_b")
            return result

        # Step 2: Generate candidate pairs
        pair_candidates = self._generate_pairs(
            arrow, near_a, near_b, candidates
        )

        if not pair_candidates:
            result.rejection_reasons.append("no_valid_pairs")
            return result

        result.all_candidates = pair_candidates

        # Step 3: Score and rank pairs
        self._score_pairs(
            pair_candidates, candidates, arrow.length, arrow
        )

        # Step 4: Select best pair
        pair_candidates.sort(key=lambda p: p.score, reverse=True)
        best = pair_candidates[0]

        # Check for competing pairs
        if len(pair_candidates) > 1:
            second = pair_candidates[1]
            margin = best.score - second.score
            best.metadata["margin_vs_second"] = margin

            if margin < self.confirmed_margin:
                best.status = PairingStatus.PAIR_REVIEW_REQUIRED
            elif best.score >= 0.5:
                best.status = PairingStatus.PAIR_CONFIRMED
            else:
                best.status = PairingStatus.PAIR_CONTEXTUALLY_RESOLVED
        else:
            if best.score >= 0.5:
                best.status = PairingStatus.PAIR_CONFIRMED
            else:
                best.status = PairingStatus.PAIR_CONTEXTUALLY_RESOLVED

        # Check if best score is too low — reject noise before applying
        # evidence-gated review overrides (a near-zero score is not worth
        # human review; it should be discarded outright).
        if best.score < 0.1:
            best.status = PairingStatus.PAIR_REJECTED
            result.rejection_reasons.append("all_pairs_too_weak")

        # Evidence-gated review overrides: prefer REVIEW over a wrong
        # automatic dependency when the geometry contradicts the arrow.
        if (best.status != PairingStatus.PAIR_REJECTED and
                best.evidence.unanchored_fragment >= 0.5):
            best.status = PairingStatus.PAIR_REVIEW_REQUIRED
        elif (best.status != PairingStatus.PAIR_REJECTED and
                best.evidence.arrow_opposition_penalty >= 0.5 and
                self._same_box_fragment(
                    best.endpoint_1, best.endpoint_2, candidates
                ) is not None):
            # Arrow fully contained inside one node box AND pointing into
            # void: a stub/label fragment, not a connector.
            best.status = PairingStatus.PAIR_REVIEW_REQUIRED
        elif (best.status != PairingStatus.PAIR_REJECTED and
                self._ambiguous_long_low_confidence_pair(
                    arrow, best, pair_candidates
                )):
            best.status = PairingStatus.PAIR_REVIEW_REQUIRED

        result.selected = best
        result.confidence = best.score
        result.status = best.status
        result.alternatives = pair_candidates[1:] if len(pair_candidates) > 1 else []

        return result

    def resolve_all(
        self,
        arrows: List[DetectedArrow],
        candidates: List[CandidateNode],
    ) -> List[NodePairResult]:
        """Resolve pairing for all arrows."""
        return [self.resolve_pairing(a, candidates) for a in arrows]

    # =========================================================================
    # Node Finding
    # =========================================================================

    def _find_nearby_nodes(
        self,
        point: Point,
        candidates: List[CandidateNode],
    ) -> List[Tuple[str, float, BoundingBox]]:
        """Find node candidates near a point, sorted by distance."""
        scored = []
        for cand in candidates:
            bb = cand.bounding_box
            dist = self._distance_to_boundary(point, bb)
            if dist > self.max_endpoint_distance:
                continue
            score = max(0.0, 1.0 - dist / self.max_endpoint_distance)
            # Bonus for being inside the rectangle
            if self._point_inside_rect(point, bb):
                score += 0.4
            scored.append((cand.node_id, score, bb))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:self.max_candidates_per_endpoint]

    # =========================================================================
    # Pair Generation
    # =========================================================================

    def _generate_pairs(
        self,
        arrow: DetectedArrow,
        near_a: List[Tuple[str, float, BoundingBox]],
        near_b: List[Tuple[str, float, BoundingBox]],
        all_candidates: List[CandidateNode],
    ) -> List[NodePairCandidate]:
        """Generate and deduplicate candidate pairs."""
        seen_pairs: Set[Tuple[str, str]] = set()
        pairs: List[NodePairCandidate] = []

        for na in near_a:
            for nb in near_b:
                a_id, a_score, a_bb = na
                b_id, b_score, b_bb = nb

                # Skip self-loops
                if a_id == b_id:
                    continue

                # Deduplicate (A,B) and (B,A)
                pair_key = tuple(sorted([a_id, b_id]))
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                pair = NodePairCandidate(
                    arrow_id=arrow.arrow_id,
                    node_a_id=a_id,
                    node_b_id=b_id,
                    endpoint_1=arrow.start,
                    endpoint_2=arrow.end,
                )
                pair.metadata["endpoint_node_a_score"] = a_score
                pair.metadata["endpoint_node_b_score"] = b_score
                pairs.append(pair)

        return pairs

    # =========================================================================
    # Pair Scoring
    # =========================================================================

    def _score_pairs(
        self,
        pairs: List[NodePairCandidate],
        all_candidates: List[CandidateNode],
        arrow_length: float = 0.0,
        arrow: Optional[DetectedArrow] = None,
    ) -> None:
        """Score all pair candidates."""
        for pair in pairs:
            score = self._score_single_pair(
                pair, all_candidates, arrow_length, arrow
            )
            pair.score = score

    def _score_single_pair(
        self,
        pair: NodePairCandidate,
        all_candidates: List[CandidateNode],
        arrow_length: float = 0.0,
        arrow: Optional[DetectedArrow] = None,
    ) -> float:
        """Score a single node pair."""
        evidence = PairingEvidence()

        # Find the actual CandidateNode objects
        cand_a = self._find_candidate(pair.node_a_id, all_candidates)
        cand_b = self._find_candidate(pair.node_b_id, all_candidates)
        if not cand_a or not cand_b:
            return 0.0

        bb_a = cand_a.bounding_box
        bb_b = cand_b.bounding_box

        # 1. Boundary contact at endpoint A
        evidence.boundary_contact_a = self._score_boundary_contact(
            pair.endpoint_1, bb_a
        )

        # 2. Boundary contact at endpoint B
        evidence.boundary_contact_b = self._score_boundary_contact(
            pair.endpoint_2, bb_b
        )

        # 3. Endpoint distance to node boundary
        evidence.endpoint_distance_a = self._score_endpoint_distance(
            pair.endpoint_1, bb_a
        )
        evidence.endpoint_distance_b = self._score_endpoint_distance(
            pair.endpoint_2, bb_b
        )

        # 4. Ray intersection (does line from endpoint hit the node)
        evidence.ray_intersection_a = self._score_ray_intersection(
            pair.endpoint_1, pair.endpoint_2, bb_a
        )
        evidence.ray_intersection_b = self._score_ray_intersection(
            pair.endpoint_2, pair.endpoint_1, bb_b
        )

        # 5. Line continuity (does segment form a path between the two nodes)
        evidence.line_continuity = self._score_line_continuity(
            pair.endpoint_1, pair.endpoint_2, bb_a, bb_b
        )

        # 6. Angle consistency (boundary normal vs line direction)
        evidence.angle_consistency = self._score_angle_consistency(
            pair.endpoint_1, pair.endpoint_2, bb_a, bb_b
        )

        # 7. Crossing penalty (does line cross other nodes)
        evidence.crossing_penalty = self._score_crossing_penalty(
            pair.endpoint_1, pair.endpoint_2, bb_a, bb_b,
            pair.node_a_id, pair.node_b_id, all_candidates
        )

        # 8. Node proximity (weak: closer nodes slightly preferred)
        evidence.node_proximity = self._score_node_proximity(bb_a, bb_b)

        # 9. Unanchored fragment: no arrowhead AND neither endpoint is near
        #    any node boundary → floating trunk/bus segment, not a real arrow.
        evidence.unanchored_fragment = self._score_unanchored_fragment(
            pair.endpoint_1, pair.endpoint_2, all_candidates, arrow
        )

        # 10. Arrow opposition: the arrowhead's forward ray must reach the pair's
        #     partner node (or at least a pillar of the pair).  If the arrow
        #     is wholly inside one box and its tip points away toward a THIRD
        #     node (or into void), the stroke is a label/stub fragment.
        evidence.arrow_opposition_penalty = self._score_arrow_opposition(
            pair.endpoint_1, pair.endpoint_2,
            pair.node_a_id, pair.node_b_id, all_candidates, arrow
        )

        # Weighted combination
        score = (
            0.22 * evidence.boundary_contact_a
            + 0.22 * evidence.boundary_contact_b
            + 0.12 * evidence.endpoint_distance_a
            + 0.12 * evidence.endpoint_distance_b
            + 0.08 * evidence.ray_intersection_a
            + 0.08 * evidence.ray_intersection_b
            + 0.10 * evidence.line_continuity
            + 0.06 * evidence.angle_consistency
            - 0.15 * evidence.crossing_penalty
            + 0.03 * evidence.node_proximity
            - 0.25 * evidence.unanchored_fragment
        )

        # Penalize very short arrows — they can't reliably span nodes
        if arrow_length < 10:
            score *= 0.3
        elif arrow_length < 20:
            score *= 0.6

        score = max(0.0, min(1.0, score))
        pair.evidence = evidence
        pair.metadata["components"] = evidence.to_dict()
        return score

    # =========================================================================
    # Evidence Factors
    # =========================================================================

    def _score_boundary_contact(
        self, point: Point, bb: BoundingBox
    ) -> float:
        """Score how close endpoint is to rectangle boundary."""
        dist = self._distance_to_boundary(point, bb)
        if dist < 2:
            return 1.0
        if dist < 15:
            return 0.9
        if dist < 30:
            return 0.7
        if dist < 50:
            return 0.5
        if dist < 80:
            return 0.3
        return 0.0

    def _score_endpoint_distance(
        self, point: Point, bb: BoundingBox
    ) -> float:
        """Score endpoint proximity to node (inverse distance)."""
        dist = self._distance_to_boundary(point, bb)
        if dist < 5:
            return 1.0
        if dist < 20:
            return 0.8
        if dist < 50:
            return 0.5
        if dist < self.min_boundary_distance:
            return 0.3
        return 0.1

    def _score_ray_intersection(
        self,
        origin: Point,
        other_end: Point,
        target_bb: BoundingBox,
    ) -> float:
        """Score whether a ray from origin toward other_end hits the target node."""
        dx = other_end.x - origin.x
        dy = other_end.y - origin.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return 0.0
        ndx, ndy = dx / length, dy / length

        intersections = self._ray_rectangle_intersections(origin, ndx, ndy, target_bb)
        valid = [t for _, _, _, t in intersections if t >= 0]
        if valid:
            closest_t = min(valid)
            if closest_t < 5:
                return 1.0
            if closest_t < 30:
                return 0.8
            if closest_t < 80:
                return 0.5
            return 0.3
        return 0.0

    def _score_line_continuity(
        self,
        endpoint_a: Point,
        endpoint_b: Point,
        bb_a: BoundingBox,
        bb_b: BoundingBox,
    ) -> float:
        """Score geometric continuity of the segment between the two nodes."""
        center_a = bb_a.center
        center_b = bb_b.center

        # Direction from center_a to center_b
        path_dx = center_b.x - center_a.x
        path_dy = center_b.y - center_a.y
        path_len = math.sqrt(path_dx**2 + path_dy**2)
        if path_len < 1e-6:
            return 0.0

        # Segment direction
        seg_dx = endpoint_b.x - endpoint_a.x
        seg_dy = endpoint_b.y - endpoint_a.y
        seg_len = math.sqrt(seg_dx**2 + seg_dy**2)
        if seg_len < 1e-6:
            return 0.0

        # Collinearity: does segment align with center-to-center path?
        dot = (seg_dx / seg_len) * (path_dx / path_len) + \
              (seg_dy / seg_len) * (path_dy / path_len)

        # Also check: endpoints should be near the nodes
        a_near = self._distance_to_boundary(endpoint_a, bb_a)
        b_near = self._distance_to_boundary(endpoint_b, bb_b)
        near_score = 1.0 - (a_near + b_near) / (2 * self.min_boundary_distance)
        near_score = max(0.0, min(1.0, near_score))

        return max(0.0, dot) * 0.6 + near_score * 0.4

    def _score_angle_consistency(
        self,
        endpoint_a: Point,
        endpoint_b: Point,
        bb_a: BoundingBox,
        bb_b: BoundingBox,
    ) -> float:
        """Score angle consistency at both boundaries."""
        # Direction of the segment
        dx = endpoint_b.x - endpoint_a.x
        dy = endpoint_b.y - endpoint_a.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return 0.0
        ndx, ndy = dx / length, dy / length

        # At endpoint A: check angle to node center
        center_a = bb_a.center
        to_center_a_x = center_a.x - endpoint_a.x
        to_center_a_y = center_a.y - endpoint_a.y
        tc_len_a = math.sqrt(to_center_a_x**2 + to_center_a_y**2)
        if tc_len_a < 1e-6:
            score_a = 0.5
        else:
            tcx_a = to_center_a_x / tc_len_a
            tcy_a = to_center_a_y / tc_len_a
            # Arrow should point away from node_a center (if node_a is source)
            dot_a = -(ndx * tcx_a + ndy * tcy_a)
            score_a = max(0.0, dot_a)

        # At endpoint B: check angle from other end to node center
        center_b = bb_b.center
        to_center_b_x = center_b.x - endpoint_b.x
        to_center_b_y = center_b.y - endpoint_b.y
        tc_len_b = math.sqrt(to_center_b_x**2 + to_center_b_y**2)
        if tc_len_b < 1e-6:
            score_b = 0.5
        else:
            tcx_b = to_center_b_x / tc_len_b
            tcy_b = to_center_b_y / tc_len_b
            # Arrow should point toward node_b center (if node_b is target)
            dot_b = ndx * tcx_b + ndy * tcy_b
            score_b = max(0.0, dot_b)

        return (score_a + score_b) / 2

    def _score_crossing_penalty(
        self,
        endpoint_a: Point,
        endpoint_b: Point,
        bb_a: BoundingBox,
        bb_b: BoundingBox,
        node_a_id: str,
        node_b_id: str,
        all_candidates: List[CandidateNode],
    ) -> float:
        """Penalize pairs whose arrow crosses unrelated nodes."""
        dx = endpoint_b.x - endpoint_a.x
        dy = endpoint_b.y - endpoint_a.y
        length = math.sqrt(dx**2 + dy**2)
        if length < 1e-6:
            return 0.0
        ndx, ndy = dx / length, dy / length

        penalty = 0.0
        for cand in all_candidates:
            if cand.node_id in (node_a_id, node_b_id):
                continue
            bb = cand.bounding_box
            intersections = self._ray_rectangle_intersections(
                endpoint_a, ndx, ndy, bb
            )
            for x, y, edge, t in intersections:
                # Only penalize intersections along the actual segment
                if 0 < t < length:
                    penalty += 0.3
                    break

        return min(1.0, penalty)

    def _score_node_proximity(
        self, bb_a: BoundingBox, bb_b: BoundingBox
    ) -> float:
        """Weak proximity score — closer nodes slightly preferred."""
        center_a = bb_a.center
        center_b = bb_b.center
        dist = math.sqrt(
            (center_b.x - center_a.x)**2 + (center_b.y - center_a.y)**2
        )
        # Normalized: closer = higher score
        if dist < 100:
            return 0.8
        if dist < 300:
            return 0.6
        if dist < 600:
            return 0.4
        return 0.2

    def _score_unanchored_fragment(
        self,
        endpoint_1: Point,
        endpoint_2: Point,
        all_candidates: List[CandidateNode],
        arrow: Optional[DetectedArrow],
    ) -> float:
        """Detect floating trunk/bus fragments with no node anchor.

        A real inter-node arrow always has its tail inside or immediately
        next to its source node, so at least one endpoint is near a node
        boundary.  Arrows with NO arrowhead AND both endpoints a meaningful
        gap away from EVERY node are orthogonal routing buses merged into
        an arrow candidate — they must not create a dependency pair.
        """
        if arrow is not None and arrow.arrowhead_confidence > 0.0:
            # An arrowhead is stronger evidence this is an intentional arrow.
            return 0.0

        gap = 12.0
        if (self._min_distance_to_any_node(endpoint_1, all_candidates) > gap and
                self._min_distance_to_any_node(endpoint_2, all_candidates) > gap):
            return 1.0
        return 0.0

    def _score_arrow_opposition(
        self,
        endpoint_1: Point,
        endpoint_2: Point,
        node_a_id: str,
        node_b_id: str,
        all_candidates: List[CandidateNode],
        arrow: Optional[DetectedArrow],
    ) -> float:
        """Flag fragments whose arrowhead does not point at the pair partner.

        The arrowhead tip is the only reliable semantic cue available at the
        pairing stage.  For a connector drawn as a stub fully inside a single
        node box, the arrow's forward ray (far end → tip) extended from the
        tip must emerge from that box and first reach the OTHER member of the
        candidate pair.  If the ray instead enters a third node, or reaches
        no node at all, the stroke is a label/glyph fragment and the pair
        must not be accepted from it.
        """
        if arrow is None or arrow.arrowhead_point is None:
            return 0.0

        tip = arrow.arrowhead_point

        # Identify the node box containing the tip.
        containing_id = None
        containing_bb = None
        for cand in all_candidates:
            bb = cand.bounding_box
            if self._point_inside_rect(tip, bb):
                containing_id = cand.node_id
                containing_bb = bb
                break
        if containing_id is None:
            return 0.0

        # Rule applies only to stubs fully inside one node box.
        if not (self._point_inside_rect(endpoint_1, containing_bb) and
                self._point_inside_rect(endpoint_2, containing_bb)):
            return 0.0

        far_end = (
            endpoint_1 if self._dist_points(tip, endpoint_1) >=
            self._dist_points(tip, endpoint_2) else endpoint_2
        )
        dx, dy = tip.x - far_end.x, tip.y - far_end.y
        length = math.sqrt(dx * dx + dy * dy)
        if length < 1e-6:
            return 0.0
        ndx, ndy = dx / length, dy / length

        # First node reached beyond the containing box.
        best_t, best_node = None, None
        for cand in all_candidates:
            if cand.node_id == containing_id:
                continue
            d = self._ray_rect_distance(tip, ndx, ndy, cand.bounding_box, pad=4.0)
            if d is not None and (best_t is None or d < best_t):
                best_t, best_node = d, cand.node_id

        if best_node is None:
            # Points into void.
            return 1.0
        if best_node in (node_a_id, node_b_id):
            # Points at the candidate pair's other member — plausible arrow.
            return 0.0
        # Points at a third node through/away from the box — fragment noise.
        return 1.0

    def _ray_rect_distance(
        self,
        point: Point,
        ndx: float,
        ndy: float,
        bb: BoundingBox,
        pad: float = 0.0,
    ) -> Optional[float]:
        """Parametric distance (t >= 0) along the ray to enter the rect, if any."""
        lo_x, hi_x = bb.x - pad, bb.x + bb.width + pad
        lo_y, hi_y = bb.y - pad, bb.y + bb.height + pad
        tmin, tmax = 0.0, float("inf")
        for org, d, lo, hi in (
            (point.x, ndx, lo_x, hi_x),
            (point.y, ndy, lo_y, hi_y),
        ):
            if abs(d) < 1e-9:
                if org < lo or org > hi:
                    return None
            else:
                t1 = (lo - org) / d
                t2 = (hi - org) / d
                if t1 > t2:
                    t1, t2 = t2, t1
                tmin = max(tmin, t1)
                tmax = min(tmax, t2)
                if tmin > tmax:
                    return None
        return tmin

    def _same_box_fragment(
        self,
        endpoint_1: Point,
        endpoint_2: Point,
        candidates: List[CandidateNode],
    ) -> Optional[str]:
        """Return the node whose box contains BOTH endpoints, else None.

        An arrow wholly inside a single node's box is only a meaningful
        connector if it also points toward another node; otherwise it is a
        stub/label fragment.
        """
        for cand in candidates:
            bb = cand.bounding_box
            if self._point_inside_rect(endpoint_1, bb) and self._point_inside_rect(endpoint_2, bb):
                return cand.node_id
        return None

    def _min_distance_to_any_node(
        self, point: Point, candidates: List[CandidateNode]
    ) -> float:
        """Minimum distance from a point to any candidate node boundary."""
        best = float("inf")
        for cand in candidates:
            d = self._distance_to_boundary(point, cand.bounding_box)
            if d < best:
                best = d
        return best

    def _dist_points(self, a: Point, b: Point) -> float:
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

    def _ambiguous_long_low_confidence_pair(
        self,
        arrow: Optional[DetectedArrow],
        best: NodePairCandidate,
        candidates: List[NodePairCandidate],
    ) -> bool:
        """Long arrows with a weak arrowhead and no decisive margin → review.

        A long arrow whose arrowhead is detected at LOW confidence and whose
        best pair only beats the runner-up by a small margin carries
        insufficient evidence to emit a dependency.  REVIEW instead of
        guessing (the tail may be part of shared orthogonal routing).
        """
        if arrow is None:
            return False
        ah_conf = arrow.arrowhead_confidence
        if not (0.0 < ah_conf < 0.3):
            return False
        if arrow.length < 100:
            return False
        if best.score >= 0.8:
            return False
        if len(candidates) < 2:
            return False
        margin = best.metadata.get("margin_vs_second", 1.0)
        return margin < 0.25

    # =========================================================================
    # Geometry Helpers
    # =========================================================================

    def _find_candidate(
        self, node_id: str, candidates: List[CandidateNode]
    ) -> Optional[CandidateNode]:
        """Find a candidate by node_id."""
        for c in candidates:
            if c.node_id == node_id:
                return c
        return None

    def _distance_to_boundary(self, point: Point, bb: BoundingBox) -> float:
        """Distance from point to nearest rectangle boundary."""
        cx = max(bb.x, min(point.x, bb.x + bb.width))
        cy = max(bb.y, min(point.y, bb.y + bb.height))
        dx = point.x - cx
        dy = point.y - cy
        return math.sqrt(dx**2 + dy**2)

    def _point_inside_rect(self, point: Point, bb: BoundingBox) -> bool:
        """Check if point is inside rectangle."""
        return (bb.x <= point.x <= bb.x + bb.width and
                bb.y <= point.y <= bb.y + bb.height)

    def _ray_rectangle_intersections(
        self,
        origin: Point,
        rdx: float,
        rdy: float,
        bb: BoundingBox,
    ) -> List[Tuple[float, float, str, float]]:
        """Find ray-rectangle intersections."""
        intersections = []
        edges = [
            (bb.x, "left"), (bb.x + bb.width, "right"),
            (bb.y, "top"), (bb.y + bb.height, "bottom"),
        ]
        for edge_val, edge_name in edges:
            if edge_name in ("left", "right"):
                if abs(rdx) < 1e-10:
                    continue
                t = (edge_val - origin.x) / rdx
                if t < 0:
                    continue
                y = origin.y + t * rdy
                if bb.y - 1 <= y <= bb.y + bb.height + 1:
                    intersections.append((edge_val, y, edge_name, t))
            else:
                if abs(rdy) < 1e-10:
                    continue
                t = (edge_val - origin.y) / rdy
                if t < 0:
                    continue
                x = origin.x + t * rdx
                if bb.x - 1 <= x <= bb.x + bb.width + 1:
                    intersections.append((x, edge_val, edge_name, t))
        return intersections

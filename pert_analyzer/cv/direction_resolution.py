"""
Bidirectional arrow direction resolution for AON diagrams.

Separates arrow geometry from arrow orientation. Each detected arrow
is treated as an unoriented segment with two possible orientations.
Both orientations are scored independently using multiple evidence
factors. The margin between scores determines whether the result is
CONFIRMED or REVIEW_REQUIRED.

Pipeline:
  DetectedArrow
  → ArrowSegment (unoriented geometry)
  → score orientation A→B
  → score orientation B→A
  → compare margin
  → ArrowDirectionResult with source, target, confidence, alternatives
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.direction_evidence import DirectionEvidenceLayer
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType

logger = logging.getLogger(__name__)


class DirectionStatus(Enum):
    """Status of the direction resolution."""
    CONFIRMED = "confirmed"
    CONTEXTUALLY_RESOLVED = "contextually_resolved"
    REVIEW_REQUIRED = "review_required"
    REJECTED = "rejected"


@dataclass
class OrientationEvidence:
    """Evidence vector for a single orientation (source→target)."""
    arrowhead_at_target: float = 0.0
    source_boundary_contact: float = 0.0
    target_boundary_contact: float = 0.0
    source_outward_normal: float = 0.0
    target_inward_normal: float = 0.0
    direction_alignment: float = 0.0
    angular_consistency: float = 0.0
    spatial_flow: float = 0.0
    continuity: float = 0.0
    node_distance: float = 0.0
    crossing_penalty: float = 0.0
    ambiguity_penalty: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "arrowhead_at_target": self.arrowhead_at_target,
            "source_boundary_contact": self.source_boundary_contact,
            "target_boundary_contact": self.target_boundary_contact,
            "source_outward_normal": self.source_outward_normal,
            "target_inward_normal": self.target_inward_normal,
            "direction_alignment": self.direction_alignment,
            "angular_consistency": self.angular_consistency,
            "spatial_flow": self.spatial_flow,
            "continuity": self.continuity,
            "node_distance": self.node_distance,
            "crossing_penalty": self.crossing_penalty,
            "ambiguity_penalty": self.ambiguity_penalty,
        }


@dataclass
class OrientationScore:
    """Scored orientation with full evidence."""
    source_id: str = ""
    target_id: str = ""
    source_point: Point = field(default_factory=lambda: Point(0, 0))
    target_point: Point = field(default_factory=lambda: Point(0, 0))
    score: float = 0.0
    evidence: OrientationEvidence = field(default_factory=OrientationEvidence)
    components: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArrowSegment:
    """An unoriented geometric arrow segment with two endpoints.

    Endpoint ordering is purely geometric (sorted by x, then y).
    It must NOT be interpreted as source → target.
    """
    arrow_id: str = ""
    endpoint_1: Point = field(default_factory=lambda: Point(0, 0))
    endpoint_2: Point = field(default_factory=lambda: Point(0, 0))
    geometric_axis: Tuple[float, float] = (0.0, 0.0)
    arrowhead_candidates: List[Point] = field(default_factory=list)
    arrowhead_confidence: float = 0.0
    arrow_length: float = 0.0
    arrow_confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Normalize endpoint ordering to be purely geometric."""
        if (self.endpoint_1.x > self.endpoint_2.x or
                (self.endpoint_1.x == self.endpoint_2.x and
                 self.endpoint_1.y > self.endpoint_2.y)):
            self.endpoint_1, self.endpoint_2 = self.endpoint_2, self.endpoint_1
            # Recompute geometric axis to match normalized endpoints
            dx = self.endpoint_2.x - self.endpoint_1.x
            dy = self.endpoint_2.y - self.endpoint_1.y
            dl = (dx**2 + dy**2) ** 0.5
            if dl > 0:
                self.geometric_axis = (dx / dl, dy / dl)
            else:
                self.geometric_axis = (0.0, 0.0)

    @property
    def midpoint(self) -> Point:
        """Get the midpoint of the segment."""
        return Point(
            (self.endpoint_1.x + self.endpoint_2.x) / 2,
            (self.endpoint_1.y + self.endpoint_2.y) / 2,
        )


@dataclass
class ArrowDirectionResult:
    """Result of bidirectional arrow direction resolution."""
    arrow_id: str = ""
    segment: Optional[ArrowSegment] = None
    status: DirectionStatus = DirectionStatus.REJECTED
    source_id: str = ""
    target_id: str = ""
    confidence: float = 0.0
    orientation_a_to_b: Optional[OrientationScore] = None
    orientation_b_to_a: Optional[OrientationScore] = None
    margin: float = 0.0
    evidence: Dict[str, Any] = field(default_factory=dict)
    rejection_reasons: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


class ArrowDirectionResolver:
    """
    Resolves arrow direction through bidirectional scoring.

    For every valid arrow segment connecting two node candidates,
    both orientations (A→B and B→A) are scored independently.
    The margin between scores determines the resolution status.
    """

    def __init__(
        self,
        confirmed_margin: float = 0.20,
        min_boundary_distance: float = 80.0,
        max_direction_tolerance_deg: float = 40.0,
        spatial_flow_weight: float = 0.10,
    ):
        self.confirmed_margin = confirmed_margin
        self.min_boundary_distance = min_boundary_distance
        self.max_direction_tolerance_deg = max_direction_tolerance_deg
        self.spatial_flow_weight = spatial_flow_weight
        self._evidence_layer = DirectionEvidenceLayer(
            min_boundary_distance=min_boundary_distance,
            max_direction_tolerance_deg=max_direction_tolerance_deg,
        )

    def resolve_direction(
        self,
        arrow: DetectedArrow,
        candidates: List[CandidateNode],
    ) -> ArrowDirectionResult:
        """
        Resolve direction for a single detected arrow.

        Args:
            arrow: The detected arrow.
            candidates: Rectangle candidate nodes.

        Returns:
            ArrowDirectionResult with both orientations scored.
        """
        segment = self._arrow_to_segment(arrow)
        return self._resolve_segment(segment, candidates, arrow)

    def resolve_direction_for_pair(
        self,
        arrow: DetectedArrow,
        node_a_id: str,
        node_b_id: str,
        candidates: List[CandidateNode],
    ) -> ArrowDirectionResult:
        """
        Resolve direction for a FIXED node pair.

        This is called after NodePairingResolver selects the pair.
        Direction resolver does NOT reconsider arbitrary third nodes.

        Args:
            arrow: The detected arrow.
            node_a_id: The first node of the selected pair.
            node_b_id: The second node of the selected pair.
            candidates: Rectangle candidate nodes.

        Returns:
            ArrowDirectionResult with both orientations scored.
        """
        segment = self._arrow_to_segment(arrow)
        result = ArrowDirectionResult(
            arrow_id=segment.arrow_id,
            segment=segment,
        )

        # Find the fixed pair candidates
        cand_a = self._find_candidate_by_id(node_a_id, candidates)
        cand_b = self._find_candidate_by_id(node_b_id, candidates)

        if not cand_a or not cand_b:
            result.status = DirectionStatus.REJECTED
            if not cand_a:
                result.rejection_reasons.append("node_a_not_found")
            if not cand_b:
                result.rejection_reasons.append("node_b_not_found")
            return result

        # Score both orientations for the fixed pair
        score_ab = self._score_orientation(
            segment=segment,
            source_node=(cand_a.node_id, 1.0, cand_a.bounding_box),
            target_node=(cand_b.node_id, 1.0, cand_b.bounding_box),
            source_point=segment.endpoint_1,
            target_point=segment.endpoint_2,
            all_candidates=candidates,
        )
        result.orientation_a_to_b = score_ab

        score_ba = self._score_orientation(
            segment=segment,
            source_node=(cand_b.node_id, 1.0, cand_b.bounding_box),
            target_node=(cand_a.node_id, 1.0, cand_a.bounding_box),
            source_point=segment.endpoint_2,
            target_point=segment.endpoint_1,
            all_candidates=candidates,
        )
        result.orientation_b_to_a = score_ba

        # Determine winner
        result.margin = abs(score_ab.score - score_ba.score)

        if score_ab.score > score_ba.score:
            winner, loser = score_ab, score_ba
        else:
            winner, loser = score_ba, score_ab

        result.source_id = winner.source_id
        result.target_id = winner.target_id
        result.confidence = winner.score
        result.evidence = winner.evidence.to_dict()
        result.alternatives = [{
            "source_id": loser.source_id,
            "target_id": loser.target_id,
            "score": loser.score,
            "evidence": loser.evidence.to_dict(),
        }]

        # Determine status
        has_arrowhead = len(segment.arrowhead_candidates) > 0
        if (not has_arrowhead and segment.arrow_length >= 150.0
                and result.margin < self.confirmed_margin):
            # Long routing segment with no arrowhead: orientation is
            # undecidable geometry (the real arrowhead was lost on an
            # orthogonal riser/bus).  REVIEW instead of guessing a direction.
            result.status = DirectionStatus.REVIEW_REQUIRED
        elif result.margin >= self.confirmed_margin and winner.score >= 0.5:
            result.status = DirectionStatus.CONFIRMED
        elif winner.score >= 0.3:
            result.status = DirectionStatus.CONTEXTUALLY_RESOLVED
        elif winner.score >= 0.1:
            result.status = DirectionStatus.REVIEW_REQUIRED
        else:
            result.status = DirectionStatus.REJECTED
            result.rejection_reasons.append("both_orientations_weak")

        return result

    def resolve_all(
        self,
        arrows: List[DetectedArrow],
        candidates: List[CandidateNode],
    ) -> List[ArrowDirectionResult]:
        """Resolve direction for all arrows."""
        results = []
        for arrow in arrows:
            result = self.resolve_direction(arrow, candidates)
            results.append(result)
        return results

    # =========================================================================
    # Segment Construction
    # =========================================================================

    def _arrow_to_segment(self, arrow: DetectedArrow) -> ArrowSegment:
        """Convert a DetectedArrow to an unoriented ArrowSegment.

        The DetectedArrow endpoints are already normalized (sorted by x, then y)
        by __post_init__. We preserve that ordering here.
        """
        arrowhead_candidates = []
        if arrow.arrowhead_point:
            arrowhead_candidates.append(arrow.arrowhead_point)

        # Compute geometric axis from normalized endpoints
        dx = arrow.end.x - arrow.start.x
        dy = arrow.end.y - arrow.start.y
        dl = (dx**2 + dy**2) ** 0.5
        if dl > 0:
            geometric_axis = (dx / dl, dy / dl)
        else:
            geometric_axis = (0.0, 0.0)

        return ArrowSegment(
            arrow_id=arrow.arrow_id,
            endpoint_1=arrow.start,
            endpoint_2=arrow.end,
            geometric_axis=geometric_axis,
            arrowhead_candidates=arrowhead_candidates,
            arrowhead_confidence=arrow.arrowhead_confidence,
            arrow_length=arrow.length,
            arrow_confidence=arrow.confidence,
            metadata=dict(arrow.metadata) if arrow.metadata else {},
        )

    # =========================================================================
    # Core Resolution
    # =========================================================================

    def _resolve_segment(
        self,
        segment: ArrowSegment,
        candidates: List[CandidateNode],
        original_arrow: DetectedArrow,
    ) -> ArrowDirectionResult:
        """Resolve direction for a segment by scoring both orientations."""
        result = ArrowDirectionResult(
            arrow_id=segment.arrow_id,
            segment=segment,
        )

        # Find the two best node candidates for each endpoint
        node_a = self._find_best_node(segment.endpoint_1, candidates)
        node_b = self._find_best_node(segment.endpoint_2, candidates)

        if not node_a or not node_b:
            result.status = DirectionStatus.REJECTED
            if not node_a:
                result.rejection_reasons.append("no_node_at_endpoint_a")
            if not node_b:
                result.rejection_reasons.append("no_node_at_endpoint_b")
            return result

        # Self-loop check
        if node_a[0] == node_b[0]:
            result.status = DirectionStatus.REJECTED
            result.rejection_reasons.append("self_loop")
            return result

        # Score orientation A→B
        score_ab = self._score_orientation(
            segment=segment,
            source_node=node_a,
            target_node=node_b,
            source_point=segment.endpoint_1,
            target_point=segment.endpoint_2,
            all_candidates=candidates,
        )
        result.orientation_a_to_b = score_ab

        # Score orientation B→A
        score_ba = self._score_orientation(
            segment=segment,
            source_node=node_b,
            target_node=node_a,
            source_point=segment.endpoint_2,
            target_point=segment.endpoint_1,
            all_candidates=candidates,
        )
        result.orientation_b_to_a = score_ba

        # Determine winner by margin
        result.margin = abs(score_ab.score - score_ba.score)

        if score_ab.score > score_ba.score:
            winner = score_ab
            loser = score_ba
        else:
            winner = score_ba
            loser = score_ab

        result.source_id = winner.source_id
        result.target_id = winner.target_id
        result.confidence = winner.score
        result.evidence = winner.evidence.to_dict()
        result.alternatives = [{
            "source_id": loser.source_id,
            "target_id": loser.target_id,
            "score": loser.score,
            "evidence": loser.evidence.to_dict(),
        }]

        # Determine status by margin
        has_arrowhead = len(segment.arrowhead_candidates) > 0
        if (not has_arrowhead and segment.arrow_length >= 150.0
                and result.margin < self.confirmed_margin):
            # Long routing segment with no arrowhead: orientation is
            # undecidable geometry (the real arrowhead was lost on an
            # orthogonal riser/bus).  REVIEW instead of guessing a direction.
            result.status = DirectionStatus.REVIEW_REQUIRED
        elif result.margin >= self.confirmed_margin and winner.score >= 0.5:
            result.status = DirectionStatus.CONFIRMED
        elif winner.score >= 0.3:
            result.status = DirectionStatus.CONTEXTUALLY_RESOLVED
        elif winner.score >= 0.1:
            result.status = DirectionStatus.REVIEW_REQUIRED
        else:
            result.status = DirectionStatus.REJECTED
            result.rejection_reasons.append("both_orientations_weak")

        return result

    # =========================================================================
    # Node Matching
    # =========================================================================

    def _find_best_node(
        self,
        point: Point,
        candidates: List[CandidateNode],
    ) -> Optional[Tuple[str, float, BoundingBox]]:
        """Find the best node candidate near a point."""
        best = None
        best_score = -1.0

        for cand in candidates:
            bb = cand.bounding_box
            dist = self._distance_to_boundary(point, bb)
            if dist > self.min_boundary_distance * 1.5:
                continue

            # Score: closer to boundary is better
            score = max(0.0, 1.0 - dist / self.min_boundary_distance)

            # Bonus for being inside the rectangle
            if self._point_inside_rect(point, bb):
                score += 0.3

            # Bonus for being near the center
            center = bb.center
            center_dist = math.sqrt(
                (point.x - center.x) ** 2 + (point.y - center.y) ** 2
            )
            diag = math.sqrt(bb.width ** 2 + bb.height ** 2)
            if diag > 0:
                center_score = max(0.0, 1.0 - center_dist / diag)
                score += 0.1 * center_score

            if score > best_score:
                best_score = score
                best = (cand.node_id, score, bb)

        return best

    def _find_candidate_by_id(
        self, node_id: str, candidates: List[CandidateNode]
    ) -> Optional[CandidateNode]:
        """Find a candidate by node_id."""
        for c in candidates:
            if c.node_id == node_id:
                return c
        return None

    # =========================================================================
    # Orientation Scoring
    # =========================================================================

    def _score_orientation(
        self,
        segment: ArrowSegment,
        source_node: Tuple[str, float, BoundingBox],
        target_node: Tuple[str, float, BoundingBox],
        source_point: Point,
        target_point: Point,
        all_candidates: List[CandidateNode],
    ) -> OrientationScore:
        """Score a single source→target orientation."""
        evidence = OrientationEvidence()
        src_id, src_match, src_bb = source_node
        tgt_id, tgt_match, tgt_bb = target_node

        # Direction from source center to target center
        src_center = src_bb.center
        tgt_center = tgt_bb.center
        center_dx = tgt_center.x - src_center.x
        center_dy = tgt_center.y - src_center.y
        center_len = math.sqrt(center_dx ** 2 + center_dy ** 2)
        if center_len > 1e-6:
            center_ndx = center_dx / center_len
            center_ndy = center_dy / center_len
        else:
            center_ndx = 0.0
            center_ndy = 0.0

        # Segment direction (A→B as given, not assumed semantic)
        seg_dx = segment.endpoint_2.x - segment.endpoint_1.x
        seg_dy = segment.endpoint_2.y - segment.endpoint_1.y
        seg_len = math.sqrt(seg_dx ** 2 + seg_dy ** 2)
        if seg_len > 1e-6:
            seg_ndx = seg_dx / seg_len
            seg_ndy = seg_dy / seg_len
        else:
            seg_ndx = 0.0
            seg_ndy = 0.0

        # Compute the orientation's own direction vector
        if source_point is segment.endpoint_1:
            ori_ndx, ori_ndy = seg_ndx, seg_ndy
        else:
            ori_ndx, ori_ndy = -seg_ndx, -seg_ndy

        # 1. Arrowhead evidence
        evidence.arrowhead_at_target = self._score_arrowhead_at_target(
            segment, target_point, tgt_bb
        )

        # Use independent evidence layer for boundary contacts (handles projection)
        all_bbs_for_proj = [c.bounding_box for c in all_candidates]
        indep_src = self._evidence_layer._classify_contact(source_point, src_bb)
        indep_tgt = self._evidence_layer._classify_contact(target_point, tgt_bb)

        # Project inside endpoints to boundary
        if indep_src.side.value == "inside":
            projected = self._evidence_layer._project_to_boundary(
                source_point, (ori_ndx, ori_ndy), src_bb, forward=True
            )
            if projected:
                indep_src = projected
        if indep_tgt.side.value == "inside":
            projected = self._evidence_layer._project_to_boundary(
                target_point, (ori_ndx, ori_ndy), tgt_bb, forward=False
            )
            if projected:
                indep_tgt = projected

        # 2. Source boundary contact (use projected if available)
        evidence.source_boundary_contact = self._score_boundary_contact(
            source_point, src_bb, -ori_ndx, -ori_ndy, is_source=True
        )
        if indep_src.confidence > 0.5 and indep_src.side.value not in ("far", "inside"):
            evidence.source_boundary_contact = max(
                evidence.source_boundary_contact,
                indep_src.confidence * 0.8
            )

        # 3. Target boundary contact (use projected if available)
        evidence.target_boundary_contact = self._score_boundary_contact(
            target_point, tgt_bb, ori_ndx, ori_ndy, is_source=False
        )
        if indep_tgt.confidence > 0.5 and indep_tgt.side.value not in ("far", "inside"):
            evidence.target_boundary_contact = max(
                evidence.target_boundary_contact,
                indep_tgt.confidence * 0.8
            )

        # 4. Source outward normal
        evidence.source_outward_normal = self._score_outward_normal(
            source_point, src_bb, ori_ndx, ori_ndy
        )
        if indep_src.confidence > 0.5 and indep_src.normal != (0.0, 0.0):
            # Use projected normal for scoring
            dot = indep_src.normal[0] * ori_ndx + indep_src.normal[1] * ori_ndy
            if dot > 0:
                evidence.source_outward_normal = max(
                    evidence.source_outward_normal,
                    min(1.0, dot * 1.5) * indep_src.confidence
                )

        # 5. Target inward normal
        evidence.target_inward_normal = self._score_inward_normal(
            target_point, tgt_bb, ori_ndx, ori_ndy
        )
        if indep_tgt.confidence > 0.5 and indep_tgt.normal != (0.0, 0.0):
            # Target inward = negated outward
            tgt_inward = (-indep_tgt.normal[0], -indep_tgt.normal[1])
            dot = tgt_inward[0] * ori_ndx + tgt_inward[1] * ori_ndy
            if dot > 0:
                evidence.target_inward_normal = max(
                    evidence.target_inward_normal,
                    min(1.0, dot * 1.5) * indep_tgt.confidence
                )

        # 6. Direction alignment (center-to-center vs segment)
        evidence.direction_alignment = self._score_direction_alignment(
            center_ndx, center_ndy, ori_ndx, ori_ndy
        )

        # 7. Angular consistency
        evidence.angular_consistency = self._score_angular_consistency(
            source_point, src_bb, tgt_bb, ori_ndx, ori_ndy
        )

        # 8. Spatial flow (weak contextual)
        evidence.spatial_flow = self._score_spatial_flow(
            src_bb, tgt_bb, ori_ndx, ori_ndy
        )

        # 9. Continuity
        evidence.continuity = self._score_continuity(
            segment, source_point, target_point, src_bb, tgt_bb, ori_ndx, ori_ndy
        )

        # 10. Node distance
        evidence.node_distance = self._score_node_distance(
            src_center, tgt_center, segment.arrow_length
        )

        # 11. Crossing penalty
        evidence.crossing_penalty = self._score_crossing_penalty(
            source_point, ori_ndx, ori_ndy, tgt_bb, all_candidates
        )

        # 12. Independent geometry evidence layer
        all_bbs = [c.bounding_box for c in all_candidates]
        indep = self._evidence_layer.score_orientation(
            source_bb=src_bb,
            target_bb=tgt_bb,
            segment_endpoint_1=segment.endpoint_1,
            segment_endpoint_2=segment.endpoint_2,
            geometric_axis=segment.geometric_axis,
            arrowhead_candidates=segment.arrowhead_candidates,
            arrowhead_confidence=segment.arrowhead_confidence,
            all_candidates=all_bbs,
        )

        # Weighted combination
        # When no arrowhead is detected, reduce weight of direction-dependent
        # factors (direction_alignment, continuity) because arrow start/end
        # assignment is unreliable without arrowhead confirmation.
        has_arrowhead = len(segment.arrowhead_candidates) > 0
        if has_arrowhead:
            da_weight = 0.10
            cont_weight = 0.05
        else:
            da_weight = 0.02
            cont_weight = 0.01

        # Blend independent geometry evidence with existing factors
        # indep.total_score gives a strong geometry-only signal
        indep_weight = 0.35 if not has_arrowhead else 0.15
        legacy_weight = 1.0 - indep_weight

        score = (
            indep_weight * indep.total_score
            + legacy_weight * (
                0.18 * evidence.arrowhead_at_target
                + 0.15 * evidence.source_boundary_contact
                + 0.15 * evidence.target_boundary_contact
                + 0.12 * evidence.source_outward_normal
                + 0.12 * evidence.target_inward_normal
                + da_weight * evidence.direction_alignment
                + 0.07 * evidence.angular_consistency
                + self.spatial_flow_weight * evidence.spatial_flow
                + cont_weight * evidence.continuity
                + 0.03 * evidence.node_distance
                - 0.13 * evidence.crossing_penalty
            )
        )

        score = max(0.0, min(1.0, score))

        components = evidence.to_dict()
        components["total"] = score
        components["indep_geometry"] = indep.evidence

        return OrientationScore(
            source_id=src_id,
            target_id=tgt_id,
            source_point=source_point,
            target_point=target_point,
            score=score,
            evidence=evidence,
            components=components,
        )

    # =========================================================================
    # Evidence Factors
    # =========================================================================

    def _score_arrowhead_at_target(
        self,
        segment: ArrowSegment,
        target_point: Point,
        target_bb: BoundingBox,
    ) -> float:
        """Score whether arrowhead evidence is near the target."""
        if not segment.arrowhead_candidates:
            return 0.3  # No arrowhead detected — neutral

        best = 0.0
        for ah in segment.arrowhead_candidates:
            dist = self._distance_to_boundary(ah, target_bb)
            if dist < 10:
                best = max(best, 1.0)
            elif dist < 30:
                best = max(best, 0.7)
            elif dist < 60:
                best = max(best, 0.4)
            else:
                best = max(best, 0.1)

        return best * segment.arrowhead_confidence

    def _score_boundary_contact(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
        is_source: bool,
    ) -> float:
        """Score whether point is in contact with rectangle boundary."""
        dist = self._distance_to_boundary(point, bb)

        if dist < 2:
            # Very close to boundary — check if pointing right way
            if is_source:
                # Source should be outside or on boundary
                return 0.95 if self._point_outside_rect(point, bb) else 0.8
            else:
                # Target should be outside or on boundary
                return 0.95 if self._point_outside_rect(point, bb) else 0.8

        if dist < self.min_boundary_distance * 0.3:
            return 0.7
        if dist < self.min_boundary_distance * 0.6:
            return 0.4
        if dist < self.min_boundary_distance:
            return 0.2
        return 0.0

    def _score_outward_normal(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
    ) -> float:
        """Score whether direction points outward from rectangle."""
        center = bb.center
        to_point_x = point.x - center.x
        to_point_y = point.y - center.y
        to_point_len = math.sqrt(to_point_x ** 2 + to_point_y ** 2)
        if to_point_len < 1e-6:
            return 0.0

        tpx = to_point_x / to_point_len
        tpy = to_point_y / to_point_len

        # For source: direction should point away from center
        dot = ndx * tpx + ndy * tpy
        if dot > 0:
            return min(1.0, dot * 1.5)
        return 0.0

    def _score_inward_normal(
        self,
        point: Point,
        bb: BoundingBox,
        ndx: float,
        ndy: float,
    ) -> float:
        """Score whether direction points toward rectangle center."""
        center = bb.center
        to_center_x = center.x - point.x
        to_center_y = center.y - point.y
        to_center_len = math.sqrt(to_center_x ** 2 + to_center_y ** 2)
        if to_center_len < 1e-6:
            return 0.0

        tcx = to_center_x / to_center_len
        tcy = to_center_y / to_center_len

        dot = ndx * tcx + ndy * tcy
        if dot > 0:
            return min(1.0, dot * 1.5)
        return 0.0

    def _score_direction_alignment(
        self,
        center_ndx: float,
        center_ndy: float,
        ori_ndx: float,
        ori_ndy: float,
    ) -> float:
        """Score alignment between center-to-center and segment direction."""
        dot = center_ndx * ori_ndx + center_ndy * ori_ndy
        return max(0.0, dot)

    def _score_angular_consistency(
        self,
        source_point: Point,
        src_bb: BoundingBox,
        tgt_bb: BoundingBox,
        ori_ndx: float,
        ori_ndy: float,
    ) -> float:
        """Score angular consistency of the orientation."""
        src_center = src_bb.center
        tgt_center = tgt_bb.center

        angle_to_target = math.atan2(
            tgt_center.y - source_point.y,
            tgt_center.x - source_point.x,
        )
        arrow_angle = math.atan2(ori_ndy, ori_ndx)

        angle_diff = abs(angle_to_target - arrow_angle)
        angle_diff = angle_diff % (2 * math.pi)
        if angle_diff > math.pi:
            angle_diff = 2 * math.pi - angle_diff

        tolerance = math.radians(self.max_direction_tolerance_deg)
        if angle_diff > tolerance:
            return 0.0
        return 1.0 - (angle_diff / tolerance) * 0.5

    def _score_spatial_flow(
        self,
        src_bb: BoundingBox,
        tgt_bb: BoundingBox,
        ori_ndx: float,
        ori_ndy: float,
    ) -> float:
        """Weak spatial flow score (left→right, top→bottom bias)."""
        src_cx = src_bb.x + src_bb.width / 2
        src_cy = src_bb.y + src_bb.height / 2
        tgt_cx = tgt_bb.x + tgt_bb.width / 2
        tgt_cy = tgt_bb.y + tgt_bb.height / 2

        # Compute spatial flow direction
        dx = tgt_cx - src_cx
        dy = tgt_cy - src_cy
        dist = math.sqrt(dx ** 2 + dy ** 2)
        if dist < 1e-6:
            return 0.0

        # Normalize
        ndx = dx / dist
        ndy = dy / dist

        # Score: how well does orientation match spatial flow
        dot = ndx * ori_ndx + ndy * ori_ndy

        # Soft score — spatial flow is WEAK evidence
        if dot > 0.5:
            return 0.6 + 0.4 * dot
        if dot > 0:
            return 0.3 * dot
        return 0.0

    def _score_continuity(
        self,
        segment: ArrowSegment,
        source_point: Point,
        target_point: Point,
        src_bb: BoundingBox,
        tgt_bb: BoundingBox,
        ori_ndx: float,
        ori_ndy: float,
    ) -> float:
        """Score geometric continuity along the path."""
        # Check if segment is roughly collinear with source→target
        src_center = src_bb.center
        tgt_center = tgt_bb.center

        path_dx = tgt_center.x - src_center.x
        path_dy = tgt_center.y - src_center.y
        path_len = math.sqrt(path_dx ** 2 + path_dy ** 2)
        if path_len < 1e-6:
            return 0.0

        # Segment vs path direction
        seg_dx = segment.endpoint_2.x - segment.endpoint_1.x
        seg_dy = segment.endpoint_2.y - segment.endpoint_1.y
        seg_len = math.sqrt(seg_dx ** 2 + seg_dy ** 2)
        if seg_len < 1e-6:
            return 0.0

        dot = (seg_dx / seg_len) * (path_dx / path_len) + \
              (seg_dy / seg_len) * (path_dy / path_len)

        return max(0.0, dot)

    def _score_node_distance(
        self,
        src_center: Point,
        tgt_center: Point,
        arrow_length: float,
    ) -> float:
        """Score based on node distance vs arrow length."""
        center_dist = math.sqrt(
            (tgt_center.x - src_center.x) ** 2 +
            (tgt_center.y - src_center.y) ** 2
        )
        if center_dist < 1e-6:
            return 0.0

        ratio = arrow_length / center_dist if center_dist > 0 else 0
        # Good: arrow length ~ matches center distance
        if 0.5 < ratio < 2.0:
            return 0.8
        if 0.3 < ratio < 3.0:
            return 0.5
        return 0.2

    def _score_crossing_penalty(
        self,
        source_point: Point,
        ndx: float,
        ndy: float,
        target_bb: BoundingBox,
        all_candidates: List[CandidateNode],
    ) -> float:
        """Penalize arrows crossing unrelated node interiors."""
        penalty = 0.0
        for cand in all_candidates:
            if cand.bounding_box == target_bb:
                continue
            bb = cand.bounding_box
            intersections = self._ray_rectangle_intersections(
                source_point, ndx, ndy, bb
            )
            for x, y, edge, t in intersections:
                if t > 0:
                    penalty += 0.3
                    break
        return min(1.0, penalty)

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

    def _point_inside_rect(self, point: Point, bb: BoundingBox) -> bool:
        """Check if point is inside rectangle."""
        return (bb.x <= point.x <= bb.x + bb.width and
                bb.y <= point.y <= bb.y + bb.height)

    def _point_outside_rect(self, point: Point, bb: BoundingBox) -> bool:
        """Check if point is outside rectangle."""
        return not self._point_inside_rect(point, bb)

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

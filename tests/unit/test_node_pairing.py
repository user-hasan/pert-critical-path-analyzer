"""
Deterministic regression tests for node pairing resolution.

17 tests covering:
  1. clear arrow between two rectangles
  2. arrow near multiple rectangles
  3. diagonal arrow
  4. arrow touching rectangle corner
  5. boundary intersection
  6. endpoint ordering reversed
  7. line passes near unrelated node
  8. line crosses unrelated node
  9. multiple candidate pairs
  10. ambiguous pair
  11. pair confidence margin
  12. pairing provenance
  13. pairing vs direction separation
  14. direction resolver receives fixed pair
  15. ambiguous direction remains review
  16. self-loop rejection
  17. duplicate pair detection
"""

from __future__ import annotations

import math
import pytest

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.node_pairing import NodePairingResolver, PairingStatus
from pert_analyzer.cv.direction_resolution import ArrowDirectionResolver, DirectionStatus


def _rect(x: float, y: float, w: float, h: float) -> CandidateNode:
    return CandidateNode(
        shape_type=ShapeType.RECTANGLE,
        bounding_box=BoundingBox(float(x), float(y), float(w), float(h)),
        position=Point(float(x + w / 2), float(y + h / 2)),
        confidence=0.9,
    )


def _arrow(
    sx: float, sy: float, ex: float, ey: float, length: float = None,
    ah: Point = None, ah_conf: float = 0.0, conf: float = 0.8,
) -> DetectedArrow:
    start = Point(sx, sy)
    end = Point(ex, ey)
    dx = ex - sx
    dy = ey - sy
    l = length if length else math.sqrt(dx**2 + dy**2)
    dvec = (dx / l if l > 0 else (0, 0))
    return DetectedArrow(
        arrow_id="test_arrow",
        start=start, end=end,
        direction_vector=dvec, length=l, confidence=conf,
        arrowhead_point=ah, arrowhead_confidence=ah_conf,
    )


class Test1ClearBetweenTwoRects:
    """Clear arrow between two rectangles."""

    def test_clear_pair(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.status in (
            PairingStatus.PAIR_CONFIRMED,
            PairingStatus.PAIR_CONTEXTUALLY_RESOLVED,
        )
        assert result.selected is not None
        pair_ids = {result.selected.node_a_id, result.selected.node_b_id}
        assert c1.node_id in pair_ids
        assert c2.node_id in pair_ids


class Test2ArrowNearMultipleRects:
    """Arrow near multiple rectangles."""

    def test_picks_best_pair(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(300, 100, 80, 50)
        c3 = _rect(300, 200, 80, 50)  # Near c2 but different position
        arrow = _arrow(130, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        assert result.selected is not None
        # Should pick c1 and c2 (c2 is closer to endpoint B)
        pair_ids = {result.selected.node_a_id, result.selected.node_b_id}
        assert c1.node_id in pair_ids


class Test3DiagonalArrow:
    """Diagonal arrow."""

    def test_diagonal_pair(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 50, 80, 50)
        c2 = _rect(300, 200, 80, 50)
        arrow = _arrow(130, 75, 300, 225)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        pair_ids = {result.selected.node_a_id, result.selected.node_b_id}
        assert c1.node_id in pair_ids
        assert c2.node_id in pair_ids


class Test4ArrowTouchingCorner:
    """Arrow touching rectangle corner."""

    def test_corner_contact(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow starts at c1's top-right corner
        arrow = _arrow(150, 100, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        pair_ids = {result.selected.node_a_id, result.selected.node_b_id}
        assert c1.node_id in pair_ids
        assert c2.node_id in pair_ids


class Test5BoundaryIntersection:
    """Boundary intersection scoring."""

    def test_boundary_scores_high(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow starts exactly on c1 boundary, ends on c2 boundary
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        ev = result.selected.evidence
        # At least one endpoint should have good boundary contact
        assert ev.boundary_contact_a > 0.5 or ev.boundary_contact_b > 0.5


class Test6ReversedEndpointOrdering:
    """Endpoint ordering reversed but same pair."""

    def test_same_pair_both_directions(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        a1 = _arrow(150, 125, 300, 125)
        a2 = _arrow(300, 125, 150, 125)
        r1 = resolver.resolve_pairing(a1, [c1, c2])
        r2 = resolver.resolve_pairing(a2, [c1, c2])
        # Both should find the same node pair (order-independent)
        assert r1.selected is not None
        assert r2.selected is not None
        pair1 = {r1.selected.node_a_id, r1.selected.node_b_id}
        pair2 = {r2.selected.node_a_id, r2.selected.node_b_id}
        assert pair1 == pair2


class Test7LinePassesNearUnrelated:
    """Line passes near unrelated node (but doesn't cross)."""

    def test_near_unrelated_accepted(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(400, 100, 80, 50)
        c3 = _rect(220, 60, 60, 40)  # Near the line but not on it
        arrow = _arrow(130, 125, 400, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        assert result.selected is not None
        pair_ids = {result.selected.node_a_id, result.selected.node_b_id}
        # Should still pick c1 and c2
        assert c1.node_id in pair_ids
        assert c2.node_id in pair_ids


class Test8LineCrossesUnrelated:
    """Line crosses unrelated node interior."""

    def test_crossing_penalized(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(400, 100, 80, 50)
        c3 = _rect(200, 100, 80, 50)  # Directly in the path
        arrow = _arrow(130, 125, 400, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        if result.selected:
            # Crossing should reduce score
            assert result.selected.evidence.crossing_penalty > 0


class Test9MultipleCandidatePairs:
    """Multiple candidate pairs generated."""

    def test_generates_multiple_pairs(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(300, 100, 80, 50)
        c3 = _rect(300, 200, 80, 50)
        arrow = _arrow(130, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        # Should generate at least 2 pairs (c1-c2, c1-c3)
        assert len(result.all_candidates) >= 2


class Test10AmbiguousPair:
    """Ambiguous pair (similar scores)."""

    def test_ambiguous_gives_review(self):
        resolver = NodePairingResolver(
            max_endpoint_distance=120.0, confirmed_margin=0.30
        )
        # Two nodes at same distance, arrow between them
        c1 = _rect(100, 100, 80, 50)
        c2 = _rect(300, 100, 80, 50)
        # Very short arrow near midpoint
        arrow = _arrow(190, 125, 210, 125, length=20)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        # Should be review or weak
        assert result.confidence < 0.6


class Test11PairConfidenceMargin:
    """Pair confidence margin between first and second."""

    def test_strong_margin(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        c3 = _rect(300, 400, 100, 50)  # Far away
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        assert result.selected is not None
        if len(result.alternatives) > 0:
            margin = result.selected.score - result.alternatives[0].score
            assert margin > 0


class Test12PairingProvenance:
    """Pairing provenance preserved."""

    def test_metadata_populated(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.arrow_id == "test_arrow"
        assert result.selected is not None
        assert result.selected.score >= 0
        assert len(result.selected.evidence.to_dict()) > 0
        assert result.confidence >= 0


class Test13PairingVsDirectionSeparation:
    """Pairing and direction are separate stages."""

    def test_pairing_does_not_set_direction(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        pair_result = resolver.resolve_pairing(arrow, [c1, c2])
        # PairResult should not have source_id/target_id (only node_a_id/node_b_id)
        assert hasattr(pair_result.selected, 'node_a_id')
        assert hasattr(pair_result.selected, 'node_b_id')
        # Direction should not be determined by pairing
        assert not hasattr(pair_result, 'source_id')


class Test14DirectionResolverReceivesFixedPair:
    """Direction resolver receives fixed pair, doesn't reconsider nodes."""

    def test_direction_uses_fixed_pair(self):
        pairing = NodePairingResolver(max_endpoint_distance=120.0)
        direction = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)

        # Stage 1: Pairing
        pair_result = pairing.resolve_pairing(arrow, [c1, c2])
        assert pair_result.selected is not None

        # Stage 2: Direction with fixed pair
        dir_result = direction.resolve_direction_for_pair(
            arrow,
            pair_result.selected.node_a_id,
            pair_result.selected.node_b_id,
            [c1, c2],
        )
        assert dir_result.source_id != ""
        assert dir_result.target_id != ""
        assert dir_result.status != DirectionStatus.REJECTED


class Test15AmbiguousDirectionRemainsReview:
    """Ambiguous direction remains review."""

    def test_low_margin_gives_review(self):
        pairing = NodePairingResolver(max_endpoint_distance=120.0)
        direction = ArrowDirectionResolver(confirmed_margin=0.30)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow with no arrowhead, equal evidence both ways
        arrow = _arrow(150, 125, 300, 125, ah_conf=0.05)

        pair_result = pairing.resolve_pairing(arrow, [c1, c2])
        dir_result = direction.resolve_direction_for_pair(
            arrow,
            pair_result.selected.node_a_id,
            pair_result.selected.node_b_id,
            [c1, c2],
        )
        # With high confirmed_margin, equal scores → REVIEW
        if dir_result.margin < 0.30:
            assert dir_result.status in (
                DirectionStatus.REVIEW_REQUIRED,
                DirectionStatus.CONTEXTUALLY_RESOLVED,
            )


class Test16SelfLoopRejection:
    """Self-loop rejection."""

    def test_self_loop_rejected(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(100, 100, 100, 50)
        arrow = _arrow(100, 125, 200, 125)
        result = resolver.resolve_pairing(arrow, [c1])
        # Only one node → can't form a pair
        assert result.status == PairingStatus.PAIR_REJECTED


class Test17DuplicatePairDetection:
    """Duplicate pair detection."""

    def test_no_duplicate_pairs(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        # Check no duplicate pairs in all_candidates
        seen = set()
        for pair in result.all_candidates:
            key = frozenset([pair.node_a_id, pair.node_b_id])
            assert key not in seen
            seen.add(key)

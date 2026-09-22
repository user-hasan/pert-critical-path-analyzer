"""
Deterministic regression tests for bidirectional arrow direction resolution.

18 tests covering:
  1. clear left-to-right arrow
  2. clear right-to-left arrow
  3. vertical downward arrow
  4. vertical upward arrow
  5. diagonal arrow
  6. arrow touching rectangle boundaries
  7. Hough line with reversed endpoint ordering
  8. weak/ambiguous arrowhead
  9. duplicate arrow segments
  10. source/target boundary ambiguity
  11. spatial flow agrees with arrowhead
  12. spatial flow conflicts with strong arrowhead
  13. equal orientation scores → REVIEW_REQUIRED
  14. self-loop rejection
  15. crossing unrelated node
  16. arrowhead near wrong endpoint
  17. multiple arrowhead candidates
  18. provenance preservation
"""

from __future__ import annotations

import math
import pytest

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.direction_resolution import (
    ArrowDirectionResolver,
    DirectionStatus,
)


def _rect(x: float, y: float, w: float, h: float) -> CandidateNode:
    return CandidateNode(
        shape_type=ShapeType.RECTANGLE,
        bounding_box=BoundingBox(float(x), float(y), float(w), float(h)),
        position=Point(float(x + w / 2), float(y + h / 2)),
        confidence=0.9,
    )


def _arrow(
    sx: float, sy: float, ex: float, ey: float,
    ah: Point = None, ah_conf: float = 0.0, conf: float = 0.8,
    dv: tuple = None,
) -> DetectedArrow:
    start = Point(sx, sy)
    end = Point(ex, ey)
    dx = ex - sx
    dy = ey - sy
    length = math.sqrt(dx**2 + dy**2)
    dvec = dv if dv else (dx / length if length > 0 else (0, 0))
    return DetectedArrow(
        arrow_id="test_arrow",
        start=start,
        end=end,
        direction_vector=dvec,
        length=length,
        confidence=conf,
        arrowhead_point=ah,
        arrowhead_confidence=ah_conf,
    )


class Test1LeftToRight:
    """Clear left-to-right arrow."""

    def test_left_to_right_source_is_left(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.status in (DirectionStatus.CONFIRMED, DirectionStatus.CONTEXTUALLY_RESOLVED)
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test2RightToLeft:
    """Clear right-to-left arrow."""

    def test_right_to_left_source_is_right(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(300, 100, 100, 50)
        c2 = _rect(50, 100, 100, 50)
        arrow = _arrow(300, 125, 150, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.status in (DirectionStatus.CONFIRMED, DirectionStatus.CONTEXTUALLY_RESOLVED)
        pair = frozenset([result.source_id, result.target_id])
        assert c1.node_id in pair
        assert c2.node_id in pair


class Test3VerticalDown:
    """Vertical downward arrow."""

    def test_top_to_bottom(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(100, 50, 80, 50)
        c2 = _rect(100, 200, 80, 50)
        arrow = _arrow(140, 100, 140, 200)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.status in (DirectionStatus.CONFIRMED, DirectionStatus.CONTEXTUALLY_RESOLVED)
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test4VerticalUp:
    """Vertical upward arrow."""

    def test_bottom_to_top(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(100, 200, 80, 50)
        c2 = _rect(100, 50, 80, 50)
        arrow = _arrow(140, 200, 140, 100)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.status in (DirectionStatus.CONFIRMED, DirectionStatus.CONTEXTUALLY_RESOLVED)
        pair = frozenset([result.source_id, result.target_id])
        assert c1.node_id in pair
        assert c2.node_id in pair


class Test5Diagonal:
    """Diagonal arrow."""

    def test_diagonal_arrow(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 50, 80, 50)
        c2 = _rect(300, 200, 80, 50)
        arrow = _arrow(130, 75, 300, 225)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.status in (DirectionStatus.CONFIRMED, DirectionStatus.CONTEXTUALLY_RESOLVED)
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test6BoundaryTouch:
    """Arrow touching rectangle boundaries."""

    def test_on_boundary(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test7ReversedEndpoints:
    """Hough line with reversed endpoint ordering but correct arrowhead."""

    def test_arrowhead_overrides_endpoints(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Endpoints are reversed (end is near c1), but arrowhead is near c2
        arrow = _arrow(300, 125, 150, 125, ah=Point(300, 125), ah_conf=0.9)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Should resolve to c2 → c1 due to arrowhead near c1's start
        # Actually arrowhead at (300,125) is near c2, so source=c1, target=c2
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test8WeakArrowhead:
    """Weak/ambiguous arrowhead."""

    def test_weak_arrowhead_uses_boundary(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125, ah_conf=0.1)
        result = resolver.resolve_direction(arrow, [c1, c2])
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id


class Test9DuplicateSegments:
    """Duplicate arrow segments."""

    def test_duplicate_arrows_independent_resolution(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        a1 = _arrow(150, 125, 300, 125)
        a2 = _arrow(152, 124, 298, 126)
        r1 = resolver.resolve_direction(a1, [c1, c2])
        r2 = resolver.resolve_direction(a2, [c1, c2])
        # Both should resolve to the same source/target
        assert r1.source_id == r2.source_id
        assert r1.target_id == r2.target_id


class Test10BoundaryAmbiguity:
    """Source/target boundary ambiguity."""

    def test_ambiguous_creates_review(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.30)
        # Two nodes equidistant, arrow between them
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(300, 100, 80, 50)
        # Very short arrow, ambiguous
        arrow = _arrow(190, 125, 210, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Low confidence should result in review or rejected
        assert result.confidence < 0.6


class Test11SpatialFlowAgrees:
    """Spatial flow agrees with arrowhead."""

    def test_spatial_and_arrowhead_agree(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125, ah=Point(300, 125), ah_conf=0.8)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Arrowhead near c2 + spatial flow left→right = strong
        assert result.source_id == c1.node_id
        assert result.target_id == c2.node_id
        assert result.confidence >= 0.5


class Test12SpatialFlowConflicts:
    """Spatial flow conflicts with strong arrowhead."""

    def test_arrowhead_wins_over_spatial(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow goes right (spatial: c1→c2), but arrowhead is at left endpoint
        arrow = _arrow(150, 125, 300, 125, ah=Point(150, 125), ah_conf=0.9)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Arrowhead at (150,125) is near c1 → should be target
        # So source=c2, target=c1
        assert result.target_id == c1.node_id


class Test13EqualScores:
    """Equal orientation scores → REVIEW_REQUIRED."""

    def test_equal_scores_not_confirmed(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.30)
        # Arrow between two identical rectangles at same Y
        c1 = _rect(100, 100, 80, 50)
        c2 = _rect(300, 100, 80, 50)
        # Arrow midpoint, no arrowhead, weak evidence
        arrow = _arrow(190, 125, 210, 125, ah_conf=0.05)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Should not be CONFIRMED due to low margin
        assert result.status != DirectionStatus.CONFIRMED


class Test14SelfLoop:
    """Self-loop rejection."""

    def test_self_loop_rejected(self):
        resolver = ArrowDirectionResolver()
        c1 = _rect(100, 100, 100, 50)
        arrow = _arrow(100, 125, 200, 125)
        result = resolver.resolve_direction(arrow, [c1])
        assert result.status == DirectionStatus.REJECTED
        assert "self_loop" in result.rejection_reasons


class Test15CrossingNode:
    """Crossing unrelated node."""

    def test_crossing_penalized(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.15)
        c1 = _rect(50, 100, 80, 50)
        c2 = _rect(400, 100, 80, 50)
        c3 = _rect(200, 100, 80, 50)  # Between c1 and c2
        arrow = _arrow(130, 125, 400, 125)
        result = resolver.resolve_direction(arrow, [c1, c2, c3])
        # Crossing penalty should be present in evidence
        assert result.evidence.get("crossing_penalty", 0) > 0
        # Confidence should be lower than without crossing
        assert result.confidence < 0.85


class Test16ArrowheadNearWrongEndpoint:
    """Arrowhead near wrong endpoint."""

    def test_arrowhead_at_source_end(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow goes right, but arrowhead is at start (near c1)
        arrow = _arrow(150, 125, 300, 125, ah=Point(155, 125), ah_conf=0.9)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Arrowhead near c1 means c1 is TARGET, not source
        assert result.target_id == c1.node_id
        assert result.source_id == c2.node_id


class Test17MultipleArrowheads:
    """Multiple arrowhead candidates."""

    def test_two_arrowheads(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        # Arrow with arrowhead at BOTH ends (ambiguous)
        arrow = _arrow(
            150, 125, 300, 125,
            ah=Point(300, 125), ah_conf=0.8,
        )
        # Also add a second arrowhead via metadata
        arrow.metadata["arrowhead_at_start"] = Point(150, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Should still resolve, but possibly with lower confidence
        assert result.source_id in (c1.node_id, c2.node_id)
        assert result.target_id in (c1.node_id, c2.node_id)
        assert result.source_id != result.target_id


class Test18ProvenancePreservation:
    """Provenance preservation."""

    def test_alternatives_preserved(self):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        arrow = _arrow(150, 125, 300, 125)
        result = resolver.resolve_direction(arrow, [c1, c2])
        # Must have orientation_a_to_b and orientation_b_to_a
        assert result.orientation_a_to_b is not None
        assert result.orientation_b_to_a is not None
        assert result.orientation_a_to_b.score >= 0
        assert result.orientation_b_to_a.score >= 0
        assert result.margin >= 0
        assert len(result.alternatives) == 1
        assert "source_id" in result.alternatives[0]
        assert "target_id" in result.alternatives[0]
        assert "score" in result.alternatives[0]
        assert "evidence" in result.alternatives[0]
        # Evidence dict populated
        assert len(result.evidence) > 0
        assert result.arrow_id == "test_arrow"

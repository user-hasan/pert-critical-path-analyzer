"""
Regression tests for the reference diagram review behaviors.

Locks in the "prefer REVIEW over a wrong automatic dependency" rules that
fixed the four wrong geometric pairings and the direction reversals on
reference_aon.png:

  * floating trunk/bus with no arrowhead and no node anchor  -> REVIEW
  * label-glyph fragment fully inside one node box whose arrowhead
    points into void or at a third node                       -> REVIEW
  * long no-arrowhead routing segment with ambiguous margin   -> DIR REVIEW
  * a same-box stub whose arrowhead points at its partner     -> NO penalty
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.node_pairing import (
    NodePairCandidate,
    NodePairingResolver,
    PairingStatus,
)
from pert_analyzer.cv.direction_resolution import (
    ArrowDirectionResolver,
    DirectionStatus,
)


def _candidate(node_id: str, x: float, y: float, w: float, h: float) -> CandidateNode:
    return CandidateNode(
        node_id=node_id,
        shape_type=ShapeType.RECTANGLE,
        bounding_box=BoundingBox(float(x), float(y), float(w), float(h)),
        position=Point(float(x + w / 2), float(y + h / 2)),
        confidence=0.9,
    )


def _arrow(
    arrow_id: str,
    sx: float, sy: float,
    ex: float, ey: float,
    ah_x: Optional[float] = None,
    ah_y: Optional[float] = None,
    ah_conf: float = 0.0,
) -> DetectedArrow:
    start = Point(sx, sy)
    end = Point(ex, ey)
    dx = ex - sx
    dy = ey - sy
    dl = (dx**2 + dy**2) ** 0.5
    ah_point = Point(ah_x, ah_y) if ah_x is not None else None
    return DetectedArrow(
        arrow_id=arrow_id,
        start=start,
        end=end,
        length=dl,
        confidence=0.8,
        arrowhead_point=ah_point,
        arrowhead_confidence=ah_conf,
        direction_vector=(dx / dl if dl > 0 else 0, dy / dl if dl > 0 else 0),
    )


def _pair_status(
    arrow: DetectedArrow, candidates: List[CandidateNode]
) -> Tuple[PairingStatus, Optional[NodePairCandidate]]:
    resolver = NodePairingResolver(max_endpoint_distance=120.0)
    result = resolver.resolve_pairing(arrow, candidates)
    return result.status, result.selected


class TestUnanchoredTrunkReview:
    """E<->H ghost buses: long arrow with no arrowhead, no node anchor."""

    def test_floating_trunk_between_far_nodes_reviewed(self):
        c_top = _candidate("top", 100, 100, 100, 60)
        c_bottom = _candidate("bottom", 100, 400, 100, 60)
        arrow = _arrow("trunk_a", 150, 180, 150, 360)
        status, selected = _pair_status(arrow, [c_top, c_bottom])
        assert status == PairingStatus.PAIR_REVIEW_REQUIRED
        if selected:
            assert selected.evidence.unanchored_fragment >= 0.5

    def test_short_weak_fragment_kept_rejected(self):
        """Near-zero score must stay REJECTED, not escalated to REVIEW."""
        c1 = _candidate("c1", 50, 100, 100, 50)
        c2 = _candidate("c2", 300, 100, 100, 50)
        arrow = _arrow("noise_b", 200, 125, 205, 125)
        status, _ = _pair_status(arrow, [c1, c2])
        assert status == PairingStatus.PAIR_REJECTED


class TestSameBoxGlyphReview:
    """Label-glyph fragments: arrow inside one box pointing away from partner."""

    def _scenario(self):
        src = _candidate("src", 400, 220, 150, 150)
        dst = _candidate("dst", 431, 420, 118, 78)
        return [src, dst]

    def test_stub_pointing_into_void_reviewed(self):
        candidates = self._scenario()
        # Both endpoints inside src; tip at top pointing UP into void.
        arrow = _arrow(
            "glyph_a", 450, 360, 458, 368,
            ah_x=450, ah_y=360, ah_conf=0.4,
        )
        status, selected = _pair_status(arrow, candidates)
        assert status == PairingStatus.PAIR_REVIEW_REQUIRED
        if selected:
            assert selected.evidence.arrow_opposition_penalty >= 0.5

    def test_stub_pointing_at_partner_not_penalized(self):
        """Recovered arrows (like V->U stubs) must not be demoted."""
        candidates = self._scenario()
        # Both endpoints inside src; tip at bottom pointing DOWN at dst.
        arrow = _arrow(
            "stub_b", 440, 240, 465, 360,
            ah_x=465, ah_y=360, ah_conf=0.5,
        )
        status, selected = _pair_status(arrow, candidates)
        assert status in (
            PairingStatus.PAIR_CONFIRMED,
            PairingStatus.PAIR_CONTEXTUALLY_RESOLVED,
        )
        if selected:
            assert selected.evidence.arrow_opposition_penalty < 0.5


class TestAmbiguousLongNoArrowheadDirection:
    """M<->N long router: no arrowhead + tight margin -> DIR REVIEW."""

    def test_long_no_arrowhead_low_margin_reviewed(self):
        m_low = _candidate("m", 1092, 301, 116, 77)
        n_up = _candidate("n", 1228, 222, 116, 77)
        arrow = _arrow("router_a", 1150, 380, 1240, 210)
        resolver = ArrowDirectionResolver(confirmed_margin=0.20)
        result = resolver.resolve_direction_for_pair(arrow, "m", "n", [m_low, n_up])
        assert result.status == DirectionStatus.REVIEW_REQUIRED

    def test_short_no_arrowhead_clear_margin_not_reviewed(self):
        """The gate must not fire for clear short connectors."""
        c1 = _candidate("c1", 0, 0, 100, 60)
        c2 = _candidate("c2", 200, 0, 100, 60)
        arrow = _arrow("clear_a", 100, 30, 200, 30)
        resolver = ArrowDirectionResolver(confirmed_margin=0.20)
        result = resolver.resolve_direction_for_pair(arrow, "c1", "c2", [c1, c2])
        assert result.status in (
            DirectionStatus.CONFIRMED,
            DirectionStatus.CONTEXTUALLY_RESOLVED,
        )
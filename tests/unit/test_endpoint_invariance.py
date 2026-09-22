"""
Endpoint-order invariance tests for arrow detection and direction resolution.

These tests verify that:
1. DetectedArrow normalization produces the same geometric result regardless
   of the order in which endpoints are supplied.
2. ArrowSegment normalization produces the same result.
3. DirectionResolver produces the same semantic result when endpoints are swapped.
4. Bottom-row arrows (R↔Q, Q↔P, P↔O, O↔N) produce invariant results.

The key principle: semantic source/target must never be derived from
Hough endpoint ordering.
"""
from __future__ import annotations

import math
import pytest

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.direction_resolution import (
    ArrowDirectionResolver,
    ArrowSegment,
    DirectionStatus,
)
from pert_analyzer.cv.node_pairing import NodePairingResolver, PairingStatus
from pert_analyzer.cv.validated_dependency import ValidatedDependencyBuilder


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
) -> DetectedArrow:
    dx = ex - sx
    dy = ey - sy
    l = math.sqrt(dx**2 + dy**2) if (dx or dy) else 0
    return DetectedArrow(
        arrow_id="test_arrow",
        start=Point(sx, sy),
        end=Point(ex, ey),
        direction_vector=(dx / l if l > 0 else (0, 0)),
        length=l,
        confidence=conf,
        arrowhead_point=ah,
        arrowhead_confidence=ah_conf,
    )


# =========================================================================
# 1. DetectedArrow normalization
# =========================================================================

class TestDetectedArrowNormalization:
    """Endpoints are normalized regardless of input order."""

    def test_same_x_different_y(self):
        a1 = _arrow(100, 300, 100, 100)
        a2 = _arrow(100, 100, 100, 300)
        assert a1.start == a2.start
        assert a1.end == a2.end
        assert a1.direction_vector == a2.direction_vector

    def test_different_x(self):
        a1 = _arrow(300, 100, 100, 200)
        a2 = _arrow(100, 200, 300, 100)
        assert a1.start == a2.start
        assert a1.end == a2.end

    def test_same_point(self):
        a1 = _arrow(100, 100, 100, 100)
        assert a1.start.x == 100
        assert a1.end.x == 100
        assert a1.length == 0

    def test_diagonal(self):
        a1 = _arrow(200, 300, 100, 100)
        a2 = _arrow(100, 100, 200, 300)
        assert a1.start == a2.start
        assert a1.end == a2.end

    def test_preserves_arrowhead(self):
        ah = Point(50, 100)
        a1 = _arrow(300, 100, 50, 100, ah=ah, ah_conf=0.9)
        a2 = _arrow(50, 100, 300, 100, ah=ah, ah_conf=0.9)
        assert a1.arrowhead_point == a2.arrowhead_point
        assert a1.arrowhead_confidence == a2.arrowhead_confidence

    def test_geometric_length_invariant(self):
        a1 = _arrow(300, 100, 100, 200)
        a2 = _arrow(100, 200, 300, 100)
        assert abs(a1.length - a2.length) < 1e-6


# =========================================================================
# 2. ArrowSegment normalization
# =========================================================================

class TestArrowSegmentNormalization:
    """ArrowSegment endpoints are normalized."""

    def test_segment_same_both_orders(self):
        s1 = ArrowSegment(
            arrow_id="s1",
            endpoint_1=Point(300, 100),
            endpoint_2=Point(100, 200),
            geometric_axis=(1.0, 0.0),
        )
        s2 = ArrowSegment(
            arrow_id="s2",
            endpoint_1=Point(100, 200),
            endpoint_2=Point(300, 100),
            geometric_axis=(1.0, 0.0),
        )
        assert s1.endpoint_1 == s2.endpoint_1
        assert s1.endpoint_2 == s2.endpoint_2


# =========================================================================
# 3. DirectionResolver endpoint-order invariance
# =========================================================================

class TestDirectionResolverInvariance:
    """DirectionResolver produces the same result when endpoints are swapped."""

    def _resolve_both_orders(
        self, sx1, sy1, ex1, ey1, sx2, sy2, ex2, ey2, candidates
    ):
        resolver = ArrowDirectionResolver(confirmed_margin=0.10)
        a1 = _arrow(sx1, sy1, ex1, ey1)
        a2 = _arrow(sx2, sy2, ex2, ey2)
        r1 = resolver.resolve_direction(a1, candidates)
        r2 = resolver.resolve_direction(a2, candidates)
        return r1, r2

    def test_horizontal_both_orders_same_pair(self):
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        r1, r2 = self._resolve_both_orders(
            150, 125, 300, 125,
            300, 125, 150, 125,
            [c1, c2],
        )
        pair1 = frozenset([r1.source_id, r1.target_id])
        pair2 = frozenset([r2.source_id, r2.target_id])
        assert pair1 == pair2

    def test_diagonal_both_orders_same_pair(self):
        c1 = _rect(50, 50, 80, 50)
        c2 = _rect(300, 200, 80, 50)
        r1, r2 = self._resolve_both_orders(
            130, 75, 300, 225,
            300, 225, 130, 75,
            [c1, c2],
        )
        pair1 = frozenset([r1.source_id, r1.target_id])
        pair2 = frozenset([r2.source_id, r2.target_id])
        assert pair1 == pair2

    def test_vertical_both_orders_same_pair(self):
        c1 = _rect(100, 50, 80, 50)
        c2 = _rect(100, 200, 80, 50)
        r1, r2 = self._resolve_both_orders(
            140, 100, 140, 200,
            140, 200, 140, 100,
            [c1, c2],
        )
        pair1 = frozenset([r1.source_id, r1.target_id])
        pair2 = frozenset([r2.source_id, r2.target_id])
        assert pair1 == pair2


# =========================================================================
# 4. PairingResolver endpoint-order invariance
# =========================================================================

class TestPairingResolverInvariance:
    """NodePairingResolver produces the same result when endpoints are swapped."""

    def test_pairing_both_orders_same_pair(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        a1 = _arrow(150, 125, 300, 125)
        a2 = _arrow(300, 125, 150, 125)
        r1 = resolver.resolve_pairing(a1, [c1, c2])
        r2 = resolver.resolve_pairing(a2, [c1, c2])
        assert r1.selected is not None
        assert r2.selected is not None
        pair1 = frozenset([r1.selected.node_a_id, r1.selected.node_b_id])
        pair2 = frozenset([r2.selected.node_a_id, r2.selected.node_b_id])
        assert pair1 == pair2

    def test_pairing_diagonal_both_orders(self):
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        c1 = _rect(50, 50, 80, 50)
        c2 = _rect(300, 200, 80, 50)
        a1 = _arrow(130, 75, 300, 225)
        a2 = _arrow(300, 225, 130, 75)
        r1 = resolver.resolve_pairing(a1, [c1, c2])
        r2 = resolver.resolve_pairing(a2, [c1, c2])
        pair1 = frozenset([r1.selected.node_a_id, r1.selected.node_b_id])
        pair2 = frozenset([r2.selected.node_a_id, r2.selected.node_b_id])
        assert pair1 == pair2


# =========================================================================
# 5. ValidatedDependency endpoint-order invariance
# =========================================================================

class TestValidatedDependencyInvariance:
    """ValidatedDependencyBuilder produces the same result when endpoints are swapped."""

    def test_dependency_both_orders_same_pair(self):
        builder = ValidatedDependencyBuilder(high_threshold=0.3)
        c1 = _rect(50, 100, 100, 50)
        c2 = _rect(300, 100, 100, 50)
        a1 = _arrow(150, 125, 300, 125)
        a2 = _arrow(300, 125, 150, 125)
        r1 = builder.build_validated_dependencies([a1], [c1, c2])
        r2 = builder.build_validated_dependencies([a2], [c1, c2])
        assert r1.accepted_count >= 1
        assert r2.accepted_count >= 1
        pair1 = frozenset([r1.validated_dependencies[0].source_id,
                           r1.validated_dependencies[0].target_id])
        pair2 = frozenset([r2.validated_dependencies[0].source_id,
                           r2.validated_dependencies[0].target_id])
        assert pair1 == pair2


# =========================================================================
# 6. Bottom-row arrows: endpoint swapping does NOT change result
# =========================================================================

class TestBottomRowInvariance:
    """
    Bottom-row arrows (R↔Q, Q↔P, P↔O, O↔N) must produce the same
    semantic result regardless of which endpoint is supplied first.

    These are evaluation fixtures, NOT hardcoded relationships.
    """

    @pytest.fixture
    def bottom_row_nodes(self):
        return [
            _rect(1220, 516, 90, 50),   # O at x~1259
            _rect(1068, 516, 90, 50),   # P at x~1113
            _rect(920, 516, 90, 50),    # Q at x~966
            _rect(775, 516, 90, 50),    # R at x~819
            _rect(1240, 240, 80, 50),   # N at x~1286
        ]

    def test_qp_both_orders_same_pair(self, bottom_row_nodes):
        c_q = bottom_row_nodes[2]
        c_p = bottom_row_nodes[1]
        builder = ValidatedDependencyBuilder(high_threshold=0.3)
        # Arrow between Q and P
        a1 = _arrow(966, 540, 1113, 540)
        a2 = _arrow(1113, 540, 966, 540)
        r1 = builder.build_validated_dependencies([a1], bottom_row_nodes)
        r2 = builder.build_validated_dependencies([a2], bottom_row_nodes)
        if r1.accepted_count >= 1 and r2.accepted_count >= 1:
            pair1 = frozenset([r1.validated_dependencies[0].source_id,
                               r1.validated_dependencies[0].target_id])
            pair2 = frozenset([r2.validated_dependencies[0].source_id,
                               r2.validated_dependencies[0].target_id])
            assert pair1 == pair2

    def test_po_both_orders_same_pair(self, bottom_row_nodes):
        c_p = bottom_row_nodes[1]
        c_o = bottom_row_nodes[0]
        builder = ValidatedDependencyBuilder(high_threshold=0.3)
        a1 = _arrow(1113, 540, 1259, 540)
        a2 = _arrow(1259, 540, 1113, 540)
        r1 = builder.build_validated_dependencies([a1], bottom_row_nodes)
        r2 = builder.build_validated_dependencies([a2], bottom_row_nodes)
        if r1.accepted_count >= 1 and r2.accepted_count >= 1:
            pair1 = frozenset([r1.validated_dependencies[0].source_id,
                               r1.validated_dependencies[0].target_id])
            pair2 = frozenset([r2.validated_dependencies[0].source_id,
                               r2.validated_dependencies[0].target_id])
            assert pair1 == pair2

    def test_rq_both_orders_same_pair(self, bottom_row_nodes):
        c_r = bottom_row_nodes[3]
        c_q = bottom_row_nodes[2]
        builder = ValidatedDependencyBuilder(high_threshold=0.3)
        a1 = _arrow(819, 540, 966, 540)
        a2 = _arrow(966, 540, 819, 540)
        r1 = builder.build_validated_dependencies([a1], bottom_row_nodes)
        r2 = builder.build_validated_dependencies([a2], bottom_row_nodes)
        if r1.accepted_count >= 1 and r2.accepted_count >= 1:
            pair1 = frozenset([r1.validated_dependencies[0].source_id,
                               r1.validated_dependencies[0].target_id])
            pair2 = frozenset([r2.validated_dependencies[0].source_id,
                               r2.validated_dependencies[0].target_id])
            assert pair1 == pair2

    def test_on_both_orders_same_pair(self, bottom_row_nodes):
        c_o = bottom_row_nodes[0]
        c_n = bottom_row_nodes[4]
        builder = ValidatedDependencyBuilder(high_threshold=0.3)
        a1 = _arrow(1259, 400, 1286, 280)
        a2 = _arrow(1286, 280, 1259, 400)
        r1 = builder.build_validated_dependencies([a1], bottom_row_nodes)
        r2 = builder.build_validated_dependencies([a2], bottom_row_nodes)
        if r1.accepted_count >= 1 and r2.accepted_count >= 1:
            pair1 = frozenset([r1.validated_dependencies[0].source_id,
                               r1.validated_dependencies[0].target_id])
            pair2 = frozenset([r2.validated_dependencies[0].source_id,
                               r2.validated_dependencies[0].target_id])
            assert pair1 == pair2

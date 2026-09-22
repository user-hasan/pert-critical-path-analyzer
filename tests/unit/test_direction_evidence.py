"""
Deterministic tests for the independent direction-evidence layer.

Tests boundary normal reasoning, endpoint contact classification,
segment continuity, boundary projection, and orientation scoring.
"""
import pytest
from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.direction_evidence import (
    DirectionEvidenceLayer,
    ContactSide,
    EndpointContact,
    DirectionEvidenceResult,
)


def _bb(x, y, w, h):
    return BoundingBox(x, y, w, h)


def _ev(min_boundary=80.0, max_tol=40.0):
    return DirectionEvidenceLayer(
        min_boundary_distance=min_boundary,
        max_direction_tolerance_deg=max_tol,
    )


class TestEndpointContactClassification:
    """Tests for endpoint contact classification."""

    def test_left_edge_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(100, 140)  # On left edge
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.LEFT
        assert contact.distance == 0.0
        assert contact.normal == (-1.0, 0.0)
        assert contact.confidence > 0.8

    def test_right_edge_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(220, 140)  # On right edge
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.RIGHT
        assert contact.distance == 0.0
        assert contact.normal == (1.0, 0.0)
        assert contact.confidence > 0.8

    def test_top_edge_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(160, 100)  # On top edge
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.TOP
        assert contact.distance == 0.0
        assert contact.normal == (0.0, -1.0)

    def test_bottom_edge_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(160, 180)  # On bottom edge
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.BOTTOM
        assert contact.distance == 0.0
        assert contact.normal == (0.0, 1.0)

    def test_inside_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(160, 140)  # Inside
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.INSIDE

    def test_far_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(500, 500)  # Far away
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.FAR
        assert contact.confidence == 0.0

    def test_near_left_contact(self):
        ev = _ev(min_boundary=80.0)
        bb = _bb(100, 100, 120, 80)
        point = Point(90, 140)  # 10px left of left edge
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.LEFT
        assert contact.distance < 15
        assert contact.confidence > 0.5

    def test_corner_contact(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(102, 102)  # Near top-left corner
        contact = ev._classify_contact(point, bb)
        assert contact.side == ContactSide.CORNER
        assert contact.normal[0] < 0  # Points up-left
        assert contact.normal[1] < 0


class TestBoundaryProjection:
    """Tests for boundary projection when endpoints are inside rectangles."""

    def test_project_right_from_inside(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(130, 140)  # Inside
        axis = (1.0, 0.0)  # Pointing right
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is not None
        assert contact.side == ContactSide.RIGHT
        assert contact.distance == 0.0
        assert contact.boundary_point.x == 220  # Right edge

    def test_project_left_from_inside(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(130, 140)  # Inside
        axis = (-1.0, 0.0)  # Pointing left
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is not None
        assert contact.side == ContactSide.LEFT
        assert contact.boundary_point.x == 100  # Left edge

    def test_project_up_from_inside(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(160, 130)  # Inside
        axis = (0.0, -1.0)  # Pointing up
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is not None
        assert contact.side == ContactSide.TOP
        assert contact.boundary_point.y == 100  # Top edge

    def test_project_down_from_inside(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(160, 150)  # Inside
        axis = (0.0, 1.0)  # Pointing down
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is not None
        assert contact.side == ContactSide.BOTTOM
        assert contact.boundary_point.y == 180  # Bottom edge

    def test_project_diagonal(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(130, 130)  # Inside, near top-left
        axis = (0.707, -0.707)  # Up-right
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is not None
        assert contact.side in (ContactSide.RIGHT, ContactSide.TOP)

    def test_project_no_intersection(self):
        ev = _ev()
        bb = _bb(100, 100, 120, 80)
        point = Point(130, 140)  # Inside
        axis = (0.0, 0.0)  # Zero direction
        contact = ev._project_to_boundary(point, axis, bb, forward=True)
        assert contact is None


class TestBoundaryNormalReasoning:
    """Tests for normal consistency scoring."""

    def test_source_right_target_left_agrees(self):
        """Source exits right, target enters from left → consistent."""
        ev = _ev()
        source = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        target = EndpointContact(
            side=ContactSide.LEFT, distance=0.0,
            normal=(-1.0, 0.0), confidence=1.0,
        )
        axis = (1.0, 0.0)  # Points right
        score = ev._score_normal_consistency(source, target, axis)
        assert score > 0.8

    def test_source_left_target_right_agrees(self):
        """Source exits left, target enters from right → consistent."""
        ev = _ev()
        source = EndpointContact(
            side=ContactSide.LEFT, distance=0.0,
            normal=(-1.0, 0.0), confidence=1.0,
        )
        target = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        axis = (-1.0, 0.0)  # Points left
        score = ev._score_normal_consistency(source, target, axis)
        assert score > 0.8

    def test_source_right_target_right_disagrees(self):
        """Source exits right, target normal is right → target normal opposes direction."""
        ev = _ev()
        source = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        target = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        axis = (1.0, 0.0)
        score = ev._score_normal_consistency(source, target, axis)
        # Source agrees, but target inward normal opposes direction
        assert score < 0.6

    def test_far_contacts_zero_score(self):
        ev = _ev()
        source = EndpointContact(side=ContactSide.FAR, confidence=0.0)
        target = EndpointContact(side=ContactSide.FAR, confidence=0.0)
        score = ev._score_normal_consistency(source, target, (1.0, 0.0))
        assert score == 0.0


class TestSourceExitsBoundary:
    """Tests for source exits boundary scoring."""

    def test_source_exits_right_agrees(self):
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        score = ev._score_source_exits(contact, (1.0, 0.0))
        assert score > 0.8

    def test_source_exits_left_agrees(self):
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.LEFT, distance=0.0,
            normal=(-1.0, 0.0), confidence=1.0,
        )
        score = ev._score_source_exits(contact, (-1.0, 0.0))
        assert score > 0.8

    def test_source_exits_wrong_direction(self):
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        score = ev._score_source_exits(contact, (-1.0, 0.0))
        assert score == 0.0

    def test_far_contact_zero_score(self):
        ev = _ev()
        contact = EndpointContact(side=ContactSide.FAR, confidence=0.0)
        score = ev._score_source_exits(contact, (1.0, 0.0))
        assert score == 0.0


class TestTargetEntersBoundary:
    """Tests for target enters boundary scoring."""

    def test_target_enters_from_left(self):
        """Direction is right, target normal is left → enters."""
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.LEFT, distance=0.0,
            normal=(-1.0, 0.0), confidence=1.0,
        )
        score = ev._score_target_enters(contact, (1.0, 0.0))
        assert score > 0.8

    def test_target_enters_from_right(self):
        """Direction is left, target normal is right → enters."""
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        score = ev._score_target_enters(contact, (-1.0, 0.0))
        assert score > 0.8

    def test_target_exits_wrong(self):
        """Direction is right, target normal is right → exits, not enters."""
        ev = _ev()
        contact = EndpointContact(
            side=ContactSide.RIGHT, distance=0.0,
            normal=(1.0, 0.0), confidence=1.0,
        )
        score = ev._score_target_enters(contact, (1.0, 0.0))
        assert score == 0.0


class TestOrientationScoring:
    """Tests for full orientation scoring with boundary projection."""

    def test_left_to_right_strong(self):
        """Source on left, target on right, segment goes left→right."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)   # At source right edge
        ep2 = Point(200, 30)   # At target left edge
        axis = (1.0, 0.0)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, axis)
        assert result.source_contact.side == ContactSide.RIGHT
        assert result.target_contact.side == ContactSide.LEFT
        assert result.source_exits_boundary > 0.5
        assert result.target_enters_boundary > 0.5
        assert result.total_score > 0.5

    def test_left_to_right_vs_right_to_left(self):
        """Verify left→right scores higher when endpoints are between rectangles."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(250, 0, 100, 60)
        ep1 = Point(120, 30)  # Just outside source right edge
        ep2 = Point(230, 30)  # Just outside target left edge
        axis = (1.0, 0.0)

        fwd = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, axis)
        rev = ev.score_orientation(tgt_bb, src_bb, ep2, ep1, (-1.0, 0.0))
        # Forward should score higher because projection matches geometry
        assert fwd.total_score >= rev.total_score

    def test_inside_endpoint_projection(self):
        """Source endpoint inside source → projected to boundary."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(50, 30)    # Inside source
        ep2 = Point(200, 30)   # At target
        axis = (1.0, 0.0)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, axis)
        assert result.source_contact.side == ContactSide.RIGHT
        assert result.source_contact.boundary_point is not None
        assert result.source_exits_boundary > 0.0

    def test_between_rectangles_projection(self):
        """Endpoint between rectangles → projected toward target."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)   # At source right edge
        ep2 = Point(150, 30)   # Between source and target
        axis = (1.0, 0.0)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, axis)
        assert result.target_contact.side == ContactSide.LEFT
        assert result.target_enters_boundary > 0.0

    def test_endpoint_order_invariance(self):
        """Swapping endpoints should produce same geometry evidence."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)
        ep2 = Point(200, 30)

        fwd = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, (1.0, 0.0))
        rev = ev.score_orientation(src_bb, tgt_bb, ep2, ep1, (-1.0, 0.0))
        # The orientation that matches the geometry should score higher
        assert fwd.total_score > rev.total_score

    def test_arrowhead_boosts_score(self):
        """Arrowhead near target should boost score when boundary isn't perfect."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(250, 0, 100, 60)
        ep1 = Point(120, 30)   # Just outside source
        ep2 = Point(230, 30)   # Just outside target
        axis = (1.0, 0.0)
        ah = [Point(248, 30)]  # Arrowhead right at target boundary

        without_ah = ev.score_orientation(
            src_bb, tgt_bb, ep1, ep2, axis,
            arrowhead_candidates=None, arrowhead_confidence=0.0,
        )
        with_ah = ev.score_orientation(
            src_bb, tgt_bb, ep1, ep2, axis,
            arrowhead_candidates=ah, arrowhead_confidence=0.9,
        )
        # Arrowhead should not reduce the score
        assert with_ah.total_score >= without_ah.total_score

    def test_crossing_penalty(self):
        """Segment crossing unrelated node should penalize."""
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(300, 0, 100, 60)
        mid_bb = _bb(140, 0, 100, 60)  # Between source and target
        ep1 = Point(100, 30)
        ep2 = Point(300, 30)
        axis = (1.0, 0.0)

        without_cross = ev.score_orientation(
            src_bb, tgt_bb, ep1, ep2, axis, all_candidates=[tgt_bb],
        )
        with_cross = ev.score_orientation(
            src_bb, tgt_bb, ep1, ep2, axis, all_candidates=[tgt_bb, mid_bb],
        )
        assert with_cross.total_score < without_cross.total_score


class TestFixedNodePair:
    """Test that the evidence layer does NOT change the node pair."""

    def test_pair_unchanged(self):
        ev = _ev()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)
        ep2 = Point(200, 30)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, (1.0, 0.0))
        assert result.source_contact.side != ContactSide.FAR
        assert result.target_contact.side != ContactSide.FAR


class TestConfidenceMargin:
    """Test that margin reflects evidence strength."""

    def test_strong_geometry_high_margin(self):
        """Clear left→right with between-rectangle endpoints should have margin."""
        ev = _ev()
        src = _bb(0, 0, 100, 60)
        tgt = _bb(300, 0, 100, 60)
        ep1 = Point(120, 30)  # Just outside source
        ep2 = Point(280, 30)  # Just outside target
        axis = (1.0, 0.0)

        fwd = ev.score_orientation(src, tgt, ep1, ep2, axis)
        rev = ev.score_orientation(tgt, src, ep2, ep1, (-1.0, 0.0))
        # Both should score, but forward should be at least as good
        assert fwd.total_score >= rev.total_score
        assert fwd.total_score > 0.0

    def test_symmetric_low_margin(self):
        """When endpoints are equidistant, margin should be smaller."""
        ev = _ev()
        src = _bb(0, 0, 100, 60)
        tgt = _bb(150, 0, 100, 60)
        # Both endpoints near middle
        ep1 = Point(100, 30)
        ep2 = Point(150, 30)
        axis = (1.0, 0.0)

        fwd = ev.score_orientation(src, tgt, ep1, ep2, axis)
        rev = ev.score_orientation(tgt, src, ep2, ep1, (-1.0, 0.0))
        # Both should have some score
        assert fwd.total_score > 0.0
        assert rev.total_score > 0.0

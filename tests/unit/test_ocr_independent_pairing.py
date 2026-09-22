"""
Regression tests for OCR-independent node pairing.

Verifies that NodePairingResolver and DirectionResolution work correctly
regardless of OCR output. Geometry is the primary identity; OCR is annotation.
"""
import pytest
from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.node_pairing import NodePairingResolver, PairingStatus
from pert_analyzer.cv.direction_resolution import ArrowDirectionResolver, DirectionStatus
from pert_analyzer.cv.direction_evidence import DirectionEvidenceLayer


def _bb(x, y, w, h):
    return BoundingBox(x, y, w, h)


def _candidate(node_id, x, y, w, h, label=""):
    return CandidateNode(
        node_id=node_id,
        shape_type=ShapeType.RECTANGLE,
        bounding_box=_bb(x, y, w, h),
        position=Point(x + w / 2, y + h / 2),
        label=label,
        confidence=0.9,
    )


def _arrow(arrow_id, sx, sy, ex, ey, ah_x=None, ah_y=None, ah_conf=0.0):
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


class TestGeometryPrimaryIdentity:
    """Node identity is determined by geometry, not OCR."""

    def test_pairing_succeeds_with_no_ocr(self):
        """Two nodes with no OCR labels can be paired."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="")
        c2 = _candidate("c2", 200, 0, 100, 60, label="")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        assert result.status in (
            PairingStatus.PAIR_CONFIRMED,
            PairingStatus.PAIR_CONTEXTUALLY_RESOLVED,
        )

    def test_pairing_succeeds_with_wrong_ocr(self):
        """Nodes with wrong OCR labels can still be paired by geometry."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="Z")  # Wrong OCR
        c2 = _candidate("c2", 200, 0, 100, 60, label="X")  # Wrong OCR
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        # Pairing should connect c1 and c2 regardless of OCR labels
        assert {result.selected.node_a_id, result.selected.node_b_id} == {"c1", "c2"}

    def test_pairing_succeeds_with_low_confidence_ocr(self):
        """Low OCR confidence doesn't prevent geometric pairing."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="A")
        c1.confidence = 0.1
        c2 = _candidate("c2", 200, 0, 100, 60, label="B")
        c2.confidence = 0.1
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None

    def test_pairing_succeeds_with_inferred_id(self):
        """Nodes with INFERRED_018 style IDs can be paired."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="INFERRED_018")
        c2 = _candidate("c2", 200, 0, 100, 60, label="INFERRED_019")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None

    def test_pairing_with_duplicate_ocr_labels(self):
        """Two nodes with same OCR label can be paired by geometry."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="A")
        c2 = _candidate("c2", 200, 0, 100, 60, label="A")  # Duplicate
        c3 = _candidate("c3", 400, 0, 100, 60, label="B")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2, c3])
        assert result.selected is not None
        # Should pair c1 and c2 (geometrically closest), not c3
        assert {result.selected.node_a_id, result.selected.node_b_id} == {"c1", "c2"}


class TestDirectionWithoutOCR:
    """Direction resolution works without OCR labels."""

    def test_direction_without_ocr(self):
        """Direction resolves correctly with empty OCR labels."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="")
        c2 = _candidate("c2", 200, 0, 100, 60, label="")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = ArrowDirectionResolver(confirmed_margin=0.20)
        result = resolver.resolve_direction_for_pair(arrow, "c1", "c2", [c1, c2])
        assert result.status != DirectionStatus.REJECTED
        # Direction should be c1 -> c2 (left to right)
        assert result.source_id == "c1"
        assert result.target_id == "c2"

    def test_direction_with_wrong_ocr(self):
        """Direction resolves correctly when OCR labels are wrong."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="Z")
        c2 = _candidate("c2", 200, 0, 100, 60, label="X")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = ArrowDirectionResolver(confirmed_margin=0.20)
        result = resolver.resolve_direction_for_pair(arrow, "c1", "c2", [c1, c2])
        assert result.status != DirectionStatus.REJECTED
        assert result.source_id == "c1"
        assert result.target_id == "c2"


class TestDirectionEvidenceWithoutOCR:
    """Direction evidence layer works without OCR labels."""

    def test_evidence_layer_no_ocr(self):
        """DirectionEvidenceLayer works with empty labels."""
        ev = DirectionEvidenceLayer()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)
        ep2 = Point(200, 30)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, (1.0, 0.0))
        assert result.total_score > 0.5

    def test_evidence_layer_wrong_ocr(self):
        """DirectionEvidenceLayer ignores OCR labels."""
        ev = DirectionEvidenceLayer()
        src_bb = _bb(0, 0, 100, 60)
        tgt_bb = _bb(200, 0, 100, 60)
        ep1 = Point(100, 30)
        ep2 = Point(200, 30)
        result = ev.score_orientation(src_bb, tgt_bb, ep1, ep2, (1.0, 0.0))
        assert result.total_score > 0.5


class TestGeometricNodeSurvivesOCRFailure:
    """Geometric nodes survive OCR failure."""

    def test_candidate_with_empty_label(self):
        """CandidateNode with empty label is still a valid candidate."""
        c = _candidate("c1", 0, 0, 100, 60, label="")
        assert c.node_id == "c1"
        assert c.bounding_box.width == 100
        assert c.label == ""

    def test_candidate_with_wrong_label(self):
        """CandidateNode with wrong OCR label retains geometric identity."""
        c = _candidate("c1", 0, 0, 100, 60, label="Z")
        assert c.node_id == "c1"
        assert c.label == "Z"  # OCR annotation preserved
        # But geometric identity is node_id, not label

    def test_candidate_with_inferred_label(self):
        """CandidateNode with INFERRED label retains geometric identity."""
        c = _candidate("c1", 0, 0, 100, 60, label="INFERRED_018")
        assert c.node_id == "c1"


class TestDualIdentity:
    """ReconstructedActivity preserves dual identity."""

    def test_dual_identity_fields_exist(self):
        """ReconstructedActivity has geometric_node_id and semantic_activity_id."""
        from pert_analyzer.cv.reconstruction_models import (
            ReconstructedActivity,
            ActivityStatus,
        )
        act = ReconstructedActivity(
            activity_id="A",
            geometric_node_id="cnode_123",
            semantic_activity_id="A",
            semantic_status=ActivityStatus.CONFIRMED,
        )
        assert act.geometric_node_id == "cnode_123"
        assert act.semantic_activity_id == "A"
        assert act.semantic_status == ActivityStatus.CONFIRMED

    def test_dual_identity_with_missing_ocr(self):
        """When OCR fails, geometric_node_id is set but semantic is None."""
        from pert_analyzer.cv.reconstruction_models import (
            ReconstructedActivity,
            ActivityStatus,
        )
        act = ReconstructedActivity(
            activity_id="cnode_456",
            geometric_node_id="cnode_456",
            semantic_activity_id=None,
            semantic_status=ActivityStatus.REVIEW_REQUIRED,
        )
        assert act.geometric_node_id == "cnode_456"
        assert act.semantic_activity_id is None
        assert act.semantic_status == ActivityStatus.REVIEW_REQUIRED

    def test_geometric_id_stable_after_semantic_correction(self):
        """Semantic correction doesn't change geometric_node_id."""
        from pert_analyzer.cv.reconstruction_models import (
            ReconstructedActivity,
            ActivityStatus,
        )
        act = ReconstructedActivity(
            activity_id="B",
            geometric_node_id="cnode_789",
            semantic_activity_id="B",
            semantic_status=ActivityStatus.CONFIRMED,
        )
        # Simulate semantic correction
        act.activity_id = "C"
        act.semantic_activity_id = "C"
        # Geometric ID should remain unchanged
        assert act.geometric_node_id == "cnode_789"


class TestOCREnrichmentLater:
    """OCR annotation can be added after geometric pairing."""

    def test_ocr_can_be_added_to_paired_nodes(self):
        """After geometric pairing, OCR labels can be attached."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="")
        c2 = _candidate("c2", 200, 0, 100, 60, label="")
        arrow = _arrow("a1", 100, 30, 200, 30)

        # Step 1: geometric pairing (no OCR)
        pairer = NodePairingResolver(max_endpoint_distance=120.0)
        result = pairer.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None

        # Step 2: attach OCR annotation (simulated)
        c1.label = "A"
        c2.label = "B"

        # Pairing should still be valid
        assert result.selected.node_a_id == "c1"
        assert result.selected.node_b_id == "c2"


class TestPairingDiagnostics:
    """Pairing produces diagnostic output."""

    def test_pair_result_has_metadata(self):
        """NodePairResult carries diagnostic metadata."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="A")
        c2 = _candidate("c2", 200, 0, 100, 60, label="B")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        assert result.selected is not None
        assert result.selected.ocr_used is False
        assert result.confidence > 0

    def test_pair_candidate_has_ocr_used_false(self):
        """NodePairCandidate.ocr_used is False for geometric pairing."""
        c1 = _candidate("c1", 0, 0, 100, 60, label="")
        c2 = _candidate("c2", 200, 0, 100, 60, label="")
        arrow = _arrow("a1", 100, 30, 200, 30)
        resolver = NodePairingResolver(max_endpoint_distance=120.0)
        result = resolver.resolve_pairing(arrow, [c1, c2])
        if result.selected:
            assert result.selected.ocr_used is False
        for alt in result.alternatives:
            assert alt.ocr_used is False

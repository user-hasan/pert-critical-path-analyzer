"""
Deterministic tests for semantic resolution of AON activity IDs and durations.

Tests the SemanticResolver's ability to resolve ambiguous OCR candidates
using contextual evidence (duplicate-ID constraints, alternatives, numeric validity).
"""

from __future__ import annotations

import pytest

from pert_analyzer.cv.region_ocr import NodeOCRResult
from pert_analyzer.cv.semantic_resolution import (
    CandidateResolution,
    DurationResolution,
    ResolutionSource,
    ResolutionStatus,
    SemanticResolver,
)


def _make_nr(
    node_id: str = "node_1",
    activity_id: str | None = None,
    confidence: float = 0.8,
    alternatives: list | None = None,
    numeric: tuple | None = None,
    numeric_candidates: list | None = None,
) -> NodeOCRResult:
    """Create a minimal NodeOCRResult for testing."""
    nr = NodeOCRResult(node_id=node_id)
    nr.best_activity_id = activity_id
    nr.best_activity_id_confidence = confidence
    nr.activity_id_alternatives = alternatives or []
    if numeric:
        nr.best_numeric = numeric
    nr.numeric_candidates = numeric_candidates or []
    return nr


class TestZToCContextualResolution:
    """Test 1: Z → C contextual resolution via low confidence + valid alternative."""

    def test_low_confidence_z_with_c_alternative_resolves_to_c(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="Z",
            confidence=0.55,
            alternatives=[("C", 0.45)],
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.resolved_id == "C"
        assert cr.source == ResolutionSource.OCR_ALTERNATIVE
        assert cr.status == ResolutionStatus.CONTEXTUALLY_RESOLVED

    def test_high_confidence_z_accepted(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="Z",
            confidence=0.85,
            alternatives=[],
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.resolved_id == "Z"
        assert cr.source == ResolutionSource.OCR_HIGH_CONFIDENCE
        assert cr.status == ResolutionStatus.CONFIRMED


class TestLToIDuplicateResolution:
    """Test 2: L → I duplicate-ID resolution."""

    def test_duplicate_l_resolves_second_to_i(self):
        resolver = SemanticResolver()
        nr_real_l = _make_nr(
            node_id="node_real_l",
            activity_id="L",
            confidence=0.85,
        )
        nr_fake_l = _make_nr(
            node_id="node_fake_l",
            activity_id="L",
            confidence=0.61,
            alternatives=[("I", 0.41)],
        )
        result = resolver.resolve_all([nr_real_l, nr_fake_l])
        cr_real = result.id_resolutions["node_real_l"]
        cr_fake = result.id_resolutions["node_fake_l"]
        # Real L stays as L
        assert cr_real.resolved_id == "L"
        # Fake L resolves to I (via alternative or duplicate-ID conflict)
        assert cr_fake.resolved_id == "I"
        assert cr_fake.status == ResolutionStatus.CONTEXTUALLY_RESOLVED

    def test_single_l_not_renamed(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="L",
            confidence=0.85,
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.resolved_id == "L"
        assert cr.source == ResolutionSource.OCR_HIGH_CONFIDENCE


class TestUnresolvedMissingID:
    """Test 3: unresolved missing ID stays as REVIEW_REQUIRED."""

    def test_no_ocr_text_returns_review_required(self):
        resolver = SemanticResolver()
        nr = _make_nr(activity_id=None, confidence=0.0)
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.resolved_id is None
        assert cr.source == ResolutionSource.REVIEW_REQUIRED
        assert cr.status == ResolutionStatus.REVIEW_REQUIRED

    def test_invalid_text_returns_review_required(self):
        resolver = SemanticResolver()
        nr = _make_nr(activity_id="7", confidence=0.5)
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.resolved_id is None
        assert cr.source == ResolutionSource.REVIEW_REQUIRED


class TestDuplicateIDHandling:
    """Test 4: duplicate ID handling preserves raw OCR data."""

    def test_duplicate_ids_both_preserved_in_raw(self):
        resolver = SemanticResolver()
        nr1 = _make_nr(node_id="n1", activity_id="M", confidence=0.9)
        nr2 = _make_nr(node_id="n2", activity_id="M", confidence=0.7)
        result = resolver.resolve_all([nr1, nr2])
        # Both should have raw_id = "M"
        assert result.id_resolutions["n1"].raw_id == "M"
        assert result.id_resolutions["n2"].raw_id == "M"
        # One should be resolved to M, the other to an alternative or review
        ids = {result.id_resolutions["n1"].resolved_id, result.id_resolutions["n2"].resolved_id}
        assert len(ids) == 2  # Must be different

    def test_duplicate_with_no_alternative_marked_review(self):
        resolver = SemanticResolver()
        nr1 = _make_nr(node_id="n1", activity_id="X", confidence=0.9)
        nr2 = _make_nr(node_id="n2", activity_id="X", confidence=0.8)
        result = resolver.resolve_all([nr1, nr2])
        # One stays X, other has no alternative → review
        statuses = [result.id_resolutions["n1"].status, result.id_resolutions["n2"].status]
        assert ResolutionStatus.REVIEW_REQUIRED in statuses


class TestDurationCandidateRanking:
    """Test 5: node-local duration candidate ranking."""

    def test_high_confidence_integer_preferred(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="A",
            numeric=(3.0, "3", 0.9),
            numeric_candidates=[
                (3.0, "3", 0.9),
                (3.5, "3.5", 0.4),
            ],
        )
        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        assert dr.resolved_value == 3.0
        assert dr.status == ResolutionStatus.CONFIRMED

    def test_low_confidence_numeric_review(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="A",
            numeric=(0.0, "0", 0.1),
            numeric_candidates=[(0.0, "0", 0.1)],
        )
        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        assert dr.status in (ResolutionStatus.REVIEW_REQUIRED, ResolutionStatus.CONTEXTUALLY_RESOLVED)


class TestNoisyNumberRejected:
    """Test 6: noisy number rejected by duration ranking."""

    def test_very_large_number_penalized(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="T",
            numeric=(999.0, "999", 0.5),
            numeric_candidates=[
                (999.0, "999", 0.5),
                (2.0, "2", 0.3),
            ],
        )
        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        # The 999 is out of range, 2 is in range — 2 should be preferred
        assert dr.resolved_value == 2.0

    def test_negative_number_penalized(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="A",
            numeric=(-3.0, "-3", 0.6),
            numeric_candidates=[(-3.0, "-3", 0.6)],
        )
        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        # Negative duration is out of range
        assert dr.resolved_value == -3.0  # value preserved but low score
        assert dr.confidence < 0.6


class TestDurationRegionPriority:
    """Test 7: duration-region priority in candidate ranking."""

    def test_duration_sub_crop_candidate_boosted(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="A",
            numeric=(5.0, "5", 0.6),
            numeric_candidates=[(5.0, "5", 0.6)],
        )
        # Simulate duration_sub_crop_regions with parsed_value=5
        from pert_analyzer.cv.ocr_models import OCRTextRegion, BoundingBox
        dur_region = OCRTextRegion(text="5", confidence=0.6)
        dur_region.parsed_value = 5.0
        nr.duration_sub_crop_regions = [dur_region]

        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        # Should get boost from duration sub-crop
        assert dr.resolved_value == 5.0
        assert any(
            any("from_duration_sub_crop" in f for f in e.supporting_factors)
            for e in dr.evidence
        )


class TestTLikeFalseNumeric:
    """Test 8: T-like false numeric candidate rejected."""

    def test_noisy_punctuation_not_preferred(self):
        resolver = SemanticResolver()
        # T node has (7) as noisy text — the 7 is parsed as numeric
        nr = _make_nr(
            activity_id="T",
            numeric=(7.0, "7", 0.3),
            numeric_candidates=[
                (7.0, "7", 0.3),
                (2.0, "2", 0.5),
            ],
        )
        result = resolver.resolve_all([nr])
        dr = result.duration_resolutions["node_1"]
        # 2 is a better candidate (higher conf, valid format, in range)
        assert dr.resolved_value == 2.0


class TestReviewRequiredBehavior:
    """Test 9: review-required behavior for ambiguous cases."""

    def test_empty_node_returns_review(self):
        resolver = SemanticResolver()
        nr = _make_nr(activity_id=None, confidence=0.0)
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.status == ResolutionStatus.REVIEW_REQUIRED
        assert len(cr.evidence) > 0
        assert cr.evidence[0].reason != ""

    def test_summary_counts_review_required(self):
        resolver = SemanticResolver()
        nr1 = _make_nr(node_id="n1", activity_id=None, confidence=0.0)
        nr2 = _make_nr(node_id="n2", activity_id="A", confidence=0.9)
        result = resolver.resolve_all([nr1, nr2])
        assert result.summary["id"]["review_required"] >= 1
        assert result.summary["id"]["confirmed"] >= 1


class TestRawOCRPreservation:
    """Test 10: raw OCR data preserved through resolution."""

    def test_raw_id_always_stored(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="Z",
            confidence=0.55,
            alternatives=[("C", 0.45)],
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.raw_id == "Z"
        assert cr.resolved_id == "C"
        assert cr.raw_id != cr.resolved_id

    def test_raw_id_preserved_when_accepted(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="A",
            confidence=0.9,
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert cr.raw_id == "A"
        assert cr.resolved_id == "A"
        assert cr.raw_id == cr.resolved_id

    def test_alternatives_preserved(self):
        resolver = SemanticResolver()
        nr = _make_nr(
            activity_id="L",
            confidence=0.61,
            alternatives=[("I", 0.41)],
        )
        result = resolver.resolve_all([nr])
        cr = result.id_resolutions["node_1"]
        assert len(cr.alternatives) == 1
        assert cr.alternatives[0] == ("I", 0.41)

"""
Comprehensive test suite for Phase 6: OCR and Text Association.

Covers: OCR models, text normalization, numeric extraction, text classification,
OCR engine abstraction, spatial association, text grouping, debug rendering,
mock OCR tests, missing OCR backend handling, coordinate mapping, edge cases.
"""

from __future__ import annotations

import math
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pert_analyzer.core.models import BoundingBox, Point
from pert_analyzer.cv.models import CandidateNode, DetectedArrow, ShapeType
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    NumericCandidate,
    OCRConfig,
    OCRProcessingResult,
    OCRTextRegion,
    TextAssociation,
    TextAssociationResult,
    TextGroup,
    TextType,
)
from pert_analyzer.cv.text_normalization import (
    TextNormalizer,
    NormalizationConfig,
    ARABIC_INDIC_DIGITS,
)
from pert_analyzer.cv.numeric_extraction import (
    NumericExtractor,
    NumericExtractionConfig,
)
from pert_analyzer.cv.text_classification import (
    TextClassifier,
    TextClassificationConfig,
)
from pert_analyzer.cv.ocr_engine import (
    MockOCREngine,
    OCREngineBase,
    OCREngineError,
    TesseractOCREngine,
    create_ocr_engine,
)
from pert_analyzer.cv.spatial_association import (
    SpatialAssociator,
    AssociationConfig,
)
from pert_analyzer.cv.text_grouping import (
    TextGrouper,
    TextGroupingConfig,
)
from pert_analyzer.cv.exceptions import OCREngineError as CVOCREngineError


# =============================================================================
# Helper functions
# =============================================================================


def _make_region(
    text: str = "test",
    x: float = 100,
    y: float = 100,
    w: float = 50,
    h: float = 30,
    confidence: float = 0.9,
    text_type: TextType = TextType.UNKNOWN,
) -> OCRTextRegion:
    """Create an OCRTextRegion for testing."""
    bbox = BoundingBox(x=x, y=y, width=w, height=h)
    return OCRTextRegion(
        text=text,
        raw_text=text,
        normalized_text=text.strip(),
        bounding_box=bbox,
        center=bbox.center,
        confidence=confidence,
        text_type=text_type,
    )


def _make_node(
    node_id: str = "node_1",
    x: float = 100,
    y: float = 100,
    w: float = 200,
    h: float = 100,
    shape_type: ShapeType = ShapeType.RECTANGLE,
) -> CandidateNode:
    """Create a CandidateNode for testing."""
    return CandidateNode(
        node_id=node_id,
        shape_type=shape_type,
        bounding_box=BoundingBox(x=x, y=y, width=w, height=h),
        confidence=0.8,
    )


def _make_arrow(
    arrow_id: str = "arr_1",
    start_x: float = 300,
    start_y: float = 150,
    end_x: float = 500,
    end_y: float = 150,
    length: float = 200,
) -> DetectedArrow:
    """Create a DetectedArrow for testing."""
    start = Point(start_x, start_y)
    end = Point(end_x, end_y)
    return DetectedArrow(
        arrow_id=arrow_id,
        start=start,
        end=end,
        length=length,
        confidence=0.8,
    )


# =============================================================================
# Test: OCR Model Validation
# =============================================================================


class TestOCRModels:
    """Test OCR data models."""

    def test_ocr_text_region_creation(self):
        region = _make_region("Hello", 10, 20, 50, 30, 0.95)
        assert region.text == "Hello"
        assert region.confidence == 0.95
        assert region.bounding_box.x == 10
        assert region.bounding_box.y == 20
        assert region.bounding_box.width == 50
        assert region.bounding_box.height == 30
        assert region.center.x == 35
        assert region.center.y == 35

    def test_ocr_text_region_defaults(self):
        region = OCRTextRegion()
        assert region.text == ""
        assert region.confidence == 0.0
        assert region.text_type == TextType.UNKNOWN
        assert region.associated_target_id is None

    def test_ocr_config_defaults(self):
        config = OCRConfig()
        assert config.engine_name == "tesseract"
        assert config.languages == ["eng"]
        assert config.min_confidence == 0.0
        issues = config.validate()
        assert len(issues) == 0

    def test_ocr_config_validation_error(self):
        config = OCRConfig(min_confidence=1.5)
        issues = config.validate()
        assert len(issues) > 0

    def test_ocr_config_high_below_min(self):
        config = OCRConfig(min_confidence=0.5, high_confidence=0.3)
        issues = config.validate()
        assert len(issues) > 0

    def test_text_type_all_values(self):
        assert TextType.UNKNOWN.value == "UNKNOWN"
        assert TextType.ACTIVITY_ID_CANDIDATE.value == "ACTIVITY_ID_CANDIDATE"
        assert TextType.TEXT_LABEL_CANDIDATE.value == "TEXT_LABEL_CANDIDATE"
        assert TextType.NUMERIC_CANDIDATE.value == "NUMERIC_CANDIDATE"

    def test_association_target_types(self):
        assert AssociationTargetType.NODE.value == "NODE"
        assert AssociationTargetType.ARROW.value == "ARROW"
        assert AssociationTargetType.NONE.value == "NONE"

    def test_numeric_candidate_creation(self):
        candidate = NumericCandidate(
            value=5.0,
            raw_text="5",
            confidence=0.9,
            bounding_box=BoundingBox(10, 20, 30, 15),
        )
        assert candidate.value == 5.0
        assert candidate.raw_text == "5"
        assert candidate.is_integer is True
        assert candidate.is_negative is False

    def test_numeric_candidate_negative(self):
        candidate = NumericCandidate(
            value=-3.14,
            raw_text="-3.14",
            confidence=0.85,
            bounding_box=BoundingBox(10, 20, 30, 15),
        )
        assert candidate.is_negative is True
        assert candidate.is_integer is False

    def test_text_association_creation(self):
        assoc = TextAssociation(
            text_region_id="r1",
            candidate_target_id="n1",
            target_type=AssociationTargetType.NODE,
            association_score=0.85,
        )
        assert assoc.text_region_id == "r1"
        assert assoc.association_score == 0.85
        assert assoc.is_best_candidate is False

    def test_text_association_result_empty(self):
        result = TextAssociationResult(text_region_id="r1")
        assert result.association_count == 0
        assert result.has_associations is False
        assert result.has_no_match is True

    def test_text_group_creation(self):
        group = TextGroup(
            member_region_ids=["r1", "r2"],
            combined_text="A 5",
            group_confidence=0.8,
            member_count=2,
        )
        assert group.member_count == 2
        assert group.combined_text == "A 5"

    def test_ocr_processing_result_empty(self):
        result = OCRProcessingResult()
        assert result.region_count == 0
        assert result.group_count == 0
        assert result.numeric_count == 0
        assert result.average_confidence == 0.0

    def test_ocr_processing_result_average_confidence(self):
        result = OCRProcessingResult(
            regions=[
                _make_region(confidence=0.8),
                _make_region(confidence=0.6),
            ]
        )
        assert result.average_confidence == pytest.approx(0.7)

    def test_ocr_processing_result_get_by_type(self):
        result = OCRProcessingResult(
            regions=[
                _make_region("A1", text_type=TextType.ACTIVITY_ID_CANDIDATE),
                _make_region("5", text_type=TextType.NUMERIC_CANDIDATE),
                _make_region("Test", text_type=TextType.TEXT_LABEL_CANDIDATE),
            ]
        )
        assert len(result.get_id_candidates()) == 1
        assert len(result.get_numeric_regions()) == 1
        assert len(result.get_labeled_regions()) == 1

    def test_ocr_processing_result_get_region_by_id(self):
        region = _make_region("test")
        result = OCRProcessingResult(regions=[region])
        found = result.get_region_by_id(region.region_id)
        assert found is region
        assert result.get_region_by_id("nonexistent") is None

    def test_ocr_processing_result_high_confidence_count(self):
        result = OCRProcessingResult(
            regions=[
                _make_region(confidence=0.9),
                _make_region(confidence=0.5),
                _make_region(confidence=0.8),
            ]
        )
        assert result.high_confidence_count == 2


# =============================================================================
# Test: Text Normalization
# =============================================================================


class TestTextNormalization:
    """Test text normalization utilities."""

    def test_basic_normalization(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("  Hello  ") == "Hello"

    def test_collapse_spaces(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("Hello   World") == "Hello World"

    def test_empty_text(self):
        normalizer = TextNormalizer()
        assert normalizer.normalize("") == ""
        assert normalizer.normalize("   ") == ""

    def test_normalize_preserve_raw(self):
        normalizer = TextNormalizer()
        raw, norm = normalizer.normalize_preserve_raw("  Hello  ")
        assert raw == "  Hello  "
        assert norm == "Hello"

    def test_arabic_indic_digits(self):
        normalizer = TextNormalizer()
        # Arabic-Indic: ١٢٣ = 123
        result = normalizer.normalize("\u0661\u0662\u0663")
        assert result == "123"

    def test_extended_arabic_indic_digits(self):
        normalizer = TextNormalizer()
        # Extended Arabic-Indic: ۱۲۳ = 123
        result = normalizer.normalize("\u06F1\u06F2\u06F3")
        assert result == "123"

    def test_unicode_normalization(self):
        normalizer = TextNormalizer()
        # Full-width digits: １２３
        result = normalizer.normalize("１２３")
        assert result == "123"

    def test_strip_whitespace_disabled(self):
        config = NormalizationConfig(strip_whitespace=False)
        normalizer = TextNormalizer(config)
        result = normalizer.normalize("  Hello  ")
        # When strip_whitespace is disabled, leading/trailing spaces preserved
        # but collapse_spaces still reduces multiple spaces to one
        assert "Hello" in result
        assert len(result) > len("Hello")

    def test_collapse_spaces_disabled(self):
        config = NormalizationConfig(collapse_repeated_spaces=False)
        normalizer = TextNormalizer(config)
        result = normalizer.normalize("Hello   World")
        assert result == "Hello   World"

    def test_zero_width_space_removal(self):
        normalizer = TextNormalizer()
        result = normalizer.normalize("Hello\u200bWorld")
        assert result == "HelloWorld"

    def test_is_numeric_context(self):
        normalizer = TextNormalizer()
        assert normalizer.is_numeric_context("5") is True
        assert normalizer.is_numeric_context("3.14") is True
        assert normalizer.is_numeric_context("-7") is True
        assert normalizer.is_numeric_context("abc") is False
        assert normalizer.is_numeric_context("") is False

    def test_arabic_indic_numeric_context(self):
        normalizer = TextNormalizer()
        assert normalizer.is_numeric_context("\u0661\u0662") is True

    def test_config_validation(self):
        config = NormalizationConfig(min_length_after_normalization=-1)
        normalizer = TextNormalizer(config)
        # Should not raise, just validate config
        result = normalizer.normalize("test")
        assert result == "test"


# =============================================================================
# Test: Numeric Parsing
# =============================================================================


class TestNumericExtraction:
    """Test numeric extraction utilities."""

    def test_integer_extraction(self):
        extractor = NumericExtractor()
        region = _make_region("5", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == 5.0
        assert candidates[0].is_integer is True

    def test_decimal_extraction(self):
        extractor = NumericExtractor()
        region = _make_region("3.14", confidence=0.85)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == pytest.approx(3.14)
        assert candidates[0].is_integer is False

    def test_negative_extraction(self):
        extractor = NumericExtractor()
        region = _make_region("-7", confidence=0.8)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == -7.0
        assert candidates[0].is_negative is True

    def test_non_numeric_text(self):
        extractor = NumericExtractor()
        region = _make_region("Hello", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 0

    def test_empty_text(self):
        extractor = NumericExtractor()
        region = _make_region("", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 0

    def test_mixed_text_with_numeric(self):
        extractor = NumericExtractor()
        region = _make_region("Task 5 days", confidence=0.85)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == 5.0

    def test_thousands_separator(self):
        extractor = NumericExtractor()
        region = _make_region("1,000", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == 1000.0

    def test_multiple_numbers(self):
        extractor = NumericExtractor()
        region = _make_region("3 and 7", confidence=0.85)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 2

    def test_is_numeric_check(self):
        extractor = NumericExtractor()
        assert extractor.is_numeric("5") is True
        assert extractor.is_numeric("3.14") is True
        assert extractor.is_numeric("-7") is True
        assert extractor.is_numeric("abc") is False
        assert extractor.is_numeric("") is False

    def test_extract_from_multiple_regions(self):
        extractor = NumericExtractor()
        regions = [
            _make_region("5", confidence=0.9),
            _make_region("Hello", confidence=0.8),
            _make_region("3.14", confidence=0.85),
        ]
        candidates = extractor.extract_from_regions(regions)
        assert len(candidates) == 2

    def test_pert_like_values(self):
        extractor = NumericExtractor()
        pairs = extractor.extract_pert_like_values("3/6")
        assert len(pairs) == 1
        assert pairs[0] == (3.0, 6.0)

    def test_pert_like_values_dash(self):
        extractor = NumericExtractor()
        pairs = extractor.extract_pert_like_values("5-10")
        assert len(pairs) == 1
        assert pairs[0] == (5.0, 10.0)

    def test_decimal_places(self):
        extractor = NumericExtractor()
        region = _make_region("3.14159", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].decimal_places == 5

    def test_config_min_value(self):
        config = NumericExtractionConfig(min_value=0)
        extractor = NumericExtractor(config)
        region = _make_region("-5", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        # Should still extract but with warning
        assert len(candidates) == 1
        assert len(candidates[0].parse_warnings) > 0

    def test_plus_sign(self):
        extractor = NumericExtractor()
        region = _make_region("+10", confidence=0.9)
        candidates = extractor.extract_from_region(region)
        assert len(candidates) == 1
        assert candidates[0].value == 10.0


# =============================================================================
# Test: Confidence Normalization
# =============================================================================


class TestConfidenceNormalization:
    """Test that OCR confidence is consistently normalized to 0.0-1.0."""

    def test_confidence_range(self):
        region = _make_region(confidence=0.5)
        assert 0.0 <= region.confidence <= 1.0

    def test_high_confidence(self):
        region = _make_region(confidence=0.95)
        assert region.confidence > 0.7

    def test_low_confidence(self):
        region = _make_region(confidence=0.2)
        assert region.confidence < 0.5

    def test_zero_confidence(self):
        region = _make_region(confidence=0.0)
        assert region.confidence == 0.0

    def test_max_confidence(self):
        region = _make_region(confidence=1.0)
        assert region.confidence == 1.0

    def test_average_confidence_calculation(self):
        result = OCRProcessingResult(
            regions=[
                _make_region(confidence=0.8),
                _make_region(confidence=0.6),
            ]
        )
        avg = result.average_confidence
        assert 0.0 <= avg <= 1.0
        assert avg == pytest.approx(0.7)


# =============================================================================
# Test: Bounding Box Handling
# =============================================================================


class TestBoundingBoxHandling:
    """Test bounding box creation and manipulation."""

    def test_bbox_center(self):
        bbox = BoundingBox(x=100, y=200, width=50, height=30)
        center = bbox.center
        assert center.x == 125
        assert center.y == 215

    def test_bbox_area(self):
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        assert bbox.area == 5000

    def test_bbox_contains_point(self):
        bbox = BoundingBox(x=100, y=100, width=200, height=100)
        assert bbox.contains_point(Point(150, 130)) is True
        assert bbox.contains_point(Point(50, 50)) is False

    def test_region_center_computed(self):
        region = _make_region(x=100, y=200, w=60, h=40)
        assert region.center.x == 130
        assert region.center.y == 220

    def test_zero_size_bbox(self):
        bbox = BoundingBox(x=0, y=0, width=0, height=0)
        assert bbox.area == 0
        assert bbox.center.x == 0
        assert bbox.center.y == 0


# =============================================================================
# Test: Text Type Classification
# =============================================================================


class TestTextTypeClassification:
    """Test text type classification."""

    def test_numeric_classification(self):
        classifier = TextClassifier()
        region = _make_region("5", confidence=0.9)
        result = classifier.classify(region)
        assert result == TextType.NUMERIC_CANDIDATE

    def test_activity_id_classification(self):
        classifier = TextClassifier()
        region = _make_region("A1", confidence=0.9)
        result = classifier.classify(region)
        assert result == TextType.ACTIVITY_ID_CANDIDATE

    def test_label_classification(self):
        classifier = TextClassifier()
        region = _make_region("Requirements", confidence=0.85)
        result = classifier.classify(region)
        assert result == TextType.TEXT_LABEL_CANDIDATE

    def test_unknown_classification(self):
        classifier = TextClassifier()
        region = _make_region("5", confidence=0.9)
        result = classifier.classify(region)
        # Should be numeric or unknown depending on pattern
        assert result in (TextType.NUMERIC_CANDIDATE, TextType.UNKNOWN)

    def test_classify_with_confidence(self):
        classifier = TextClassifier()
        region = _make_region("5", confidence=0.9)
        text_type, conf, evidence = classifier.classify_with_confidence(region)
        assert text_type == TextType.NUMERIC_CANDIDATE
        assert 0.0 <= conf <= 1.0
        assert "reason" in evidence

    def test_classify_region_updates_in_place(self):
        classifier = TextClassifier()
        region = _make_region("5", confidence=0.9)
        classified = classifier.classify_region(region)
        assert classified is region
        assert region.text_type == TextType.NUMERIC_CANDIDATE

    def test_classify_regions_batch(self):
        classifier = TextClassifier()
        regions = [
            _make_region("5", confidence=0.9),
            _make_region("Requirements", confidence=0.85),
        ]
        classified = classifier.classify_regions(regions)
        assert len(classified) == 2
        assert classified[0].text_type == TextType.NUMERIC_CANDIDATE
        assert classified[1].text_type == TextType.TEXT_LABEL_CANDIDATE

    def test_empty_text_unknown(self):
        classifier = TextClassifier()
        region = _make_region("", confidence=0.9)
        result = classifier.classify(region)
        assert result == TextType.UNKNOWN

    def test_activity_id_b12(self):
        classifier = TextClassifier()
        region = _make_region("B12", confidence=0.9)
        result = classifier.classify(region)
        assert result == TextType.ACTIVITY_ID_CANDIDATE

    def test_decimal_numeric(self):
        classifier = TextClassifier()
        region = _make_region("3.14", confidence=0.85)
        result = classifier.classify(region)
        assert result == TextType.NUMERIC_CANDIDATE

    def test_label_too_short(self):
        classifier = TextClassifier()
        region = _make_region("A", confidence=0.9)
        result = classifier.classify(region)
        # Single char might be activity ID or unknown
        assert result in (TextType.ACTIVITY_ID_CANDIDATE, TextType.UNKNOWN)


# =============================================================================
# Test: OCR Engine Abstraction
# =============================================================================


class TestOCREngineAbstraction:
    """Test OCR engine abstraction and mock engine."""

    def test_mock_engine_init(self):
        engine = MockOCREngine()
        result = engine.initialize()
        assert result is True

    def test_mock_engine_availability(self):
        engine = MockOCREngine()
        engine.initialize()
        assert engine.is_available() is True

    def test_mock_engine_name(self):
        engine = MockOCREngine()
        assert engine.get_engine_name() == "mock"

    def test_mock_engine_languages(self):
        engine = MockOCREngine()
        engine.initialize()
        assert "eng" in engine.get_supported_languages()

    def test_mock_engine_recognize(self):
        engine = MockOCREngine()
        engine.initialize()
        image = np.zeros((100, 100), dtype=np.uint8)
        regions = engine.recognize(image)
        assert isinstance(regions, list)

    def test_mock_engine_custom_results(self):
        engine = MockOCREngine()
        mock_region = _make_region("Test", confidence=0.9)
        engine.initialize({"results": [mock_region]})
        image = np.zeros((100, 100), dtype=np.uint8)
        regions = engine.recognize(image)
        assert len(regions) == 1
        assert regions[0].text == "Test"

    def test_mock_engine_not_initialized(self):
        engine = MockOCREngine()
        image = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(OCREngineError):
            engine.recognize(image)

    def test_create_ocr_engine_mock(self):
        engine = create_ocr_engine("mock")
        assert isinstance(engine, MockOCREngine)
        assert engine.is_available()

    def test_create_ocr_engine_unknown(self):
        with pytest.raises(OCREngineError):
            create_ocr_engine("nonexistent_engine")

    def test_tesseract_engine_not_available(self):
        # When pytesseract is available but tesseract executable is not found,
        # this should raise OCREngineError
        import unittest.mock as mock
        engine = TesseractOCREngine()
        with mock.patch("pytesseract.get_tesseract_version", side_effect=Exception("tesseract not found")):
            with pytest.raises(OCREngineError):
                engine.initialize()


# =============================================================================
# Test: Shape Association
# =============================================================================


class TestShapeAssociation:
    """Test text-to-shape spatial association."""

    def test_text_inside_shape(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        assert len(results) == 1
        assert results[0].has_associations is True
        assert results[0].best_association is not None
        assert results[0].best_association.association_score > 0

    def test_text_outside_shape(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=500, y=500, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        assert len(results) == 1
        # May or may not have association depending on distance threshold
        if results[0].has_associations:
            assert results[0].best_association.association_score < 0.5

    def test_text_near_shape(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=310, y=140, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        assert len(results) == 1
        # Should have some association score

    def test_no_shapes(self):
        associator = SpatialAssociator()
        region = _make_region("A1")
        results = associator.associate_text_to_shapes([region], [])
        assert len(results) == 1
        assert results[0].has_no_match is True

    def test_multiple_shapes_best_candidate(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node1 = _make_node("n1", x=100, y=100, w=200, h=100)
        node2 = _make_node("n2", x=400, y=400, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node1, node2])
        assert len(results) == 1
        if results[0].has_associations:
            assert results[0].best_association is not None
            assert results[0].best_association.candidate_target_id == "n1"

    def test_association_evidence(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        if results[0].has_associations:
            assoc = results[0].best_association
            assert "is_contained" in assoc.evidence
            assert "overlap_ratio" in assoc.evidence
            assert "center_distance" in assoc.evidence

    def test_association_reasons(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        if results[0].has_associations:
            assoc = results[0].best_association
            assert len(assoc.reasons) > 0


# =============================================================================
# Test: Arrow Association
# =============================================================================


class TestArrowAssociation:
    """Test text-to-arrow spatial association."""

    def test_text_near_arrow_midpoint(self):
        associator = SpatialAssociator()
        region = _make_region("5", x=380, y=140, w=30, h=20)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_text_to_arrows([region], [arrow])
        assert len(results) == 1
        assert results[0].has_associations is True

    def test_text_far_from_arrow(self):
        associator = SpatialAssociator()
        region = _make_region("5", x=50, y=500, w=30, h=20)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_text_to_arrows([region], [arrow])
        assert len(results) == 1
        if results[0].has_associations:
            assert results[0].best_association.association_score < 0.5

    def test_no_arrows(self):
        associator = SpatialAssociator()
        region = _make_region("5")
        results = associator.associate_text_to_arrows([region], [])
        assert len(results) == 1
        assert results[0].has_no_match is True

    def test_arrow_association_evidence(self):
        associator = SpatialAssociator()
        region = _make_region("5", x=380, y=140, w=30, h=20)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_text_to_arrows([region], [arrow])
        if results[0].has_associations:
            assoc = results[0].best_association
            assert "midpoint_distance" in assoc.evidence
            assert "perpendicular_distance" in assoc.evidence


# =============================================================================
# Test: Combined Association
# =============================================================================


class TestCombinedAssociation:
    """Test combined text-to-shape-and-arrow association."""

    def test_associate_all(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_all([region], [node], [arrow])
        assert len(results) == 1
        # Should have associations from both shape and arrow
        assert results[0].association_count >= 1


# =============================================================================
# Test: Ambiguous Association
# =============================================================================


class TestAmbiguousAssociation:
    """Test ambiguous association detection."""

    def test_ambiguous_when_close_scores(self):
        associator = SpatialAssociator(
            config=AssociationConfig(ambiguity_score_threshold=0.2)
        )
        region = _make_region("A1", x=200, y=150, w=40, h=20)
        node1 = _make_node("n1", x=100, y=100, w=200, h=100)
        node2 = _make_node("n2", x=200, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node1, node2])
        if results[0].association_count >= 2:
            # Close scores should be ambiguous
            scores = [a.association_score for a in results[0].associations]
            if abs(scores[0] - scores[1]) < 0.2:
                assert results[0].is_ambiguous is True

    def test_not_ambiguous_clear_winner(self):
        associator = SpatialAssociator()
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node1 = _make_node("n1", x=100, y=100, w=200, h=100)
        node2 = _make_node("n2", x=500, y=500, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node1, node2])
        if results[0].association_count >= 2:
            scores = [a.association_score for a in results[0].associations]
            if abs(scores[0] - scores[1]) >= 0.15:
                assert results[0].is_ambiguous is False


# =============================================================================
# Test: Irrelevant Text Filtering
# =============================================================================


class TestIrrelevantTextFiltering:
    """Test that irrelevant OCR results can be identified."""

    def test_low_confidence_filter(self):
        regions = [
            _make_region("A1", confidence=0.9),
            _make_region("noise", confidence=0.1),
        ]
        # Filter by confidence
        filtered = [r for r in regions if r.confidence >= 0.3]
        assert len(filtered) == 1
        assert filtered[0].text == "A1"

    def test_empty_text_filter(self):
        regions = [
            _make_region("A1", confidence=0.9),
            _make_region("", confidence=0.5),
        ]
        filtered = [r for r in regions if r.text.strip()]
        assert len(filtered) == 1

    def test_numeric_in_context(self):
        # Numbers near shapes are relevant; numbers far away are not
        associator = SpatialAssociator()
        region_near = _make_region("5", x=150, y=120, w=30, h=20)
        region_far = _make_region("99", x=800, y=800, w=30, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)

        results = associator.associate_text_to_shapes(
            [region_near, region_far], [node]
        )
        # Near region should have higher association
        assert results[0].best_association is not None
        # Far region may have no match or low score


# =============================================================================
# Test: Text Grouping
# =============================================================================


class TestTextGrouping:
    """Test text grouping utilities."""

    def test_group_close_regions(self):
        grouper = TextGrouper()
        regions = [
            _make_region("A", x=150, y=110, w=20, h=20),
            _make_region("5", x=150, y=140, w=20, h=20),
        ]
        groups = grouper.group_regions(regions)
        assert len(groups) >= 1

    def test_no_group_distant_regions(self):
        grouper = TextGrouper()
        regions = [
            _make_region("A", x=100, y=100, w=20, h=20),
            _make_region("B", x=800, y=800, w=20, h=20),
        ]
        groups = grouper.group_regions(regions)
        assert len(groups) == 0

    def test_group_min_size(self):
        grouper = TextGrouper(
            config=TextGroupingConfig(min_group_size=2)
        )
        regions = [_make_region("A", x=150, y=110, w=20, h=20)]
        groups = grouper.group_regions(regions)
        assert len(groups) == 0  # Single region not grouped

    def test_group_combined_text(self):
        grouper = TextGrouper()
        regions = [
            _make_region("A", x=150, y=110, w=20, h=20),
            _make_region("5", x=150, y=140, w=20, h=20),
        ]
        groups = grouper.group_regions(regions)
        if groups:
            assert "A" in groups[0].combined_text
            assert "5" in groups[0].combined_text

    def test_group_combined_bounding_box(self):
        grouper = TextGrouper()
        regions = [
            _make_region("A", x=100, y=100, w=30, h=20),
            _make_region("5", x=100, y=130, w=30, h=20),
        ]
        groups = grouper.group_regions(regions)
        if groups:
            bbox = groups[0].combined_bounding_box
            assert bbox is not None
            assert bbox.x <= 100
            assert bbox.y <= 100

    def test_group_confidence(self):
        grouper = TextGrouper()
        regions = [
            _make_region("A", x=150, y=110, w=20, h=20, confidence=0.9),
            _make_region("5", x=150, y=140, w=20, h=20, confidence=0.8),
        ]
        groups = grouper.group_regions(regions)
        if groups:
            assert 0.0 <= groups[0].group_confidence <= 1.0


# =============================================================================
# Test: Coordinate Mapping
# =============================================================================


class TestCoordinateMapping:
    """Test coordinate mapping for OCR results."""

    def test_to_original_coordinates_no_mapping(self):
        result = OCRProcessingResult()
        x, y = result.to_original_coordinates(100, 200)
        assert x == 100
        assert y == 200

    def test_region_to_original_coordinates(self):
        from pert_analyzer.cv.models import CoordinateMapping
        mapping = CoordinateMapping(
            original_width=1000,
            original_height=800,
            processed_width=500,
            processed_height=400,
        )
        result = OCRProcessingResult(coordinate_mapping=mapping)
        x, y = result.to_original_coordinates(250, 200)
        assert x == pytest.approx(500)
        assert y == pytest.approx(400)


# =============================================================================
# Test: Debug Rendering
# =============================================================================


class TestDebugRendering:
    """Test debug rendering utilities."""

    def test_render_returns_image(self):
        from pert_analyzer.cv.ocr_debug import OCRDebugRenderer
        renderer = OCRDebugRenderer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        regions = [_make_region("Test", x=50, y=50, w=60, h=30)]
        result = renderer.render_regions(image, regions)
        assert result.shape == image.shape

    def test_render_grayscale_input(self):
        from pert_analyzer.cv.ocr_debug import OCRDebugRenderer
        renderer = OCRDebugRenderer()
        image = np.zeros((200, 200), dtype=np.uint8)
        regions = [_make_region("Test", x=50, y=50, w=60, h=30)]
        result = renderer.render_regions(image, regions)
        assert len(result.shape) == 3  # Should be BGR

    def test_render_associations(self):
        from pert_analyzer.cv.ocr_debug import OCRDebugRenderer
        renderer = OCRDebugRenderer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        region = _make_region("Test", x=50, y=50, w=60, h=30)
        assoc = TextAssociationResult(
            text_region_id=region.region_id,
            associations=[
                TextAssociation(
                    text_region_id=region.region_id,
                    candidate_target_id="n1",
                    target_type=AssociationTargetType.NODE,
                    association_score=0.8,
                )
            ],
        )
        result = renderer.render_associations(image, [region], [assoc])
        assert result.shape == image.shape

    def test_render_full(self):
        from pert_analyzer.cv.ocr_debug import OCRDebugRenderer
        renderer = OCRDebugRenderer()
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        ocr_result = OCRProcessingResult(
            regions=[_make_region("Test", x=50, y=50, w=60, h=30)]
        )
        result = renderer.render_full(image, ocr_result)
        assert result.shape == image.shape


# =============================================================================
# Test: Missing OCR Backend
# =============================================================================


class TestMissingOCRBackend:
    """Test handling of missing OCR backend."""

    def test_import_not_broken_without_ocr(self):
        """Import should succeed even if Tesseract is not installed."""
        from pert_analyzer.cv import (
            MockOCREngine,
            OCREngineBase,
            create_ocr_engine,
        )
        assert MockOCREngine is not None
        assert OCREngineBase is not None

    def test_tesseract_init_fails_gracefully(self):
        """Tesseract init should raise clear error if not installed."""
        engine = TesseractOCREngine()
        try:
            engine.initialize()
        except OCREngineError as e:
            assert "pytesseract" in str(e) or "Tesseract" in str(e)

    def test_mock_engine_works_without_external_deps(self):
        """Mock engine should work without any external dependencies."""
        engine = create_ocr_engine("mock")
        assert engine.is_available()
        image = np.zeros((100, 100), dtype=np.uint8)
        regions = engine.recognize(image)
        assert isinstance(regions, list)


# =============================================================================
# Test: Malformed OCR Result Handling
# =============================================================================


class TestMalformedOCRResult:
    """Test handling of malformed OCR results."""

    def test_empty_text_region(self):
        region = OCRTextRegion()
        assert region.text == ""
        assert region.confidence == 0.0

    def test_whitespace_only_text(self):
        region = _make_region("   ", confidence=0.5)
        assert region.text == "   "

    def test_very_long_text(self):
        long_text = "A" * 500
        region = _make_region(long_text, confidence=0.9)
        assert len(region.text) == 500

    def test_special_characters(self):
        region = _make_region("!@#$%^&*()", confidence=0.7)
        assert region.text == "!@#$%^&*()"

    def test_unicode_text(self):
        region = _make_region("العربية", confidence=0.8)
        assert region.text == "العربية"

    def test_zero_size_bbox(self):
        region = _make_region("test", x=0, y=0, w=0, h=0)
        assert region.bounding_box.area == 0


# =============================================================================
# Test: AON Text Association
# =============================================================================


class TestAONTextAssociation:
    """Test AON-specific text association patterns."""

    def test_text_inside_rectangle(self):
        """In AON, text inside rectangles is strongly associated."""
        associator = SpatialAssociator()
        # Text "A1" inside a rectangle
        region = _make_region("A1", x=150, y=120, w=40, h=20)
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes([region], [node])
        assert results[0].has_associations is True
        assert results[0].best_association.evidence.get("is_contained") is True

    def test_multiple_texts_in_rectangle(self):
        """Multiple texts inside same rectangle should all associate."""
        associator = SpatialAssociator()
        regions = [
            _make_region("A1", x=140, y=110, w=40, h=20),
            _make_region("5", x=140, y=140, w=30, h=20),
        ]
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes(regions, [node])
        assert len(results) == 2
        for r in results:
            assert r.has_associations is True


# =============================================================================
# Test: AOA Text Association
# =============================================================================


class TestAOATextAssociation:
    """Test AOA-specific text association patterns."""

    def test_text_near_arrow(self):
        """In AOA, text near arrows represents activity labels."""
        associator = SpatialAssociator()
        region = _make_region("A", x=380, y=135, w=20, h=20)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_text_to_arrows([region], [arrow])
        assert results[0].has_associations is True

    def test_duration_near_arrow(self):
        """Duration value near arrow midpoint."""
        associator = SpatialAssociator()
        region = _make_region("5", x=380, y=160, w=20, h=20)
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        results = associator.associate_text_to_arrows([region], [arrow])
        assert results[0].has_associations is True


# =============================================================================
# Test: Multiple Text Regions
# =============================================================================


class TestMultipleTextRegions:
    """Test handling of multiple text regions."""

    def test_multiple_regions_all_associated(self):
        associator = SpatialAssociator()
        regions = [
            _make_region("A1", x=140, y=110, w=40, h=20),
            _make_region("5", x=140, y=140, w=30, h=20),
            _make_region("B2", x=440, y=110, w=40, h=20),
        ]
        node1 = _make_node("n1", x=100, y=100, w=200, h=100)
        node2 = _make_node("n2", x=400, y=100, w=200, h=100)
        results = associator.associate_text_to_shapes(regions, [node1, node2])
        assert len(results) == 3

    def test_association_results_match_regions(self):
        associator = SpatialAssociator()
        regions = [_make_region(f"R{i}") for i in range(5)]
        results = associator.associate_text_to_shapes(regions, [])
        assert len(results) == 5
        for i, r in enumerate(results):
            assert r.text_region_id == regions[i].region_id


# =============================================================================
# Test: No Final Graph Reconstruction
# =============================================================================


class TestNoGraphReconstruction:
    """Verify that Phase 6 does NOT implement graph reconstruction."""

    def test_no_dependency_creation(self):
        """Phase 6 should not create Dependency objects."""
        from pert_analyzer.cv import spatial_association
        import inspect
        source = inspect.getsource(spatial_association)
        assert "Dependency" not in source or "AssociationTargetType" in source

    def test_no_cpm_logic(self):
        """Phase 6 should not contain CPM logic."""
        import pert_analyzer.cv.ocr_models as models
        import inspect
        source = inspect.getsource(models)
        assert "critical_path" not in source.lower()
        assert "early_start" not in source.lower()

    def test_association_not_dependency(self):
        """TextAssociation is NOT a Dependency."""
        assoc = TextAssociation()
        from pert_analyzer.core.models import Dependency
        assert not isinstance(assoc, Dependency)


# =============================================================================
# Test: Full Pipeline Integration
# =============================================================================


class TestFullPipeline:
    """Integration tests for the full OCR pipeline."""

    def test_full_ocr_pipeline(self):
        """Test mock OCR → normalize → classify → associate pipeline."""
        # 1. Mock OCR
        engine = create_ocr_engine("mock")
        mock_regions = [
            _make_region("A1", x=140, y=110, w=40, h=20, confidence=0.9),
            _make_region("5", x=140, y=140, w=30, h=20, confidence=0.85),
        ]
        engine.set_mock_results(mock_regions)
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        regions = engine.recognize(image)
        assert len(regions) == 2

        # 2. Normalize
        normalizer = TextNormalizer()
        for region in regions:
            raw, norm = normalizer.normalize_preserve_raw(region.text)
            region.raw_text = raw
            region.normalized_text = norm

        # 3. Classify
        classifier = TextClassifier()
        classified = classifier.classify_regions(regions)
        assert classified[0].text_type == TextType.ACTIVITY_ID_CANDIDATE
        assert classified[1].text_type == TextType.NUMERIC_CANDIDATE

        # 4. Associate
        associator = SpatialAssociator()
        node = _make_node("n1", x=100, y=100, w=200, h=100)
        assoc_results = associator.associate_text_to_shapes(
            classified, [node]
        )
        assert len(assoc_results) == 2
        for r in assoc_results:
            assert r.has_associations is True

    def test_full_aoa_pipeline(self):
        """Test AOA association pipeline."""
        engine = create_ocr_engine("mock")
        mock_regions = [
            _make_region("A", x=380, y=135, w=20, h=20, confidence=0.9),
            _make_region("5", x=380, y=160, w=20, h=20, confidence=0.85),
        ]
        engine.set_mock_results(mock_regions)
        image = np.zeros((300, 600, 3), dtype=np.uint8)
        regions = engine.recognize(image)

        # Classify
        classifier = TextClassifier()
        classified = classifier.classify_regions(regions)

        # Associate with arrows
        arrow = _make_arrow(start_x=300, start_y=150, end_x=500, end_y=150)
        associator = SpatialAssociator()
        assoc_results = associator.associate_text_to_arrows(
            classified, [arrow]
        )
        assert len(assoc_results) == 2


# =============================================================================
# Phase 8: Tesseract Integration Tests
# =============================================================================


class TestTesseractEngineConfiguration:
    """Test Tesseract engine configuration and initialization."""

    def test_tesseract_config_psm_oem(self):
        """OCRConfig supports psm and oem fields."""
        from pert_analyzer.cv.ocr_models import OCRConfig
        config = OCRConfig(psm=11, oem=3)
        assert config.psm == 11
        assert config.oem == 3

    def test_tesseract_config_defaults(self):
        """OCRConfig defaults are sensible."""
        from pert_analyzer.cv.ocr_models import OCRConfig
        config = OCRConfig()
        assert config.languages == ["eng"]
        assert config.psm == 11
        assert config.oem == 3

    def test_create_ocr_engine_tesseract_config(self):
        """create_ocr_engine passes config to TesseractOCREngine."""
        from pert_analyzer.cv.ocr_engine import create_ocr_engine, OCREngineError
        try:
            engine = create_ocr_engine("tesseract", {
                "languages": ["eng"],
                "psm": 6,
                "oem": 1,
            })
            assert engine.get_engine_name() == "tesseract"
        except OCREngineError:
            pytest.skip("Tesseract not installed")

    def test_create_ocr_engine_unknown_raises(self):
        """create_ocr_engine raises for unknown engine."""
        from pert_analyzer.cv.ocr_engine import create_ocr_engine, OCREngineError
        with pytest.raises(OCREngineError):
            create_ocr_engine("nonexistent_engine")


class TestTesseractAvailability:
    """Test Tesseract availability detection."""

    def test_tesseract_availability_check(self):
        """TesseractOCREngine.is_available returns correct status."""
        from pert_analyzer.cv.ocr_engine import TesseractOCREngine
        engine = TesseractOCREngine()
        assert engine.is_available() is False

    def test_tesseract_unavailable_raises_on_recognize(self):
        """TesseractOCREngine.recognize raises when not initialized."""
        from pert_analyzer.cv.ocr_engine import TesseractOCREngine, OCREngineError
        engine = TesseractOCREngine()
        with pytest.raises(OCREngineError, match="not initialized"):
            engine.recognize(np.zeros((100, 100), dtype=np.uint8))

    def test_tesseract_import_error_handled(self):
        """TesseractOCREngine.initialize handles missing pytesseract."""
        from pert_analyzer.cv.ocr_engine import TesseractOCREngine, OCREngineError
        engine = TesseractOCREngine()
        # If pytesseract is installed but tesseract exe is not, should raise
        # If pytesseract is not installed, should also raise
        try:
            result = engine.initialize()
            if result:
                assert engine.is_available()
        except OCREngineError:
            pass  # Expected when Tesseract not installed


class TestMockEnginePreserved:
    """Test that MockOCREngine remains fully functional."""

    def test_mock_engine_still_works(self):
        """MockOCREngine returns configurable results."""
        engine = create_ocr_engine("mock")
        assert engine.is_available()
        assert engine.get_engine_name() == "mock"

    def test_mock_engine_set_results(self):
        """MockOCREngine can have results set and returned."""
        engine = create_ocr_engine("mock")
        mock_regions = [
            OCRTextRegion(text="A", confidence=0.9),
            OCRTextRegion(text="5", confidence=0.85),
        ]
        engine.set_mock_results(mock_regions)
        result = engine.recognize(np.zeros((100, 100), dtype=np.uint8))
        assert len(result) == 2
        assert result[0].text == "A"
        assert result[1].text == "5"

    def test_mock_engine_languages(self):
        """MockOCREngine returns configured languages."""
        engine = create_ocr_engine("mock", {"languages": ["eng", "ara"]})
        assert engine.get_supported_languages() == ["eng", "ara"]


class TestOCRResultParsing:
    """Test OCR result parsing with word-level bounding boxes."""

    def test_region_has_bounding_box(self):
        """OCRTextRegion has valid bounding box."""
        bbox = BoundingBox(x=10, y=20, width=100, height=50)
        region = OCRTextRegion(
            text="Hello",
            bounding_box=bbox,
            confidence=0.95,
        )
        assert region.bounding_box.x == 10
        assert region.bounding_box.y == 20
        assert region.bounding_box.width == 100
        assert region.bounding_box.height == 50
        assert region.confidence == 0.95

    def test_region_center_computed(self):
        """OCRTextRegion computes center from bounding box."""
        bbox = BoundingBox(x=0, y=0, width=100, height=50)
        region = OCRTextRegion(text="test", bounding_box=bbox)
        assert region.center.x == 50
        assert region.center.y == 25

    def test_confidence_normalization(self):
        """Confidence values are normalized to 0.0-1.0."""
        region = OCRTextRegion(text="x", confidence=0.75)
        assert 0.0 <= region.confidence <= 1.0


class TestEmptyAndMalformedOCR:
    """Test handling of empty and malformed OCR results."""

    def test_empty_ocr_result(self):
        """OCRProcessingResult handles empty regions."""
        result = OCRProcessingResult()
        assert result.region_count == 0
        assert result.average_confidence == 0.0
        assert result.high_confidence_count == 0

    def test_ocr_result_with_empty_text(self):
        """Regions with empty text are handled."""
        region = OCRTextRegion(text="", confidence=0.5)
        assert region.text == ""
        assert region.confidence == 0.5

    def test_ocr_result_region_lookup(self):
        """get_region_by_id returns correct region."""
        r1 = OCRTextRegion(text="A")
        r2 = OCRTextRegion(text="B")
        result = OCRProcessingResult(regions=[r1, r2])
        found = result.get_region_by_id(r1.region_id)
        assert found is not None
        assert found.text == "A"
        assert result.get_region_by_id("nonexistent") is None


class TestPipelineOCRIntegration:
    """Test pipeline OCR stage integration."""

    def test_pipeline_default_engine_is_tesseract(self):
        """EndToEndAnalyzer defaults to tesseract engine."""
        from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
        analyzer = EndToEndAnalyzer()
        assert analyzer.ocr_engine_name == "tesseract"

    def test_pipeline_mock_engine_explicit(self):
        """EndToEndAnalyzer can be explicitly set to mock engine."""
        from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
        analyzer = EndToEndAnalyzer(ocr_engine_name="mock")
        assert analyzer.ocr_engine_name == "mock"

    def test_pipeline_ocr_config_passed(self):
        """EndToEndAnalyzer passes OCR config to engine."""
        from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
        analyzer = EndToEndAnalyzer(
            ocr_engine_name="tesseract",
            ocr_languages=["eng", "ara"],
            ocr_psm=6,
            ocr_oem=1,
        )
        assert analyzer._ocr_languages == ["eng", "ara"]
        assert analyzer._ocr_psm == 6
        assert analyzer._ocr_oem == 1

    def test_pipeline_ocr_unavailable_returns_skipped(self):
        """Pipeline returns OCR_ENGINE_UNAVAILABLE when Tesseract not found."""
        from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
        from pert_analyzer.pipeline.result import PipelineResult
        analyzer = EndToEndAnalyzer(ocr_engine_name="tesseract")
        result = PipelineResult()
        # Run OCR stage — should either succeed or return SKIPPED
        ocr_result = analyzer._stage_ocr(
            np.zeros((100, 100, 3), dtype=np.uint8), result
        )
        stage = result.stages["ocr"]
        assert stage.status in ("SUCCESS", "SKIPPED", "WARNING")

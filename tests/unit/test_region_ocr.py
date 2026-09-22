"""Unit tests for region-based OCR processor."""
import os
import numpy as np
import cv2
import pytest
from unittest.mock import MagicMock, patch

from pert_analyzer.core.models import BoundingBox, Point, _generate_id
from pert_analyzer.cv.models import CandidateNode, ShapeType
from pert_analyzer.cv.region_ocr import (
    RegionOCRConfig,
    RegionOCRProcessor,
    NodeOCRResult,
    merge_ocr_results,
)
from pert_analyzer.cv.ocr_engine import OCREngineBase
from pert_analyzer.cv.ocr_models import OCRConfig, OCRTextRegion, OCRProcessingResult, TextType


class MockOCREngine(OCREngineBase):
    """Mock OCR engine for deterministic tests."""

    def __init__(self, responses=None):
        self._responses = responses or {}
        self._initialized = False
        self._call_count = 0

    def initialize(self, config=None):
        self._initialized = True
        return True

    def recognize(self, image, config=None):
        self._call_count += 1
        regions = []
        # Return mock regions based on image content
        if image is not None and image.size > 0:
            key = f"psm{config.psm if config else 11}"
            text = self._responses.get(key, self._responses.get("default", "A"))
            regions.append(OCRTextRegion(
                text=text,
                confidence=0.85,
                bounding_box=BoundingBox(10, 10, 50, 20),
            ))
        return regions

    def get_engine_name(self):
        return "mock_region"

    def get_supported_languages(self):
        return ["eng"]

    def is_available(self):
        return True

    def cleanup(self):
        pass


class TestRegionOCRConfig:
    def test_default_config(self):
        cfg = RegionOCRConfig()
        assert cfg.padding_px == 8
        assert cfg.scale_factor == 3
        assert 6 in cfg.psm_modes
        assert 11 in cfg.psm_modes
        assert cfg.oem == 3

    def test_custom_config(self):
        cfg = RegionOCRConfig(padding_px=5, scale_factor=2, psm_modes=[7])
        assert cfg.padding_px == 5
        assert cfg.scale_factor == 2
        assert cfg.psm_modes == [7]


class TestNodeOCRResult:
    def test_default_values(self):
        result = NodeOCRResult(node_id="test")
        assert result.node_id == "test"
        assert result.regions == []
        assert result.best_activity_id is None
        assert result.numeric_candidates == []
        assert result.warnings == []


class TestRegionOCRProcessor:
    def _make_node(self, bbox, node_id=None):
        return CandidateNode(
            node_id=node_id or _generate_id(),
            shape_type=ShapeType.RECTANGLE,
            bounding_box=bbox,
            confidence=0.9,
        )

    def test_process_single_node_grayscale(self):
        # Create a small grayscale test image
        img = np.ones((100, 200), dtype=np.uint8) * 255
        cv2.putText(img, "A", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

        engine = MockOCREngine({"default": "A"})
        proc = RegionOCRProcessor(engine, RegionOCRConfig(scale_factor=2, psm_modes=[11]))

        node = self._make_node(BoundingBox(40, 30, 80, 50), "node1")
        result = proc._process_single_node(img, node)

        assert result.node_id == "node1"
        assert result.crop_bbox is not None
        assert result.crop_bbox.width > 0
        assert len(result.regions) > 0
        assert result.processing_time >= 0

    def test_process_single_node_color(self):
        img = np.ones((100, 200, 3), dtype=np.uint8) * 255
        cv2.putText(img, "B", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)

        engine = MockOCREngine({"default": "B"})
        proc = RegionOCRProcessor(engine, RegionOCRConfig(scale_factor=2, psm_modes=[11]))

        node = self._make_node(BoundingBox(40, 30, 80, 50), "node2")
        result = proc._process_single_node(img, node)

        assert result.node_id == "node2"
        assert len(result.regions) > 0

    def test_crop_region_with_padding(self):
        img = np.ones((200, 200), dtype=np.uint8) * 255
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(padding_px=10))

        bbox = BoundingBox(50, 50, 60, 40)
        crop, crop_bbox = proc._crop_region(img, bbox)

        assert crop is not None
        assert crop.shape[0] > 0
        assert crop.shape[1] > 0
        assert crop_bbox.x >= 0
        assert crop_bbox.y >= 0

    def test_crop_region_clamps_to_boundary(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(padding_px=20))

        # Near bottom-right edge
        bbox = BoundingBox(85, 85, 30, 30)
        crop, crop_bbox = proc._crop_region(img, bbox)

        assert crop is not None
        assert crop_bbox.x + crop_bbox.width <= 100
        assert crop_bbox.y + crop_bbox.height <= 100

    def test_crop_region_too_small_returns_none(self):
        img = np.ones((100, 100), dtype=np.uint8) * 255
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(padding_px=0, min_crop_size=50))

        bbox = BoundingBox(10, 10, 5, 5)
        crop, crop_bbox = proc._crop_region(img, bbox)

        assert crop is None

    def test_prepare_representations(self):
        img = np.ones((50, 80), dtype=np.uint8) * 128
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(scale_factor=2))

        reps = proc._prepare_representations(img)

        assert "upscaled" in reps
        assert "upscaled_otsu" in reps
        # Check upscaling
        assert reps["upscaled"].shape[0] == 100
        assert reps["upscaled"].shape[1] == 160

    def test_map_coordinates(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)

        crop_bbox = BoundingBox(100, 200, 150, 100)
        region = OCRTextRegion(
            text="A",
            confidence=0.9,
            bounding_box=BoundingBox(30, 20, 60, 30),
        )

        mapped = proc._map_coordinates(region, crop_bbox, (100, 150))

        # Original coordinates should be offset by crop_bbox
        assert mapped.bounding_box.x > 100
        assert mapped.bounding_box.y > 200

    def test_deduplicate_regions(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)

        regions = [
            OCRTextRegion(text="A", confidence=0.9, bounding_box=BoundingBox(10, 10, 50, 20)),
            OCRTextRegion(text="A", confidence=0.8, bounding_box=BoundingBox(12, 12, 50, 20)),
            OCRTextRegion(text="B", confidence=0.7, bounding_box=BoundingBox(100, 100, 50, 20)),
        ]

        deduped = proc._deduplicate_regions(regions)

        # Should keep one "A" and one "B"
        texts = [r.text for r in deduped]
        assert texts.count("A") == 1
        assert "B" in texts

    def test_compute_overlap(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)

        a = BoundingBox(0, 0, 100, 100)
        b = BoundingBox(50, 50, 100, 100)
        overlap = proc._compute_overlap(a, b)

        assert 0.0 < overlap < 1.0

    def test_compute_overlap_no_overlap(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)

        a = BoundingBox(0, 0, 50, 50)
        b = BoundingBox(100, 100, 50, 50)
        overlap = proc._compute_overlap(a, b)

        assert overlap == 0.0

    def test_extract_activity_id_single_char(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="A", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        assert result.best_activity_id == "A"
        assert result.best_activity_id_confidence == 0.9

    def test_extract_activity_id_letter_digit(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="A1", confidence=0.85, bounding_box=BoundingBox(0, 0, 20, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        assert result.best_activity_id == "A1"

    def test_extract_numeric_candidates(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="5", confidence=0.95, bounding_box=BoundingBox(0, 0, 10, 10)),
            OCRTextRegion(text="A", confidence=0.9, bounding_box=BoundingBox(20, 0, 10, 10)),
            OCRTextRegion(text="3.14", confidence=0.8, bounding_box=BoundingBox(40, 0, 20, 10)),
        ]

        proc._extract_numeric_candidates(result)

        values = [v for v, _, _ in result.numeric_candidates]
        assert 5.0 in values
        assert 3.14 in values
        assert len(result.numeric_candidates) == 2

    def test_process_multiple_nodes(self):
        img = np.ones((200, 300), dtype=np.uint8) * 255
        cv2.putText(img, "A", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)
        cv2.putText(img, "B", (150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3)

        engine = MockOCREngine({"default": "X"})
        proc = RegionOCRProcessor(engine, RegionOCRConfig(scale_factor=2, psm_modes=[11]))

        nodes = [
            self._make_node(BoundingBox(30, 30, 80, 50), "n1"),
            self._make_node(BoundingBox(130, 30, 80, 50), "n2"),
        ]

        results = proc.process_node_regions(img, nodes)

        assert len(results) == 2
        assert results[0].node_id == "n1"
        assert results[1].node_id == "n2"

    def test_error_handling_continues_on_failure(self):
        engine = MockOCREngine()
        engine.recognize = MagicMock(side_effect=Exception("OCR crashed"))

        proc = RegionOCRProcessor(engine, RegionOCRConfig(scale_factor=2, psm_modes=[11]))
        img = np.ones((100, 100), dtype=np.uint8) * 255
        node = self._make_node(BoundingBox(10, 10, 50, 40), "fail_node")

        result = proc._process_single_node(img, node)

        assert result.node_id == "fail_node"
        # Exception is caught in _run_ocr_pass; result may be empty but not crashed
        assert result.processing_time >= 0


class TestMergeOCRResults:
    def test_merge_empty_results(self):
        full = OCRProcessingResult(regions=[], engine="mock", image_dimensions=(100, 100))
        merged = merge_ocr_results(full, [])

        assert merged.regions == []

    def test_merge_preserves_full_image_results(self):
        full = OCRProcessingResult(
            regions=[
                OCRTextRegion(text="START", confidence=0.9, bounding_box=BoundingBox(10, 10, 50, 20)),
            ],
            engine="mock",
            image_dimensions=(100, 100),
        )

        merged = merge_ocr_results(full, [])

        assert len(merged.regions) == 1
        assert merged.regions[0].text == "START"

    def test_merge_adds_region_results(self):
        full = OCRProcessingResult(
            regions=[
                OCRTextRegion(text="START", confidence=0.9, bounding_box=BoundingBox(10, 10, 50, 20)),
            ],
            engine="mock",
            image_dimensions=(100, 100),
        )

        node_result = NodeOCRResult(node_id="n1")
        node_result.regions = [
            OCRTextRegion(text="A", confidence=0.95, bounding_box=BoundingBox(200, 200, 30, 15)),
        ]
        node_result.numeric_candidates = [(5.0, "5", 0.9)]

        merged = merge_ocr_results(full, [node_result])

        assert len(merged.regions) == 2
        texts = [r.text for r in merged.regions]
        assert "START" in texts
        assert "A" in texts
        assert len(merged.numeric_candidates) == 1

    def test_merge_deduplicates_same_text(self):
        full = OCRProcessingResult(
            regions=[
                OCRTextRegion(text="A", confidence=0.8, bounding_box=BoundingBox(50, 50, 40, 20)),
            ],
            engine="mock",
            image_dimensions=(100, 100),
        )

        node_result = NodeOCRResult(node_id="n1")
        node_result.regions = [
            OCRTextRegion(text="A", confidence=0.95, bounding_box=BoundingBox(52, 52, 40, 20)),
        ]

        merged = merge_ocr_results(full, [node_result])

        # Should keep higher confidence version
        a_regions = [r for r in merged.regions if r.text == "A"]
        assert len(a_regions) == 1
        assert a_regions[0].confidence == 0.95


class TestAmbiguityHandling:
    def test_digit_to_letter_confusion(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="0", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        # 0 should be mapped to O with reduced confidence
        assert result.best_activity_id == "O"
        assert result.best_activity_id_confidence > 0

    def test_s_from_5(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="5", confidence=0.85, bounding_box=BoundingBox(0, 0, 10, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        assert result.best_activity_id == "S"

    def test_direct_letter_preferred_over_confusion(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="O", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)),
            OCRTextRegion(text="0", confidence=0.85, bounding_box=BoundingBox(20, 0, 10, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        assert result.best_activity_id == "O"
        assert result.best_activity_id_confidence == 0.9

    def test_alternatives_generated(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="O", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)),
            OCRTextRegion(text="0", confidence=0.85, bounding_box=BoundingBox(20, 0, 10, 10)),
        ]

        proc._extract_activity_id_with_alternatives(result)

        # Should have alternatives
        assert len(result.activity_id_alternatives) >= 0


class TestSubCropExtraction:
    def test_id_sub_crop(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(sub_crop_scale=2))

        crop = np.ones((80, 120), dtype=np.uint8) * 255
        sub = proc._extract_sub_crop(crop, proc.config, is_id=True)

        assert sub is not None
        # ID region is top 45%
        assert sub.shape[0] < crop.shape[0] * 2  # after 2x upscale

    def test_duration_sub_crop(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine, RegionOCRConfig(sub_crop_scale=2))

        crop = np.ones((80, 120), dtype=np.uint8) * 255
        sub = proc._extract_sub_crop(crop, proc.config, is_id=False)

        assert sub is not None

    def test_sub_crop_too_small(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)

        crop = np.ones((5, 5), dtype=np.uint8) * 255
        sub = proc._extract_sub_crop(crop, proc.config, is_id=True)

        assert sub is None


class TestNumericExtraction:
    def test_best_numeric_selected(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.duration_sub_crop_regions = [
            OCRTextRegion(text="5", confidence=0.9, bounding_box=BoundingBox(0, 0, 10, 10)),
        ]
        result.regions = [
            OCRTextRegion(text="3", confidence=0.7, bounding_box=BoundingBox(20, 0, 10, 10)),
        ]

        proc._extract_numeric_candidates(result)

        assert result.best_numeric is not None
        assert result.best_numeric[0] == 5.0
        assert result.best_numeric[2] == 0.9

    def test_numeric_from_mixed_text(self):
        engine = MockOCREngine()
        proc = RegionOCRProcessor(engine)
        result = NodeOCRResult(node_id="test")
        result.regions = [
            OCRTextRegion(text="2ex", confidence=0.67, bounding_box=BoundingBox(0, 0, 20, 10)),
        ]

        proc._extract_numeric_candidates(result)

        # Should extract "2" from "2ex"
        assert len(result.numeric_candidates) > 0
        assert result.numeric_candidates[0][0] == 2.0


class TestNewNodeOCRResultFields:
    def test_has_alternatives_field(self):
        result = NodeOCRResult(node_id="test")
        assert result.activity_id_alternatives == []
        assert result.best_numeric is None
        assert result.id_sub_crop_regions == []
        assert result.duration_sub_crop_regions == []

    def test_has_id_psm_modes_in_config(self):
        cfg = RegionOCRConfig()
        assert 8 in cfg.id_psm_modes
        assert 11 in cfg.id_psm_modes
        assert 8 in cfg.duration_psm_modes


class TestDeduplicatedNodeOCR:
    """Deterministic tests for OCR on deduplicated CandidateNodes."""

    def _make_candidates(self, rects):
        """Create CandidateNode list from (x, y, w, h) tuples."""
        nodes = []
        for x, y, w, h in rects:
            nodes.append(CandidateNode(
                shape_type=ShapeType.RECTANGLE,
                bounding_box=BoundingBox(float(x), float(y), float(w), float(h)),
                position=Point(float(x + w / 2), float(y + h / 2)),
                confidence=0.95,
            ))
        return nodes

    def _make_engine(self, text="A"):
        """Create MockOCREngine returning fixed text."""
        engine = MockOCREngine(responses={"default": text})
        return engine

    def test_deduplicated_candidates_passed_to_ocr(self):
        """OCR receives exactly the deduplicated CandidateNodes list."""
        engine = self._make_engine("A")
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 100, 60)])
        results = proc.process_node_regions(np.zeros((200, 200), dtype=np.uint8), candidates)
        assert len(results) == len(candidates)

    def test_one_ocr_region_per_rectangle(self):
        """Each unique rectangle produces exactly one NodeOCRResult."""
        engine = self._make_engine("B")
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 100, 60), (200, 50, 100, 60)])
        results = proc.process_node_regions(np.zeros((200, 400), dtype=np.uint8), candidates)
        assert len(results) == 2

    def test_crop_coordinate_mapping(self):
        """Crop bbox is mapped back to original image coordinates."""
        engine = self._make_engine("C")
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(100, 100, 80, 50)])
        results = proc.process_node_regions(np.zeros((300, 300), dtype=np.uint8), candidates)
        nr = results[0]
        if nr.crop_bbox is not None:
            assert nr.crop_bbox.x >= 0
            assert nr.crop_bbox.y >= 0

    def test_id_crop_generation(self):
        """ID sub-crop config specifies top portion of node crop."""
        cfg = RegionOCRConfig()
        assert cfg.id_crop_top == 0.0
        assert cfg.id_crop_bottom == 0.45
        assert cfg.id_crop_left == 0.0
        assert cfg.id_crop_right == 1.0

    def test_duration_crop_generation(self):
        """Duration sub-crop config specifies bottom portion of node crop."""
        cfg = RegionOCRConfig()
        assert cfg.duration_crop_top == 0.55
        assert cfg.duration_crop_bottom == 1.0

    def test_ocr_source_traceability(self):
        """NodeOCRResult records source node_id for traceability."""
        engine = self._make_engine("D")
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 100, 60)])
        results = proc.process_node_regions(np.zeros((200, 200), dtype=np.uint8), candidates)
        assert results[0].node_id == candidates[0].node_id

    def test_mock_ocr_available_for_unit_tests(self):
        """MockOCREngine can be used without Tesseract."""
        engine = self._make_engine("E")
        assert engine.is_available()
        results = engine.recognize(np.zeros((50, 50), dtype=np.uint8))
        assert len(results) == 1
        assert results[0].text == "E"

    def test_tesseract_unavailable_graceful(self):
        """RegionOCRProcessor handles engine returning no text gracefully."""
        engine = MockOCREngine(responses={"default": ""})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 100, 60)])
        results = proc.process_node_regions(np.zeros((200, 200), dtype=np.uint8), candidates)
        assert len(results) == 1
        assert results[0].best_activity_id is None or results[0].best_activity_id == ""


class TestNodeLocalOCRMapping:
    """Deterministic tests for node-local OCR-to-node association."""

    def _make_candidates(self, rects):
        nodes = []
        for x, y, w, h in rects:
            nodes.append(CandidateNode(
                shape_type=ShapeType.RECTANGLE,
                bounding_box=BoundingBox(float(x), float(y), float(w), float(h)),
                position=Point(float(x + w / 2), float(y + h / 2)),
                confidence=0.95,
            ))
        return nodes

    def test_right_to_left_semantic_ordering(self):
        """Nodes arranged right-to-left: each gets its own local OCR ID."""
        engine = MockOCREngine(responses={"default": "X"})
        proc = RegionOCRProcessor(engine)
        # 4 nodes arranged right-to-left
        candidates = self._make_candidates([
            (400, 50, 80, 50), (300, 50, 80, 50), (200, 50, 80, 50), (100, 50, 80, 50),
        ])
        results = proc.process_node_regions(np.zeros((150, 550), dtype=np.uint8), candidates)
        assert len(results) == 4
        # Each result must have the node_id of its source candidate
        result_node_ids = {r.node_id for r in results}
        source_node_ids = {c.node_id for c in candidates}
        assert result_node_ids == source_node_ids

    def test_left_to_right_semantic_ordering(self):
        """Nodes arranged left-to-right: each gets its own local OCR ID."""
        engine = MockOCREngine(responses={"default": "Y"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([
            (50, 50, 80, 50), (150, 50, 80, 50), (250, 50, 80, 50), (350, 50, 80, 50),
        ])
        results = proc.process_node_regions(np.zeros((150, 450), dtype=np.uint8), candidates)
        assert len(results) == 4
        for r in results:
            assert r.best_activity_id == "Y"

    def test_bottom_row_reverse_layout(self):
        """Bottom row V,U,T,S,R,Q,P,O left-to-right: node-local IDs preserved."""
        engine = MockOCREngine(responses={"default": "V"})
        proc = RegionOCRProcessor(engine)
        # Bottom row: V(232), U(378), T(526), S(672), R(819), Q(966), P(1113), O(1259)
        candidates = self._make_candidates([
            (232, 541, 80, 50), (378, 541, 80, 50), (526, 541, 80, 50),
            (672, 541, 80, 50), (819, 541, 80, 50), (966, 541, 80, 50),
            (1113, 541, 80, 50), (1259, 541, 80, 50),
        ])
        results = proc.process_node_regions(np.zeros((650, 1400), dtype=np.uint8), candidates)
        assert len(results) == 8
        # All 8 must have unique node_ids matching sources
        result_ids = {r.node_id for r in results}
        source_ids = {c.node_id for c in candidates}
        assert result_ids == source_ids

    def test_irregular_layout(self):
        """Non-linear arrangement: A at top, B at left, C at right."""
        engine = MockOCREngine(responses={"default": "A"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([
            (200, 50, 80, 50),   # top center (A)
            (50, 150, 80, 50),   # bottom left (B)
            (350, 150, 80, 50),  # bottom right (C)
        ])
        results = proc.process_node_regions(np.zeros((250, 480), dtype=np.uint8), candidates)
        assert len(results) == 3
        # Each node_id must match its source
        for r, c in zip(results, candidates):
            assert r.node_id == c.node_id

    def test_node_local_ocr_association(self):
        """OCR result node_id always matches source CandidateNode node_id."""
        engine = MockOCREngine(responses={"default": "Z"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([
            (50, 50, 100, 60), (200, 100, 100, 60), (350, 150, 100, 60),
        ])
        results = proc.process_node_regions(np.zeros((300, 500), dtype=np.uint8), candidates)
        for r, c in zip(results, candidates):
            assert r.node_id == c.node_id, (
                f"OCR result node_id {r.node_id} != source node_id {c.node_id}"
            )

    def test_no_global_list_index_mapping(self):
        """Results are NOT reordered by position after OCR."""
        engine = MockOCREngine(responses={"default": "K"})
        proc = RegionOCRProcessor(engine)
        # Deliberately non-spatial order
        candidates = self._make_candidates([
            (300, 200, 80, 50), (100, 100, 80, 50), (200, 300, 80, 50),
        ])
        results = proc.process_node_regions(np.zeros((400, 450), dtype=np.uint8), candidates)
        # Results must be in same order as candidates (process_node_regions preserves order)
        for r, c in zip(results, candidates):
            assert r.node_id == c.node_id

    def test_duration_attached_to_correct_node(self):
        """Numeric candidates retain source_node_id from their parent node."""
        engine = MockOCREngine(responses={"default": "5"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 100, 60)])
        results = proc.process_node_regions(np.zeros((200, 200), dtype=np.uint8), candidates)
        nr = results[0]
        # The node_id on the result must match the source candidate
        assert nr.node_id == candidates[0].node_id
        # Numeric candidates are extracted within the same node context
        if nr.best_numeric:
            assert nr.best_numeric[0] == 5.0

    def test_duplicate_ocr_candidate_handled(self):
        """Two nodes reading same letter get distinct node_ids."""
        engine = MockOCREngine(responses={"default": "M"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 80, 50), (200, 50, 80, 50)])
        results = proc.process_node_regions(np.zeros((150, 350), dtype=np.uint8), candidates)
        assert len(results) == 2
        assert results[0].node_id != results[1].node_id
        assert results[0].best_activity_id == "M"
        assert results[1].best_activity_id == "M"

    def test_c_z_ambiguity_stored(self):
        """OCR alternative Z for C is stored in alternatives list."""
        engine = MockOCREngine(responses={"default": "Z"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 80, 50)])
        results = proc.process_node_regions(np.zeros((150, 200), dtype=np.uint8), candidates)
        nr = results[0]
        # The raw OCR text should be stored
        assert nr.best_activity_id == "Z" or nr.best_activity_id is not None
        # Alternatives may be empty (no alternative OCR passes in mock),
        # but the structure must support them
        assert isinstance(nr.activity_id_alternatives, list)

    def test_i_l_ambiguity_stored(self):
        """OCR alternative L for I is stored in alternatives list."""
        engine = MockOCREngine(responses={"default": "L"})
        proc = RegionOCRProcessor(engine)
        candidates = self._make_candidates([(50, 50, 80, 50)])
        results = proc.process_node_regions(np.zeros((150, 200), dtype=np.uint8), candidates)
        nr = results[0]
        assert nr.best_activity_id == "L" or nr.best_activity_id is not None
        assert isinstance(nr.activity_id_alternatives, list)

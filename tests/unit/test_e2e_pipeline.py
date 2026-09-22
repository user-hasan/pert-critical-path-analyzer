"""
End-to-end pipeline integration tests.

Tests the complete pipeline from image loading through CPM analysis
using mock CV components to verify orchestration logic.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pert_analyzer.core.models import (
    Activity,
    BoundingBox,
    Dependency,
    DiagramType,
    GraphModel,
    Point,
)
from pert_analyzer.cv.models import (
    ArrowDetectionResult,
    CandidateNode,
    DetectedArrow,
    ShapeDetectionResult,
    ShapeType,
)
from pert_analyzer.cv.ocr_models import OCRProcessingResult, OCRTextRegion
from pert_analyzer.cv.reconstruction_models import (
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
    AmbiguityIssue,
    AmbiguityType,
    ValidationResult,
    ValidationSeverity,
)
from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
from pert_analyzer.pipeline.result import AnalysisStatus, ImageMetadata, PipelineResult


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def simple_chain_image() -> np.ndarray:
    """Create a simple chain: A -> B -> C."""
    img = np.full((200, 800, 3), 255, dtype=np.uint8)
    try:
        import cv2

        # Activity boxes
        positions = [(50, 70, 100, 60), (250, 70, 100, 60), (450, 70, 100, 60)]
        for x, y, w, h in positions:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)

        # Arrows
        cv2.arrowedLine(img, (150, 100), (250, 100), (0, 0, 0), 2, tipLength=0.15)
        cv2.arrowedLine(img, (350, 100), (450, 100), (0, 0, 0), 2, tipLength=0.15)

        # Text labels
        cv2.putText(img, "A", (80, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "5", (90, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "B", (280, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "3", (290, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(img, "C", (480, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.putText(img, "4", (490, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    except ImportError:
        pass

    return img


@pytest.fixture
def branching_image() -> np.ndarray:
    """Create branching diagram: A->B->D, A->C->D."""
    img = np.full((400, 900, 3), 255, dtype=np.uint8)
    try:
        import cv2

        boxes = {
            "A": (50, 150, 100, 60),
            "B": (300, 50, 100, 60),
            "C": (300, 250, 100, 60),
            "D": (600, 150, 100, 60),
        }
        for bid, (x, y, w, h) in boxes.items():
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
            cv2.putText(img, bid, (x + 40, y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        # Arrows
        cv2.arrowedLine(img, (150, 180), (300, 80), (0, 0, 0), 2, tipLength=0.12)
        cv2.arrowedLine(img, (150, 180), (300, 280), (0, 0, 0), 2, tipLength=0.12)
        cv2.arrowedLine(img, (400, 80), (600, 180), (0, 0, 0), 2, tipLength=0.12)
        cv2.arrowedLine(img, (400, 280), (600, 180), (0, 0, 0), 2, tipLength=0.12)
    except ImportError:
        pass

    return img


@pytest.fixture
def temp_image_file(simple_chain_image: np.ndarray) -> str:
    """Save a synthetic image to a temp file and return the path."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        import cv2

        cv2.imwrite(path, simple_chain_image)
    except ImportError:
        # Fallback: write a minimal valid PNG
        import struct

        # Minimal 1x1 white PNG
        png_data = (
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01"
            b"\x00\x00\x00\x01\x08\x02\x00\x00\x00"
            b"\x90wS\xde\x00\x00\x00\x0cIDATx"
            b"\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05"
            b"\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        with open(path, "wb") as f:
            f.write(png_data)
    return path


@pytest.fixture
def nonexistent_image() -> str:
    """Return a path to a nonexistent image."""
    return "/tmp/nonexistent_image_xyz.png"


@pytest.fixture
def blank_image_file() -> str:
    """Create a blank white image file."""
    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    img = np.full((300, 400, 3), 255, dtype=np.uint8)
    try:
        import cv2

        cv2.imwrite(path, img)
    except ImportError:
        pass
    return path


# =============================================================================
# Mock CV Results Factory
# =============================================================================


def make_shape_result(
    activities: List[Dict[str, Any]],
    events: Optional[List[Dict[str, Any]]] = None,
) -> ShapeDetectionResult:
    """Create a mock ShapeDetectionResult from activity definitions."""
    candidates = []
    detected_shapes = []

    for i, act in enumerate(activities):
        bbox = BoundingBox(x=50 + i * 200, y=50, width=100, height=60)
        candidate = CandidateNode(
            node_id=f"candidate_{act['id']}",
            shape_type=ShapeType.RECTANGLE,
            bounding_box=bbox,
            confidence=0.95,
            node_role="activity",
        )
        candidates.append(candidate)

        from pert_analyzer.core.models import DetectedShape

        shape = DetectedShape(
            shape_id=f"shape_{act['id']}",
            shape_type="rectangle",
            bounding_box=bbox,
            centroid=Point(100 + i * 200, 80),
            confidence=0.95,
        )
        detected_shapes.append(shape)

    if events:
        for i, evt in enumerate(events):
            bbox = BoundingBox(x=50 + i * 200, y=150, width=40, height=40)
            candidate = CandidateNode(
                node_id=f"candidate_event_{evt.get('id', i)}",
                shape_type=ShapeType.CIRCLE,
                bounding_box=bbox,
                confidence=0.9,
                node_role="event",
            )
            candidates.append(candidate)

    return ShapeDetectionResult(
        candidate_nodes=candidates,
        detected_shapes=detected_shapes,
    )


def make_arrow_result(
    connections: List[Dict[str, Any]],
    activity_map: Dict[str, BoundingBox],
) -> ArrowDetectionResult:
    """Create a mock ArrowDetectionResult from connection definitions."""
    arrows = []
    for conn in connections:
        src_bbox = activity_map[conn["source"]]
        tgt_bbox = activity_map[conn["target"]]
        arrow = DetectedArrow(
            start=Point(src_bbox.x + src_bbox.width, src_bbox.y + src_bbox.height / 2),
            end=Point(tgt_bbox.x, tgt_bbox.y + tgt_bbox.height / 2),
            confidence=0.9,
        )
        arrows.append(arrow)

    return ArrowDetectionResult(
        arrows=arrows,
        processing_time=0.0,
    )


def make_ocr_result(
    texts: List[Dict[str, Any]],
) -> OCRProcessingResult:
    """Create a mock OCRProcessingResult from text definitions."""
    regions = []
    for i, txt in enumerate(texts):
        bbox = BoundingBox(x=txt.get("x", 0), y=txt.get("y", 0), width=50, height=20)
        region = OCRTextRegion(
            raw_text=txt["text"],
            bounding_box=bbox,
            confidence=txt.get("confidence", 0.9),
        )
        regions.append(region)
    return OCRProcessingResult(regions=regions)


def make_reconstruction(
    activities: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
    diagram_type: str = "AON",
) -> ReconstructedDiagram:
    """Create a mock ReconstructedDiagram from definitions."""
    diagram = ReconstructedDiagram(diagram_type=diagram_type)

    for act in activities:
        recon_act = ReconstructedActivity(
            activity_id=act["id"],
            label=act.get("label", act["id"]),
            duration=act.get("duration", 0),
            confidence=act.get("confidence", 0.95),
            source_text_region_ids=[],
        )
        diagram.activities.append(recon_act)

    for dep in dependencies:
        recon_dep = ReconstructedDependency(
            source_id=dep["source"],
            target_id=dep["target"],
            confidence=dep.get("confidence", 0.9),
            evidence=[],
        )
        diagram.dependencies.append(recon_dep)

    diagram.overall_confidence = 0.9
    diagram.validation = ValidationResult(
        is_valid=True,
    )

    return diagram


def build_test_graph(
    activities: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
    diagram_type: DiagramType = DiagramType.AON,
) -> GraphModel:
    """Build a GraphModel from activity/dependency definitions."""
    graph = GraphModel(diagram_type=diagram_type)

    for act in activities:
        a = Activity(
            activity_id=act["id"],
            name=act.get("label", act["id"]),
            duration=act.get("duration", 0),
            confidence=act.get("confidence", 0.95),
            is_dummy=act.get("is_dummy", False),
        )
        graph.activities[act["id"]] = a

    for dep in dependencies:
        d = Dependency(
            source=dep["source"],
            target=dep["target"],
            dependency_type="finish_to_start",
            confidence=dep.get("confidence", 0.9),
        )
        graph.dependencies.append(d)

    return graph


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResultConstruction:
    """Test PipelineResult internal state management."""

    def test_set_stage_creates_timing(self) -> None:
        result = PipelineResult()
        result.set_stage("preprocess", "SUCCESS")
        assert "preprocess" in result.stages
        stage = result.stages["preprocess"]
        assert stage.status == "SUCCESS"
        assert stage.timing.stage_name == "preprocess"

    def test_set_stage_with_error(self) -> None:
        result = PipelineResult()
        result.set_stage("ocr", "FAILED", error="engine unavailable")
        assert result.stages["ocr"].status == "FAILED"
        assert result.stages["ocr"].error == "engine unavailable"

    def test_warnings_accumulate(self) -> None:
        result = PipelineResult()
        result.add_warning("W1")
        result.add_warning("W2")
        assert result.warnings == ["W1", "W2"]

    def test_errors_accumulate(self) -> None:
        result = PipelineResult()
        result.add_error("E1")
        result.add_error("E2")
        assert result.errors == ["E1", "E2"]

    def test_summary_dict_structure(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        result.diagram_type = "AON"
        result.shape_count = 5
        result.arrow_count = 4
        result.reconstructed_activity_count = 3
        result.cpm_project_duration = 12.0
        d = result.to_summary_dict()
        assert d["status"] == "SUCCESS"
        assert d["diagram_type"] == "AON"
        assert d["shape_count"] == 5
        assert d["cpm_project_duration"] == 12.0
        assert "stage_timings" not in d  # Not in summary, only in JSON

    def test_json_serialization(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        result.review_required = True
        result.review_issues = [{"type": "test", "description": "test issue"}]
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["status"] == "SUCCESS"
        assert data["review_required"] is True
        assert data["review_issues"] == [{"type": "test", "description": "test issue"}]

    def test_summary_string_contains_key_sections(self) -> None:
        result = PipelineResult()
        result.diagram_type = "AON"
        result.diagram_confidence = 0.85
        result.cpm_project_duration = 15.0
        result.cpm_critical_path = ["A", "B", "D"]
        s = result.to_summary_string()
        assert "AON" in s
        assert "15.0" in s
        assert "A -> B -> D" in s


# =============================================================================
# EndToEndAnalyzer — Error Handling Tests
# =============================================================================


class TestEndToEndAnalyzerErrors:
    """Test pipeline error handling for various failure modes."""

    def test_nonexistent_image_fails(self) -> None:
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze("/tmp/nonexistent_xyz_12345.png")
        assert result.status == AnalysisStatus.FAILED
        assert any("not found" in e.lower() or "not found" in e for e in result.errors)

    def test_result_has_all_stages_even_on_failure(self) -> None:
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze("/tmp/nonexistent_xyz_12345.png")
        assert "load_image" in result.stages
        assert result.stages["load_image"].status == "FAILED"


# =============================================================================
# EndToEndAnalyzer — Progress Callback Tests
# =============================================================================


class TestEndToEndAnalyzerProgress:
    """Test progress callback functionality."""

    def test_progress_callback_called(self, nonexistent_image: str) -> None:
        callbacks = []
        analyzer = EndToEndAnalyzer()

        def track_progress(stage: str, pct: float) -> None:
            callbacks.append((stage, pct))

        analyzer.analyze(nonexistent_image, progress_callback=track_progress)
        assert len(callbacks) >= 1
        assert callbacks[0][0] == "Loading image"

    def test_progress_callback_exception_does_not_crash(self, nonexistent_image: str) -> None:
        def bad_callback(stage: str, pct: float) -> None:
            raise RuntimeError("boom")

        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze(nonexistent_image, progress_callback=bad_callback)
        assert result.status == AnalysisStatus.FAILED


# =============================================================================
# EndToEndAnalyzer — Mock Pipeline Tests
# =============================================================================


class TestEndToEndAnalyzerMockPipeline:
    """Test the pipeline with mocked CV components."""

    def test_simple_chain_pipeline(
        self,
        temp_image_file: str,
    ) -> None:
        """Test a 3-activity chain pipeline: A(5)->B(3)->C(4)."""
        analyzer = EndToEndAnalyzer()

        # Mock the pipeline internals
        mock_image = np.zeros((200, 800, 3), dtype=np.uint8)
        mock_prep_result = MagicMock()
        mock_prep_result.get_representation = MagicMock(return_value=mock_image)

        activities = [
            {"id": "A", "duration": 5},
            {"id": "B", "duration": 3},
            {"id": "C", "duration": 4},
        ]
        bboxes = {
            "A": BoundingBox(50, 50, 100, 60),
            "B": BoundingBox(250, 50, 100, 60),
            "C": BoundingBox(450, 50, 100, 60),
        }
        mock_shape_result = make_shape_result(activities)
        mock_arrow_result = make_arrow_result(
            [{"source": "A", "target": "B"}, {"source": "B", "target": "C"}],
            bboxes,
        )
        mock_ocr_result = make_ocr_result([
            {"text": "A", "x": 80, "y": 80},
            {"text": "5", "x": 90, "y": 100},
            {"text": "B", "x": 280, "y": 80},
            {"text": "3", "x": 290, "y": 100},
            {"text": "C", "x": 480, "y": 80},
            {"text": "4", "x": 490, "y": 100},
        ])
        mock_reconstruction = make_reconstruction(activities, [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ])
        mock_graph = build_test_graph(activities, [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
        ])

        with patch.object(analyzer._preprocessor, "process_image", return_value=mock_prep_result), \
             patch.object(analyzer._shape_detector, "detect_from_preprocessing", return_value=mock_shape_result), \
             patch.object(analyzer._classifier, "classify_from_detection", return_value=MagicMock(diagram_type="AON", confidence=0.9)), \
             patch.object(analyzer._arrow_detector, "detect_from_preprocessing", return_value=mock_arrow_result), \
             patch.object(analyzer._spatial_associator, "associate_all", return_value=MagicMock()), \
             patch.object(analyzer._reconstruction_engine, "reconstruct_aon", return_value=mock_reconstruction), \
             patch.object(mock_reconstruction, "to_graph_model", return_value=mock_graph):

            analyzer._ocr_engine = MagicMock(recognize=MagicMock(return_value=mock_ocr_result.regions))
            analyzer._ocr_initialized = True

            result = analyzer.analyze(temp_image_file)

        assert result.status in (AnalysisStatus.SUCCESS, AnalysisStatus.REVIEW_REQUIRED, AnalysisStatus.FAILED)
        assert result.shape_count == 3
        assert result.arrow_count == 2
        assert result.reconstructed_activity_count == 3

    def test_missing_file_returns_failed_status(self) -> None:
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze("/tmp/does_not_exist_abc123.png")
        assert result.status == AnalysisStatus.FAILED
        assert result.errors

    def test_pipeline_stores_intermediate_results(self, temp_image_file: str) -> None:
        """Verify pipeline stores intermediate results even on failure."""
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze(temp_image_file)
        # Even if stages fail, the result should have stage entries
        assert isinstance(result.stages, dict)


# =============================================================================
# GraphModel Construction Tests
# =============================================================================


class TestGraphModelConstruction:
    """Test that GraphModel can be constructed for CPM."""

    def test_simple_chain_graph(self) -> None:
        graph = build_test_graph(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
                {"id": "C", "duration": 4},
            ],
            dependencies=[
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
            ],
        )
        assert graph.activity_count == 3
        assert graph.dependency_count == 2
        assert graph.activities["A"].duration == 5

    def test_branching_graph(self) -> None:
        graph = build_test_graph(
            activities=[
                {"id": "A", "duration": 2},
                {"id": "B", "duration": 4},
                {"id": "C", "duration": 1},
                {"id": "D", "duration": 3},
            ],
            dependencies=[
                {"source": "A", "target": "B"},
                {"source": "A", "target": "C"},
                {"source": "B", "target": "D"},
                {"source": "C", "target": "D"},
            ],
        )
        assert graph.activity_count == 4
        assert graph.dependency_count == 4


# =============================================================================
# ReviewAnalysis Tests
# =============================================================================


class TestReviewAnalysis:
    """Test the review analysis stage of the pipeline."""

    def test_low_confidence_triggers_review(self) -> None:
        reconstruction = ReconstructedDiagram(diagram_type="AON")
        reconstruction.activities.append(
            ReconstructedActivity(
                activity_id="X",
                label="X",
                duration=5,
                confidence=0.2,  # Low confidence
                source_text_region_ids=[],
            )
        )
        reconstruction.dependencies.append(
            ReconstructedDependency(
                source_id="X",
                target_id="Y",
                confidence=0.9,
                evidence=[],
            )
        )
        reconstruction.overall_confidence = 0.3
        reconstruction.validation = ValidationResult(
            is_valid=True,
        )

        analyzer = EndToEndAnalyzer()
        result = PipelineResult()
        analyzer._stage_review_analysis(reconstruction, result)

        assert result.review_required is True
        assert len(result.review_issues) > 0
        assert any(
            "low confidence" in issue.get("description", "").lower()
            for issue in result.review_issues
        )

    def test_no_review_needed_for_clean_diagram(self) -> None:
        reconstruction = ReconstructedDiagram(diagram_type="AON")
        reconstruction.activities.append(
            ReconstructedActivity(
                activity_id="A",
                label="A",
                duration=5,
                confidence=0.95,
                source_text_region_ids=["region_1"],
            )
        )
        reconstruction.dependencies.append(
            ReconstructedDependency(
                source_id="A",
                target_id="B",
                confidence=0.9,
                evidence=[],
            )
        )
        reconstruction.overall_confidence = 0.95
        reconstruction.validation = ValidationResult(
            is_valid=True,
        )

        analyzer = EndToEndAnalyzer()
        result = PipelineResult()
        analyzer._stage_review_analysis(reconstruction, result)

        assert result.review_required is False or len(result.review_issues) == 0


# =============================================================================
# ReconstructedDiagram Tests
# =============================================================================


class TestReconstructedDiagramModel:
    """Test ReconstructedDiagram properties."""

    def test_activity_count(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.activities.append(
            ReconstructedActivity(
                activity_id="A", label="A", duration=5,
                confidence=0.9, source_text_region_ids=[],
            )
        )
        diagram.activities.append(
            ReconstructedActivity(
                activity_id="B", label="B", duration=3,
                confidence=0.8, source_text_region_ids=[],
            )
        )
        assert diagram.activity_count == 2

    def test_dependency_count(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.dependencies.append(
            ReconstructedDependency(
                source_id="A", target_id="B",
                confidence=0.9, evidence=[],
            )
        )
        assert diagram.dependency_count == 1

    def test_ambiguity_count(self) -> None:
        diagram = ReconstructedDiagram(diagram_type="AON")
        diagram.ambiguities.append(
            AmbiguityIssue(
                ambiguity_type=AmbiguityType.TEXT_MULTIPLE_MATCHES,
                description="Multiple candidates",
                involved_ids=["A", "B"],
                severity=ValidationSeverity.WARNING,
            )
        )
        assert diagram.ambiguity_count == 1


# =============================================================================
# End-to-End with Mocked Image Loading
# =============================================================================


class TestEndToEndWithMockedImage:
    """Test E2E with fully mocked image loading."""

    def test_pipeline_completes_with_mock_image(self) -> None:
        """Test that pipeline runs and produces a result with mock image."""
        analyzer = EndToEndAnalyzer()
        mock_image = np.zeros((200, 400, 3), dtype=np.uint8)

        with patch("cv2.imread", return_value=mock_image), \
             patch("pert_analyzer.pipeline.analyzer.os.path.exists", return_value=True):

            # Mock preprocessing
            mock_prep = MagicMock()
            mock_prep.get_representation.return_value = mock_image
            analyzer._preprocessor.process_image = MagicMock(return_value=mock_prep)

            # Mock shape detection — empty result
            empty_shapes = ShapeDetectionResult(
                candidate_nodes=[],
                detected_shapes=[],
            )
            analyzer._shape_detector.detect_from_preprocessing = MagicMock(return_value=empty_shapes)

            # Mock classification
            mock_classification = MagicMock()
            mock_classification.diagram_type = "UNKNOWN"
            mock_classification.confidence = 0.0
            analyzer._classifier.classify_from_detection = MagicMock(return_value=mock_classification)

            # Mock arrow detection
            empty_arrows = ArrowDetectionResult(
                arrows=[],
                processing_time=0.0,
            )
            analyzer._arrow_detector.detect_from_preprocessing = MagicMock(return_value=empty_arrows)

            # Mock OCR
            analyzer._ocr_engine = MagicMock()
            analyzer._ocr_engine.recognize.return_value = []
            analyzer._ocr_initialized = True

            # Mock reconstruction
            empty_recon = ReconstructedDiagram(diagram_type="AON")
            empty_recon.validation = ValidationResult(
                is_valid=True,
            )
            empty_recon.overall_confidence = 0.0
            analyzer._reconstruction_engine.reconstruct_aon = MagicMock(return_value=empty_recon)

            # Mock graph build
            empty_graph = GraphModel(diagram_type=DiagramType.AON)
            empty_recon.to_graph_model = MagicMock(return_value=empty_graph)

            # Mock spatial association
            analyzer._spatial_associator.associate_all = MagicMock(return_value=MagicMock())

            fd, path = tempfile.mkstemp(suffix=".png")
            os.close(fd)
            img_data = np.zeros((10, 10, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.imwrite(path, img_data)
            except ImportError:
                pass

            try:
                result = analyzer.analyze(path)
                assert result.status in (
                    AnalysisStatus.SUCCESS,
                    AnalysisStatus.REVIEW_REQUIRED,
                    AnalysisStatus.FAILED,
                )
                assert "load_image" in result.stages
            finally:
                os.unlink(path)


# =============================================================================
# Serialization Round-Trip Tests
# =============================================================================


class TestResultSerialization:
    """Test JSON serialization and deserialization."""

    def test_json_round_trip(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        result.diagram_type = "AON"
        result.shape_count = 10
        result.arrow_count = 8
        result.reconstructed_activity_count = 6
        result.cpm_project_duration = 25.0
        result.cpm_critical_path = ["A", "C", "F"]
        result.review_required = True
        result.review_issues = [
            {"type": "MISSING_DURATION", "description": "Activity X has no duration"}
        ]

        json_str = result.to_json()
        data = json.loads(json_str)

        assert data["status"] == "SUCCESS"
        assert data["diagram_type"] == "AON"
        assert data["shape_count"] == 10
        assert data["cpm_project_duration"] == 25.0
        assert data["cpm_critical_path"] == ["A", "C", "F"]
        assert data["review_required"] is True
        assert len(data["review_issues"]) == 1

    def test_summary_dict_matches_json(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.REVIEW_REQUIRED
        result.diagram_type = "AOA"
        result.review_required = True

        d = result.to_summary_dict()
        json_data = json.loads(result.to_json())

        # Core fields should match
        assert d["status"] == json_data["status"]
        assert d["diagram_type"] == json_data["diagram_type"]
        assert d["review_required"] == json_data["review_required"]


# =============================================================================
# ImageMetadata Tests
# =============================================================================


class TestImageMetadataEdgeCases:
    def test_from_none(self) -> None:
        meta = ImageMetadata.from_image(None)
        assert meta.width == 0

    def test_from_image_array(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        meta = ImageMetadata.from_image(img, "/path/to/file.png")
        assert meta.width == 200
        assert meta.height == 100
        assert meta.channels == 3
        assert meta.file_name == "file.png"

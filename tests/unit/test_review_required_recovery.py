"""
Regression tests: OCR/reconstruction defects become recoverable
REVIEW_REQUIRED outcomes (never fatal Analysis ERROR) and route into the
human-review pipeline, then Validate -> CPM -> Results.

Covers the classifier contract:
- Fatal (FAILED/ERROR): undecodable image, missing file.
- Reviewable (REVIEW_REQUIRED): invalid/missing duration, missing or
  duplicate activity ID, ambiguous dependency reconstruction.
- SUCCESS: clean reconstruction with a valid graph and CPM results.

Also verifies the review session, DurationReview creation, the CPM
gate, and that correcting a duration unblocks validation and CPM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pert_analyzer.core.models import BoundingBox
from pert_analyzer.cv.reconstruction_models import (
    ActivityStatus,
    AmbiguityIssue,
    AmbiguityType,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ValidationResult,
)
from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
from pert_analyzer.pipeline.human_review import (
    ReviewStatus,
    build_review_session,
)
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult
from pert_analyzer.pipeline.review_api import ReviewWorkflow

from tests.unit.test_e2e_pipeline import (
    make_arrow_result,
    make_ocr_result,
    make_shape_result,
)


# =============================================================================
# Helpers
# =============================================================================


@pytest.fixture
def temp_image_file() -> str:
    """Save a synthetic image to a temp file and return the path."""
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    try:
        import cv2

        img = np.full((200, 800, 3), 255, dtype=np.uint8)
        cv2.imwrite(path, img)
    except ImportError:
        pass
    return path


@pytest.fixture
def nonexistent_image() -> str:
    """Return a path to a nonexistent image."""
    return "/tmp/nonexistent_image_xyz_12345.png"


def _recon(
    activities: List[Dict[str, Any]],
    dependencies: List[Dict[str, Any]],
    ambiguities: Optional[List[AmbiguityIssue]] = None,
) -> ReconstructedDiagram:
    """Build a ReconstructedDiagram whose elements pass review checks."""
    diagram = ReconstructedDiagram(diagram_type="AON")
    for i, act in enumerate(activities):
        recon_act = ReconstructedActivity(
            activity_id=act["id"],
            label=act.get("label", act["id"]),
            duration=act.get("duration", 0.0),
            confidence=act.get("confidence", 0.95),
            is_dummy=act.get("is_dummy", False),
            source_text_region_ids=act.get("source_text_region_ids") or [f"t{i}"],
            status=ActivityStatus.CONFIRMED,
            semantic_status=ActivityStatus.CONFIRMED,
            needs_review=False,
        )
        diagram.activities.append(recon_act)
    for dep in dependencies:
        diagram.dependencies.append(
            ReconstructedDependency(
                source_id=dep["source"],
                target_id=dep["target"],
                confidence=dep.get("confidence", 0.9),
                evidence=[],
            )
        )
    diagram.overall_confidence = 0.9
    if ambiguities:
        diagram.ambiguities.extend(ambiguities)
    diagram.validation = ValidationResult(is_valid=True)
    return diagram


def _run_pipeline(temp_image_file: str, reconstruction: ReconstructedDiagram) -> PipelineResult:
    """Run the full analyzer with mocked CV stages returning `reconstruction`."""
    analyzer = EndToEndAnalyzer()
    mock_image = np.zeros((200, 800, 3), dtype=np.uint8)
    mock_prep_result = MagicMock()
    mock_prep_result.get_representation = MagicMock(return_value=mock_image)

    mock_shape_result = make_shape_result(
        [{"id": a.activity_id} for a in reconstruction.activities]
    )
    bboxes = {
        a.activity_id: BoundingBox(50 + i * 200, 50, 100, 60)
        for i, a in enumerate(reconstruction.activities)
    }
    connections = [
        {"source": d.source_id, "target": d.target_id}
        for d in reconstruction.dependencies
    ]
    mock_arrow_result = make_arrow_result(connections, bboxes)
    mock_ocr_result = make_ocr_result([])

    with patch.object(analyzer._preprocessor, "process_image", return_value=mock_prep_result), \
         patch.object(analyzer._shape_detector, "detect_from_preprocessing", return_value=mock_shape_result), \
         patch.object(analyzer._classifier, "classify_from_detection", return_value=MagicMock(diagram_type="AON", confidence=0.9)), \
         patch.object(analyzer._arrow_detector, "detect_from_preprocessing", return_value=mock_arrow_result), \
         patch.object(analyzer._spatial_associator, "associate_all", return_value=MagicMock()), \
         patch.object(analyzer._reconstruction_engine, "reconstruct_aon", return_value=reconstruction):
        analyzer._ocr_engine = MagicMock(recognize=MagicMock(return_value=mock_ocr_result.regions))
        analyzer._ocr_initialized = True
        return analyzer.analyze(temp_image_file)


def _q_reconstruction() -> ReconstructedDiagram:
    """Reference scenario: chain A(5) -> B(3) -> Q(0.0 invalid duration)."""
    return _recon(
        [
            {"id": "A", "duration": 5},
            {"id": "B", "duration": 3},
            {"id": "Q", "duration": 0.0},
        ],
        [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "Q"},
        ],
    )


# =============================================================================
# Classification: reviewable defects -> REVIEW_REQUIRED
# =============================================================================


class TestReviewableDefectsBecomeReviewRequired:
    def test_invalid_duration_becomes_review_required(self, temp_image_file) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())

        assert result.status == AnalysisStatus.REVIEW_REQUIRED
        assert result.errors == []
        assert result.review_required is True
        assert result.stages["graph_build"].status == "REVIEW_REQUIRED"
        assert "Q" in result.stages["graph_build"].warning
        assert "invalid duration" in result.stages["graph_build"].warning

    def test_missing_duration_becomes_review_required(self, temp_image_file) -> None:
        recon = _recon(
            [{"id": "X", "duration": 0.0}],
            [],
        )
        result = _run_pipeline(temp_image_file, recon)

        assert result.status == AnalysisStatus.REVIEW_REQUIRED
        assert result.errors == []
        assert result.stages["graph_build"].status == "REVIEW_REQUIRED"
        assert "invalid duration" in result.stages["graph_build"].warning

    def test_invalid_activity_id_becomes_review_required(self, temp_image_file) -> None:
        recon = _recon(
            [{"id": "", "duration": 5}],
            [],
        )
        result = _run_pipeline(temp_image_file, recon)

        assert result.status == AnalysisStatus.REVIEW_REQUIRED
        assert result.errors == []
        assert result.stages["graph_build"].status == "REVIEW_REQUIRED"
        assert "missing an identifier" in result.stages["graph_build"].warning

    def test_duplicate_activity_id_becomes_review_required(self, temp_image_file) -> None:
        recon = _recon(
            [{"id": "A", "duration": 5}, {"id": "A", "duration": 3}],
            [],
        )
        result = _run_pipeline(temp_image_file, recon)

        assert result.status == AnalysisStatus.REVIEW_REQUIRED
        assert result.errors == []
        assert result.stages["graph_build"].status == "REVIEW_REQUIRED"
        assert "Duplicate activity identifier 'A'" in result.stages["graph_build"].warning

    def test_ambiguous_dependency_becomes_review_required(self, temp_image_file) -> None:
        recon = _recon(
            [
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
            ],
            [{"source": "A", "target": "B"}],
            ambiguities=[
                AmbiguityIssue(
                    ambiguity_type=AmbiguityType.ARROW_AMBIGUOUS,
                    description="Arrow ambiguous between A and B",
                    involved_ids=["A", "B"],
                )
            ],
        )
        result = _run_pipeline(temp_image_file, recon)

        assert result.status == AnalysisStatus.REVIEW_REQUIRED
        assert result.errors == []
        assert result.review_required is True


# =============================================================================
# Classification: fatal failures stay FAILED
# =============================================================================


class TestFatalFailuresStayFailed:
    def test_undecodable_image_remains_failed(self, nonexistent_image: str) -> None:
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze(nonexistent_image)
        assert result.status == AnalysisStatus.FAILED
        assert result.errors


# =============================================================================
# Review session / graph / CPM gate for the blocked reference scenario
# =============================================================================


class TestReviewPipelineForBlockedScenario:
    def test_review_session_and_duration_review_created_for_q(
        self, temp_image_file
    ) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())
        session = build_review_session(result, source_image_id="ref.png")

        duration_items = [
            d for d in session.durations if d.activity_id == "Q"
        ]
        assert len(duration_items) == 1
        dur = duration_items[0]
        assert dur.current_duration == 0.0
        assert dur.status == ReviewStatus.PENDING
        assert "missing/invalid duration" in dur.reason
        assert dur.provenance == "cv_pipeline"

    def test_graph_model_not_built_when_duration_unresolved(
        self, temp_image_file
    ) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())
        assert result._graph_model is None

    def test_cpm_remains_blocked_when_duration_unresolved(
        self, temp_image_file
    ) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())
        workflow = ReviewWorkflow.from_pipeline_result(result)
        candidate = workflow.apply()

        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"
        assert candidate.cpm is None
        with pytest.raises(ValueError):
            workflow.run_cpm()

    def test_corrected_duration_produces_valid_validation(
        self, temp_image_file
    ) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())
        workflow = ReviewWorkflow.from_pipeline_result(result)
        assert workflow.decide_duration(
            "Q", "CORRECT",
            corrected_duration=6.0,
            reason="corrected during review",
        )
        candidate = workflow.apply()

        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.validation.is_valid
        assert candidate.cpm is not None

    def test_corrected_duration_enables_cpm_run(self, temp_image_file) -> None:
        result = _run_pipeline(temp_image_file, _q_reconstruction())
        workflow = ReviewWorkflow.from_pipeline_result(result)
        assert workflow.decide_duration(
            "Q", "CORRECT", corrected_duration=6.0
        )
        workflow.apply()
        cpm = workflow.run_cpm()

        # Chain A(5) -> B(3) -> Q(6) => project duration 14.
        assert cpm.project_duration == pytest.approx(14.0)
        assert list(cpm.critical_path) == ["A", "B", "Q"]


# =============================================================================
# Clean success + threshold warning handling
# =============================================================================


class TestCleanSuccessAndThresholdHandling:
    def test_clean_success_analysis_is_success(self, temp_image_file) -> None:
        recon = _recon(
            [
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
                {"id": "C", "duration": 4},
            ],
            [
                {"source": "A", "target": "B"},
                {"source": "B", "target": "C"},
            ],
        )
        result = _run_pipeline(temp_image_file, recon)

        assert result.status == AnalysisStatus.SUCCESS
        assert result.errors == []
        assert result.review_required is False
        assert result.validation_passed is True
        assert result._graph_model is not None
        assert result.cpm_project_duration == pytest.approx(12.0)

    def test_default_adaptive_threshold_is_not_fatal(
        self, temp_image_file
    ) -> None:
        """ADAPTIVE_GAUSSIAN (the default) must never fail preprocessing."""
        analysis = EndToEndAnalyzer()
        preprocessor = analysis._preprocessor

        try:
            import cv2

            image = np.full((100, 200, 3), 255, dtype=np.uint8)
            cv2.putText(
                image, "A 5", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )
            result = preprocessor.process_image(image)
        except ImportError:
            pytest.skip("OpenCV not available")

        assert result is not None
        assert result.binary is not None
        assert result.adaptive_binary is not None
"""
Tests for the ReviewCorrectionAPI.

Verifies programmatic correction of reconstructed diagram elements
and re-analysis without re-running CV.
"""

from __future__ import annotations

import pytest

from pert_analyzer.core.models import DiagramType, GraphModel, Activity, Dependency
from pert_analyzer.cv.reconstruction_models import (
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
    ValidationResult,
)
from pert_analyzer.pipeline.review_api import ReviewCorrectionAPI
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult


# =============================================================================
# Fixtures
# =============================================================================


def make_pipeline_result(
    activities=None,
    dependencies=None,
    diagram_type="AON",
    status=AnalysisStatus.SUCCESS,
):
    """Create a PipelineResult with a ReconstructedDiagram."""
    result = PipelineResult()
    result.status = status
    result.diagram_type = diagram_type

    recon = ReconstructedDiagram(diagram_type=diagram_type)
    if activities:
        for act in activities:
            recon.activities.append(
                ReconstructedActivity(
                    activity_id=act["id"],
                    label=act.get("label", act["id"]),
                    duration=act.get("duration", 0),
                    confidence=act.get("confidence", 0.9),
                    source_text_region_ids=[],
                )
            )
    if dependencies:
        for dep in dependencies:
            recon.dependencies.append(
                ReconstructedDependency(
                    source_id=dep["source"],
                    target_id=dep["target"],
                    confidence=dep.get("confidence", 0.9),
                    evidence=[],
                )
            )
    recon.overall_confidence = 0.9
    recon.validation = ValidationResult(is_valid=True)

    result._reconstruction = recon
    return result


# =============================================================================
# correct_activity_duration Tests
# =============================================================================


class TestCorrectActivityDuration:
    def test_correct_existing_activity(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 3}])
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_duration("A", 5.0, "OCR misread 5 as 3")
        assert success is True
        assert api.reconstruction.get_activity_by_id("A").duration == 5.0
        assert len(api.corrections) == 1
        assert api.corrections[0].old_value == 3
        assert api.corrections[0].new_value == 5.0

    def test_correct_nonexistent_returns_false(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 3}])
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_duration("X", 5.0)
        assert success is False

    def test_correct_increases_confidence(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 3, "confidence": 0.4}])
        api = ReviewCorrectionAPI(pr)

        api.correct_activity_duration("A", 5.0)
        assert api.reconstruction.get_activity_by_id("A").confidence == 1.0

    def test_correct_clears_warnings(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 3}])
        api = ReviewCorrectionAPI(pr)
        api.reconstruction.get_activity_by_id("A").warnings = ["duration uncertain"]

        api.correct_activity_duration("A", 5.0)
        assert "duration uncertain" not in api.reconstruction.get_activity_by_id("A").warnings

    def test_no_reconstruction_returns_false(self) -> None:
        pr = PipelineResult()
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_duration("A", 5.0)
        assert success is False


# =============================================================================
# correct_activity_id Tests
# =============================================================================


class TestCorrectActivityId:
    def test_rename_activity(self) -> None:
        pr = make_pipeline_result(
            activities=[{"id": "X1", "duration": 5}],
            dependencies=[{"source": "X1", "target": "Y"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_id("X1", "A", "OCR misidentified")
        assert success is True
        assert api.reconstruction.get_activity_by_id("A") is not None
        assert api.reconstruction.get_activity_by_id("X1") is None

    def test_rename_updates_dependencies(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "X1", "duration": 5},
                {"id": "Y", "duration": 3},
            ],
            dependencies=[{"source": "X1", "target": "Y"}],
        )
        api = ReviewCorrectionAPI(pr)

        api.correct_activity_id("X1", "A")
        deps = api.reconstruction.dependencies
        assert deps[0].source_id == "A"
        assert deps[0].target_id == "Y"

    def test_nonexistent_returns_false(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_id("X", "Y")
        assert success is False


# =============================================================================
# correct_activity_label Tests
# =============================================================================


class TestCorrectActivityLabel:
    def test_correct_label(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_label("A", "Excavate")
        assert success is True
        assert api.reconstruction.get_activity_by_id("A").label == "Excavate"

    def test_nonexistent_returns_false(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.correct_activity_label("X", "Label")
        assert success is False


# =============================================================================
# correct_dependency_source Tests
# =============================================================================


class TestCorrectDependencySource:
    def test_correct_source(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
                {"id": "C", "duration": 4},
            ],
            dependencies=[{"source": "A", "target": "C"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.correct_dependency_source("A", "C", "B", "Wrong arrow")
        assert success is True
        assert api.reconstruction.dependencies[0].source_id == "B"

    def test_nonexistent_dep_returns_false(self) -> None:
        pr = make_pipeline_result(
            activities=[{"id": "A", "duration": 5}],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.correct_dependency_source("X", "Y", "Z")
        assert success is False


# =============================================================================
# correct_dependency_target Tests
# =============================================================================


class TestCorrectDependencyTarget:
    def test_correct_target(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
                {"id": "C", "duration": 4},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.correct_dependency_target("A", "B", "C")
        assert success is True
        assert api.reconstruction.dependencies[0].target_id == "C"


# =============================================================================
# remove_false_detection Tests
# =============================================================================


class TestRemoveFalseDetection:
    def test_remove_activity(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.remove_false_detection("B", "False positive")
        assert success is True
        assert api.reconstruction.activity_count == 1
        assert api.reconstruction.get_activity_by_id("B") is None

    def test_remove_cleans_dependencies(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)

        api.remove_false_detection("B")
        assert len(api.reconstruction.dependencies) == 0

    def test_nonexistent_returns_false(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.remove_false_detection("X")
        assert success is False


# =============================================================================
# add_missing_dependency Tests
# =============================================================================


class TestAddMissingDependency:
    def test_add_dependency(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
            ],
            dependencies=[],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.add_missing_dependency("A", "B", "Missing link")
        assert success is True
        assert len(api.reconstruction.dependencies) == 1
        assert api.reconstruction.dependencies[0].source_id == "A"
        assert api.reconstruction.dependencies[0].target_id == "B"

    def test_duplicate_dependency_not_added(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 5},
                {"id": "B", "duration": 3},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)

        success = api.add_missing_dependency("A", "B")
        assert success is False
        assert len(api.reconstruction.dependencies) == 1


# =============================================================================
# mark_dummy_activity Tests
# =============================================================================


class TestMarkDummyActivity:
    def test_mark_dummy(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "X", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.mark_dummy_activity("X", "Dummy activity")
        assert success is True
        act = api.reconstruction.get_activity_by_id("X")
        assert act.is_dummy is True
        assert act.duration == 0.0

    def test_nonexistent_returns_false(self) -> None:
        pr = make_pipeline_result(activities=[{"id": "A", "duration": 5}])
        api = ReviewCorrectionAPI(pr)

        success = api.mark_dummy_activity("X")
        assert success is False


# =============================================================================
# reanalyze Tests
# =============================================================================


class TestReanalyze:
    def test_reanalyze_applies_corrections(self) -> None:
        pr = make_pipeline_result(
            activities=[
                {"id": "A", "duration": 3},
                {"id": "B", "duration": 2},
            ],
            dependencies=[{"source": "A", "target": "B"}],
        )
        api = ReviewCorrectionAPI(pr)
        api.correct_activity_duration("A", 5.0)

        new_result = api.reanalyze()
        assert new_result._reconstruction is not None
        assert new_result._reconstruction.get_activity_by_id("A").duration == 5.0

    def test_reanalyze_preserves_metadata(self) -> None:
        pr = make_pipeline_result(
            activities=[{"id": "A", "duration": 5}],
            dependencies=[],
        )
        pr.image_metadata.file_path = "/test/path.png"
        pr.diagram_type = "AON"

        api = ReviewCorrectionAPI(pr)
        new_result = api.reanalyze()

        assert new_result.diagram_type == "AON"
        assert new_result.image_metadata.file_path == "/test/path.png"

    def test_reanalyze_without_reconstruction_returns_failed(self) -> None:
        pr = PipelineResult()
        api = ReviewCorrectionAPI(pr)

        new_result = api.reanalyze()
        assert new_result.status == AnalysisStatus.FAILED


# =============================================================================
# get_review_result Tests
# =============================================================================


class TestGetReviewResult:
    def test_review_result_from_corrections(self) -> None:
        pr = make_pipeline_result(
            activities=[{"id": "A", "duration": 3}],
            dependencies=[],
        )
        api = ReviewCorrectionAPI(pr)
        api.correct_activity_duration("A", 5.0, "OCR misread")

        review = api.get_review_result()
        assert review.correction_count == 1
        assert review.corrections[0].action_type == "correct_duration"
        assert review.is_fully_resolved is True

    def test_review_result_empty_when_no_corrections(self) -> None:
        pr = make_pipeline_result(
            activities=[{"id": "A", "duration": 3}],
            dependencies=[],
        )
        api = ReviewCorrectionAPI(pr)

        review = api.get_review_result()
        assert review.correction_count == 0

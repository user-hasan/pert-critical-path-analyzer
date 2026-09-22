"""
Tests for pipeline result models and human review data structures.
"""

from __future__ import annotations

import json

import pytest

from pert_analyzer.pipeline.result import (
    AnalysisStatus,
    ImageMetadata,
    PipelineResult,
    StageResult,
    StageTiming,
)
from pert_analyzer.pipeline.human_review import (
    CandidateValue,
    CorrectionAction,
    HumanReviewItem,
    HumanReviewResult,
    ReviewIssue,
    ReviewIssueType,
)


# =============================================================================
# StageTiming Tests
# =============================================================================


class TestStageTiming:
    def test_default_values(self) -> None:
        timing = StageTiming(stage_name="test")
        assert timing.stage_name == "test"
        assert timing.duration_seconds == 0.0

    def test_duration_calculation(self) -> None:
        timing = StageTiming(stage_name="test", start_time=1.0, end_time=3.5)
        assert timing.duration_seconds == 2.5


# =============================================================================
# StageResult Tests
# =============================================================================


class TestStageResult:
    def test_default_values(self) -> None:
        result = StageResult(stage_name="preprocess")
        assert result.stage_name == "preprocess"
        assert result.status == "PENDING"
        assert result.error is None

    def test_with_error(self) -> None:
        result = StageResult(stage_name="ocr", status="FAILED", error="engine unavailable")
        assert result.status == "FAILED"
        assert result.error == "engine unavailable"


# =============================================================================
# ImageMetadata Tests
# =============================================================================


class TestImageMetadata:
    def test_default_values(self) -> None:
        meta = ImageMetadata()
        assert meta.file_path == ""
        assert meta.width == 0

    def test_from_image(self) -> None:
        import numpy as np

        img = np.zeros((100, 200, 3), dtype=np.uint8)
        meta = ImageMetadata.from_image(img, "/path/to/test.png")
        assert meta.width == 200
        assert meta.height == 100
        assert meta.channels == 3
        assert meta.file_name == "test.png"

    def test_from_none_image(self) -> None:
        meta = ImageMetadata.from_image(None)
        assert meta.width == 0


# =============================================================================
# PipelineResult Tests
# =============================================================================


class TestPipelineResult:
    def test_default_values(self) -> None:
        result = PipelineResult()
        assert result.status == AnalysisStatus.FAILED
        assert result.total_time == 0.0
        assert result.diagram_type == "UNKNOWN"

    def test_set_stage(self) -> None:
        result = PipelineResult()
        result.set_stage("preprocess", "SUCCESS")
        assert "preprocess" in result.stages
        assert result.stages["preprocess"].status == "SUCCESS"

    def test_add_warning(self) -> None:
        result = PipelineResult()
        result.add_warning("Low confidence")
        assert len(result.warnings) == 1

    def test_add_error(self) -> None:
        result = PipelineResult()
        result.add_error("File not found")
        assert len(result.errors) == 1

    def test_to_summary_dict(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        result.diagram_type = "AON"
        result.shape_count = 10
        summary = result.to_summary_dict()
        assert summary["status"] == "SUCCESS"
        assert summary["diagram_type"] == "AON"
        assert summary["shape_count"] == 10

    def test_to_json(self) -> None:
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        json_str = result.to_json()
        data = json.loads(json_str)
        assert data["status"] == "SUCCESS"

    def test_to_summary_string(self) -> None:
        result = PipelineResult()
        result.diagram_type = "AON"
        result.diagram_confidence = 0.9
        result.shape_count = 5
        summary = result.to_summary_string()
        assert "AON" in summary
        assert "0.900" in summary

    def test_review_required_flag(self) -> None:
        result = PipelineResult()
        result.review_required = True
        result.review_issues.append({"type": "MISSING_DURATION", "description": "test"})
        summary = result.to_summary_dict()
        assert summary["review_required"] is True
        assert summary["review_issues"] == 1


# =============================================================================
# ReviewIssueType Tests
# =============================================================================


class TestReviewIssueType:
    def test_all_values_exist(self) -> None:
        types = [
            "UNCERTAIN_DURATION", "UNCERTAIN_ACTIVITY_ID", "UNCERTAIN_ACTIVITY_LABEL",
            "AMBIGUOUS_ARROW_TARGET", "AMBIGUOUS_ARROW_SOURCE", "DUPLICATE_ACTIVITY_ID",
            "MISSING_DURATION", "ISOLATED_NODE", "LOW_OCR_CONFIDENCE",
            "LOW_SHAPE_CONFIDENCE", "ORPHAN_ARROW", "DIAGRAM_TYPE_UNCERTAIN",
            "DISCONNECTED_GRAPH", "CYCLE_DETECTED", "MISSING_DEPENDENCY",
            "FALSE_DETECTION",
        ]
        for t in types:
            assert ReviewIssueType(t).value == t


# =============================================================================
# CandidateValue Tests
# =============================================================================


class TestCandidateValue:
    def test_to_dict(self) -> None:
        cv = CandidateValue(value=5, confidence=0.9, source="ocr")
        d = cv.to_dict()
        assert d["value"] == 5
        assert d["confidence"] == 0.9
        assert d["source"] == "ocr"


# =============================================================================
# ReviewIssue Tests
# =============================================================================


class TestReviewIssue:
    def test_to_dict(self) -> None:
        issue = ReviewIssue(
            issue_type=ReviewIssueType.MISSING_DURATION,
            description="Activity A has no duration",
            element_id="A",
            element_type="activity",
        )
        d = issue.to_dict()
        assert d["type"] == "MISSING_DURATION"
        assert d["element_id"] == "A"


# =============================================================================
# CorrectionAction Tests
# =============================================================================


class TestCorrectionAction:
    def test_to_dict(self) -> None:
        corr = CorrectionAction(
            action_type="correct_duration",
            target_id="A1",
            field_name="duration",
            old_value=3,
            new_value=5,
            reason="OCR misread",
        )
        d = corr.to_dict()
        assert d["action_type"] == "correct_duration"
        assert d["old_value"] == 3
        assert d["new_value"] == 5


# =============================================================================
# HumanReviewItem Tests
# =============================================================================


class TestHumanReviewItem:
    def test_to_dict(self) -> None:
        item = HumanReviewItem(
            element_id="A1",
            element_type="activity",
            field_name="duration",
            detected_value=3,
            detected_confidence=0.7,
        )
        d = item.to_dict()
        assert d["element_id"] == "A1"
        assert d["detected_value"] == 3
        assert d["is_resolved"] is False


# =============================================================================
# HumanReviewResult Tests
# =============================================================================


class TestHumanReviewResult:
    def test_default_values(self) -> None:
        result = HumanReviewResult()
        assert result.issue_count == 0
        assert result.correction_count == 0
        assert result.unresolved_count == 0

    def test_add_issue(self) -> None:
        result = HumanReviewResult()
        issue = ReviewIssue(
            issue_type=ReviewIssueType.MISSING_DURATION,
            description="test",
        )
        result.add_issue(issue)
        assert result.issue_count == 1

    def test_add_correction(self) -> None:
        result = HumanReviewResult()
        corr = CorrectionAction(action_type="correct_duration", target_id="A1")
        result.add_correction(corr)
        assert result.correction_count == 1

    def test_get_issues_by_type(self) -> None:
        result = HumanReviewResult()
        result.add_issue(ReviewIssue(
            issue_type=ReviewIssueType.MISSING_DURATION, description="1"
        ))
        result.add_issue(ReviewIssue(
            issue_type=ReviewIssueType.MISSING_DURATION, description="2"
        ))
        result.add_issue(ReviewIssue(
            issue_type=ReviewIssueType.LOW_OCR_CONFIDENCE, description="3"
        ))
        missing = result.get_issues_by_type(ReviewIssueType.MISSING_DURATION)
        assert len(missing) == 2

    def test_to_dict(self) -> None:
        result = HumanReviewResult()
        result.add_issue(ReviewIssue(
            issue_type=ReviewIssueType.MISSING_DURATION, description="test"
        ))
        d = result.to_dict()
        assert d["issue_count"] == 1

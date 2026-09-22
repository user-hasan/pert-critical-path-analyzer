"""
Structured result models for the end-to-end analysis pipeline.

Preserves all intermediate results, timing, validation, and status
information from the complete image-to-CPM pipeline.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AnalysisStatus(Enum):
    """Overall pipeline status."""

    SUCCESS = "SUCCESS"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    FAILED = "FAILED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


@dataclass
class StageTiming:
    """Timing information for a single pipeline stage."""

    stage_name: str
    start_time: float = 0.0
    end_time: float = 0.0

    @property
    def duration_seconds(self) -> float:
        return self.end_time - self.start_time


@dataclass
class StageResult:
    """Result from a single pipeline stage."""

    stage_name: str
    status: str = "PENDING"
    error: Optional[str] = None
    warning: Optional[str] = None
    data: Any = None
    timing: StageTiming = field(default_factory=lambda: StageTiming(""))


@dataclass
class ImageMetadata:
    """Metadata about the input image."""

    file_path: str = ""
    file_name: str = ""
    file_size_bytes: int = 0
    width: int = 0
    height: int = 0
    channels: int = 0
    dtype: str = ""

    @classmethod
    def from_image(cls, image: Any, file_path: str = "") -> ImageMetadata:
        """Create metadata from a loaded image array."""
        import os

        meta = cls(file_path=file_path, file_name=os.path.basename(file_path))
        if image is not None:
            try:
                import numpy as np

                if isinstance(image, np.ndarray):
                    meta.height, meta.width = image.shape[:2]
                    meta.channels = (
                        image.shape[2] if len(image.shape) > 2 else 1
                    )
                    meta.dtype = str(image.dtype)
            except Exception:
                pass
        if file_path:
            try:
                meta.file_size_bytes = os.path.getsize(file_path)
            except OSError:
                pass
        return meta


@dataclass
class PipelineResult:
    """
    Complete result of the end-to-end analysis pipeline.

    Preserves all intermediate results, timing, validation outcome,
    and human review data.
    """

    # Overall status
    status: AnalysisStatus = AnalysisStatus.FAILED

    # Image metadata
    image_metadata: ImageMetadata = field(default_factory=ImageMetadata)

    # Stage results
    stages: Dict[str, StageResult] = field(default_factory=dict)

    # Timing
    total_time: float = 0.0

    # Diagram info
    diagram_type: str = "UNKNOWN"
    diagram_confidence: float = 0.0

    # Counts
    shape_count: int = 0
    arrow_count: int = 0
    ocr_region_count: int = 0
    reconstructed_activity_count: int = 0
    reconstructed_event_count: int = 0
    dependency_count: int = 0

    # Validation
    validation_passed: bool = False
    validation_errors: List[str] = field(default_factory=list)
    validation_warnings: List[str] = field(default_factory=list)

    # CPM results
    cpm_project_duration: Optional[float] = None
    cpm_critical_path: List[str] = field(default_factory=list)
    cpm_critical_paths: List[List[str]] = field(default_factory=list)
    cpm_critical_activity_count: int = 0

    # Human review
    review_required: bool = False
    review_issues: List[Dict[str, Any]] = field(default_factory=list)

    # Warnings and errors
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    # References to intermediate objects (not serialized)
    _preprocessing_result: Any = field(default=None, repr=False)
    _shape_result: Any = field(default=None, repr=False)
    _classification_result: Any = field(default=None, repr=False)
    _arrow_result: Any = field(default=None, repr=False)
    _ocr_result: Any = field(default=None, repr=False)
    _association_results: Any = field(default=None, repr=False)
    _reconstruction: Any = field(default=None, repr=False)
    _graph_model: Any = field(default=None, repr=False)
    _cpm_result: Any = field(default=None, repr=False)

    def set_stage(
        self,
        stage_name: str,
        status: str = "SUCCESS",
        error: Optional[str] = None,
        warning: Optional[str] = None,
        data: Any = None,
    ) -> None:
        """Set the result for a pipeline stage."""
        timing = StageTiming(stage_name)
        self.stages[stage_name] = StageResult(
            stage_name=stage_name,
            status=status,
            error=error,
            warning=warning,
            data=data,
            timing=timing,
        )

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)

    def to_summary_dict(self) -> Dict[str, Any]:
        """Convert to a summary dictionary for display."""
        return {
            "status": self.status.value,
            "diagram_type": self.diagram_type,
            "diagram_confidence": round(self.diagram_confidence, 3),
            "shape_count": self.shape_count,
            "arrow_count": self.arrow_count,
            "ocr_region_count": self.ocr_region_count,
            "reconstructed_activities": self.reconstructed_activity_count,
            "reconstructed_events": self.reconstructed_event_count,
            "dependencies": self.dependency_count,
            "validation_passed": self.validation_passed,
            "validation_errors": len(self.validation_errors),
            "validation_warnings": len(self.validation_warnings),
            "cpm_project_duration": self.cpm_project_duration,
            "cpm_critical_path": self.cpm_critical_path,
            "cpm_critical_activity_count": self.cpm_critical_activity_count,
            "review_required": self.review_required,
            "review_issues": len(self.review_issues),
            "total_warnings": len(self.warnings),
            "total_errors": len(self.errors),
            "total_time_seconds": round(self.total_time, 3),
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string (excludes non-serializable fields)."""
        data = self.to_summary_dict()
        # Add stage timing
        data["stage_timings"] = {
            name: {
                "status": sr.status,
                "duration_seconds": round(sr.timing.duration_seconds, 4),
                "error": sr.error,
                "warning": sr.warning,
            }
            for name, sr in self.stages.items()
        }
        # Add review issues
        data["review_issues"] = self.review_issues
        # Add warnings and errors
        data["warnings"] = self.warnings
        data["errors"] = self.errors
        return json.dumps(data, indent=indent, default=str)

    def to_summary_string(self) -> str:
        """Format a human-readable summary string."""
        lines = [
            "=" * 50,
            "PERT & Critical Path Analyzer",
            "=" * 50,
            "",
            f"Input:            {self.image_metadata.file_name}",
            f"Image Size:       {self.image_metadata.width}x{self.image_metadata.height}",
            "",
            f"Diagram Type:     {self.diagram_type}",
            f"Confidence:       {self.diagram_confidence:.3f}",
            "",
            f"Shapes:           {self.shape_count}",
            f"Arrows:           {self.arrow_count}",
            f"OCR Regions:      {self.ocr_region_count}",
            "",
            f"Activities:       {self.reconstructed_activity_count}",
            f"Events:           {self.reconstructed_event_count}",
            f"Dependencies:     {self.dependency_count}",
            "",
            f"Validation:       {'PASSED' if self.validation_passed else 'FAILED'}",
        ]

        if self.validation_errors:
            lines.append(f"  Errors: {len(self.validation_errors)}")
            for err in self.validation_errors[:5]:
                lines.append(f"    - {err}")

        if self.validation_warnings:
            lines.append(f"  Warnings: {len(self.validation_warnings)}")

        if self.cpm_project_duration is not None:
            lines.extend([
                "",
                f"CPM Duration:     {self.cpm_project_duration}",
                f"Critical Path:    {' -> '.join(self.cpm_critical_path)}",
                f"Critical Acts:    {self.cpm_critical_activity_count}",
                f"Critical Paths:   {len(self.cpm_critical_paths)}",
            ])
        else:
            lines.extend(["", "CPM:              NOT AVAILABLE"])

        if self.review_required:
            lines.extend([
                "",
                "REVIEW REQUIRED:",
            ])
            for issue in self.review_issues[:5]:
                lines.append(f"  - {issue.get('type', '?')}: {issue.get('description', '')}")

        lines.extend([
            "",
            f"Total Time:       {self.total_time:.3f}s",
            f"Status:           {self.status.value}",
            "=" * 50,
        ])

        return "\n".join(lines)

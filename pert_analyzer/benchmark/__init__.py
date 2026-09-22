"""
Generalization benchmark infrastructure.

Runs the real production analysis pipeline against a corpus of unseen
diagram images, captures measured stage metrics, classifies the first
abnormal expansion stage, optionally compares against ground truth,
and writes a deterministic JSON + Markdown report.

The benchmark never runs mocked data, never hardcodes per-image
expectations, and never modifies the CV pipeline to improve scores.
"""

from pert_analyzer.benchmark.accuracy import (
    compare_dependencies,
    compare_durations,
    compute_accuracy,
    match_nodes,
)
from pert_analyzer.benchmark.annotations import (
    SCHEMA_VERSION,
    AnnotationError,
    GroundTruthAnnotation,
    GroundTruthValidator,
    annotation_path_for_image,
    default_ground_truth_dir,
    from_dict,
    load_annotation,
    validate_annotation_dict,
)
from pert_analyzer.benchmark.discovery import (
    SUPPORTED_IMAGE_EXTENSIONS,
    DatasetScanner,
    discover_images,
    relative_image_path,
)
from pert_analyzer.benchmark.classification import (
    FailureClass,
    classify_first_abnormal_stage,
    summarize_bottlenecks,
)
from pert_analyzer.benchmark.ground_truth import (
    GroundTruth,
    GroundTruthError,
    load_ground_truth,
)
from pert_analyzer.benchmark.metrics import (
    UNAVAILABLE,
    STAGE_METRIC_KEYS,
    classify_outcome,
    extract_stage_metrics,
    extract_stage_progression,
    extract_stage_timings,
)
from pert_analyzer.benchmark.runner import (
    ImageRunResult,
    BenchmarkRunError,
    analyze_single,
    run_benchmark,
)
from pert_analyzer.benchmark.reporting import build_markdown, write_reports

__all__ = [
    "SUPPORTED_IMAGE_EXTENSIONS",
    "DatasetScanner",
    "discover_images",
    "relative_image_path",
    "FailureClass",
    "classify_first_abnormal_stage",
    "classify_outcome",
    "summarize_bottlenecks",
    "SCHEMA_VERSION",
    "AnnotationError",
    "GroundTruthAnnotation",
    "GroundTruthValidator",
    "annotation_path_for_image",
    "default_ground_truth_dir",
    "from_dict",
    "load_annotation",
    "validate_annotation_dict",
    "match_nodes",
    "compare_dependencies",
    "compare_durations",
    "compute_accuracy",
    "GroundTruth",
    "GroundTruthError",
    "load_ground_truth",
    "UNAVAILABLE",
    "STAGE_METRIC_KEYS",
    "extract_stage_metrics",
    "extract_stage_progression",
    "extract_stage_timings",
    "ImageRunResult",
    "BenchmarkRunError",
    "analyze_single",
    "run_benchmark",
    "build_markdown",
    "write_reports",
]
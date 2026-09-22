"""
Measured stage metrics for the generalization benchmark.

Every metric is read from the real ``PipelineResult`` intermediates the
production pipeline already produces (shape detection, arrow detection,
OCR, reconstruction, validation, CPM).  A metric is ``UNAVAILABLE``
whenever the pipeline did not produce that stage (failure or skipped),
never fabricated.

Canonical metric mapping (kept in sync with the docs report):
    RAW SHAPES                  = len(shape_result.detected_shapes)
    FINAL CANDIDATE NODES       = len(shape_result.candidate_nodes)
    RAW ARROW CANDIDATES        = len(arrow_result.raw_line_segments)
    DETECTED (DEDUPLICATED)     = len(arrow_result.arrows)
    VALIDATED NODE PAIRS        = report.deduplicated_arrow_count
    VALIDATED DEPENDENCIES      = report.accepted_count
    OCR REGIONS                 = ocr_result.region_count
    REVIEW ITEMS                = activities + dependencies + durations
    GRAPH / CPM STATUS          = reviewed candidate (validation.status,
                                  cpm_gate, project_duration,
                                  critical_path_count)
    PERT STATUS                 = UNAVAILABLE (pipeline never runs PERT)
"""

from __future__ import annotations

import numbers
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.cv.models import ArrowDetectionResult, ShapeDetectionResult
from pert_analyzer.cv.ocr_models import OCRProcessingResult, TextType
from pert_analyzer.cv.reconstruction_models import ReconstructedDiagram
from pert_analyzer.pipeline.human_review import CpmGateStatus
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult

UNAVAILABLE = "UNAVAILABLE"

# Ordered list of stage metrics surfaced in the report tables.
STAGE_METRIC_KEYS: Tuple[str, ...] = (
    "raw_shapes",
    "contours_analyzed",
    "shapes_filtered",
    "duplicates_suppressed",
    "final_candidate_nodes",
    "reconstructed_activities",
    "reconstructed_events",
    "reconstructed_dependencies",
    "ocr_regions",
    "ocr_labels",
    "ocr_id_candidates",
    "ocr_numeric_candidates",
    "raw_arrow_segments",
    "lines_detected",
    "lines_filtered",
    "segments_merged",
    "arrows_detected",
    "raw_arrow_candidates",
    "deduplicated_arrows",
    "validated_node_pairs",
    "validated_dependencies",
    "review_candidate_dependencies",
    "rejected_dependencies",
    "self_loops",
    "duplicate_edges",
    "review_items",
    "graph_status",
    "graph_is_valid",
    "cpm_gate",
    "cpm_project_duration",
    "critical_path_count",
    "pert_status",
)

_STAGE_TIMING_KEYS: Tuple[str, ...] = (
    "Loading image",
    "Preprocessing",
    "Detecting shapes",
    "Classifying diagram",
    "Detecting arrows",
    "Extracting text (OCR)",
    "Associating text",
    "Reconstructing diagram",
    "Building graph",
    "Validating and analyzing",
    "Complete",
    "Review preparation",
)


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return float(value)
    return None


def _safe_int(value: Any) -> Optional[int]:
    if isinstance(value, numbers.Number) and not isinstance(value, bool):
        return int(value)
    return None


def _len_of(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return len(value)
    except TypeError:
        return default


def _str(value: Any) -> str:
    if hasattr(value, "value"):
        return str(value.value)
    return UNAVAILABLE if value is None else str(value)


def _shape_metrics(shape: Optional[ShapeDetectionResult]) -> Dict[str, Any]:
    if shape is None:
        return {
            "raw_shapes": UNAVAILABLE,
            "contours_analyzed": UNAVAILABLE,
            "shapes_filtered": UNAVAILABLE,
            "duplicates_suppressed": UNAVAILABLE,
            "final_candidate_nodes": UNAVAILABLE,
        }
    return {
        "raw_shapes": _len_of(shape.detected_shapes, 0),
        "contours_analyzed": _safe_int(shape.contours_analyzed),
        "shapes_filtered": _safe_int(shape.shapes_filtered),
        "duplicates_suppressed": _safe_int(shape.duplicates_suppressed),
        "final_candidate_nodes": _len_of(shape.candidate_nodes, 0),
    }


def _arrow_metrics(arrow: Optional[ArrowDetectionResult]) -> Dict[str, Any]:
    if arrow is None:
        return {
            "raw_arrow_segments": UNAVAILABLE,
            "lines_detected": UNAVAILABLE,
            "lines_filtered": UNAVAILABLE,
            "segments_merged": UNAVAILABLE,
            "arrows_detected": UNAVAILABLE,
            "arrow_candidates": UNAVAILABLE,
            "arrow_deduplicated": UNAVAILABLE,
            "validated_aoa_arrows": UNAVAILABLE,
            "aoa_rejected_arrows": UNAVAILABLE,
            "aoa_context": UNAVAILABLE,
        }
    return {
        "raw_arrow_segments": _len_of(arrow.raw_line_segments, 0),
        "lines_detected": _safe_int(arrow.lines_detected),
        "lines_filtered": _safe_int(arrow.lines_filtered),
        "segments_merged": _safe_int(arrow.segments_merged),
        "arrows_detected": (
            _safe_int(arrow.arrows_detected)
            if getattr(arrow, "arrows_detected", None) is not None
            else _len_of(arrow.arrows, 0)
        ),
        "arrow_candidates": _safe_int(
            getattr(arrow, "arrow_candidates", None)
        ),
        "arrow_deduplicated": _safe_int(
            getattr(arrow, "deduplicated_arrow_count", None)
        ),
        "validated_aoa_arrows": _safe_int(
            getattr(arrow, "validated_arrow_count", None)
        ),
        "aoa_rejected_arrows": _len_of(
            getattr(arrow, "rejected_arrows", None), None
        ),
        "aoa_context": (
            bool(getattr(arrow, "aoa_context", False))
            if getattr(arrow, "aoa_context", None) is not None
            else UNAVAILABLE
        ),
    }


def _ocr_metrics(
    ocr: Optional[OCRProcessingResult],
) -> Dict[str, Any]:
    if ocr is None:
        return {
            "ocr_regions": UNAVAILABLE,
            "ocr_labels": UNAVAILABLE,
            "ocr_id_candidates": UNAVAILABLE,
            "ocr_numeric_candidates": UNAVAILABLE,
        }
    regions = list(getattr(ocr, "regions", None) or [])
    labels = 0
    id_candidates = 0
    numeric_candidates = 0
    id_type = TextType.ACTIVITY_ID_CANDIDATE
    numeric_types = {
        TextType.NUMERIC_CANDIDATE,
        TextType.POSSIBLE_PERT_VALUE,
        TextType.DURATION_CANDIDATE,
    }
    for region in regions:
        text = getattr(region, "normalized_text", None) or ""
        if not text.strip():
            text = getattr(region, "text", "") or ""
        if text.strip():
            labels += 1
        text_type = getattr(region, "text_type", None)
        if text_type == id_type:
            id_candidates += 1
        if text_type in numeric_types:
            numeric_candidates += 1
    region_count = getattr(ocr, "region_count", None)
    if region_count is None:
        region_count = len(regions)
    return {
        "ocr_regions": _safe_int(region_count),
        "ocr_labels": labels,
        "ocr_id_candidates": id_candidates,
        "ocr_numeric_candidates": numeric_candidates,
    }


def _reconstruction_metrics(
    reconstruction: Optional[ReconstructedDiagram],
) -> Dict[str, Any]:
    if reconstruction is None:
        return {
            "reconstructed_activities": UNAVAILABLE,
            "reconstructed_events": UNAVAILABLE,
            "reconstructed_dependencies": UNAVAILABLE,
            "raw_arrow_candidates": UNAVAILABLE,
            "deduplicated_arrows": UNAVAILABLE,
            "validated_node_pairs": UNAVAILABLE,
            "validated_dependencies": UNAVAILABLE,
            "review_candidate_dependencies": UNAVAILABLE,
            "rejected_dependencies": UNAVAILABLE,
            "self_loops": UNAVAILABLE,
            "duplicate_edges": UNAVAILABLE,
        }
    metadata = dict(getattr(reconstruction, "metadata", None) or {})
    report = metadata.get("dependency_validation")
    out: Dict[str, Any] = {
        "reconstructed_activities": _safe_int(reconstruction.activity_count),
        "reconstructed_events": _safe_int(reconstruction.event_count),
        "reconstructed_dependencies": _safe_int(reconstruction.dependency_count),
    }
    if report is None:
        out.update(
            {
                "raw_arrow_candidates": UNAVAILABLE,
                "deduplicated_arrows": UNAVAILABLE,
                "validated_node_pairs": UNAVAILABLE,
                "validated_dependencies": UNAVAILABLE,
                "review_candidate_dependencies": UNAVAILABLE,
                "rejected_dependencies": UNAVAILABLE,
                "self_loops": UNAVAILABLE,
                "duplicate_edges": UNAVAILABLE,
            }
        )
        return out
    out.update(
        {
            "raw_arrow_candidates": _safe_int(
                getattr(report, "raw_arrow_count", None)
            ),
            "deduplicated_arrows": _safe_int(
                getattr(report, "deduplicated_arrow_count", None)
            ),
            "validated_node_pairs": _safe_int(
                getattr(report, "deduplicated_arrow_count", None)
            ),
            "validated_dependencies": _safe_int(
                getattr(report, "accepted_count", None)
            ),
            "review_candidate_dependencies": _safe_int(
                getattr(report, "review_count", None)
            ),
            "rejected_dependencies": _safe_int(
                getattr(report, "rejected_count", None)
            ),
            "self_loops": _safe_int(getattr(report, "self_loop_count", None)),
            "duplicate_edges": _safe_int(
                getattr(report, "duplicate_edge_count", None)
            ),
        }
    )
    return out


def _candidate_metrics(candidate: Any) -> Dict[str, Any]:
    if candidate is None:
        return {
            "graph_status": UNAVAILABLE,
            "graph_is_valid": UNAVAILABLE,
            "cpm_gate": UNAVAILABLE,
            "cpm_project_duration": UNAVAILABLE,
            "critical_path_count": UNAVAILABLE,
        }
    validation = getattr(candidate, "validation", None)
    status = None
    is_valid = None
    if validation is not None:
        status = _str(getattr(validation, "status", None))
        is_valid = bool(getattr(validation, "is_valid", False))
    gate = getattr(candidate, "cpm_gate", None)
    if hasattr(gate, "value"):
        gate_value = gate.value
    else:
        gate_value = gate
    cpm = getattr(candidate, "cpm", None)
    duration = None
    critical_count = None
    if cpm is not None:
        duration = _safe_float(getattr(cpm, "project_duration", None))
        critical_count = _len_of(
            getattr(candidate, "pure_critical_paths", None), None
        )
        if critical_count is None:
            critical_count = _len_of(getattr(cpm, "critical_paths", None))
    return {
        "graph_status": status if status is not None else UNAVAILABLE,
        "graph_is_valid": is_valid if is_valid is not None else UNAVAILABLE,
        "cpm_gate": (
            str(gate_value) if gate_value is not None else UNAVAILABLE
        ),
        "cpm_project_duration": (
            duration if duration is not None else UNAVAILABLE
        ),
        "critical_path_count": (
            critical_count if critical_count is not None else UNAVAILABLE
        ),
    }


def _review_item_metrics(session: Any) -> Dict[str, Any]:
    if session is None:
        return {"review_items": UNAVAILABLE}
    n_act = _len_of(getattr(session, "activities", None), 0)
    n_dep = _len_of(getattr(session, "dependencies", None), 0)
    n_dur = _len_of(getattr(session, "durations", None), 0)
    total = sum(x for x in (n_act, n_dep, n_dur) if x is not None)
    return {"review_items": total}


def extract_stage_metrics(workflow: Any) -> Dict[str, Any]:
    """Extract every measured stage metric from a ``ReviewWorkflow``-like object.

    ``workflow`` must expose ``pipeline_result``, ``review_session`` and
    ``apply()``.  Deterministic: safe accessors only; a missing stage
    yields ``UNAVAILABLE``.
    """
    result: Optional[PipelineResult] = getattr(
        workflow, "pipeline_result", None
    ) or getattr(workflow, "result", None)
    session = getattr(workflow, "review_session", None)

    candidate = None
    apply = getattr(workflow, "apply", None)
    if callable(apply):
        try:
            candidate = apply()
        except Exception:
            candidate = None

    shape = getattr(result, "_shape_result", None) if result is not None else None
    arrow = getattr(result, "_arrow_result", None) if result is not None else None
    ocr = getattr(result, "_ocr_result", None) if result is not None else None
    reconstruction = (
        getattr(result, "_reconstruction", None) if result is not None else None
    )

    metrics: Dict[str, Any] = {}
    metrics.update(_shape_metrics(shape))
    metrics.update(_arrow_metrics(arrow))
    metrics.update(_ocr_metrics(ocr))
    metrics.update(_reconstruction_metrics(reconstruction))
    metrics.update(_review_item_metrics(session))
    metrics.update(_candidate_metrics(candidate))
    metrics["pert_status"] = UNAVAILABLE
    return metrics


def extract_stage_progression(metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the ordered stage-progression list for the Markdown report."""
    # Primary measured counts through the AON pipeline (upward "expansion"
    # metrics) and their intermediate difference signals.
    counts: List[Tuple[str, Any]] = [
        ("contours_analyzed", metrics.get("contours_analyzed")),
        ("raw_shapes", metrics.get("raw_shapes")),
        ("final_candidate_nodes", metrics.get("final_candidate_nodes")),
        ("reconstructed_activities", metrics.get("reconstructed_activities")),
    ]
    row = {"stage": "Shape/candidate expansion (measured)", "counts": dict(counts)}
    pipeline_counts: List[Tuple[str, Any]] = [
        ("raw_arrow_segments", metrics.get("raw_arrow_segments")),
        ("deduplicated_arrows", metrics.get("deduplicated_arrows")),
        ("validated_node_pairs", metrics.get("validated_node_pairs")),
        ("validated_dependencies", metrics.get("validated_dependencies")),
        ("reconstructed_dependencies", metrics.get("reconstructed_dependencies")),
    ]
    row2 = {"stage": "Arrow/dependency pipeline (measured)", "counts": dict(pipeline_counts)}
    ocr_counts: List[Tuple[str, Any]] = [
        ("ocr_regions", metrics.get("ocr_regions")),
        ("ocr_labels", metrics.get("ocr_labels")),
        ("ocr_id_candidates", metrics.get("ocr_id_candidates")),
        ("ocr_numeric_candidates", metrics.get("ocr_numeric_candidates")),
    ]
    row3 = {"stage": "OCR text (measured)", "counts": dict(ocr_counts)}
    graph_counts: List[Tuple[str, Any]] = [
        ("graph_status", metrics.get("graph_status")),
        ("graph_is_valid", metrics.get("graph_is_valid")),
        ("cpm_gate", metrics.get("cpm_gate")),
        ("cpm_project_duration", metrics.get("cpm_project_duration")),
        ("critical_path_count", metrics.get("critical_path_count")),
        ("pert_status", metrics.get("pert_status")),
    ]
    row4 = {"stage": "Graph validation / CPM / PERT", "counts": dict(graph_counts)}
    return [row, row2, row3, row4]


def extract_stage_timings(result: Optional[PipelineResult]) -> Dict[str, Any]:
    """Extract per-stage elapsed seconds when present in ``stages``."""
    timing: Dict[str, Any] = {}
    if result is None:
        return {k: UNAVAILABLE for k in _STAGE_TIMING_KEYS}
    stages = getattr(result, "stages", None)
    if not isinstance(stages, dict):
        return {k: UNAVAILABLE for k in _STAGE_TIMING_KEYS}
    for key in _STAGE_TIMING_KEYS:
        entry = stages.get(key)
        if isinstance(entry, dict):
            timing[key] = _safe_float(
                entry.get("elapsed_seconds")
                or entry.get("duration_seconds")
                or entry.get("elapsed")
            )
        elif isinstance(entry, numbers.Number):
            timing[key] = float(entry)
        else:
            timing[key] = UNAVAILABLE
    return timing


def classify_outcome(workflow: Any) -> str:
    """Classify the benchmark outcome for one image.

    OUTCOME CLASSES
    ---------------
    AUTOMATIC_SUCCESS
        Pipeline finished with AnalysisStatus.SUCCESS and CPM ran.
    AUTOMATIC_REVIEW_REQUIRED
        Pipeline correctly routed the image to review (REVIEW_REQUIRED),
        CPM blocked until review, or final status is not SUCCESS/FAILED.
    FATAL_FAILURE
        Pipeline crashed or explicitly failed the analysis.
    """
    result: Optional[PipelineResult] = getattr(
        workflow, "pipeline_result", None
    ) or getattr(workflow, "result", None)

    final_status = None
    if result is not None:
        status = getattr(result, "status", None)
        if status is not None:
            final_status = (
                status.value if hasattr(status, "value") else str(status)
            )

    if final_status in (str(AnalysisStatus.FAILED.value), "FAILED"):
        return "FATAL_FAILURE"
    if final_status in (str(AnalysisStatus.SUCCESS.value), "SUCCESS"):
        return "AUTOMATIC_SUCCESS"
    if final_status in (
        str(AnalysisStatus.REVIEW_REQUIRED.value),
        str(AnalysisStatus.NOT_SUPPORTED.value),
        "REVIEW_REQUIRED",
        "NOT_SUPPORTED",
    ):
        return "AUTOMATIC_REVIEW_REQUIRED"

    # Last-resort signal: did CPM actually run on the reviewed candidate?
    candidate = None
    apply = getattr(workflow, "apply", None)
    if callable(apply):
        try:
            candidate = apply()
        except Exception:
            candidate = None
    if candidate is not None:
        cpm = getattr(candidate, "cpm", None)
        if cpm is not None and getattr(cpm, "project_duration", None) is not None:
            return "AUTOMATIC_SUCCESS"
        gate = getattr(candidate, "cpm_gate", None)
        gate_value = gate.value if hasattr(gate, "value") else gate
        if gate_value == CpmGateStatus.RUNNABLE.value:
            return "AUTOMATIC_SUCCESS"
    return "AUTOMATIC_REVIEW_REQUIRED" if final_status else "FATAL_FAILURE"
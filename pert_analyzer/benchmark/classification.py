"""
Evidence-based failure classification for the generalization benchmark.

Every classification is derived from measured pipeline counts only;
rules are conservative, correctly routed REVIEW_REQUIRED outcomes are
never real failures, and when no measured signal is conclusive the
class is ``UNKNOWN`` rather than guessed.

Heuristic thresholds are documented constants, deliberately generous:

* ``ACTIVITY_COUNT_ABNORMAL`` — a measured final candidate count above
  this is treated as abnormal for this small corpus (the 11 real
  images contain roughly 3-25 visual network nodes depending on
  diagram scale; the reference diagram used for the system's dev data
  has 22).
* ``DUPLICATE_FRACTION_MEANINGFUL`` — when deduplication removed less
  than this fraction of raw shapes, the count explosion is attributed
  to over-detection rather than failed deduplication.

The corpus is mixed: rectangular diagrams reconstruct as AON (each
candidate node becomes one activity), while circle-event diagrams
reconstruct as AOA (each detected arrow becomes one activity).  For
AOA the activity count is therefore 1:1 with the arrow count, so an
inflated final activity count originates at arrow detection; for AON
the same inflation originates at node reconstruction.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

# -------------------------------------------------------------------------
# Failure class vocabulary (stable, documented in the report).
# -------------------------------------------------------------------------


class FailureClass:
    SHAPE_DETECTION = "SHAPE_DETECTION"
    SHAPE_DEDUPLICATION = "SHAPE_DEDUPLICATION"
    NODE_RECONSTRUCTION = "NODE_RECONSTRUCTION"
    OCR = "OCR"
    DURATION_EXTRACTION = "DURATION_EXTRACTION"
    ARROW_DETECTION = "ARROW_DETECTION"
    NODE_PAIRING = "NODE_PAIRING"
    DIRECTION = "DIRECTION"
    SEMANTIC_RECONSTRUCTION = "SEMANTIC_RECONSTRUCTION"
    GRAPH_BUILD = "GRAPH_BUILD"
    OTHER = "OTHER"
    UNKNOWN = "UNKNOWN"


OUTCOME_SUCCESS = "AUTOMATIC_SUCCESS"
OUTCOME_REVIEW_REQUIRED = "AUTOMATIC_REVIEW_REQUIRED"
OUTCOME_FAILURE = "FATAL_FAILURE"

ACTIVITY_COUNT_ABNORMAL = 40
DUPLICATE_FRACTION_MEANINGFUL = 0.25

_UNAVAILABLE = "UNAVAILABLE"


def _as_int(value: Any) -> Optional[int]:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _evidence(**kwargs: Any) -> Dict[str, Any]:
    return {k: v for k, v in kwargs.items() if v is not None}


def classify_first_abnormal_stage(
    metrics: Dict[str, Any],
) -> Tuple[Optional[str], str, Dict[str, Any]]:
    """Return (pipeline_stage, failure_class, evidence) for one image.

    The pipeline stage string uses the canonical names from
    ``pipeline/progress.py``; ``None`` means no abnormal stage was found.
    """
    candidates = _as_int(metrics.get("final_candidate_nodes"))
    raw_shapes = _as_int(metrics.get("raw_shapes"))
    duplicates = _as_int(metrics.get("duplicates_suppressed"))
    recon_activities = _as_int(metrics.get("reconstructed_activities"))
    arrows = _as_int(metrics.get("arrows_detected"))
    dedup_arrows = _as_int(metrics.get("deduplicated_arrows"))
    validated_deps = _as_int(metrics.get("validated_dependencies"))
    recon_deps = _as_int(metrics.get("reconstructed_dependencies"))
    ocr_regions = _as_int(metrics.get("ocr_regions"))
    ocr_labels = _as_int(metrics.get("ocr_labels"))
    graph_status = metrics.get("graph_status")
    graph_is_valid = metrics.get("graph_is_valid")
    cpm_gate = metrics.get("cpm_gate")

    is_aoa = str(metrics.get("diagram_type") or "").upper() == "AOA"

    def _count_inflation() -> Tuple[str, str, Dict[str, Any]]:
        """Attribute an inflated activity count to its origin stage.

        In AOA every detected arrow becomes one activity
        (``reconstruct_aoa`` maps arrows to activities 1:1), so the
        count explosion originates in arrow detection.  In AON each
        candidate node becomes one activity and the explosion
        originates in node reconstruction.
        """
        if is_aoa:
            return (
                "Detecting arrows",
                FailureClass.ARROW_DETECTION,
                _evidence(
                    arrows_detected=arrows,
                    deduplicated_arrows=dedup_arrows,
                    reconstructed_activities=recon_activities,
                ),
            )
        return (
            "Reconstructing diagram",
            FailureClass.NODE_RECONSTRUCTION,
            _evidence(
                final_candidate_nodes=candidates,
                reconstructed_activities=recon_activities,
            ),
        )

    # --- 1. Activity-count expansion (earliest abnormal node signal) ---
    if candidates is not None and candidates >= ACTIVITY_COUNT_ABNORMAL:
        if (
            recon_activities is not None
            and recon_activities >= ACTIVITY_COUNT_ABNORMAL
            and (candidates == 0 or recon_activities > candidates * 2)
        ):
            return _count_inflation()
        duplicate_fraction = (
            (duplicates / raw_shapes) if (raw_shapes or 0) > 0 else None
        )
        if duplicate_fraction is None or duplicate_fraction < DUPLICATE_FRACTION_MEANINGFUL:
            return (
                "Detecting shapes",
                FailureClass.SHAPE_DETECTION,
                _evidence(
                    raw_shapes=raw_shapes,
                    duplicates_suppressed=duplicates,
                    final_candidate_nodes=candidates,
                    duplicate_fraction=(
                        round(duplicate_fraction, 3)
                        if duplicate_fraction is not None
                        else None
                    ),
                ),
            )
        return (
            "Detecting shapes",
            FailureClass.SHAPE_DEDUPLICATION,
            _evidence(
                raw_shapes=raw_shapes,
                duplicates_suppressed=duplicates,
                final_candidate_nodes=candidates,
                duplicate_fraction=round(duplicate_fraction, 3),
            ),
        )

    if (
        recon_activities is not None
        and recon_activities >= ACTIVITY_COUNT_ABNORMAL
    ):
        return _count_inflation()

    # --- 2. Arrow / dependency collapse (middle of the pipeline) ---
    if candidates is not None and candidates > 0:
        no_dedup_arrows = dedup_arrows is not None and dedup_arrows == 0
        no_detected_arrows = arrows is not None and arrows == 0
        if no_detected_arrows or no_dedup_arrows:
            return (
                "Detecting arrows",
                FailureClass.ARROW_DETECTION,
                _evidence(
                    final_candidate_nodes=candidates,
                    arrows_detected=arrows,
                    deduplicated_arrows=dedup_arrows,
                ),
            )
        if validated_deps == 0 or recon_deps == 0:
            stage = (
                "Associating text"
                if ocr_labels is not None and ocr_labels > 0
                else "Building graph"
            )
            cls = (
                FailureClass.ARROW_DETECTION
                if dedup_arrows is None
                else FailureClass.NODE_PAIRING
            )
            return (
                stage,
                cls,
                _evidence(
                    final_candidate_nodes=candidates,
                    deduplicated_arrows=dedup_arrows,
                    validated_dependencies=validated_deps,
                ),
            )

    # --- 3. OCR / duration extraction (later stages) ---
    if candidates is not None and candidates > 0 and recon_activities is not None and recon_activities > 0:
        if graph_is_valid is False or cpm_gate == "BLOCKED_REVIEW":
            if ocr_labels == 0 and (ocr_regions is None or ocr_regions == 0):
                return (
                    "Extracting text (OCR)",
                    FailureClass.OCR,
                    _evidence(
                        final_candidate_nodes=candidates,
                        ocr_regions=ocr_regions,
                        ocr_labels=ocr_labels,
                        graph_status=graph_status,
                    ),
                )
            if ocr_labels is not None and ocr_labels > 0 and recon_activities > ocr_labels * 3:
                return (
                    "Associating text",
                    FailureClass.OCR,
                    _evidence(
                        reconstructed_activities=recon_activities,
                        ocr_labels=ocr_labels,
                    ),
                )

    # --- 4. Graph build ---
    if graph_is_valid is False:
        return (
            "Building graph",
            FailureClass.GRAPH_BUILD,
            _evidence(
                graph_status=graph_status,
                cpm_gate=cpm_gate,
                reconstructed_activities=recon_activities,
            ),
        )

    return (None, FailureClass.UNKNOWN, {})


def summarize_bottlenecks(
    results: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rank failure classes by how many images they first explain."""
    counts: Dict[str, int] = {}
    for entry in results:
        cls = (entry.get("failure_classification") or {}).get("failure_class")
        if cls is None or cls == FailureClass.UNKNOWN:
            cls = FailureClass.UNKNOWN
        counts[cls] = counts.get(cls, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"failure_class": cls, "image_count": count} for cls, count in ranked
    ]
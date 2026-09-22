"""
Graph-based activity reconciliation.

Reconciles visual activity candidates into the correct semantic
activity set by analyzing graph connectivity, identifying START/FINISH
nodes, filtering false positives, and reconciling IDs and durations.

Pipeline:

    Detected Rectangles
            ↓
    Candidate Activities
            ↓
    Graph Connectivity Analysis
            ↓
    START / FINISH identification
            ↓
    False-positive filtering
            ↓
    Activity identity reconciliation
            ↓
    GraphModel
            ↓
    CPM / Review

This module does NOT modify the ArrowDetector geometry algorithm.
It operates on the already-resolved dependency graph.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pert_analyzer.cv.reconstruction_models import (
    ActivityStatus,
    AmbiguityIssue,
    AmbiguityType,
    EvidenceTrace,
    IDSource,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ValidationResult,
    ValidationSeverity,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Enums
# =============================================================================


class CandidateVerdict(Enum):
    """Verdict for a candidate activity after reconciliation."""

    LIKELY_REAL = "likely_real"
    LIKELY_FALSE_POSITIVE = "likely_false_positive"
    REVIEW_REQUIRED = "review_required"
    EXCLUDED_AS_START = "excluded_as_start"
    EXCLUDED_AS_FINISH = "excluded_as_finish"


class ReconciliationStatus(Enum):
    """Overall reconciliation status."""

    COMPLETE = "complete"
    PARTIAL = "partial"
    REVIEW_REQUIRED = "review_required"


class GraphConsistencyStatus(Enum):
    """Graph consistency check result."""

    VALID = "valid"
    HAS_SELF_LOOPS = "has_self_loops"
    HAS_DUPLICATE_DEPS = "has_duplicate_deps"
    HAS_CYCLES = "has_cycles"
    HAS_ISOLATED_NODES = "has_isolated_nodes"


# =============================================================================
# Data Models
# =============================================================================


@dataclass
class CandidateActivity:
    """A single candidate activity with full evidence."""

    activity: ReconstructedActivity
    predecessor_ids: List[str] = field(default_factory=list)
    successor_ids: List[str] = field(default_factory=list)
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    arrow_confidence_sum: float = 0.0
    arrow_confidence_avg: float = 0.0
    verdict: CandidateVerdict = CandidateVerdict.REVIEW_REQUIRED
    verdict_reasons: List[str] = field(default_factory=list)
    verdict_confidence: float = 0.0
    is_start_candidate: bool = False
    is_finish_candidate: bool = False
    duplicate_ids: List[str] = field(default_factory=list)
    id_alternatives: List[str] = field(default_factory=list)
    duration_alternatives: List[float] = field(default_factory=list)
    cluster_similarity: float = 0.0


@dataclass
class FilteredCandidate:
    """A candidate that was filtered out during reconciliation."""

    candidate_id: str
    activity_id: str
    reason: str
    evidence: str
    confidence: float
    original_activity: ReconstructedActivity


@dataclass
class DependencyReconciliation:
    """Result of dependency reconciliation."""

    total_deps: int = 0
    preserved_deps: int = 0
    removed_self_arrows: int = 0
    removed_duplicate_deps: int = 0
    missing_dep_source: Optional[str] = None
    missing_dep_target: Optional[str] = None
    missing_dep_evidence: str = ""
    unresolved_arrows: int = 0


@dataclass
class GraphConsistency:
    """Result of graph consistency checks."""

    status: GraphConsistencyStatus = GraphConsistencyStatus.VALID
    self_loop_count: int = 0
    duplicate_dep_count: int = 0
    has_cycles: bool = False
    isolated_node_count: int = 0
    isolated_node_ids: List[str] = field(default_factory=list)
    issues: List[str] = field(default_factory=list)


@dataclass
class ReconciliationResult:
    """Complete result of graph-based reconciliation."""

    candidates_before: int = 0
    confirmed_activities: int = 0
    inferred_activities: int = 0
    review_required_activities: int = 0
    likely_false_positives: int = 0
    excluded_as_start: int = 0
    excluded_as_finish: int = 0
    candidates: List[CandidateActivity] = field(default_factory=list)
    filtered_candidates: List[FilteredCandidate] = field(default_factory=list)
    final_activities: List[ReconstructedActivity] = field(default_factory=list)
    final_dependencies: List[ReconstructedDependency] = field(default_factory=list)
    dependency_reconciliation: DependencyReconciliation = field(
        default_factory=DependencyReconciliation
    )
    graph_consistency: GraphConsistency = field(default_factory=GraphConsistency)
    reconciliation_status: ReconciliationStatus = ReconciliationStatus.REVIEW_REQUIRED
    cpm_status: str = "REVIEW_REQUIRED"
    cpm_project_duration: Optional[float] = None
    cpm_critical_paths: List[List[str]] = field(default_factory=list)
    diagnostic_table: str = ""
    warnings: List[str] = field(default_factory=list)


# =============================================================================
# Reconciliation Engine
# =============================================================================


class ReconciliationEngine:
    """
    Graph-based activity reconciliation engine.

    Analyzes the reconstructed diagram to:
    1. Compute graph connectivity for each candidate
    2. Identify START/FINISH nodes
    3. Filter false-positive candidates
    4. Reconcile activity IDs
    5. Reconcile durations
    6. Verify graph consistency
    7. Run CPM if possible
    """

    def __init__(
        self,
        start_finish_ocr_keywords: Optional[List[str]] = None,
        min_degree_for_real: int = 1,
        false_positive_confidence_threshold: float = 0.3,
    ):
        self.start_finish_keywords = (
            start_finish_ocr_keywords or ["START", "FINISH", "END", "BEGIN"]
        )
        self.min_degree_for_real = min_degree_for_real
        self.fp_confidence_threshold = false_positive_confidence_threshold

    # =========================================================================
    # Public API
    # =========================================================================

    def reconcile(
        self,
        diagram: ReconstructedDiagram,
        reference_activities: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> ReconciliationResult:
        """
        Run full reconciliation on a reconstructed diagram.

        Args:
            diagram: The reconstructed diagram from Phase 7.
            reference_activities: Optional reference data for evaluation
                (ONLY used in test/evaluation layer, NOT in production logic).

        Returns:
            ReconciliationResult with full diagnostic info.
        """
        result = ReconciliationResult()
        result.candidates_before = diagram.activity_count

        # Step 1: Build candidate activity list with graph connectivity
        candidates = self._analyze_connectivity(diagram)
        result.candidates = candidates

        # Step 2: Identify START/FINISH
        self._identify_start_finish(candidates, diagram)

        # Step 3: False-positive detection
        self._detect_false_positives(candidates, diagram)

        # Step 4: Activity ID reconciliation
        self._reconcile_ids(candidates)

        # Step 5: Duration reconciliation
        self._reconcile_durations(candidates)

        # Step 6: Count results
        self._count_results(candidates, result)

        # Step 7: Build final activity and dependency lists
        self._build_final_elements(candidates, diagram, result)

        # Step 8: Dependency reconciliation
        self._reconcile_dependencies(diagram, result)

        # Step 9: Graph consistency
        self._check_graph_consistency(result)

        # Step 10: CPM
        self._run_cpm(result)

        # Step 11: Generate diagnostic table
        result.diagnostic_table = self._generate_diagnostic_table(candidates)

        # Step 12: Overall status
        self._determine_overall_status(result)

        logger.info(
            "Reconciliation: %d -> %d confirmed, %d inferred, %d review, %d FP",
            result.candidates_before,
            result.confirmed_activities,
            result.inferred_activities,
            result.review_required_activities,
            result.likely_false_positives,
        )

        return result

    # =========================================================================
    # Step 1: Graph Connectivity Analysis
    # =========================================================================

    def _analyze_connectivity(
        self, diagram: ReconstructedDiagram
    ) -> List[CandidateActivity]:
        """Analyze graph connectivity for all activity candidates."""
        candidates: List[CandidateActivity] = []

        # Build adjacency info from dependencies
        predecessors: Dict[str, List[str]] = defaultdict(list)
        successors: Dict[str, List[str]] = defaultdict(list)
        arrow_confidence: Dict[str, List[float]] = defaultdict(list)

        for dep in diagram.dependencies:
            if dep.source_id and dep.target_id:
                predecessors[dep.target_id].append(dep.source_id)
                successors[dep.source_id].append(dep.target_id)
                arrow_confidence[dep.source_id].append(dep.confidence)
                arrow_confidence[dep.target_id].append(dep.confidence)

        # Also include ambiguities with resolved source/target
        for amb in diagram.ambiguities:
            if amb.ambiguity_type in (
                AmbiguityType.ARROW_NO_SOURCE,
                AmbiguityType.ARROW_NO_TARGET,
            ):
                pass  # These are unresolved, skip

        for act in diagram.activities:
            pred_ids = predecessors.get(act.activity_id, [])
            succ_ids = successors.get(act.activity_id, [])
            confs = arrow_confidence.get(act.activity_id, [])

            cand = CandidateActivity(
                activity=act,
                predecessor_ids=pred_ids,
                successor_ids=succ_ids,
                in_degree=len(pred_ids),
                out_degree=len(succ_ids),
                total_degree=len(pred_ids) + len(succ_ids),
                arrow_confidence_sum=sum(confs),
                arrow_confidence_avg=(sum(confs) / len(confs)) if confs else 0.0,
            )
            candidates.append(cand)

        return candidates

    # =========================================================================
    # Step 2: START/FINISH Identification
    # =========================================================================

    def _identify_start_finish(
        self,
        candidates: List[CandidateActivity],
        diagram: ReconstructedDiagram,
    ) -> None:
        """Identify START and FINISH candidates using multiple heuristics.

        START/FINISH are only identified when there is strong evidence:
        - OCR text explicitly says START/FINISH
        - Node is clearly a terminal with very specific characteristics

        We do NOT mark every in_degree=0 node as START. Many real
        activities legitimately have in_degree=0 because arrow detection
        misses some connections.
        """
        for cand in candidates:
            act = cand.activity
            reasons_start: List[str] = []
            reasons_finish: List[str] = []

            # Heuristic 1: OCR text matches START/FINISH keywords (STRONG evidence)
            label_upper = act.label.upper().strip()
            for kw in self.start_finish_keywords:
                if kw in label_upper:
                    if kw in ("START", "BEGIN"):
                        reasons_start.append(f"OCR text contains '{kw}'")
                    elif kw in ("FINISH", "END"):
                        reasons_finish.append(f"OCR text contains '{kw}'")

            # Heuristic 2: Node has an OCR-confirmed alphabetic ID
            # (e.g., "A", "B", etc.) — this is a real activity, NOT START/FINISH
            # Must have both a confirmed ID AND actual OCR text evidence
            has_real_id = (
                act.activity_id
                and not act.activity_id.startswith("INFERRED_")
                and len(act.activity_id) <= 3
                and act.activity_id[0].isalpha()
                and len(act.source_text_region_ids) > 0
            )

            # Heuristic 3: Graph degree — only use as WEAK signal
            # A node with in_degree=0 and HIGH out_degree (>=3) might be START
            # but only if it has no real activity ID
            if not has_real_id and not reasons_start:
                if cand.in_degree == 0 and cand.out_degree >= 3:
                    reasons_start.append(
                        f"in_degree=0, out_degree={cand.out_degree} (weak)"
                    )

            # Heuristic 4: Graph degree — only FINISH if out_degree=0 and HIGH in_degree
            if not has_real_id and not reasons_finish:
                if cand.out_degree == 0 and cand.in_degree >= 3:
                    reasons_finish.append(
                        f"out_degree=0, in_degree={cand.in_degree} (weak)"
                    )

            # Heuristic 5: Spatial position — only use as WEAK signal
            # Only consider if no other evidence exists
            if not reasons_start and not reasons_finish and not has_real_id:
                if act.position:
                    all_positions = [
                        c.activity.position
                        for c in candidates
                        if c.activity.position
                    ]
                    if all_positions:
                        min_x = min(p[0] for p in all_positions)
                        max_x = max(p[0] for p in all_positions)
                        min_y = min(p[1] for p in all_positions)
                        max_y = max(p[1] for p in all_positions)
                        range_x = max_x - min_x if max_x > min_x else 1.0
                        range_y = max_y - min_y if max_y > min_y else 1.0

                        norm_x = (act.position[0] - min_x) / range_x
                        norm_y = (act.position[1] - min_y) / range_y

                        # Only mark as START if clearly top-left AND isolated
                        if norm_x < 0.1 and norm_y < 0.1 and cand.total_degree <= 1:
                            reasons_start.append("position: top-left, isolated")
                        # Only mark as FINISH if clearly bottom-right AND isolated
                        if norm_x > 0.9 and norm_y > 0.9 and cand.total_degree <= 1:
                            reasons_finish.append("position: bottom-right, isolated")

            # Determine START/FINISH — require at least one STRONG reason
            has_strong_start = any("OCR" in r or "contains" in r for r in reasons_start)
            has_strong_finish = any("OCR" in r or "contains" in r for r in reasons_finish)

            if reasons_start and (has_strong_start or cand.in_degree == 0):
                # Only mark as START if: strong OCR evidence, OR
                # in_degree=0 AND no real activity ID
                if has_strong_start or (not has_real_id and cand.in_degree == 0):
                    cand.is_start_candidate = True
                    cand.verdict_reasons.extend(reasons_start)

            if reasons_finish and (has_strong_finish or cand.out_degree == 0):
                # Only mark as FINISH if: strong OCR evidence, OR
                # out_degree=0 AND no real activity ID
                if has_strong_finish or (not has_real_id and cand.out_degree == 0):
                    cand.is_finish_candidate = True
                    cand.verdict_reasons.extend(reasons_finish)

    # =========================================================================
    # Step 3: False-Positive Detection (Visual Consistency Dominates)
    # =========================================================================

    def _compute_shape_cluster(
        self, candidates: List[CandidateActivity]
    ) -> Dict[str, Any]:
        """Compute the dominant activity-node shape cluster.

        Uses the bounding box dimensions of all candidates to identify
        the typical activity rectangle size. Candidates matching this
        cluster are likely real activities.
        """
        widths: List[float] = []
        heights: List[float] = []
        areas: List[float] = []
        aspect_ratios: List[float] = []

        for cand in candidates:
            bb = cand.activity.bounding_box
            if bb and bb.width > 0 and bb.height > 0:
                widths.append(bb.width)
                heights.append(bb.height)
                areas.append(bb.width * bb.height)
                aspect_ratios.append(bb.width / bb.height)

        if not widths:
            return {"count": 0}

        # Compute median dimensions (robust to outliers)
        widths_sorted = sorted(widths)
        heights_sorted = sorted(heights)
        areas_sorted = sorted(areas)
        ar_sorted = sorted(aspect_ratios)

        n = len(widths_sorted)
        median_w = widths_sorted[n // 2]
        median_h = heights_sorted[n // 2]
        median_area = areas_sorted[n // 2]
        median_ar = ar_sorted[n // 2]

        # Compute IQR for outlier detection
        q1_w = widths_sorted[n // 4] if n >= 4 else widths_sorted[0]
        q3_w = widths_sorted[3 * n // 4] if n >= 4 else widths_sorted[-1]
        q1_h = heights_sorted[n // 4] if n >= 4 else heights_sorted[0]
        q3_h = heights_sorted[3 * n // 4] if n >= 4 else heights_sorted[-1]

        return {
            "count": n,
            "median_width": median_w,
            "median_height": median_h,
            "median_area": median_area,
            "median_aspect_ratio": median_ar,
            "q1_width": q1_w,
            "q3_width": q3_w,
            "q1_height": q1_h,
            "q3_height": q3_h,
        }

    def _shape_matches_cluster(
        self,
        cand: CandidateActivity,
        cluster: Dict[str, Any],
    ) -> Tuple[bool, float, List[str]]:
        """Check if a candidate matches the dominant shape cluster.

        Returns:
            (matches, similarity_score, reasons)
        """
        if cluster.get("count", 0) < 3:
            # Not enough data for cluster analysis — assume matches
            return True, 0.5, ["insufficient cluster data"]

        bb = cand.activity.bounding_box
        if not bb or bb.width <= 0 or bb.height <= 0:
            return False, 0.0, ["no bounding box"]

        reasons: List[str] = []
        similarity = 1.0

        # Check width similarity (within IQR range)
        w = bb.width
        if w < cluster["q1_width"] * 0.3 or w > cluster["q3_width"] * 3.0:
            similarity -= 0.4
            reasons.append(f"width={w:.0f} far from cluster median={cluster['median_width']:.0f}")
        elif w < cluster["q1_width"] * 0.5 or w > cluster["q3_width"] * 2.0:
            similarity -= 0.2
            reasons.append(f"width={w:.0f} moderately off from cluster")

        # Check height similarity
        h = bb.height
        if h < cluster["q1_height"] * 0.3 or h > cluster["q3_height"] * 3.0:
            similarity -= 0.4
            reasons.append(f"height={h:.0f} far from cluster median={cluster['median_height']:.0f}")
        elif h < cluster["q1_height"] * 0.5 or h > cluster["q3_height"] * 2.0:
            similarity -= 0.2
            reasons.append(f"height={h:.0f} moderately off from cluster")

        # Check aspect ratio
        ar = w / h if h > 0 else 0
        median_ar = cluster["median_aspect_ratio"]
        if median_ar > 0:
            ar_ratio = ar / median_ar
            if ar_ratio < 0.3 or ar_ratio > 3.0:
                similarity -= 0.3
                reasons.append(f"aspect_ratio={ar:.2f} far from cluster={median_ar:.2f}")
            elif ar_ratio < 0.5 or ar_ratio > 2.0:
                similarity -= 0.15
                reasons.append(f"aspect_ratio={ar:.2f} moderately off from cluster")

        similarity = max(0.0, similarity)
        matches = similarity >= 0.4

        if not reasons:
            reasons.append("matches cluster dimensions")

        return matches, similarity, reasons

    def _detect_false_positives(
        self,
        candidates: List[CandidateActivity],
        diagram: ReconstructedDiagram,
    ) -> None:
        """Detect likely false-positive activity candidates.

        CRITICAL: Visual consistency with the dominant shape cluster is the
        PRIMARY criterion. Arrow connectivity is only WEAK supporting evidence.

        A rectangle that matches the visual style of other activity nodes
        should NEVER be discarded solely because arrows are missing.
        Arrow detection is imperfect — missing arrows are expected.
        """
        # Step 1: Compute the dominant shape cluster
        cluster = self._compute_shape_cluster(candidates)

        for cand in candidates:
            act = cand.activity

            # Skip START/FINISH
            if cand.is_start_candidate or cand.is_finish_candidate:
                continue

            # Step 2: Check visual consistency with cluster (PRIMARY criterion)
            matches_cluster, cluster_similarity, cluster_reasons = (
                self._shape_matches_cluster(cand, cluster)
            )
            cand.cluster_similarity = cluster_similarity

            # Step 3: Collect negative evidence (MULTIPLE required for FP)
            negative_count = 0
            fp_reasons: List[str] = []

            # Negative 1: Shape does NOT match cluster
            if not matches_cluster:
                negative_count += 1
                fp_reasons.extend(cluster_reasons)

            # Negative 2: Very low shape confidence
            if act.confidence < 0.15:
                negative_count += 1
                fp_reasons.append(f"very low shape confidence ({act.confidence:.3f})")

            # Negative 3: Extremely small (likely noise/artifact)
            if act.bounding_box:
                area = act.bounding_box.width * act.bounding_box.height
                if area < 100:  # extremely small
                    negative_count += 1
                    fp_reasons.append(f"extremely small bounding box (area={area:.0f})")

            # Negative 4: Outside the main diagram region
            if act.position and candidates:
                all_positions = [
                    c.activity.position
                    for c in candidates
                    if c.activity.position
                ]
                if all_positions:
                    min_x = min(p[0] for p in all_positions)
                    max_x = max(p[0] for p in all_positions)
                    min_y = min(p[1] for p in all_positions)
                    max_y = max(p[1] for p in all_positions)
                    margin = 50  # pixels outside the cluster boundary
                    if (act.position[0] < min_x - margin
                            or act.position[0] > max_x + margin
                            or act.position[1] < min_y - margin
                            or act.position[1] > max_y + margin):
                        negative_count += 1
                        fp_reasons.append("outside main diagram region")

            # Step 4: Weak supporting evidence (does NOT contribute to FP score)
            # These are logged for review but do NOT cause rejection
            if cand.total_degree == 0:
                fp_reasons.append("no detected arrows (weak — arrow detection imperfect)")
            if not act.source_text_region_ids:
                fp_reasons.append("no OCR text (weak — OCR is optional)")
            if act.duration <= 0:
                fp_reasons.append("no duration (weak — duration may be missing)")

            # Step 5: Determine verdict
            # REQUIRE multiple negative signals (≥2) to mark as FP
            # Single negative signal → REVIEW_REQUIRED
            if negative_count >= 2:
                cand.verdict = CandidateVerdict.LIKELY_FALSE_POSITIVE
                cand.verdict_reasons.extend(fp_reasons)
                cand.verdict_confidence = min(0.3 * negative_count, 1.0)
            elif negative_count == 1:
                cand.verdict = CandidateVerdict.REVIEW_REQUIRED
                cand.verdict_reasons.extend(fp_reasons)
                cand.verdict_confidence = 0.5
            else:
                # No negative signals — likely real
                cand.verdict = CandidateVerdict.LIKELY_REAL
                cand.verdict_confidence = max(cluster_similarity, 0.5)
                if fp_reasons:  # log weak evidence
                    cand.verdict_reasons.extend(fp_reasons)

    # =========================================================================
    # Step 4: Activity ID Reconciliation
    # =========================================================================

    def _reconcile_ids(self, candidates: List[CandidateActivity]) -> None:
        """Reconcile activity IDs, handling duplicates and missing IDs."""
        # Build ID → candidates mapping
        id_to_candidates: Dict[str, List[CandidateActivity]] = defaultdict(list)
        for cand in candidates:
            act_id = cand.activity.activity_id
            id_to_candidates[act_id].append(cand)

        # Find duplicates
        for act_id, cands in id_to_candidates.items():
            if len(cands) > 1 and not act_id.startswith("INFERRED_"):
                for c in cands:
                    c.duplicate_ids = [
                        other.activity.activity_id
                        for other in cands
                        if other is not c
                    ]
                    if c.verdict == CandidateVerdict.LIKELY_REAL:
                        c.verdict = CandidateVerdict.REVIEW_REQUIRED
                        c.verdict_reasons.append(
                            f"Duplicate ID '{act_id}' shared with "
                            f"{len(cands) - 1} other candidate(s)"
                        )

        # For inferred IDs, check if there are better alternatives
        for cand in candidates:
            if cand.activity.activity_id.startswith("INFERRED_"):
                # Check if any OCR regions could provide an ID
                if cand.activity.source_text_region_ids:
                    cand.id_alternatives.append("OCR_evidence_available")

    # =========================================================================
    # Step 5: Duration Reconciliation
    # =========================================================================

    def _reconcile_durations(self, candidates: List[CandidateActivity]) -> None:
        """Reconcile durations, collecting alternatives."""
        for cand in candidates:
            act = cand.activity

            # Collect duration alternatives from evidence
            if act.duration > 0:
                cand.duration_alternatives.append(act.duration)

            # Check metadata for alternative durations
            if "duration_candidates" in act.metadata:
                for d in act.metadata["duration_candidates"]:
                    if d not in cand.duration_alternatives:
                        cand.duration_alternatives.append(d)

    # =========================================================================
    # Step 6: Count Results
    # =========================================================================

    def _count_results(
        self,
        candidates: List[CandidateActivity],
        result: ReconciliationResult,
    ) -> None:
        """Count activities by status."""
        for cand in candidates:
            if cand.verdict == CandidateVerdict.EXCLUDED_AS_START:
                result.excluded_as_start += 1
            elif cand.verdict == CandidateVerdict.EXCLUDED_AS_FINISH:
                result.excluded_as_finish += 1
            elif cand.verdict == CandidateVerdict.LIKELY_FALSE_POSITIVE:
                result.likely_false_positives += 1
            elif cand.activity.status == ActivityStatus.CONFIRMED:
                result.confirmed_activities += 1
            elif cand.activity.status == ActivityStatus.INFERRED:
                result.inferred_activities += 1
            else:
                result.review_required_activities += 1

    # =========================================================================
    # Step 7: Build Final Elements
    # =========================================================================

    def _build_final_elements(
        self,
        candidates: List[CandidateActivity],
        diagram: ReconstructedDiagram,
        result: ReconciliationResult,
    ) -> None:
        """Build final activity and dependency lists after reconciliation."""
        # Collect IDs of activities to keep
        kept_ids: Set[str] = set()
        for cand in candidates:
            if cand.verdict in (
                CandidateVerdict.LIKELY_REAL,
                CandidateVerdict.REVIEW_REQUIRED,
            ):
                kept_ids.add(cand.activity.activity_id)
            elif cand.verdict == CandidateVerdict.EXCLUDED_AS_START:
                # Keep START as a special activity
                kept_ids.add(cand.activity.activity_id)
            elif cand.verdict == CandidateVerdict.EXCLUDED_AS_FINISH:
                # Keep FINISH as a special activity
                kept_ids.add(cand.activity.activity_id)

        # Build final activities
        result.final_activities = []
        for cand in candidates:
            if cand.activity.activity_id in kept_ids:
                result.final_activities.append(cand.activity)

        # Build final dependencies (only between kept activities)
        result.final_dependencies = []
        for dep in diagram.dependencies:
            if dep.source_id in kept_ids and dep.target_id in kept_ids:
                result.final_dependencies.append(dep)

        # Store filtered candidates
        for cand in candidates:
            if cand.verdict == CandidateVerdict.LIKELY_FALSE_POSITIVE:
                result.filtered_candidates.append(
                    FilteredCandidate(
                        candidate_id=cand.activity.source_node_id or "",
                        activity_id=cand.activity.activity_id,
                        reason="; ".join(cand.verdict_reasons),
                        evidence=f"confidence={cand.verdict_confidence:.3f}",
                        confidence=cand.verdict_confidence,
                        original_activity=cand.activity,
                    )
                )

    # =========================================================================
    # Step 8: Dependency Reconciliation
    # =========================================================================

    def _reconcile_dependencies(
        self,
        diagram: ReconstructedDiagram,
        result: ReconciliationResult,
    ) -> None:
        """Reconcile dependencies, removing self-arrows and duplicates."""
        dep_recon = DependencyReconciliation()
        dep_recon.total_deps = len(diagram.dependencies)

        # Remove self-arrows
        non_self_deps = [
            d for d in result.final_dependencies if d.source_id != d.target_id
        ]
        dep_recon.removed_self_arrows = (
            len(result.final_dependencies) - len(non_self_deps)
        )

        # Remove duplicate dependencies
        seen_deps: Set[Tuple[str, str]] = set()
        unique_deps: List[ReconstructedDependency] = []
        for dep in non_self_deps:
            key = (dep.source_id, dep.target_id)
            if key not in seen_deps:
                seen_deps.add(key)
                unique_deps.append(dep)
            else:
                dep_recon.removed_duplicate_deps += 1

        result.final_dependencies = unique_deps
        dep_recon.preserved_deps = len(unique_deps)

        # Identify the missing dependency (if any)
        # This is evaluation-only: compare to reference if provided
        dep_recon.unresolved_arrows = sum(
            1 for d in diagram.dependencies
            if not d.source_id or not d.target_id
        )

        result.dependency_reconciliation = dep_recon

    # =========================================================================
    # Step 9: Graph Consistency
    # =========================================================================

    def _check_graph_consistency(self, result: ReconciliationResult) -> None:
        """Verify graph consistency after reconciliation."""
        consistency = GraphConsistency()

        # Check self-loops
        for dep in result.final_dependencies:
            if dep.source_id == dep.target_id:
                consistency.self_loop_count += 1

        # Check duplicate dependencies
        seen: Set[Tuple[str, str]] = set()
        for dep in result.final_dependencies:
            key = (dep.source_id, dep.target_id)
            if key in seen:
                consistency.duplicate_dep_count += 1
            seen.add(key)

        # Check for cycles using DFS
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for dep in result.final_dependencies:
            if dep.source_id and dep.target_id:
                adjacency[dep.source_id].append(dep.target_id)

        all_node_ids = set()
        for act in result.final_activities:
            all_node_ids.add(act.activity_id)

        # DFS cycle detection
        visited: Set[str] = set()
        rec_stack: Set[str] = set()

        def _has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            for neighbor in adjacency.get(node, []):
                if neighbor not in visited:
                    if _has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            rec_stack.discard(node)
            return False

        for node in all_node_ids:
            if node not in visited:
                if _has_cycle(node):
                    consistency.has_cycles = True
                    break

        # Check isolated nodes
        connected_nodes: Set[str] = set()
        for dep in result.final_dependencies:
            connected_nodes.add(dep.source_id)
            connected_nodes.add(dep.target_id)

        isolated = all_node_ids - connected_nodes
        consistency.isolated_node_count = len(isolated)
        consistency.isolated_node_ids = list(isolated)

        # Determine overall status
        if consistency.self_loop_count > 0:
            consistency.status = GraphConsistencyStatus.HAS_SELF_LOOPS
            consistency.issues.append(
                f"{consistency.self_loop_count} self-loop(s) detected"
            )
        elif consistency.duplicate_dep_count > 0:
            consistency.status = GraphConsistencyStatus.HAS_DUPLICATE_DEPS
            consistency.issues.append(
                f"{consistency.duplicate_dep_count} duplicate dependency(ies)"
            )
        elif consistency.has_cycles:
            consistency.status = GraphConsistencyStatus.HAS_CYCLES
            consistency.issues.append("Cycle detected in dependency graph")
        elif consistency.isolated_node_count > 0:
            consistency.status = GraphConsistencyStatus.HAS_ISOLATED_NODES
            consistency.issues.append(
                f"{consistency.isolated_node_count} isolated node(s): "
                f"{consistency.isolated_node_ids}"
            )
        else:
            consistency.status = GraphConsistencyStatus.VALID

        result.graph_consistency = consistency

    # =========================================================================
    # Step 10: CPM
    # =========================================================================

    def _run_cpm(self, result: ReconciliationResult) -> None:
        """Run CPM if the graph is valid and sufficient data is available."""
        # Only run CPM if graph is consistent
        if result.graph_consistency.status != GraphConsistencyStatus.VALID:
            result.cpm_status = "REVIEW_REQUIRED"
            result.warnings.append(
                f"CPM skipped: graph consistency = {result.graph_consistency.status.value}"
            )
            return

        # Check if all non-dummy activities have durations
        activities_without_duration = [
            act for act in result.final_activities
            if not act.is_dummy and act.duration <= 0
        ]
        if activities_without_duration:
            result.cpm_status = "REVIEW_REQUIRED"
            result.warnings.append(
                f"CPM skipped: {len(activities_without_duration)} activities "
                f"missing duration"
            )
            return

        # Build GraphModel and run CPM
        try:
            from pert_analyzer.analysis.cpm_engine import CPMEngine
            from pert_analyzer.core.models import DiagramType
            from pert_analyzer.graph.builder import GraphBuilder

            builder = GraphBuilder(diagram_type=DiagramType.AON)
            for act in result.final_activities:
                builder.add_activity(
                    activity_id=act.activity_id,
                    duration=act.duration,
                    name=act.label or act.activity_id,
                    is_dummy=act.is_dummy,
                )
            for dep in result.final_dependencies:
                if dep.source_id and dep.target_id:
                    try:
                        builder.add_dependency(dep.source_id, dep.target_id)
                    except Exception:
                        pass

            graph = builder.build()
            engine = CPMEngine()
            cpm_result = engine.calculate(graph)

            result.cpm_status = "VALID"
            result.cpm_project_duration = cpm_result.project_duration
            result.cpm_critical_paths = cpm_result.critical_paths

        except Exception as e:
            result.cpm_status = "ERROR"
            result.warnings.append(f"CPM calculation failed: {e}")

    # =========================================================================
    # Step 11: Diagnostic Table
    # =========================================================================

    def _generate_diagnostic_table(
        self, candidates: List[CandidateActivity]
    ) -> str:
        """Generate a diagnostic table of all candidates."""
        lines = [
            "=" * 135,
            "DIAGNOSTIC TABLE: Activity Candidate Analysis",
            "=" * 135,
            f"{'ID':<15s} {'Label':<8s} {'Dur':>4s} {'InDeg':>5s} {'OutDeg':>6s} "
            f"{'ArrowConf':>9s} {'ClstrSim':>8s} {'Status':<18s} {'Verdict':<25s}",
            "-" * 135,
        ]

        for cand in candidates:
            act = cand.activity
            lines.append(
                f"{act.activity_id:<15s} "
                f"{act.label[:7]:<8s} "
                f"{act.duration:4.0f} "
                f"{cand.in_degree:5d} "
                f"{cand.out_degree:6d} "
                f"{cand.arrow_confidence_avg:9.3f} "
                f"{cand.cluster_similarity:8.3f} "
                f"{act.status.value:<18s} "
                f"{cand.verdict.value:<25s}"
            )

        lines.append("-" * 135)
        lines.append(
            f"Total: {len(candidates)} candidates, "
            f"confirmed={sum(1 for c in candidates if c.verdict == CandidateVerdict.LIKELY_REAL and c.activity.status == ActivityStatus.CONFIRMED)}, "
            f"inferred={sum(1 for c in candidates if c.activity.status == ActivityStatus.INFERRED)}, "
            f"review={sum(1 for c in candidates if c.verdict == CandidateVerdict.REVIEW_REQUIRED)}, "
            f"FP={sum(1 for c in candidates if c.verdict == CandidateVerdict.LIKELY_FALSE_POSITIVE)}, "
            f"START={sum(1 for c in candidates if c.verdict == CandidateVerdict.EXCLUDED_AS_START)}, "
            f"FINISH={sum(1 for c in candidates if c.verdict == CandidateVerdict.EXCLUDED_AS_FINISH)}"
        )
        lines.append("=" * 135)

        return "\n".join(lines)

    # =========================================================================
    # Step 12: Overall Status
    # =========================================================================

    def _determine_overall_status(self, result: ReconciliationResult) -> None:
        """Determine overall reconciliation status."""
        if result.likely_false_positives > 0 or result.review_required_activities > 0:
            result.reconciliation_status = ReconciliationStatus.REVIEW_REQUIRED
        elif (
            result.graph_consistency.status == GraphConsistencyStatus.VALID
            and result.confirmed_activities > 0
        ):
            result.reconciliation_status = ReconciliationStatus.COMPLETE
        else:
            result.reconciliation_status = ReconciliationStatus.PARTIAL

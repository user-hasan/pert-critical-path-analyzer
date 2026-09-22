"""
Contextual semantic resolution for AON activity IDs and durations.

Resolves ambiguous OCR candidates using evidence from node-local OCR,
alternatives, geometry, duplicate-ID constraints, and numeric validity.

Does NOT use reference answers — resolution is purely evidence-based.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from pert_analyzer.cv.region_ocr import NodeOCRResult

logger = logging.getLogger(__name__)


class ResolutionSource(Enum):
    """Source of a resolved value."""

    OCR_HIGH_CONFIDENCE = "ocr_high_confidence"
    OCR_ALTERNATIVE = "ocr_alternative"
    CONTEXTUAL_RESOLUTION = "contextual_resolution"
    GRAPH_TOPOLOGY = "graph_topology"
    INFERRED = "inferred"
    REVIEW_REQUIRED = "review_required"


class ResolutionStatus(Enum):
    """Status of a resolved field."""

    CONFIRMED = "confirmed"
    CONTEXTUALLY_RESOLVED = "contextually_resolved"
    INFERRED = "inferred"
    REVIEW_REQUIRED = "review_required"


@dataclass
class ResolutionEvidence:
    """Evidence supporting a resolution decision."""

    reason: str
    supporting_factors: List[str] = field(default_factory=list)
    confidence_contribution: float = 0.0


@dataclass
class CandidateResolution:
    """Resolution result for a single activity ID."""

    node_id: str
    raw_id: Optional[str] = None
    resolved_id: Optional[str] = None
    alternatives: List[Tuple[str, float]] = field(default_factory=list)
    source: ResolutionSource = ResolutionSource.REVIEW_REQUIRED
    status: ResolutionStatus = ResolutionStatus.REVIEW_REQUIRED
    evidence: List[ResolutionEvidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class DurationResolution:
    """Resolution result for a single duration."""

    node_id: str
    raw_value: Optional[float] = None
    resolved_value: Optional[float] = None
    raw_text: Optional[str] = None
    source: ResolutionSource = ResolutionSource.REVIEW_REQUIRED
    status: ResolutionStatus = ResolutionStatus.REVIEW_REQUIRED
    evidence: List[ResolutionEvidence] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class SemanticResolutionResult:
    """Complete result of semantic resolution for all nodes."""

    id_resolutions: Dict[str, CandidateResolution] = field(default_factory=dict)
    duration_resolutions: Dict[str, DurationResolution] = field(default_factory=dict)
    summary: Dict[str, int] = field(default_factory=dict)


class SemanticResolver:
    """
    Resolves ambiguous OCR activity IDs and durations using contextual evidence.

    Evidence sources:
    - Node-local OCR confidence and alternatives
    - Duplicate-ID constraints (uniqueness)
    - Valid activity-ID format (A-Z single letter)
    - Duration numeric validity and region membership
    - Spatial relationships (no position-based ID assignment)
    """

    VALID_ACTIVITY_IDS: Set[str] = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def resolve_all(
        self,
        node_results: List[NodeOCRResult],
    ) -> SemanticResolutionResult:
        """
        Resolve all activity IDs and durations from node-local OCR results.

        Args:
            node_results: Per-node OCR results from RegionOCRProcessor.

        Returns:
            SemanticResolutionResult with resolutions for every node.
        """
        result = SemanticResolutionResult()

        # Pass 1: Resolve activity IDs
        id_resolutions = self._resolve_all_ids(node_results)
        result.id_resolutions = {r.node_id: r for r in id_resolutions}

        # Pass 2: Resolve durations
        dur_resolutions = self._resolve_all_durations(node_results)
        result.duration_resolutions = {r.node_id: r for r in dur_resolutions}

        # Summary
        result.summary = self._compute_summary(id_resolutions, dur_resolutions)
        return result

    # =========================================================================
    # ID Resolution
    # =========================================================================

    def _resolve_all_ids(
        self, node_results: List[NodeOCRResult]
    ) -> List[CandidateResolution]:
        """Resolve activity IDs for all nodes."""
        resolutions = [self._resolve_single_id(nr) for nr in node_results]
        self._resolve_duplicate_ids(resolutions)
        return resolutions

    def _resolve_single_id(self, nr: NodeOCRResult) -> CandidateResolution:
        """Resolve a single node's activity ID."""
        cr = CandidateResolution(node_id=nr.node_id)

        # Record raw OCR data
        cr.raw_id = nr.best_activity_id
        cr.alternatives = list(nr.activity_id_alternatives)
        cr.confidence = nr.best_activity_id_confidence

        if nr.best_activity_id is None:
            cr.source = ResolutionSource.REVIEW_REQUIRED
            cr.status = ResolutionStatus.REVIEW_REQUIRED
            cr.evidence.append(ResolutionEvidence(
                reason="No OCR text detected",
                supporting_factors=["OCR returned no single-letter match"],
            ))
            return cr

        raw = nr.best_activity_id.upper()
        conf = nr.best_activity_id_confidence

        # High confidence: accept directly
        if conf >= 0.70:
            cr.resolved_id = raw
            cr.source = ResolutionSource.OCR_HIGH_CONFIDENCE
            cr.status = ResolutionStatus.CONFIRMED
            cr.evidence.append(ResolutionEvidence(
                reason=f"OCR high confidence ({conf:.2f})",
                supporting_factors=[f"raw='{raw}'", f"conf={conf:.3f}"],
                confidence_contribution=conf,
            ))
            return cr

        # Medium confidence with valid single letter: check alternatives first
        if raw in self.VALID_ACTIVITY_IDS and conf >= 0.40:
            # If alternatives exist, prefer the best plausible one
            if cr.alternatives:
                sorted_alts = sorted(cr.alternatives, key=lambda x: x[1], reverse=True)
                for alt_id, alt_conf in sorted_alts:
                    if alt_id in self.VALID_ACTIVITY_IDS and alt_conf >= 0.30:
                        cr.resolved_id = alt_id
                        cr.source = ResolutionSource.OCR_ALTERNATIVE
                        cr.status = ResolutionStatus.CONTEXTUALLY_RESOLVED
                        cr.evidence.append(ResolutionEvidence(
                            reason=f"Alternative '{alt_id}' (conf={alt_conf:.2f}) preferred over raw '{raw}' (conf={conf:.2f})",
                            supporting_factors=[
                                f"raw='{raw}' conf={conf:.3f}",
                                f"alt='{alt_id}' conf={alt_conf:.3f}",
                                f"alt_conf/raw_conf ratio={alt_conf/max(conf,0.01):.2f}",
                            ],
                            confidence_contribution=alt_conf,
                        ))
                        return cr

            # No alternatives: accept raw
            cr.resolved_id = raw
            cr.source = ResolutionSource.OCR_HIGH_CONFIDENCE
            cr.status = ResolutionStatus.CONFIRMED
            cr.evidence.append(ResolutionEvidence(
                reason=f"OCR acceptable confidence ({conf:.2f}), no better alternative",
                supporting_factors=[f"raw='{raw}'", f"conf={conf:.3f}"],
                confidence_contribution=conf,
            ))
            return cr

        # Low confidence or has better alternative: try alternatives
        if cr.alternatives:
            # Sort alternatives by confidence
            sorted_alts = sorted(cr.alternatives, key=lambda x: x[1], reverse=True)
            for alt_id, alt_conf in sorted_alts:
                if alt_id in self.VALID_ACTIVITY_IDS and alt_conf >= 0.30:
                    cr.resolved_id = alt_id
                    cr.source = ResolutionSource.OCR_ALTERNATIVE
                    cr.status = ResolutionStatus.CONTEXTUALLY_RESOLVED
                    cr.evidence.append(ResolutionEvidence(
                        reason=f"Alternative '{alt_id}' (conf={alt_conf:.2f}) preferred over raw '{raw}' (conf={conf:.2f})",
                        supporting_factors=[
                            f"raw='{raw}' conf={conf:.3f}",
                            f"alt='{alt_id}' conf={alt_conf:.3f}",
                            f"alt_conf/raw_conf ratio={alt_conf/max(conf,0.01):.2f}",
                        ],
                        confidence_contribution=alt_conf,
                    ))
                    return cr

        # Accept raw if valid letter
        if raw in self.VALID_ACTIVITY_IDS:
            cr.resolved_id = raw
            cr.source = ResolutionSource.OCR_HIGH_CONFIDENCE
            cr.status = ResolutionStatus.CONFIRMED
            cr.evidence.append(ResolutionEvidence(
                reason=f"Valid letter '{raw}' at confidence {conf:.2f}",
                supporting_factors=[f"raw='{raw}'", f"conf={conf:.3f}"],
                confidence_contribution=conf,
            ))
            return cr

        # Invalid letter: mark for review
        cr.resolved_id = None
        cr.source = ResolutionSource.REVIEW_REQUIRED
        cr.status = ResolutionStatus.REVIEW_REQUIRED
        cr.evidence.append(ResolutionEvidence(
            reason=f"Invalid activity ID '{raw}'",
            supporting_factors=[f"raw='{raw}'", f"conf={conf:.3f}"],
        ))
        return cr

    def _resolve_duplicate_ids(self, resolutions: List[CandidateResolution]) -> None:
        """
        Resolve duplicate ID conflicts.

        When two nodes get the same ID, uses alternatives and evidence
        to disambiguate. Only falls back to INFERRED if no resolution found.
        """
        id_map: Dict[str, List[CandidateResolution]] = {}
        for cr in resolutions:
            rid = cr.resolved_id or cr.raw_id or ""
            if rid:
                id_map.setdefault(rid, []).append(cr)

        for aid, conflicts in id_map.items():
            if len(conflicts) <= 1:
                continue

            # Sort by confidence descending
            conflicts.sort(key=lambda c: c.confidence, reverse=True)

            # Keep the highest-confidence one as-is
            primary = conflicts[0]
            primary.evidence.append(ResolutionEvidence(
                reason=f"Primary assignment for '{aid}' (highest confidence)",
                supporting_factors=[f"conf={primary.confidence:.3f}"],
            ))

            # For each duplicate, try alternatives
            used_ids = {primary.resolved_id}
            for dup in conflicts[1:]:
                alt_resolved = False

                # Try alternatives in confidence order
                for alt_id, alt_conf in sorted(
                    dup.alternatives, key=lambda x: x[1], reverse=True
                ):
                    if (
                        alt_id in self.VALID_ACTIVITY_IDS
                        and alt_id not in used_ids
                    ):
                        dup.resolved_id = alt_id
                        dup.source = ResolutionSource.CONTEXTUAL_RESOLUTION
                        dup.status = ResolutionStatus.CONTEXTUALLY_RESOLVED
                        dup.evidence.append(ResolutionEvidence(
                            reason=f"Duplicate '{aid}' resolved to alternative '{alt_id}' (uniqueness constraint)",
                            supporting_factors=[
                                f"original='{dup.raw_id}'",
                                f"alt='{alt_id}' conf={alt_conf:.3f}",
                                f"conflict with node_id={primary.node_id}",
                            ],
                            confidence_contribution=alt_conf,
                        ))
                        used_ids.add(alt_id)
                        alt_resolved = True
                        break

                if not alt_resolved:
                    # No valid alternative: mark for review with uniqueness conflict
                    dup.source = ResolutionSource.REVIEW_REQUIRED
                    dup.status = ResolutionStatus.REVIEW_REQUIRED
                    dup.resolved_id = None
                    dup.evidence.append(ResolutionEvidence(
                        reason=f"Duplicate '{aid}' with no valid alternative (uniqueness conflict)",
                        supporting_factors=[
                            f"raw='{dup.raw_id}'",
                            f"alternatives={[a[0] for a in dup.alternatives]}",
                            f"used_ids={sorted(used_ids)}",
                        ],
                    ))

    # =========================================================================
    # Duration Resolution
    # =========================================================================

    def _resolve_all_durations(
        self, node_results: List[NodeOCRResult]
    ) -> List[DurationResolution]:
        """Resolve durations for all nodes."""
        return [self._resolve_single_duration(nr) for nr in node_results]

    def _resolve_single_duration(self, nr: NodeOCRResult) -> DurationResolution:
        """Resolve a single node's duration from numeric candidates."""
        dr = DurationResolution(node_id=nr.node_id)

        if not nr.numeric_candidates:
            dr.source = ResolutionSource.REVIEW_REQUIRED
            dr.status = ResolutionStatus.REVIEW_REQUIRED
            dr.evidence.append(ResolutionEvidence(
                reason="No numeric candidates found",
            ))
            return dr

        # Rank candidates by evidence
        ranked = self._rank_duration_candidates(nr)
        if not ranked:
            dr.source = ResolutionSource.REVIEW_REQUIRED
            dr.status = ResolutionStatus.REVIEW_REQUIRED
            dr.evidence.append(ResolutionEvidence(
                reason="No valid duration candidates after ranking",
            ))
            return dr

        best_value, best_raw, best_score, best_reasons = ranked[0]
        dr.raw_value = nr.best_numeric[0] if nr.best_numeric else None
        dr.raw_text = nr.best_numeric[1] if nr.best_numeric else None
        dr.resolved_value = best_value
        dr.confidence = best_score

        if best_score >= 0.60:
            dr.source = ResolutionSource.OCR_HIGH_CONFIDENCE
            dr.status = ResolutionStatus.CONFIRMED
        elif best_score >= 0.30:
            dr.source = ResolutionSource.CONTEXTUAL_RESOLUTION
            dr.status = ResolutionStatus.CONTEXTUALLY_RESOLVED
        else:
            dr.source = ResolutionSource.REVIEW_REQUIRED
            dr.status = ResolutionStatus.REVIEW_REQUIRED

        dr.evidence.append(ResolutionEvidence(
            reason=f"Best duration candidate: {best_value} (from '{best_raw}', score={best_score:.2f})",
            supporting_factors=best_reasons,
            confidence_contribution=best_score,
        ))

        return dr

    def _rank_duration_candidates(
        self, nr: NodeOCRResult
    ) -> List[Tuple[float, str, float, List[str]]]:
        """
        Rank duration candidates by multiple evidence factors.

        Returns list of (value, raw_text, score, reasons) sorted by score desc.
        """
        scored = []
        for value, raw_text, conf in nr.numeric_candidates:
            score = 0.0
            reasons = []

            # Factor 1: OCR confidence (0.0 - 0.3)
            score += min(conf, 1.0) * 0.3
            reasons.append(f"ocr_conf={conf:.3f}")

            # Factor 2: Numeric format validity (0.0 - 0.2)
            if self._is_valid_duration_format(raw_text):
                score += 0.2
                reasons.append("valid_format")
            else:
                reasons.append("invalid_format")

            # Factor 3: Reasonable range (0.0 - 0.2)
            if 0 < value <= 20:
                score += 0.2
                reasons.append(f"in_range(0,20]")
            else:
                reasons.append(f"out_of_range({value})")

            # Factor 4: Integer vs decimal (0.0 - 0.1)
            if value == int(value):
                score += 0.1
                reasons.append("integer")

            # Factor 5: Pure numeric (no extra text) (0.0 - 0.1)
            if raw_text.strip() == str(int(value)) if value == int(value) else raw_text.strip() == str(value):
                score += 0.1
                reasons.append("pure_numeric")

            # Factor 6: Duration sub-crop region bonus (0.0 - 0.1)
            # If numeric came from duration_sub_crop_regions, boost it
            for dur_region in nr.duration_sub_crop_regions:
                if dur_region.parsed_value == value:
                    score += 0.1
                    reasons.append("from_duration_sub_crop")
                    break

            scored.append((value, raw_text, score, reasons))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    def _is_valid_duration_format(self, raw_text: str) -> bool:
        """Check if raw text is a clean numeric format."""
        text = raw_text.strip()
        if not text:
            return False
        # Pure integer
        if text.isdigit():
            return True
        # Decimal with single dot
        parts = text.split(".")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return True
        return False

    # =========================================================================
    # Summary
    # =========================================================================

    def _compute_summary(
        self,
        id_resolutions: List[CandidateResolution],
        dur_resolutions: List[DurationResolution],
    ) -> Dict[str, int]:
        """Compute summary counts."""
        id_counts = {"confirmed": 0, "contextually_resolved": 0, "inferred": 0, "review_required": 0}
        for cr in id_resolutions:
            if cr.status == ResolutionStatus.CONFIRMED:
                id_counts["confirmed"] += 1
            elif cr.status == ResolutionStatus.CONTEXTUALLY_RESOLVED:
                id_counts["contextually_resolved"] += 1
            elif cr.status == ResolutionStatus.INFERRED:
                id_counts["inferred"] += 1
            else:
                id_counts["review_required"] += 1

        dur_counts = {"confirmed": 0, "resolved": 0, "review_required": 0}
        for dr in dur_resolutions:
            if dr.status == ResolutionStatus.CONFIRMED:
                dur_counts["confirmed"] += 1
            elif dr.status == ResolutionStatus.CONTEXTUALLY_RESOLVED:
                dur_counts["resolved"] += 1
            else:
                dur_counts["review_required"] += 1

        return {"id": id_counts, "duration": dur_counts}

    # =========================================================================
    # Graph-Consistency Correction
    # =========================================================================

    def resolve_with_topology(
        self,
        resolution_result: SemanticResolutionResult,
        activities: List[Any],
        dependencies: List[Any],
    ) -> SemanticResolutionResult:
        """
        Apply graph-consistency correction to ambiguous IDs.

        For each node with REVIEW_REQUIRED or CONTEXTUALLY_RESOLVED status,
        generates ranked candidate interpretations and scores them using
        graph topology evidence.

        Args:
            resolution_result: Initial semantic resolution result.
            activities: List of ReconstructedActivity objects.
            dependencies: List of ReconstructedDependency objects.

        Returns:
            Updated SemanticResolutionResult with topology-corrected IDs.
        """
        act_map = {a.activity_id: a for a in activities}
        existing_ids = {a.activity_id for a in activities}

        # Build adjacency for topology analysis
        adj_out: Dict[str, Set[str]] = defaultdict(set)
        adj_in: Dict[str, Set[str]] = defaultdict(set)
        for dep in dependencies:
            if dep.source_id in existing_ids and dep.target_id in existing_ids:
                adj_out[dep.source_id].add(dep.target_id)
                adj_in[dep.target_id].add(dep.source_id)

        for cr in resolution_result.id_resolutions.values():
            # Only attempt correction for REVIEW_REQUIRED or low-confidence
            if cr.status not in (ResolutionStatus.REVIEW_REQUIRED,):
                continue

            raw = cr.raw_id
            if raw is None:
                continue

            # Generate candidate interpretations
            candidates = self._generate_id_candidates(
                raw, cr.alternatives, existing_ids
            )

            # Score each candidate using topology
            scored = self._score_topology_evidence(
                candidates, cr, adj_out, adj_in, existing_ids, act_map
            )

            if scored:
                best_id, best_score, best_reasons = scored[0]
                # Only resolve if score margin is strong enough
                if best_score >= 0.6 and best_id != raw:
                    cr.resolved_id = best_id
                    cr.source = ResolutionSource.GRAPH_TOPOLOGY
                    cr.status = ResolutionStatus.CONTEXTUALLY_RESOLVED
                    cr.evidence.append(ResolutionEvidence(
                        reason=f"Graph topology: '{raw}' corrected to '{best_id}' (score={best_score:.2f})",
                        supporting_factors=best_reasons,
                        confidence_contribution=best_score,
                    ))
                    logger.debug(
                        "Topology correction: node=%s raw=%s → %s (score=%.2f)",
                        cr.node_id, raw, best_id, best_score,
                    )

        return resolution_result

    def _generate_id_candidates(
        self,
        raw_id: str,
        alternatives: List[Tuple[str, float]],
        existing_ids: Set[str],
    ) -> List[Tuple[str, float, str]]:
        """
        Generate ranked candidate interpretations for a suspicious ID.

        Returns list of (candidate_id, base_score, source_reason).
        """
        candidates = []
        seen = set()

        # 1. Raw OCR result
        if raw_id not in seen:
            candidates.append((raw_id, 0.5, "ocr_raw"))
            seen.add(raw_id)

        # 2. OCR alternatives
        for alt_id, alt_conf in alternatives:
            if alt_id not in seen:
                candidates.append((alt_id, alt_conf * 0.8, "ocr_alternative"))
                seen.add(alt_id)

        # 3. Character confusion candidates
        confusable = self._get_confusable_chars(raw_id)
        for conf_id in confusable:
            if conf_id not in seen and conf_id in existing_ids:
                candidates.append((conf_id, 0.3, "character_confusion"))
                seen.add(conf_id)

        # 4. Alphabetical neighbors (for single letters)
        if len(raw_id) == 1 and raw_id.isalpha():
            for offset in [-1, 1]:
                neighbor = chr(ord(raw_id.upper()) + offset)
                if neighbor.isalpha() and neighbor not in seen and neighbor in existing_ids:
                    candidates.append((neighbor, 0.2, "alphabetical_neighbor"))
                    seen.add(neighbor)

        return candidates

    def _get_confusable_chars(self, char: str) -> List[str]:
        """Get characters that are commonly confused with the given character."""
        confusion_map = {
            "C": ["G", "O", "Q"],
            "G": ["C", "6"],
            "O": ["Q", "D", "0"],
            "Q": ["O", "D"],
            "I": ["L", "1", "l"],
            "L": ["I", "1"],
            "S": ["5", "8"],
            "Z": ["2"],
            "B": ["8", "13"],
            "D": ["O", "0"],
            "U": ["V"],
            "V": ["U", "Y"],
            "M": ["N"],
            "N": ["M"],
        }
        upper = char.upper()
        return confusion_map.get(upper, [])

    def _score_topology_evidence(
        self,
        candidates: List[Tuple[str, float, str]],
        cr: CandidateResolution,
        adj_out: Dict[str, Set[str]],
        adj_in: Dict[str, Set[str]],
        existing_ids: Set[str],
        act_map: Dict[str, Any],
    ) -> List[Tuple[str, float, List[str]]]:
        """
        Score candidates using graph topology evidence.

        Returns list of (candidate_id, total_score, reasons) sorted desc.
        """
        scored = []
        for cand_id, base_score, source_reason in candidates:
            score = base_score
            reasons = [f"base={source_reason}({base_score:.2f})"]

            # Topology evidence: connectivity
            if cand_id in existing_ids:
                # Check if candidate fits graph structure
                out_edges = len(adj_out.get(cand_id, set()))
                in_edges = len(adj_in.get(cand_id, set()))
                total_edges = out_edges + in_edges

                if total_edges > 0:
                    score += 0.2
                    reasons.append(f"connected({total_edges} edges)")
                else:
                    score -= 0.1
                    reasons.append("disconnected")

            # Confidence from OCR
            score += cr.confidence * 0.1
            reasons.append(f"ocr_conf={cr.confidence:.2f}")

            # Penalty for invalid candidates
            if cand_id not in existing_ids and len(cand_id) == 1:
                score -= 0.3
                reasons.append("not_in_activities")

            scored.append((cand_id, max(score, 0.0), reasons))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

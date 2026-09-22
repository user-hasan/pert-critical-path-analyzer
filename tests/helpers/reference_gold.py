"""
Shared evaluation-only support for the reference AON acceptance tests.

Gold data lives in tests/test_data/reference_diagrams/reference_aon_expected.json
and is used for EVALUATION ONLY — it is never fed back into CV heuristics,
CPM math, or the Review Center.  All corrections here are test data.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pert_analyzer.pipeline.human_review import (
    ActivityReview,
    DependencyReview,
    DurationReview,
    DurationStatus,
    ReviewSession,
    ReviewStatus,
    build_review_session,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "test_data" / "reference_diagrams"
REFERENCE_AON = REFERENCE_DIR / "reference_aon.png"
GOLD_FIXTURE = REFERENCE_DIR / "reference_aon_expected.json"


def load_gold() -> dict:
    with open(GOLD_FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def run_reference_analysis():
    """Run the real CV pipeline on the reference AON once (cached)."""
    from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer

    analyzer = EndToEndAnalyzer()
    return analyzer.analyze(str(REFERENCE_AON))


def _nearest_gold(position, gold_positions):
    """Map a reconstructed activity position to the nearest gold id."""
    x = position.x if hasattr(position, "x") else position[0]
    y = position.y if hasattr(position, "y") else position[1]
    best_id, best_d2 = None, None
    for gid, (gx, gy) in gold_positions.items():
        d2 = (gx - x) ** 2 + (gy - y) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_id = gid
    return best_id


def correct_from_gold(session: ReviewSession, gold: dict) -> dict:
    """
    Record evaluation-only review decisions from the gold fixture.

    Every reconstructed activity is mapped to gold by geometry and gets a
    CORRECTED activity id + CORRECTED duration; every gold dependency is
    recorded as an ACCEPTED review decision.  Existing PENDING review items
    (from the CV review-candidate report) are resolved in place.  Returns
    the node_id -> gold activity id mapping.
    """
    gold_positions = {a["id"]: tuple(a["position"]) for a in gold["activities"]}
    gold_durations = {a["id"]: float(a["duration"]) for a in gold["activities"]}
    gold_edges = {tuple(e) for e in gold["gold_dependencies"]}

    recon = session.reconstruction
    node_to_gold: dict = {}
    for act in recon.activities:
        node = act.source_node_id or act.geometric_node_id or act.activity_id
        if act.position is None:
            continue
        gid = _nearest_gold(act.position, gold_positions)
        node_to_gold[node] = gid
        gdur = gold_durations[gid]

        arev = session.get_activity_review(node)
        if arev is not None:
            if arev.status == ReviewStatus.PENDING:
                arev.status = ReviewStatus.CORRECTED
                arev.corrected_activity_id = gid
                arev.reason += " | gold: ID corrected"
        else:
            session.activities.append(ActivityReview(
                geometric_node_id=node,
                current_activity_id=act.activity_id,
                confidence=0.0,
                reason="gold correction (evaluation only)",
                status=ReviewStatus.CORRECTED,
                corrected_activity_id=gid,
                raw_semantic_id=(act.semantic_activity_id or act.activity_id),
            ))

        drev = session.get_duration_review(node)
        if drev is not None:
            if drev.status == ReviewStatus.PENDING:
                drev.status = ReviewStatus.CORRECTED
                drev.corrected_duration = gdur
                drev.duration_status = DurationStatus.CORRECTED
                drev.reason += " | gold: duration corrected"
        else:
            session.durations.append(DurationReview(
                geometric_node_id=node,
                activity_id=gid,
                current_duration=act.duration,
                duration_status=DurationStatus.CORRECTED,
                reason="gold correction (evaluation only)",
                status=ReviewStatus.CORRECTED,
                corrected_duration=gdur,
            ))

    gold_node = {gid: node for node, gid in node_to_gold.items()}
    for s, t in sorted(gold_edges):
        session.dependencies.append(DependencyReview(
            arrow_id=f"gold:{s}->{t}",
            source_node_id=gold_node[s],
            target_node_id=gold_node[t],
            current_source_id=s,
            current_target_id=t,
            proposed_direction=f"{s}->{t}",
            confidence=1.0,
            reason="gold dependency (evaluation only)",
            status=ReviewStatus.ACCEPTED,
        ))
    return node_to_gold


def resolve_pending_dependency_reviews(session: ReviewSession, gold: dict) -> int:
    """
    Resolve every PENDING CV dependency review so the session reaches zero
    pending items.

    The CV arrow-candidate pairs are unreliable (that is why they are pending);
    after the canonical gold edges are accepted separately, each CV candidate
    is rejected as subsumed/erroneous.  Evaluation only.
    """
    gold_edges = {tuple(e) for e in gold["gold_dependencies"]}
    resolved = 0
    for drev in session.dependencies:
        if drev.status != ReviewStatus.PENDING:
            continue
        drev.status = ReviewStatus.REJECTED
        drev.reason += " | gold: superseded by canonical edge (evaluation only)"
        resolved += 1
    return resolved


def build_corrected_session(
    pipeline_result, source_image_id: str = "reference_aon.png",
) -> ReviewSession:
    """Build a review session with all gold review decisions applied."""
    gold = load_gold()
    session = build_review_session(pipeline_result, source_image_id=source_image_id)
    correct_from_gold(session, gold)
    resolve_pending_dependency_reviews(session, gold)
    return session


def gold_forward_backward(gold: dict) -> tuple:
    """
    Independent forward/backward pass over the gold fixture used ONLY to
    verify the backend CPM result (never to re-implement CPM in production).

    Returns (project_duration, analyses) where analyses[a_id] is a dict with
    early_start/early_finish/late_start/late_finish/total_float/free_float/
    is_critical.  Mirrors the engine's float definitions exactly.
    """
    durations = {a["id"]: float(a["duration"]) for a in gold["activities"]}
    edges = {tuple(e) for e in gold["gold_dependencies"]}
    preds: dict = {a_id: [] for a_id in durations}
    succs: dict = {a_id: [] for a_id in durations}
    for s, t in edges:
        preds[t].append(s)
        succs[s].append(t)

    es = {a_id: 0.0 for a_id in durations}
    ef = {a_id: durations[a_id] for a_id in durations}
    changed = True
    while changed:
        changed = False
        for a_id in durations:
            if preds[a_id]:
                late_start = max(ef[p] for p in preds[a_id])
            else:
                late_start = 0.0
            if es[a_id] < late_start:
                es[a_id], changed = late_start, True
                ef[a_id] = es[a_id] + durations[a_id]

    project_duration = max(ef.values()) if ef else 0.0

    lf = {a_id: project_duration for a_id in durations}
    ls = {a_id: project_duration - durations[a_id] for a_id in durations}
    changed = True
    while changed:
        changed = False
        for a_id in durations:
            if succs[a_id]:
                early_finish = min(ls[s] for s in succs[a_id])
            else:
                early_finish = project_duration
            if lf[a_id] > early_finish:
                lf[a_id], changed = early_finish, True
                ls[a_id] = lf[a_id] - durations[a_id]

    analyses = {}
    for a_id in durations:
        tf = ls[a_id] - es[a_id]
        ff = (
            min(es[s] for s in succs[a_id]) - ef[a_id]
            if succs[a_id]
            else 0.0
        )
        analyses[a_id] = {
            "early_start": es[a_id],
            "early_finish": ef[a_id],
            "late_start": ls[a_id],
            "late_finish": lf[a_id],
            "total_float": tf,
            "free_float": ff,
            "is_critical": abs(tf) < 1e-9,
        }
    return project_duration, analyses
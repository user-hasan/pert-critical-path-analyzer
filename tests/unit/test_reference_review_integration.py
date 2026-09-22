"""
Reference-fixture integration test for the human-review gate.

Drives the real CV pipeline on the reference AON diagram, builds a
review session, applies evaluation-only gold corrections (mapped by
geometry — OCR text is unreliable), and verifies that the reviewed
graph passes validation and CPM reproduces the canonical answer:

    project duration = 54.0, critical paths = 16.

Also verifies the CPM gate: before corrections the raw reconstruction
(e.g. activity Q with OCR duration 0) stays BLOCKED_REVIEW and never
silently reaches CPM.

Gold data lives in tests/test_data/reference_diagrams/
reference_aon_expected.json and is used for EVALUATION ONLY — it is
never fed back into CV heuristics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
from pert_analyzer.pipeline.human_review import (
    ActivityReview,
    DependencyReview,
    DurationReview,
    DurationStatus,
    ReviewSession,
    ReviewStatus,
    apply_review_decisions,
    build_review_session,
)

REFERENCE_DIR = Path(__file__).resolve().parents[1] / "test_data" / "reference_diagrams"
REFERENCE_AON = REFERENCE_DIR / "reference_aon.png"
GOLD_FIXTURE = REFERENCE_DIR / "reference_aon_expected.json"


@pytest.fixture(scope="module")
def gold() -> dict:
    with open(GOLD_FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def pipeline_result():
    """Run the real CV pipeline once for the whole module."""
    if not REFERENCE_AON.exists():
        pytest.skip(f"Reference image not found: {REFERENCE_AON}")
    analyzer = EndToEndAnalyzer()
    return analyzer.analyze(str(REFERENCE_AON))


def _nearest_gold(position, gold_positions):
    """Find the gold activity id whose position is closest to `position`."""
    x = position.x if hasattr(position, "x") else position[0]
    y = position.y if hasattr(position, "y") else position[1]
    best_id, best_d2 = None, None
    for gid, (gx, gy) in gold_positions.items():
        d2 = (gx - x) ** 2 + (gy - y) ** 2
        if best_d2 is None or d2 < best_d2:
            best_d2 = d2
            best_id = gid
    return best_id


def _correct_from_gold(session: ReviewSession, gold: dict) -> dict:
    """
    Record evaluation-only review decisions from the gold fixture.

    Every reconstructed activity is mapped to gold by geometry and gets
    a CORRECTED activity id + CORRECTED duration; every gold dependency
    is recorded as an ACCEPTED review decision.  Existing PENDING review
    items (from the CV review-candidate report) are resolved in place;
    items that CV accepted (or never created) are appended.
    """
    gold_positions = {a["id"]: tuple(a["position"]) for a in gold["activities"]}
    gold_durations = {a["id"]: a["duration"] for a in gold["activities"]}

    recon = session.reconstruction
    node_to_gold: dict = {}
    for act in recon.activities:
        node = act.source_node_id or act.geometric_node_id or act.activity_id
        if act.position is None:
            continue
        gid = _nearest_gold(act.position, gold_positions)
        node_to_gold[node] = gid
        gdur = float(gold_durations[gid])

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
                raw_semantic_id=(
                    act.semantic_activity_id or act.activity_id
                ),
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

    assigned = set(node_to_gold.values())
    assert len(node_to_gold) == len(gold["activities"]), (
        f"Expected {len(gold['activities'])} geometric nodes, "
        f"got {len(node_to_gold)}"
    )
    assert assigned == set(gold_positions), "Nearest-gold mapping collided"

    gold_node = {gid: node for node, gid in node_to_gold.items()}
    for s, t in gold["gold_dependencies"]:
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


@pytest.fixture(scope="module")
def corrected_session(pipeline_result, gold) -> ReviewSession:
    session = build_review_session(
        pipeline_result, source_image_id=REFERENCE_AON.name
    )
    _correct_from_gold(session, gold)
    return session


class TestCpmGateBeforeReview:
    def test_raw_reconstruction_is_blocked(self, pipeline_result) -> None:
        session = build_review_session(
            pipeline_result, source_image_id=REFERENCE_AON.name
        )
        candidate = apply_review_decisions(pipeline_result, session)

        assert candidate.cpm_gate.value == "BLOCKED_REVIEW"
        assert candidate.cpm is None
        assert candidate.cpm_project_duration is None

        codes = [e.code for e in candidate.validation.errors]
        # OCR produced zero duration for at least one activity (e.g. Q).
        assert "missing_duration" in codes


class TestReviewedGraphReproducesGold:
    def test_duration_and_critical_paths(self, corrected_session, gold) -> None:
        candidate = apply_review_decisions(
            corrected_session.reconstruction or corrected_session,
            corrected_session,
        )

        assert candidate.cpm_gate.value == "RUNNABLE"
        assert candidate.validation.is_valid

        assert candidate.graph.activity_count == gold["unique_activities"]
        assert candidate.graph.dependency_count == gold["unique_gold_dependencies"]

        assert candidate.cpm_project_duration == pytest.approx(
            gold["expected_project_duration"], abs=1e-9
        )
        assert candidate.critical_path_count == gold["expected_critical_path_count"]

        actual = sorted(sorted(p) for p in candidate.pure_critical_paths)
        expected = sorted(sorted(p) for p in gold["expected_critical_paths"])
        assert actual == expected

    def test_candidate_serialization_roundtrip(self, corrected_session) -> None:
        candidate = apply_review_decisions(
            corrected_session.reconstruction or corrected_session,
            corrected_session,
        )
        data = candidate.to_dict()
        assert data["cpm_gate"] == "RUNNABLE"
        assert data["cpm_project_duration"] == pytest.approx(54.0, abs=1e-9)
        assert data["critical_path_count"] == 16
        assert len(data["cpm_critical_paths"]) == 16
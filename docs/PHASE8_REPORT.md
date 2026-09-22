# Phase 8 — Recoverable Analysis: Object→Review→Validate→CPM Recovery

**Status:** COMPLETE (recoverable `REVIEW_REQUIRED` outcomes, recovery wiring through
the Human Review Center, and full regression green; user directive: continue).
**Scope gate:** all edits landed in `analyzer.py`, `preprocessing.py`,
`analysis_page.py`, `main_window.py`, `validation_page.py` + 18 new regression
tests (13 unit + 5 GUI). Full repo regression: **1224 passed** in 02:53.

---

## 1. Problem

On the reference AON diagram, activity **Q** is detected with an invalid duration
(`0.0`). Previously, `ReconstructedDiagram.to_graph_model()` handed
`GraphBuilder.add_activity` a zero duration for a non-dummy activity, which raised
`InvalidDurationError` inside the `graph_build` stage. That exception was wrapped as a
fatal Analysis `ERROR`, so the user could never reach the Human Review Center to fix
the value — a dead end even though the Review infrastructure (duration reviews,
`decide_duration`, validation, CPM) already existed.

Two root defects were addressed:

1. **Fatal graph-build deferral** — reviewable defects (missing/duplicate/invalid IDs,
   non-dummy `duration <= 0`) must defer to review instead of crashing.
2. **Missing final status assignment** — `analyze()` never set `SUCCESS` /
   `REVIEW_REQUIRED`; unhandled outcomes silently defaulted to `FAILED`.

## 2. What changed

### 2.1 Preprocessing log clarification — `cv/preprocessing.py`

`_apply_threshold` now logs `ADAPTIVE_MEAN` / `ADAPTIVE_GAUSSIAN` as *debug*
("applied by the adaptive threshold step ... expected"), reserving `logger.warning`
for genuinely unknown method names. The default config is `ADAPTIVE_GAUSSIAN`, so a
stock analysis run no longer prints a scary warning for expected behavior.

### 2.2 Analyzer — blocker pre-scan + final status — `pipeline/analyzer.py`

- New `_collect_graph_blockers(reconstruction) -> list[str]` runs **before** graph
  build and reports deferrable defects (no `add_error`):
  - missing / empty `activity_id` → "Activity at node 'X' is missing an identifier."
  - duplicate IDs → "Duplicate activity identifier 'X'."
  - non-dummy `duration <= 0` → "Activity 'X' has invalid duration {d}; a valid
    duration is required before CPM can run."
- `_stage_build_graph` consumes the blockers: when present it sets
  `result.review_required = True`, marks stage `graph_build` as `REVIEW_REQUIRED`
  with a warning (first 3 messages joined, `+N more`), and returns `None` **without
  adding an error**. The fatal `except Exception` path is retained for unexpected
  crashes (e.g. undecodable image).
- New `_classify_final_status(result)` is invoked at the end of `analyze()`:
  1. `result.errors`                          → `FAILED`
  2. `review_required` or any `REVIEW_REQUIRED` stage → `REVIEW_REQUIRED`
  3. `validation_passed`                      → `SUCCESS`
  4. otherwise                               → `FAILED`
- `analyze()` now explicitly assigns the status before the debug export / total-time
  bookkeeping, fixing the status-never-set latent bug.

### 2.3 GUI — Analysis page — `gui/pages/analysis_page.py`

- New signal `review_center_requested`.
- New `_open_review_btn` ("Open Review Center", objectName `secondary`, hidden by
  default) in the analyze row → emits `review_center_requested`.
- Badge text fixes: `REVIEWED?` → `REVIEW`, `VALIDATION?` → `VALIDATE`.
- `set_analysis_result_status(state_name, activity_count=None)` — for
  `REVIEW_REQUIRED` shows *"Analysis completed - review required (N activities
  detected. Some results need review before CPM can run.)"* and reveals the button.
- New `refresh(session)` keeps the page correct when the user navigates back
  (`REVIEW_REQUIRED` → button + count from `review_summary["total_activities"]`,
  `ERROR` → `set_error`, plus IMAGE_SELECTED/VALIDATION_REQUIRED handling).

### 2.4 GUI — MainWindow — `gui/main_window.py`

- `review_center_requested` connected → navigate to `NavDestination.REVIEW`.
- Header status color map: `BLOCKED REVIEW` (space-separated label) → `WARNING`;
  this also fixed a latent mismatch where the map used an underscored key against a
  space-separated label (would have rendered `TEXT_MUTED`).
- `_on_analysis_completed` passes `review_summary["total_activities"]` into
  `set_analysis_result_status` so the Analysis page message shows the real count.

### 2.5 GUI — Validation page hint — `gui/pages/validation_page.py`

New `_blocked_review_details(session) -> str` synthesizes concrete pending-item
messages from the real `ReviewSession` (enum-safe via `status.value`):
- pending duration with `current_duration <= 0` → "Activity 'Q' has an invalid duration."
- pending activity with no ID candidate → "Activity at node 'N' is missing an ID."

`_render_status` appends these to the `BLOCKED_REVIEW` hint after the pending-count
sentence, so the user immediately sees *which* item still blocks CPM.

## 3. Recovery flow (reference image, Activity Q with duration 0.0)

```
Analyze ──► REVIEW_REQUIRED (not ERROR)
   (blocker pre-scan: non-dummy duration <= 0 for Q)
        │
        v
Review Center: DurationReview for Q
   current_duration = 0.0, status PENDING,
   reason contains "missing/invalid duration", provenance cv_pipeline
        │  user: correct duration to 6.0 (decide_duration "CORRECT")
        v
Apply decisions → _build_graph_model: A(5) → B(3) → Q(6)
        │
        v
Validate ──► RUNNABLE (is_valid True), CPM gate open
        │
        v
CPM: project_duration = 14.0, critical path [A, B, Q]
```

Unresolved case stays safely gated: with Q still at `0.0` the graph model is not
built (`AnalysisResult._graph_model = None`), the candidate gate is `BLOCKED_REVIEW`,
`cpm is None`, and `run_cpm()` raises `ValueError` — nothing crashes, and the user is
always routed back into review.

## 4. Verification

- **Unit regression** — `tests/unit/test_review_required_recovery.py` (13 tests, new):
  - Reviewable defects (invalid duration, missing duration, empty/invalid activity ID,
    duplicate activity ID, ambiguous dependency) → `REVIEW_REQUIRED`, zero errors,
    resolution DIED before graph build.
  - Fatal path unchanged: undecodable/missing image → `FAILED`.
  - Blocked scenario: `ReviewSession`/`DurationReview` created for Q on `cv_pipeline`
    provenance; graph model not built; CPM remains `BLOCKED_REVIEW` + `run_cpm()`
    raises until corrected; correcting to 6.0 → `RUNNABLE`, valid, CPM duration 14,
    critical path `[A, B, Q]`.
  - Clean chain `A(5)→B(3)→C(4)` → `SUCCESS`, duration 12; default adaptive threshold
    is not fatal.
- **GUI regression** — `tests/gui/test_review_required_recovery.py` (5 tests, new):
  - Analysis page shows "Open Review Center" + `REVIEW` badge + activity count.
  - Clicking the button navigates to the Review Center.
  - REVIEW_REQUIRED analysis is NOT an error state.
  - Header shows `BLOCKED REVIEW` in the `WARNING` color.
  - Validation page `BLOCKED_REVIEW` hint lists "Activity 'A0' has an invalid duration."
- **No regressions:** `python -m pytest tests/ -q` → **1224 passed** (was 1206 at end
  of Phase 7), including all existing CV/CPM/PERT/review/GUI suites.

## 5. Remaining notes / known behavior

- `FakeReviewedCandidate(valid=True, cpm_gate="BLOCKED_REVIEW")` is the GUI fixture
  idiom for the blocked-until-reviewed state (tests never rebuild CPM math).
- The header color map now keys the space-separated label (`"BLOCKED REVIEW"`); keep
  any future enum-derived labels consistent with their formatting.
- CPM/PERT math, arrow detector, node pairing, direction resolver, and final Results
  calculations were **not** touched in this phase.
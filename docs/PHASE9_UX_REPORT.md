# Phase 9 — Manual Network Builder / Results UI Verification Report

**Status:** COMPLETE (visible Results widgets proven populated through the real
Analyze button, builder UI/UX refinements landed, and full regression green).
**Scope gate:** GUI layer only (builder page + badge/validation/network canvas/
relationship dropdowns + CPM-stale inspector handling + one results-page
verification script + regression tests). No CV / OCR / CPM / PERT algorithm code
touched. Full repo regression: **1432 passed** in 04:49 (was 1425 before this
phase; +7 new tests: 5 widget/state + 2 generated here, net 1432 = 1425 + 7).
2 pre-existing DeprecationWarnings (`uni=True` in pdf font embedding, unchanged).

---

## A. Problem Reproduced

Building a small AON network in the **Manual Network Builder** and clicking
**Analyze** navigated to Results, but in the real GUI the Results widgets lacked
a verifiable populated state. Prior verification only inspected backend data
(via `describe_ready()`/`extract()`), not the actual visible widgets, so the
empty-looking dashboard was unguarded.

## B. Root-Cause Status

Prior phase fixed the integration chain (`describe_ready`, session
`validation_status`, `_on_manual_analyze`, plus removal of `self._session.workflow
= None`). This phase **proves** that chain end-to-end from the real button and
adds the missing defensive tests.

## C. Verification Method

`scripts/verify_results_widgets.py` — drives the **actual** `_analyze_btn.click()`
and asserts widget state (not just backend data):

| Step | Assertion | Status |
|---|---|---|
| 01–02 | ResultsPage is a single instance | PASS |
| 03–05 | Navigate to builder; build A(1)/B(2,A)/C(2,A)/D(3,B,C) | PASS |
| 06–07 | Model valid after structure change | PASS |
| 08–09 | CPM computes (duration 6.0) | PASS |
| 10–11 | Analyze button enabled | PASS |
| 12–14 | Session `RESULTS_AVAILABLE`, validation `VALID` | PASS |
| 15 | Candidate present | PASS |
| 16–19 | Main stack on Results; **same** ResultsPage instance; dashboard active | PASS |
| 20–26 | `extract` ready; duration 6.0; 4 activities; 4 deps; 2 critical paths; path `A→B→D` | PASS |
| 27–32 | KPI cards: `6 days`, `4`, `4`, `4` (critical activities), `2` | PASS |
| 33–34 | Activities table 4 rows, IDs {A,B,C,D} | PASS |
| 35–36 | Network scene 4 nodes, 4 edges | PASS |
| 37 | Critical-paths list 2 entries | PASS |
| 38–39 | Overview path list 2 entries; embedded table 4 rows | PASS |
| 40 | **ALL PASS** | PASS |

**40/40 assertions pass** through the real button path.

## D. Results Widgets — Per-Widget PASS/FAIL

| Widget | State after Analyze | Status |
|---|---|---|
| KPI cards (duration/activities/deps/critical paths/critical activities) | 5 populated cards | PASS |
| Activities table | 4 rows, columns ID/Duration/ES/EF/LS/LF/TF/FF/Critical/Pred/Succ | PASS |
| Network scene | 4 nodes + 4 arrows | PASS |
| Critical paths list | 2 paths (`A→B→D`, `A→C→D`) | PASS |
| Overview tab | KPI mirror + path list + embedded activities table | PASS |
| PERT tab | deterministic-only projects show `PERT_NO_DATA_TITLE` + hint | PASS |

## E. Builder UI/UX Refinements Landed

| Item | Implementation | Status |
|---|---|---|
| State badge | `_state_badge` in header; `_update_state_badge()`; `RESULTS READY`/`NETWORK MODIFIED` via `_network_snapshot()` fingerprint | PASS |
| Validation message | never shows `0 issues: Unknown issue.`; counts errors/warnings; named invalid items | PASS |
| CPM stale | `⚠ Network modified. CPM results outdated — recalculate.` + `Recalculate CPM` button label | PASS |
| CPM-stale inspector | ES/EF/LS/LF hidden when stale; inline “outdated” notice shown instead | PASS |
| Real network canvas | `QGraphicsView`/`QGraphicsScene`; `_GraphNodeItem` + `_GraphEdgeItem` (arrowheads), click→select activity/relationship; hover; zoom in/out/fit | PASS |
| Relationship dropdowns | `From`/`To` `QComboBox` filtered to existing activities (no dangling-activity input path) | PASS |
| PERT deterministic hint | already present (`PERT_NO_DATA_HINT`) | PASS |
| Responsive layout | page constructs/resizes cleanly at 1500×900 and 1280×768 (smoke) | PASS |

Badge flow verified: `DRAFT → ERROR → VALID → CPM READY → CPM OUTDATED → DRAFT`
and `RESULTS READY → NETWORK MODIFIED` after editing.

## F. Edge-Case Notes

- A single isolated activity is an **invalid** AON network (no
  predecessor/successor): badge `ERROR`. This is intended grid semantics, not a bug.
- Double `refresh()` after Analyze is idempotent (proven harmless) — left as-is.
- `extract()` success path leaves `reason="no_analysis"` (default). Cosmetic only;
  flagged for optional cleanup, no functional impact.

## G. Regression Coverage Added

- `TestResultsWidgetsPopulatedFromAnalyzeButton` — real button path populates
  all visible Results widgets.
- `TestBuilderStateBadgeAndGraph` — badge transitions, canvas node/edge click
  selection, relationship dropdown contents, results-ready fingerprint,
  CPM-stale inspector hiding.

## H. Full Regression

**1432 passed, 2 warnings in 04:49.** No failures.

## I. Files Touched

- `pert_analyzer/gui/pages/network_builder_page.py` — badge, validation message,
  CPM-stale inspector, real canvas, relationship dropdowns, results fingerprint.
- `pert_analyzer/gui/main_window.py` — calls `mark_results_ready()` after Analyze.
- `scripts/verify_results_widgets.py` — new 40-assertion real-button verifier.
- `tests/gui/test_network_builder.py` — +7 regression tests.

## J. Conclusion

The empty-Results-widgets report is closed: 40/40 widget-state assertions pass
through the real Analyze button, using only the public signal chain
(Button → `analyze_requested` → slot → `GraphModel` → validation → `CPMEngine` →
`ReviewedGraphCandidate` → `GuiSession` → Results `refresh()`). Builder UI/UX
refinements are implemented and regression-green.
# Phase 7.1 - Interactive UX Refinement & Real-Time Analysis Feedback

**Status:** COMPLETE

**Scope gate:** real pipeline progress model/UI (no fabricated percentages) +
state-aware workflow indicator/header + review splitter & percent/breakdown +
validation/results readiness strips + subtle fade animation + latent review
widget paint crash fixed + `tests/gui/` (205 tests) green + full repo regression
(1271 tests) green + real-image manual verification on `reference_aon.png`.

---

## 1. What Phase 7.1 set out to do

Polish the interactive UX on top of the Phase 7 design system and Phase 8/9
review-validation flows, with the hard rule that every piece of feedback must
reflect **real** pipeline events:

- Show the actual analysis pipeline progress (stage checklist, overall %, real
  detected-object metrics) while the background worker runs.
- Make the workflow indicator and header status badge fully state-aware.
- Refine the Review page (3-pane splitter, progress percent + per-category
  breakdown), Validation page (readiness strip), Results page (readiness strip).
- Add a subtle, crash-safe page transition fade.
- No new dependencies, no backend math rewrites, no fake progress, and legacy
  `progress_callback(stage, progress)` calls stay byte-identical for CLI/library
  consumers.

## 2. Implementation

### 2.1 Pipeline progress catalog - `pert_analyzer/pipeline/progress.py` (new)
`PIPELINE_STAGES` = the exact stage ids the pipeline emits, in order:
`Loading image -> Preprocessing -> Detecting shapes -> Classifying diagram ->
Detecting arrows -> Extracting text (OCR) -> Associating text ->
Reconstructing diagram -> Building graph -> Validating and analyzing -> Complete`.
`STAGE_LABELS`, state constants (`PENDING/RUNNING/COMPLETED/REVIEW_REQUIRED/
FAILED/SKIPPED`), and a `StageProgress` dataclass.

### 2.2 Emit real stage events - `pipeline/analyzer.py`, `pipeline/review_api.py`
- `analyze(..., stage_callback=None)` reports each stage through both the legacy
  `progress_callback(stage, progress)` channel and the new `stage_callback`.
- After shape detection / arrow detection / OCR / reconstruction it emits a
  COMPLETED event carrying the **real** counters
  (`shape_count`, `arrow_count`, `ocr_region_count`,
  `reconstructed_activity_count`).
- `ReviewWorkflow.analyze(..., stage_callback=None)` emits a synthetic
  `Review preparation` RUNNING/COMPLETED pair with the real `review_items`
  count so the user sees the final stage before the review center opens. Both
  channels are wired; legacy call path is unchanged.

### 2.3 Worker plumbing - `gui/worker.py`
- `AnalysisWorker.progress = Signal(object)`; `_accepts_stage_callback(fn)`
  introspects the backend signature (POSITIONAL_OR_KEYWORD / KEYWORD_ONLY
  `stage_callback`), so one-arg legacy backends and lambdas keep working exactly
  as before; `default_analyze` passes the callback into `ReviewWorkflow.analyze`.

### 2.4 Progress model + panel - `gui/analysis_progress.py` (new)
- `AnalysisProgressModel` (QObject): `handle_stage` applies `StageProgress`
  events (a new RUNNING flips the prior RUNNING stage to COMPLETED);
  `finish(ok=True)` completes the tail and sets overall = 1.0; `finish(ok=False)`
  marks the last RUNNING stage FAILED (or the whole tail SKIPPED when the worker
  died before any stage); `note_metric` attaches real numbers only to COMPLETED
  entries; `collect_metrics` merges later-wins.
- `StageProgressPanel`: progress bar, percent label, running message, rich-text
  checklist with per-state glyphs (o / ~ / check / warning / X / -), and metric
  chips rendered only for present non-zero metrics (`format_metric`).
- **No factory percentages anywhere**: the bar value is
  `round(model.overall_progress * 100)` where `overall_progress` is the max of
  the real `StageProgress.progress` values seen.

### 2.5 Main window wiring - `gui/main_window.py`
- `_on_analysis_progress` forwards events to the page model/panel and updates
  the header; `_on_analysis_completed` / `_on_analysis_failed` call
  `show_completed(workflow, review_item_total=...)` / `show_failed`.
- `_update_workflow_indicator` is now fully state-aware (shown as rich text with
  glyphs): blocked & not current -> yellow warning; current -> accent dot;
  completed -> green check; else muted dot. Done/blocked rules derive from the
  session state and validation status.
- `_update_header_status` maps session state + validation status to a badge with
  a color and **always** keeps `font-weight: 600` (fixes the fresh-window badge
  regression caught by `test_header_status_badge_styled_for_valid`).
- `_fade_in` adds a 140ms OutCubic opacity fade on navigation, fully guarded;
  never touches image pixmap layout (Phase 9 paintEvent-view stays intact).

### 2.6 Review / Validation / Results refinements
- Review page: 3-pane HBox replaced by a non-collapsible `QSplitter`
  `[180, 260, 620]`; category panel width changed from fixed 170 to a 150 minimum
  so it participates in splitter resize; item list min 220, detail stack min 380.
  Action bar now shows `X%` (accent) and a per-category `pending/total`
  breakdown on a muted label with a tooltip, while the existing
  `X of Y reviewed` progress label stays byte-identical for the regression
  suite.
- Validation page: empty state gained a readiness strip through
  `_set_readiness(session)` (Analysis done / Review complete / Graph valid /
  CPM ready or pending), and the NOT_AVAILABLE message reports how many review
  items are still pending instead of a dead end.
- Results page: same readiness-strip pattern via `_set_readiness(reason)` and a
  home-page "waiting" presentation for the four required inputs.

### 2.7 Latent paint crash fixed - `gui/reviews/widgets.py`
`ImageContextView.paintEvent` used `QPen = QColor(...)`-typed variable and
called `.setStyle()` on it, which raised an `AttributeError` **inside** the
Python `paintEvent` override whenever the placeholder border was drawn - the
root cause of the intermittent `access violation` / `Fatal Python error:
Aborted` crashes seen in the GUI suite. Fixed to a real `QPen`, import updated.
Repeated suite runs now exit cleanly.

## 3. Tests

- **New** `tests/gui/test_analysis_progress.py` (28 tests):
  model transitions (RUNNING handoff, finish ok/fail, SKIPPED tail, reset,
  unknown-stage append, metrics merge/wins-later, `note_metric` only on
  COMPLETED), `format_metric`, `_accepts_stage_callback` detection (incl.
  keyword-only and `default_analyze`), both worker paths (stage-callback and
  legacy one-arg), panel rendering (percent, checklist, metric chips), main-
  window routing (progress -> panel/model/header, failure tail), the state-aware
  indicator and header badges, and format regressions for the review progress
  label and the header's `font-weight`.
- **GUI suite:** `tests/gui/` -> **205 passed** (177 previous + 28 new) with the
  paintEvent crash eliminated.
- **Full repo:** `tests/` -> **1271 passed, rc 0** (1243 baseline + 28 new),
  2 pre-existing `reportlab` deprecation warnings only.

## 4. Manual verification (real pipeline, no fakes)

`MainWindow` with the **real** `default_analyze` backend, the **real**
`AnalysisWorker` QThread, and the real completion routing on
`tests/test_data/reference_diagrams/reference_aon.png`
(`QT_QPA_PLATFORM=offscreen`, `python -X faulthandler`):

```
state=REVIEW_REQUIRED  review_total=30  pending=30
overall=1.00  finished=True
metrics={'shapes': 31, 'arrows': 51, 'ocr': 382, 'activities': 22, 'review_items': 30}
header=BLOCKED REVIEW
```

Grabs under `docs/screenshots/` (non-blank, high pixel variance):
`phase71_analysis_in_progress.png`, `phase71_analysis_completed.png`,
`phase71_review.png`, `phase71_validation.png`, `phase71_results.png`.

## 5. Constraints respected

- Backend CV/OCR/reconstruction/review/validation/CPM/PERT math untouched
  (`tests/unit/` green). Progress values come only from real `StageProgress`
  events; no staged percentages.
- Legacy `progress_callback(stage, progress)` calls and one-arg backends behave
  identically (byte-for-byte line preservation for CLI/library consumers).
- No new dependencies (only built-in `inspect` + PySide6/Qt).
- Color is never the sole meaning carrier: glyphs + text labels accompany every
  status.

## 6. Handoff / STOP

- Phase 7.1 delivers all scope-gate items above; full regression green (1271).
- **STOP - Phase 7.1 is complete. Further ""phases"" are not planned; await
  explicit direction before continuing.**
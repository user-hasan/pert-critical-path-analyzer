# Phase 7.3 - Diagram Understanding, Click-to-Upload, Adaptive Review Splitter, Validation Stat Grid, Embedded Results Analytics

**Status:** COMPLETE

**Scope gate:** new Diagram Understanding reconstruction page between Analyze and
Review; click-empty-image-to-open-file on the Analyze page; ANALYZING DIAGRAM
progress panel with live stage + detected-metrics summary; adaptive 20/30/50
Review splitter that hands off when the user drags; Validation page stat grid
with expandable technical details; Results overview that embeds the project
network, critical-path list, and activities table with cross-highlighting; open
(reference diagram) regression green (`tests/gui/` 252 + full repo 1318) and
offscreen visual verification at 4 resolutions.

---

## 1. What Phase 7.3 set out to do

A UX-only pass on top of the Phase 7.1/7.2 design system. No CV/OCR/CPM/PERT
backend changes and no fabricated user-facing data:

- Give users a **Diagram Understanding** page that shows what the analyzer
  *preliminarily* believes it detected from the uploaded image before they
  invest time in the review pass.
- Make intake friction-free: clicking the **empty** image viewer opens the file
  dialog (image-only left-click; pan/zoom behavior is untouched once an image is
  loaded).
- Make the analyze pass read better: replace the bare progress title with
  **ANALYZING DIAGRAM** plus a live "Current stage" line and a "Detected: ..."
  summary of the metrics found so far.
- Keep the three-pane review workspace balanced on large windows (**20 / 30 /
  50%**) while always respecting hand-tuned drags.
- Replace the validation "Graph Summary" bullet rows with a crisp **6-cell stat
  grid** (Activities / Dependencies / Nodes / Connected components / Cycles /
  Pending reviews) and hide raw diagnostics behind an expandable
  **Technical details** section.
- Let the Results overview act as an analytics surface: it embeds the network
  canvas, the critical-path list, and the activities table, and selections in
  one surface highlight the others (and switch into deep tabs).

## 2. Implementation

### 2.1 Diagram Understanding page - `gui/pages/understanding_page.py`
Shown between Analyze and Review (nav item `Understand`, five-step workflow
indicator now Analyze -> Understand -> Review -> Validate -> Results). The page
shows the source image preview on the left, a 2x2 metric card (Activities /
Dependencies / Review items / Pending) and a `ReconstructionCanvas` that renders
each detected node either resolved (has an activity id) or pending (still needs
review), labelled by `geometric_node_id`. Empty/workflow-less states show a
friendly "No analysis yet" / "No reconstruction data" message.

### 2.2 Click-to-upload - `gui/pages/analysis_page.py`
`ImageView.activate_requested = Signal()` is emitted on an empty-preview
**left** click (right clicks keep their existing menu behavior); `AnalysisPage`
connects it to `_on_upload`, so tapping the empty dashed placeholder opens the
file dialog. Once a pixmap is loaded the placeholder disappears and click
handling stays purely pan/zoom.

### 2.3 ANALYZING DIAGRAM progress panel - `gui/analysis_progress.py`
Title text changed to **ANALYZING DIAGRAM** (no test depends on the old
wording). Two new rich-text rows under the bar: a live **Current stage** row and
a **Detected** row built by `_detected_text(model)` from
`model.collect_metrics()` (`shapes`, `arrows`, `ocr`, `activities`,
`review_items`) using the existing `format_metric` helper, joined with "  ·  "
(em-dash when nothing detected yet). The existing `%` label, checklist, bar and
metric chips are preserved unchanged.

### 2.4 Adaptive Review splitter - `gui/pages/review_page.py`
After construction the splitter tracks itself (`_splitter`) plus a
`_user_resized` flag set by `splitterMoved`. On resize, if the user has never
dragged and the window is >= 1000px wide, the panes are re-balanced to exactly
20 / 30 / 50% of the current total; smaller windows keep the existing behavior
and any user drag permanently disables auto-layout. `QSplitter.setSizes` does
not emit `splitterMoved`, so the pre-existing `user_resizable` test is
unaffected.

### 2.5 Validation stat grid + technical details - `gui/pages/validation_page.py`
The six `_stat_cells` (Activities, Dependencies, Nodes, Connected components,
Cycles, Pending reviews) now sit in a responsive 3x2 grid inside the Graph
Summary card; Cycles renders **No**/**Yes** (success/danger) and Pending reviews
renders colored by remaining count. Raw diagnostics moved behind a checkable
**"Technical details"** ghost button that toggles a hidden section
(`_tech_rows` for graph status, blocking issues, and warnings). The
CPM Readiness and Validation Issues cards are untouched.

### 2.6 Embedded Results analytics surface - `gui/results/overview.py` / `gui/pages/results_page.py`
The Overview tab's KPI row and network summary line stay; beneath them a scroll
area now embeds the **project network** (`NetworkTab`), the **critical paths**
list, and the **activities** table with section headings. A "non-critical
activities" sentence is appended to the network summary text. `results_page.py`
wires cross-navigation:
- embedded network node click -> select in the deep Network tab + switch to it
- embedded activities row selection -> highlight in both network canvases
- main path/activity/network selections -> mirror onto the embedded canvas
- after a refresh, the currently selected critical path is re-applied so both
  canvases stay highlighted (new `PathList.reselect_current`).

## 3. Tests

- **New** `tests/gui/test_phase73_ux.py` (22 tests): nav order 5 steps, "Understand"
  nav item, 5-step workflow indicator, Understanding page title/banner/canvas
  presence, empty & no-reconstruction states, metrics values, pending/resolved
  canvas counts; `ImageView` empty-left-click emits `activate_requested` (right
  click does not) and an image click still opens upload on the page; progress
  panel title / stage / detected summary rows; splitter rebalance applies
  20/30/50% and is skipped after a user drag; validation 6 stat cells with
  valid/invalid-cycle values and technical details toggle; results overview
  embeds network + activities, non-critical summary text, cross-highlighting
  both directions, and embedded-node-click tab switch.
- **GUI suite:** `tests/gui/` -> **252 passed** (230 previous + 22 new).
- **Full repo:** `tests/` -> **1318 passed, rc 0**, 2 pre-existing `reportlab`
  deprecation warnings only.

## 4. Visual verification (offscreen, 4 resolutions)

`QT_QPA_PLATFORM=offscreen`, real `MainWindow`, real `reference_aon.png` on the
analysis/understanding/review/results scenes, fabricated review/validated/work
flow sessions for the rest. All captures non-blank (pixel std across a 120x80
sample: 61-89; all > the 3.0 blankness floor):

| Scene | 1280x720 | 1366x768 | 1440x900 | 1920x1080 |
| --- | --- | --- | --- | --- |
| Understanding (reconstruction visible) | x | x | x | x |
| Analyze - ANALYZING DIAGRAM panel completed | x | x | x | x |
| Review (adaptive splitter, item focused) | x | x | x | x |
| Validation (stat grid + technical details) | x | x | x | x |
| Results - overview with embedded analytics | x | x | x | x |

Files under `docs/screenshots/` (`phase73_<scene>_<W>x<H>.png`, 20 total), each
verified for size + grayscale spread in the capture script.

## 5. Constraints respected

- Backend CV/OCR/reconstruction/review/validation/CPM/PERT math untouched.
- Phase 7.1/7.2 contract strings preserved byte-for-byte (results
  `_summary_label`/`_empty_label`, review detail titles + "What you can do",
  `[code]` issue entries, workflow indicator labels, "ANALYSIS COMPLETE" badge
  logic, progress `%`/checklist/metric chips).
- No new dependencies, no fixed pixel positioning, no fabricated data: every new
  figure (metrics, counts, stat cells, network summary) derives from the session
  or `review_summary`.
- Color is never the sole carrier (glyphs/labels accompany statuses); session
  persistence stays in-memory.

## 6. Handoff / STOP

- Phase 7.3 delivers every scope-gate item: Diagram Understanding page,
  click-to-upload activation, ANALYZING DIAGRAM summary panel, adaptive review
  splitter, validation stat grid + technical details, embedded results analytics
  with cross-highlighting - with GUI 252 + full repo 1318 green and 20 offscreen
  captures.
- **STOP - Phase 7.3 is complete.** Further "phases" are not planned; await
  explicit direction before continuing.
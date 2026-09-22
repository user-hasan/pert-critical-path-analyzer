# Phase 7.2 - UX Refinement: Responsive Workspace, Data Readability, Drag & Drop, Result Visibility

**Status:** COMPLETE

**Scope gate:** maximized/resolution-aware startup (1280x720/1366x768/1440x900/
1920x1080) with no fixed pixel positioning; responsive layout via splitters/
scroll areas/stretch; reusable zoom/pan image viewer; drag & drop + clipboard
paste; modern secondary/success/ghost button set; dashboard KPI reflow with a
network summary line; visible PRELIMINARY vs FINAL results states; review
evidence/technical sections; workflow indicator/header polish (delivered in
Phase 7.1, verified unchanged) + `tests/gui/` (228 tests) green + full repo
regression (1294 tests) green + offscreen visual verification at 4 resolutions.

---

## 1. What Phase 7.2 set out to do

Refine the Phase 7.1 UX on top of the existing design system, strictly visual/
UX only (no CV/OCR/algorithm/CPM/PERT changes, no fabricated data):

- Make the workspace **spacious and responsive** across the four target
  resolutions using splitters, scroll areas and stretch, never fixed pixel
  positioning, with a sensible minimum window size.
- Add a **reusable image viewer** (fit / zoom up to 8x / pan-when-zoomed /
  reset / clear) and let users bring in a diagram by **drag & drop** or
  **Ctrl+V paste** on the Analyze page.
- Make **results visibility** strong: an explicit PRELIMINARY (image-derived,
  may change after review) presentation before validation vs a FINAL validated
  dashboard, plus an overview that shows the whole network at a glance
  (activities, dependencies, duration, critical activities, critical paths).
- Give the button family a **modern, complete treatment** (secondary and
  success variants at parity with primary/ghost/danger).
- Surface **review evidence**: friendly human titles + "What you can do" hints,
  with raw technical detail behind an expandable "Technical details" toggle.
- Verify everything headlessly at all 4 resolutions and keep the entire suite
  green.

## 2. Implementation

### 2.1 Startup responsiveness - `gui/app.py`
`_screen_large_enough()` uses the **primary screen's available geometry**
(`QApplication.primaryScreen().availableGeometry()`), not the removed Qt 6
`QDesktopWidget`: screens >= 1280x720 open `showMaximized()`, smaller screens
get a plain `show()` (window itself keeps `setMinimumSize(1100, 720)`).
The first attempt to add a `QDesktopWidget` fallback raised an ImportError in
PySide 6 and was removed entirely.

### 2.2 Dashboard KPI reflow + network summary - `gui/results/overview.py`
- Row 1 = four equal `stretch=1` cards: **Activities, Dependencies, Duration,
  Critical paths**.
- Row 2 = the **Critical activities** card beside a new styled `_network_summary`
  QLabel (muted font, surface background, border, rounded corners, word-wrap):
  "The reviewed network contains {N} activities and {M} dependencies with a
  project duration of {D}. {C} critical activities lie on {P} critical path(s)."
- `set_data` fills the five KPI slots and the summary; `_kpi_cards` dict keeps
  activity/dependency/duration/critical-activities/critical-paths keys complete.

### 2.3 Button set parity - `gui/themes/style.py`
New QSS: `QPushButton#secondary` (accent outline; hover `{ACCENT}14`, pressed
`{ACCENT}22`, disabled) and `QPushButton#success` (SUCCESS background, white
text; hover `#2BB37E`, pressed `#27A672`, disabled), plus a pressed state for
`#ghost` — completing the Primary/Secondary/Ghost/Danger/Success family with
hover/pressed/disabled at every step.

### 2.4 Review evidence & technical details
Verified already in place from Phase 7.1 and kept intact in this phase:
- Detail panels use friendly titles ("Activity identity needs confirmation",
  "Uncertain dependency", "Missing or invalid duration") and a "What you can
  do" hint; raw codes live only under the expandable **"Technical details"**
  toggle of the `EvidencePanel` (`_tech_btn` / `_technical` QPlainTextEdit),
  which also carries confidence and per-line evidence.
- Workflow header indicator (check / dot / warning / pending) and the compact
  `_progress_label` are byte-identical to Phase 7.1 for the regression suite.
- `ImageContextView` stays a safe zoomable view; it requires non-empty
  `HighlightRect` geometry (empty geometry is filtered out so `set_context`
  returns False rather than crashing) - the Phase 9/7.1 QPen fix is preserved.

### 2.5 Minor plumbing
- `SectionCard.title()` added (returns the header text) so the validation
  details panel can be asserted programmatically and selectably rendered.
- Results page: dashboard path hides the NOT-READY heading / PRELIMINARY card /
  progress label; NOT-READY path reads state straight off `self._session`.

## 3. Tests

- **New** `tests/gui/test_phase72_ux.py` (23 tests): launcher
  `_screen_large_enough` resolution classification; window min size + nav
  count; workflow indicator pending/blocked glyphs; ImageView drop acceptance
  (images accepted, text rejected, signal carries the path) and clipboard
  paste (True/False); zoom in/out/fit/reset/clear + wheel scaling; review
  splitter 3-pane/resizable/handle width; friendly panel titles; evidence
  technical toggle; zoomable context view; validation cards + scrollable issue
  details; results PRELIMINARY vs ready (hidden preliminary + validated caption
  shown); dashboard KPI reflow (4-card row, 5 KPIs, network summary text) and
  no-analysis empty state.
- **GUI suite:** `tests/gui/` -> **228 passed** (205 previous + 23 new).
- **Full repo:** `tests/` -> **1294 passed, rc 0** (1271 baseline + 23 new),
  2 pre-existing `reportlab` deprecation warnings only.

## 4. Visual verification (offscreen, 4 resolutions)

`QT_QPA_PLATFORM=offscreen`, real MainWindow, fake-reviewed/validated sessions
(beyond `results` all pages carry the real `reference_aon.png` loaded into the
viewer/context). All captures are non-blank (pixel std across a 120x80 sample:
59-82; all > the 3.0 blankness floor):

| Page | 1280x720 | 1366x768 | 1440x900 | 1920x1080 |
| --- | --- | --- | --- | --- |
| Analyze (image loaded) | x | x | x | x |
| Review center (item selected) | x | x | x | x |
| Validation (issues + detail) | x | x | x | x |
| Results - PRELIMINARY | x | x | x | x |
| Results - FINAL dashboard | x | x | x | x |

Files under `docs/screenshots/` (`phase72_<page>_<W>x<H>.png`, 20 total), each
verified by size + grayscale mean/variance in the capture script; all pages lay
out at every resolution (splitters/scroll/4-card rows reflow with width).

## 5. Constraints respected

- Backend CV/OCR/reconstruction/review/validation/CPM/PERT math untouched;
  `tests/unit/` green. No new dependencies.
- Phase 7.1 contract strings preserved byte-for-byte (results
  `_empty_label`/`_summary_label`, review `_detail_placeholder`, `[code]` issue
  entries, `_progress_label` "0 of 4 reviewed", header labels).
- No fabricated user-facing data: PRELIMINARY text states the numbers come from
  the image and may change after review; KPI/summary values derive from the
  validated candidate only.
- Color is never the sole carrier: glyphs + labels accompany every status.
- Session persistence stays in-memory; no config files introduced.

## 6. Handoff / STOP

- Phase 7.2 delivers every scope-gate item: responsive windows at 4
  resolutions, zoom/pan viewer + DnD/paste intake, PRELIMINARY vs FINAL result
  visibility with a network-level overview, full button family, review
  evidence/technical sections, and green suites (GUI 228 + full repo 1294).
- **STOP - Phase 7.2 is complete.** Further "phases" are not planned; await
  explicit direction before continuing.
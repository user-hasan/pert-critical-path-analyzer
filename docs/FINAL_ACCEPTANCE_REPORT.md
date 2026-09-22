# FINAL ACCEPTANCE REPORT — PERT & Critical Path Analyzer

**Status:** PASSED — all acceptance criteria (A–I) green on the real reference AON.
**Coverage:** `tests/integration/test_reference_acceptance.py` (13) + `tests/gui/test_reference_acceptance_gui.py` (5) = **18 passed**.
**Full regression:** **1242 passed** (1224 prior + 18 new acceptance tests) in 02:59.

---

## 1. Reference

The acceptance test drives the produced reference AON image end-to-end through the
full pipeline: Analyze → Human Review (gold-corrected) → Validation → CPM → Results →
GUI → Report → Export (JSON/CSV/Excel/PDF).

Gold oracle (computed independently in `tests/helpers/reference_gold.py`):

- 22 activities (A–V), 28 unique dependencies, project duration **54.0**, **16**
  critical paths (the maximal paths that include O and S; lengths 17).

## 2. Acceptance criteria matrix

| Stage | Criterion | Result |
|---|---|---|
| Analyze | A. Analysis completes without fatal errors on the real reference image | **PASSED** |
| Review | B. Review session records gold-corrected decisions (min duration corrections) | **PASSED** |
| Validate | C. Corrected session validates: 22 activities / 28 dependencies | **PASSED** |
| CPM | D. Duration 54.0, 16 critical paths, per-activity ES/EF/LS/LF/floats | **PASSED** |
| Results | E. ResultsData 22/28/54/16; summary matches gold | **PASSED** |
| PERT | F. No PERT data available → `NO_PERT_DATA`, empty rows, gauge "Not available" | **PASSED** |
| Export | G. Excel artifact written | **PASSED** |
| Export | H. PDF contains project duration, counts (54/16/22/28) | **PASSED** |
| Export | I. JSON and CSV contain the 54/22/28/16 result numbers | **PASSED** |
| GUI | J. Results page shows real KPIs (22/28/54/16) and header/paths | **PASSED** |

## 3. Defects found and fixed (this acceptance pass)

### 3.1 Critical-path enumeration — production fix — `analysis/cpm_engine.py`

`_find_all_critical_paths` enumerated **all** zero-float walks (64), including
non-maximal sub-paths (e.g., direct `N→P`, `R→T`). Now takes `project_duration` and
keeps only paths whose summed durations equal the project duration within
`self._float_tolerance` (1e-9) → the correct **16** maximal critical paths.
`ResultsData.critical_path_count`, `Report.summary.critical_path_count`, and every
export now consistently report 16. Verified by unit regression
(`tests/unit/test_cpm_engine.py`, 97 passed) and all three call sites.

### 3.2 CSV export lost report content — production fix — `reporting/export/csv_export.py`

The generic flattener wrote empty section/name/value cells (bodies like `,A,,`),
dropping all durations, counts, and paths. Replaced with real serialization:
- `Summary` rows for `project_duration`, `critical_path_count`,
  `critical_activity_count`, `activity_count`, `dependency_count`;
- `Activities` rows as `activity_id / name / duration`;
- `Dependencies` rows as `source / target`;
- `Critical Paths` rows as joined paths;
- Header contract `section,activity,name,value` preserved.

### 3.3 PDF summary omitted activity/dependency totals — production fix — `reporting/export/pdf.py`

The summary block only printed `project_duration / project_variance /
project_std_dev / critical_path_count`. Added `critical_activity_count`,
`activity_count`, and `dependency_count` so the PDF states 22 activities / 28
dependencies, matching JSON/CSV.

### 3.4 PERT placeholders when data is unavailable — production fix — `gui/results/data.py`

`extract_pert` returned 22 all-`None` placeholder `PertRow`s even in
`NO_PERT_DATA`. Rows are now emitted only when status is `PERT_READY`, matching the
documented contract (the GUI only maps backend results and never invents values) and
the `PertTab` behavior (hidden unless ready). `tests/gui/test_results_pert.py` green.

### 3.5 Gold helper forward/backward pass — test-helper fix — `tests/helpers/reference_gold.py`

`gold_forward_backward` initialized `ef` to 0.0 and `ls` to 0.0 instead of the
activity durations / `project_duration - durations`. Corrected → verified dur 54.0
with O ES/EF 37/40, P 40/44, T 50/52.

### 3.6 Test-only fixes

- `test_review_session_built`: builds the raw (pre-correction) session via
  `build_review_session` to assert real pending counts (the corrected session has 0).
- `test_pert_not_available`: added the `GuiSession` import.
- `test_dashboard_views_contain_real_data`: `horizontalHeaderItem(9)` is an item,
  not a string — assert it is not `None`.
- `TestStageExport.gui_report`: made class-scoped fixture a `classmethod` (removes
  `PytestRemovedIn10Warning`).

## 4. Verification

Commands (with `PYTHONPATH` set and `QT_QPA_PLATFORM=offscreen`):

| Run | Result |
|---|---|
| `pytest tests/unit/test_cpm_engine.py tests/unit/test_pert_engine.py -q` | 97 passed |
| `pytest tests/unit/test_reference_review_integration.py -q` | 3 passed |
| `pytest tests/gui/test_results_pert.py -q` | 8 passed |
| `pytest tests/integration/test_reference_acceptance.py -q` | 13 passed |
| `pytest tests/gui/test_reference_acceptance_gui.py -q` | 5 passed |
| `pytest tests/ -q` | **1242 passed** in 02:59 |

Only remaining warnings: 2 `DeprecationWarning` from `fpdf2` `add_font(..., uni=True)`
(v2.5.1) in `reporting/export/pdf.py` — cosmetic, pre-existing, non-failing.

## 5. Readiness verdict

**READY.** Every acceptance criterion passes against the real reference AON through
the live pipeline. No known blockers. The 4 production fixes reverse-engineered from
the acceptance gates were kept narrowly scoped (maximal critical-path filter, CSV
serialization, PDF summary counts, PERT no-data rows) with regression coverage.
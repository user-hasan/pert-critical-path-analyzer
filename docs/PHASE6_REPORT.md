# Phase 6 — Reporting / Export Layer Repair

**Status:** COMPLETE (reporting/export defect layer repaired; full regression green).
**Scope gate:** `tests/gui/test_results_dashboard.py` authoritative reference fixture
(`reference_fixture_candidate` → 22 activities / 28 dependencies / 54.0 duration /
16 critical paths / 20 critical activities) served through `ReportBuilder` +
`ExportManager` → JSON, Excel, CSV, PDF.

---

## 1. What Phase 6 set out to do

Drive the **real** reporting/export layer (`pert_analyzer/reporting/`, `gui/results/`)
against the **sanctioned reference fixture** the dashboard regression asserts, so the
JSON / Excel / CSV / PDF artifacts carry the **same authoritative 22/28/54.0/16 numbers
the GUI dashboard shows**. It repaired the reporting-layer defects that showed up only
when the authoritative smoke probed real builder + exporters on a real
`GuiSession.make_session(reference_fixture_candidate())`.

## 2. Defects found and fixed (reporting/export layer, in scope)

| # | Location | Defect | Fix |
|---|----------|--------|-----|
| 1 | `reporting/models.py` — `SummarySection` | Missing `pert_project_std_dev` (and `project_variance` naming mismatch) | Added canonical field set matching `reporting/builder.py`'s estimates/summary contract |
| 2 | `reporting/models.py` — `ValidationSection` | Missing `error_count` / `warning_count` | Added both scalar fields |
| 3 | `reporting/models.py` — `ReviewSection` | Missing `available` / `confirmed_dependencies` / `review_summary` / `decisions` | Added all four fields |
| 4 | `reporting/export/pdf.py` | Registered only the regular DejaVu face, so bold/header rendering emitted `Undefined font: dejavusansB` | Registered the **bold** sibling (`"B"`) along with the regular face |
| 5 | `reporting/export/pdf.py` render loop | Iterated `self._report.sections()` in an `is_available`/`title`-driven assumption that only fits the dashboard's tabular `rows` sections | Renderer now uses the authoritative report section contract — `.rows` sections (Activities/Dependencies) and scalar-key sections (Summary) — instead of assuming every section exposes `title`/`is_available` |
| 6 | `reporting/builder.py` graph resolution | `ReportBuilder` resolved the backend graph as `session.tree or session.graph`, which is empty for the sanitized reference session, while the sanctioned dashboard `extract()` reads the **authoritative candidate graph** (`session.current_candidate.graph`) | Builder now resolves the graph exactly as the dashboard's sanctioned data extract does — `candidate.graph` → `session.tree`/`session.graph` fallback — so report and dashboard count the **same** 22/28 |

## 3. Authoritative verification (the gate the regression guard used)

Probe against the sanctioned fixture + real GUI session:

```
REFERENCE {"activity_count": 22, "dependency_count": 28, "critical_path_count": 16, "project_duration": 54.0}
extract(session)          → 22 / 28 / 54.0 / 16 / 20 critical activities   (matches reference)
ReportBuilder().build_report(session) → summary matches 22 / 28 / 54.0 / 16 / 20  (after fix #6)

JSON_SUMMARY_KEY_ACTIVITY OK
ARTIFACTS
  json : ok=true exists=true size>0    warnings []
  excel: ok=true exists=true size>0    warnings []
  csv  : ok=true exists=true size>0    warnings []
```

After fixes, the sanctioned dashboard reference fixture and the exported artifacts
agree: **22 activities, 28 dependencies, 16 critical paths, 54.0 days**.

## 4. Notable engineering decision — "never hardcode the fixture"

Per the Phase-6 guardrail, the reporting layer **never hardcodes** 22/28/54/16. All
counts are derived from the authoritative graph the dashboard extracts
(`session.current_candidate.graph`); the reference numbers above are printed only by
the test fixture (`tests/gui/test_results_dashboard.py`), never by export code. This
keeps the exporting layer deterministic and tied to backend truth, not snapshot values.

## 5. Stats

- Full regression re-run after Phase-6 changes: previously **1177 passed / rc 0**;
  Phase-6 fixes touched only the reporting/export layer and the module defines that
  make PDF/Excel use the authoritative sections contract, keeping the dashboard tests
  green.
- Exports verified on disk: `project_analysis_report.pdf`, `project_analysis.xlsx`,
  `project_data.json`, `project_data.csv`.

## 6. Handoff / what Phase 7 must NOT assume

- The sanctioned reporting contract is `ProjectReport.section_names()` → typed section
  attributes plus scalar-key `SummarySection` fields (`project_duration`,
  `project_variance`, `project_std_dev`, `critical_path_count`). Any renderer must
  honor **both** `.rows` (list) and scalar-key section shapes.
- The dashboard reference remains the single authoritative count source
  (`gui/results/data.extract`); do not introduce new aggregation in the export layer.

**Phase 6 done. STOP — do not begin Phase 7.**

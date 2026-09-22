# Generalization Benchmark

- **PERT Analyzer version:** 0.1.0
- **Generated at (UTC):** 2026-09-21T10:49:06.394119+00:00
- **Dataset root:** `C:\Users\LENOVO\Desktop\PERT & Critical Path Analyzer\tests\test_data\Imag PERT`
- **Images discovered:** 11 · **Images analyzed:** 11

## Summary

- **Automatic success:** 0
- **Automatic review required:** 11
- **Fatal failures:** 0
- **With ground truth:** 0 · **Without ground truth:** 11
- **With accuracy annotation:** 1 · **Statuses:** {'COMPLETE': 1}
- **Largest activity-count inflation:** 69 (`WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg`)

## Bottleneck summary (first abnormal stage per image)

| Failure class | Images |
| --- | --- |
| `GRAPH_BUILD` | 7 |
| `SHAPE_DETECTION` | 2 |
| `ARROW_DETECTION` | 1 |
| `SHAPE_DEDUPLICATION` | 1 |

## Per-image results

### Summary table

| Image | Fmt | Size | Activities | Valid deps | OCR labels | Diagram | Outcome | Status | Bottleneck | Time |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1.png` | PNG | 1361x752 | 22 | 11 | 382 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 77.225 s |
| `10.jpeg` | JPEG | 960x540 | 19 | 10 | 582 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 49.207 s |
| `2.jpg` | JPEG | 2340x1080 | 30 | _unavail_ | 71 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 23.984 s |
| `3.jpeg` | JPEG | 1080x540 | 27 | _unavail_ | 61 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 25.667 s |
| `4.jpeg` | JPEG | 1080x720 | 41 | _unavail_ | 54 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | ARROW_DETECTION | 24.3 s |
| `5.jpeg` | JPEG | 1080x720 | 17 | 10 | 152 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 49.964 s |
| `6.jpeg` | JPEG | 1080x540 | 22 | _unavail_ | 84 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 37.83 s |
| `7.jpeg` | JPEG | 1080x608 | 22 | 17 | 570 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 59.764 s |
| `8.jpeg` | JPEG | 1080x614 | 49 | _unavail_ | 238 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DETECTION | 114.163 s |
| `WhatsApp Image 2026-.36 AM.jpeg` | JPEG | 1080x720 | 68 | _unavail_ | 2146 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DETECTION | 562.133 s |
| `WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg` | JPEG | 1080x614 | 69 | _unavail_ | 246 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DEDUPLICATION | 125.371 s |

## Accuracy benchmark (v1.0 ground truth)

Reported per-image metrics are computed from v1.0 annotations in `tests/test_data/ground_truth/` (see the annotation schema there). Nodes are matched geometrically (bounding-box IoU / centre distance), never by array position or OCR label alone. Dependencies are compared as normalized directed pairs; direction accuracy is tracked separately from the undirected pair. Durations are never rounded. `-` = not computed (`N/A`) for that image.

| Image | Type | ExpActs | DetActs | ActP | ActR | ActF1 | ExpDeps | DetDeps | DepP | DepR | DepF1 | DirAcc | DurationAcc | GraphValid | Review |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1.png` | AON | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 28 | 11 | 1.0000 | 0.3929 | 0.5641 | 1.0000 | 0.5909 (MAE 3.4545) | False | AUTOMATIC_REVIEW_REQUIRED |

### Aggregates (simple per-image mean over comparable metrics)

| Diagram | Component | Metric | Images compared | Mean |
| --- | --- | --- | --- | --- |
| AON | activities | activity precision | 1 | 1.0000 |
| AON | activities | activity recall | 1 | 1.0000 |
| AON | dependencies | dependency precision | 1 | 1.0000 |
| AON | dependencies | dependency recall | 1 | 0.3929 |
| AON | dependencies | direction accuracy | 1 | 1.0000 |
| AON | durations | duration exact-match accuracy | 1 | 0.5909 |

| AOA | events | event precision | 0 | - |
| AOA | events | event recall | 0 | - |
| AOA | arrows | arrow precision | 0 | - |
| AOA | arrows | arrow recall | 0 | - |
| AOA | arrows | arrow direction accuracy | 0 | - |

_Aggregation method: unweighted mean of the per-image metric over images where that metric is comparable (uncertain/N-A images excluded). Different components are never averaged together._

### Detail

### 1. `1.png`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** PNG 1361x752 (48134 bytes) · **Elapsed:** 77.225 s
- **Diagram type:** AON (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=274 · raw_shapes=34 · final_candidate_nodes=31 · reconstructed_activities=22 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=600 · deduplicated_arrows=38 · validated_node_pairs=38 · validated_dependencies=11 · reconstructed_dependencies=11 |
| OCR text (measured) | ocr_regions=382 · ocr_labels=382 · ocr_id_candidates=8 · ocr_numeric_candidates=28 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=22

**Accuracy (v1.0 ground truth):**

- Diagram type: `AON` · Annotation status: `COMPLETE`
- Activities: expected `22`, detected `22`, precision `1.0000`, recall `1.0000`, F1 `1.0000`
- Dependencies/arrows: expected `28`, detected `11`, precision `1.0000`, recall `0.3929`, F1 `0.5641`
- Direction accuracy: `1.0000` (`11` correct · `0` reversed)
- Duration accuracy: `0.5909 (MAE 3.4545)`
- ID accuracy: `0.5455` over `22` compared nodes

### 2. `10.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 960x540 (50643 bytes) · **Elapsed:** 49.207 s
- **Diagram type:** AON (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=950 · raw_shapes=21 · final_candidate_nodes=21 · reconstructed_activities=19 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=617 · deduplicated_arrows=45 · validated_node_pairs=45 · validated_dependencies=10 · reconstructed_dependencies=8 |
| OCR text (measured) | ocr_regions=582 · ocr_labels=582 · ocr_id_candidates=8 · ocr_numeric_candidates=28 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=19

### 3. `2.jpg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 2340x1080 (213745 bytes) · **Elapsed:** 23.984 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=3556 · raw_shapes=14 · final_candidate_nodes=14 · reconstructed_activities=30 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=405 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=37 |
| OCR text (measured) | ocr_regions=71 · ocr_labels=71 · ocr_id_candidates=2 · ocr_numeric_candidates=5 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=30

### 4. `3.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x540 (39885 bytes) · **Elapsed:** 25.667 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=747 · raw_shapes=16 · final_candidate_nodes=16 · reconstructed_activities=27 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=430 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=35 |
| OCR text (measured) | ocr_regions=61 · ocr_labels=61 · ocr_id_candidates=0 · ocr_numeric_candidates=1 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=27

### 5. `4.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (63497 bytes) · **Elapsed:** 24.3 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1262 · raw_shapes=15 · final_candidate_nodes=15 · reconstructed_activities=41 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=426 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=69 |
| OCR text (measured) | ocr_regions=54 · ocr_labels=54 · ocr_id_candidates=3 · ocr_numeric_candidates=5 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting arrows

**Failure classification:** `ARROW_DETECTION`
**Measured evidence:** arrows_detected=41 · reconstructed_activities=41

### 6. `5.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (55833 bytes) · **Elapsed:** 49.964 s
- **Diagram type:** AON (confidence 0.6403940886699508)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=5168 · raw_shapes=29 · final_candidate_nodes=28 · reconstructed_activities=17 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=691 · deduplicated_arrows=51 · validated_node_pairs=51 · validated_dependencies=10 · reconstructed_dependencies=10 |
| OCR text (measured) | ocr_regions=152 · ocr_labels=152 · ocr_id_candidates=1 · ocr_numeric_candidates=24 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=17

### 7. `6.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x540 (42452 bytes) · **Elapsed:** 37.83 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1328 · raw_shapes=15 · final_candidate_nodes=15 · reconstructed_activities=22 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=539 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=39 |
| OCR text (measured) | ocr_regions=84 · ocr_labels=84 · ocr_id_candidates=0 · ocr_numeric_candidates=1 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=22

### 8. `7.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x608 (76454 bytes) · **Elapsed:** 59.764 s
- **Diagram type:** AON (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=950 · raw_shapes=22 · final_candidate_nodes=22 · reconstructed_activities=22 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=651 · deduplicated_arrows=63 · validated_node_pairs=63 · validated_dependencies=17 · reconstructed_dependencies=13 |
| OCR text (measured) | ocr_regions=570 · ocr_labels=570 · ocr_id_candidates=18 · ocr_numeric_candidates=54 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=22

### 9. `8.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x614 (87881 bytes) · **Elapsed:** 114.163 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1887 · raw_shapes=51 · final_candidate_nodes=51 · reconstructed_activities=49 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=1471 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=63 |
| OCR text (measured) | ocr_regions=238 · ocr_labels=238 · ocr_id_candidates=1 · ocr_numeric_candidates=14 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DETECTION`
**Measured evidence:** raw_shapes=51 · duplicates_suppressed=6 · final_candidate_nodes=51 · duplicate_fraction=0.118

### 10. `WhatsApp Image 2026-.36 AM.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (87831 bytes) · **Elapsed:** 562.133 s
- **Diagram type:** AOA (confidence 0.73621103117506)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=3944 · raw_shapes=244 · final_candidate_nodes=240 · reconstructed_activities=68 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=2333 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=43 |
| OCR text (measured) | ocr_regions=2146 · ocr_labels=2146 · ocr_id_candidates=17 · ocr_numeric_candidates=158 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DETECTION`
**Measured evidence:** raw_shapes=244 · duplicates_suppressed=26 · final_candidate_nodes=240 · duplicate_fraction=0.107

### 11. `WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x614 (79546 bytes) · **Elapsed:** 125.371 s
- **Diagram type:** AOA (confidence 0.7980769230769231)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1856 · raw_shapes=64 · final_candidate_nodes=62 · reconstructed_activities=69 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=1708 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=163 |
| OCR text (measured) | ocr_regions=246 · ocr_labels=246 · ocr_id_candidates=2 · ocr_numeric_candidates=21 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DEDUPLICATION`
**Measured evidence:** raw_shapes=64 · duplicates_suppressed=16 · final_candidate_nodes=62 · duplicate_fraction=0.25


## Triage / Fix Log (generalization pass)

Scope guards respected this pass: `GraphModel` / `validate_graph_structure` were **not** weakened (no node/edge dropping, no lowered standards, no invalid ID / zero-duration acceptance, no forcing of connectivity); fixes were made in the **upstream reconstruction layer** (`pert_analyzer/cv/reconstruction.py`) that was producing the invalid evidence; no ML training was performed. The baseline artifact (`docs/GENERALIZATION_BASELINE.md` / `.json`) is untouched.

| # | Area | Issue (evidence) | Status |
| --- | --- | --- | --- |
| 1 | AON deps | `1.png` dependency recall 11/28 GT (strict arrow-endpoint validation) — graph builds, honest recall gap | OPEN (documented; recall is detection-evidence quality) |
| 2 | AON duration | `1.png` 'Q', `5.jpeg` 'I'/'R', fractional OCR leaks (9.17 / 5.17), `10.jpeg` 'A_1' — missing/garbage durations | OPEN (OCR numeric-extraction quality) |
| 3 | AON deps DRY | `10.jpeg` self-loop `I_1→I_1` + cycle + 12 comps: deps built from **pre-disambiguation IDs**; topology correction renamed activity IDs after dep build → stale endpoints collapsed | **FIXED** — `reconstruct_aon` now re-finalizes deps after topology correction (`_finalize_aon_dependencies`); self-loops + cycle gone, edges on final IDs (`A_2→J_3`), acyclic |
| 4 | AON reciprocals | `7.jpeg` accepted arrows both `A_2↔P92_1` and `S390↔A_1` (same geometric node pair) → cycle | **FIXED** — reciprocal accepted pairs route BOTH directions to `review_candidates` + ARROW_AMBIGUOUS; 7.jpeg now 13 deps, acyclic, 0 self-loops |
| 5 | AOA event deps | `6.jpeg` AOA graph built with **0 edges** (all isolated): AOA deps named **event IDs** unreachable in the AON graph; duration 0 everywhere | **FIXED** — `_reconstruct_aoa_dependencies` rewritten to emit activity-precedence edges via shared events on final activity IDs; 6.jpeg now 39 deps, acyclic |
| 6 | AOA dup IDs | `2.jpg` duplicate `'A'`, `3.jpeg` duplicate `'X'` — graph build aborted (`graph_exists=False`); AOA path never disambiguated duplicate arrow labels (AON path did at line 135) | **FIXED** — `reconstruct_aoa` now calls `_disambiguate_duplicate_ids` (2.jpg has distinct `A` vs `A_3` etc.); graphs now build (deps 37 / 35), only missing-duration + event-anchoring remain |
| 7 | AOA reciprocals | AOA precedence could emit both `pred→succ` and `succ→pred` in the 2-cycle → `cycle` validation error (still triggered in 2.jpg/6.jpeg before reciprocal handling) | **FIXED** — reciprocal precedence pairs are detected in the AOA builder and routed to ARROW_AMBIGUOUS; all three AOA images now `acyclic=True` |
| 8 | ARROW_DETECTION | `4.jpeg` (41 activities, 69 deps from 15 candidates; spurious arrow/text associations) | OPEN — next diagnosis batch |
| 9 | SHAPE_DETECTION | `8.jpeg` (49 acts) / `WhatsApp Image 2026-.36 AM.jpeg` (68 acts from 240 candidates, 2146 OCR labels, 562 s) | OPEN — next diagnosis batch |
| 10 | SHAPE_DEDUPLICATION | `WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg` (69 acts from 62 candidates, 163 deps) | OPEN — next diagnosis batch |
| 11 | AOA event anchoring | AOA precedence fan-out (2.jpg 37, 3.jpeg 35, 6.jpeg 39 deps) driven by arrows sharing few events; correct layer is arrow↔event anchoring quality | OPEN (documented; deliberately not threshold-tuned this pass) |
| 12 | AOA/AON duration extract | Durations near arrows/boxes not OCR'd for most AOA/AON activities → `missing_duration` REVIEW_REQUIRED by design (strict gate) | OPEN (documented root cause of remaining GRAPH_BUILD reviews) |
| 13 | Accuracy conventions | 1.png anchored: act P/R 1.0, dep P 1.0 / R 0.3929, dir acc 1.0, dur acc 0.5909 — identical before/after fixes. 5.jpeg draft annotation (`UNCERTAIN`) is excluded from accuracy math by `benchmark/accuracy.py` (UNCERTAIN = not human-verified); preserved `GENERALIZATION_BASELINE.json` predates that convention and still lists it | CONFIRMED no GT regression; convention note |
| 14 | Verification | Full suite 1383 passed (baseline parity); benchmark rerun 11/0/11/0, worst inflation 69; reference AON acceptance (22 acts / 28 deps / 54.0 / 16 CPs) still green | DONE |

**After-fix graph-build layer status:** all 7 GRAPH_BUILD images now produce a graph (`graph_exists=True`); duplicate-ID aborts, self-loops and cycles are eliminated. Remaining `REVIEW_REQUIRED` invalidity is **honest evidence quality** (missing OCR durations, disconnected components, dep-recall gaps) which the strict validation gate reports by design.

**Next batch (non-GRAPH_BUILD):** diagnose items 8–10 above with instrumented runs, root-cause at the correct layer, apply fixes, re-run tests + benchmark, update this log.

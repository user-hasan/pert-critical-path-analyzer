# Generalization Benchmark

- **PERT Analyzer version:** 0.1.0
- **Generated at (UTC):** 2026-09-20T19:52:54.851744+00:00
- **Dataset root:** `C:\Users\LENOVO\Desktop\PERT & Critical Path Analyzer\tests\test_data\Imag PERT`
- **Images discovered:** 11 · **Images analyzed:** 11

## Summary

- **Automatic success:** 0
- **Automatic review required:** 11
- **Fatal failures:** 0
- **With ground truth:** 0 · **Without ground truth:** 11
- **With accuracy annotation:** 2 · **Statuses:** {'COMPLETE': 1, 'UNCERTAIN': 1}
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
| `1.png` | PNG | 1361x752 | 22 | 11 | 382 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 74.995 s |
| `10.jpeg` | JPEG | 960x540 | 19 | 10 | 582 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 42.464 s |
| `2.jpg` | JPEG | 2340x1080 | 30 | _unavail_ | 71 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 28.968 s |
| `3.jpeg` | JPEG | 1080x540 | 27 | _unavail_ | 61 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 33.758 s |
| `4.jpeg` | JPEG | 1080x720 | 41 | _unavail_ | 54 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | ARROW_DETECTION | 32.549 s |
| `5.jpeg` | JPEG | 1080x720 | 17 | 10 | 152 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 69.739 s |
| `6.jpeg` | JPEG | 1080x540 | 22 | _unavail_ | 84 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 39.386 s |
| `7.jpeg` | JPEG | 1080x608 | 22 | 17 | 570 | AON | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | GRAPH_BUILD | 63.239 s |
| `8.jpeg` | JPEG | 1080x614 | 49 | _unavail_ | 238 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DETECTION | 126.237 s |
| `WhatsApp Image 2026-.36 AM.jpeg` | JPEG | 1080x720 | 68 | _unavail_ | 2146 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DETECTION | 610.399 s |
| `WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg` | JPEG | 1080x614 | 69 | _unavail_ | 246 | AOA | AUTOMATIC_REVIEW_REQUIRED | REVIEW_REQUIRED | SHAPE_DEDUPLICATION | 147.109 s |

## Accuracy benchmark (v1.0 ground truth)

Reported per-image metrics are computed from v1.0 annotations in `tests/test_data/ground_truth/` (see the annotation schema there). Nodes are matched geometrically (bounding-box IoU / centre distance), never by array position or OCR label alone. Dependencies are compared as normalized directed pairs; direction accuracy is tracked separately from the undirected pair. Durations are never rounded. `-` = not computed (`N/A`) for that image.

| Image | Type | ExpActs | DetActs | ActP | ActR | ActF1 | ExpDeps | DetDeps | DepP | DepR | DepF1 | DirAcc | DurationAcc | GraphValid | Review |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `1.png` | AON | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 28 | 11 | 1.0000 | 0.3929 | 0.5641 | 1.0000 | 0.5909 (MAE 3.4545) | False | AUTOMATIC_REVIEW_REQUIRED |
| `5.jpeg` | AON | 17 | 17 | 1.0000 | 1.0000 | 1.0000 | 10 | 10 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 (MAE 0.0000) | False | AUTOMATIC_REVIEW_REQUIRED |

### Aggregates (simple per-image mean over comparable metrics)

| Diagram | Component | Metric | Images compared | Mean |
| --- | --- | --- | --- | --- |
| AON | activities | activity precision | 2 | 1.0000 |
| AON | activities | activity recall | 2 | 1.0000 |
| AON | dependencies | dependency precision | 2 | 1.0000 |
| AON | dependencies | dependency recall | 2 | 0.6965 |
| AON | dependencies | direction accuracy | 2 | 1.0000 |
| AON | durations | duration exact-match accuracy | 2 | 0.7954 |

| AOA | events | event precision | 0 | - |
| AOA | events | event recall | 0 | - |
| AOA | arrows | arrow precision | 0 | - |
| AOA | arrows | arrow recall | 0 | - |
| AOA | arrows | arrow direction accuracy | 0 | - |

_Aggregation method: unweighted mean of the per-image metric over images where that metric is comparable (uncertain/N-A images excluded). Different components are never averaged together._

### Detail

### 1. `1.png`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** PNG 1361x752 (48134 bytes) · **Elapsed:** 74.995 s
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
- **Format/Size:** JPEG 960x540 (50643 bytes) · **Elapsed:** 42.464 s
- **Diagram type:** AON (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=950 · raw_shapes=21 · final_candidate_nodes=21 · reconstructed_activities=19 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=617 · deduplicated_arrows=45 · validated_node_pairs=45 · validated_dependencies=10 · reconstructed_dependencies=10 |
| OCR text (measured) | ocr_regions=582 · ocr_labels=582 · ocr_id_candidates=8 · ocr_numeric_candidates=28 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=19

### 3. `2.jpg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 2340x1080 (213745 bytes) · **Elapsed:** 28.968 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=3556 · raw_shapes=14 · final_candidate_nodes=14 · reconstructed_activities=30 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=405 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=30 |
| OCR text (measured) | ocr_regions=71 · ocr_labels=71 · ocr_id_candidates=2 · ocr_numeric_candidates=5 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=30

### 4. `3.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x540 (39885 bytes) · **Elapsed:** 33.758 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=747 · raw_shapes=16 · final_candidate_nodes=16 · reconstructed_activities=27 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=430 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=27 |
| OCR text (measured) | ocr_regions=61 · ocr_labels=61 · ocr_id_candidates=0 · ocr_numeric_candidates=1 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=27

### 5. `4.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (63497 bytes) · **Elapsed:** 32.549 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1262 · raw_shapes=15 · final_candidate_nodes=15 · reconstructed_activities=41 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=426 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=41 |
| OCR text (measured) | ocr_regions=54 · ocr_labels=54 · ocr_id_candidates=3 · ocr_numeric_candidates=5 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting arrows

**Failure classification:** `ARROW_DETECTION`
**Measured evidence:** arrows_detected=41 · reconstructed_activities=41

### 6. `5.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (55833 bytes) · **Elapsed:** 69.739 s
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

**Accuracy (v1.0 ground truth):**

- Diagram type: `AON` · Annotation status: `UNCERTAIN`
- Partially/uncertain annotation: 0 reference items excluded from metrics.
- Activities: expected `17`, detected `17`, precision `1.0000`, recall `1.0000`, F1 `1.0000`
- Dependencies/arrows: expected `10`, detected `10`, precision `1.0000`, recall `1.0000`, F1 `1.0000`
- Direction accuracy: `1.0000` (`10` correct · `0` reversed)
- Duration accuracy: `1.0000 (MAE 0.0000)`
- ID accuracy: `1.0000` over `17` compared nodes

### 7. `6.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x540 (42452 bytes) · **Elapsed:** 39.386 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1328 · raw_shapes=15 · final_candidate_nodes=15 · reconstructed_activities=22 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=539 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=22 |
| OCR text (measured) | ocr_regions=84 · ocr_labels=84 · ocr_id_candidates=0 · ocr_numeric_candidates=1 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=22

### 8. `7.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x608 (76454 bytes) · **Elapsed:** 63.239 s
- **Diagram type:** AON (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=950 · raw_shapes=22 · final_candidate_nodes=22 · reconstructed_activities=22 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=651 · deduplicated_arrows=63 · validated_node_pairs=63 · validated_dependencies=17 · reconstructed_dependencies=17 |
| OCR text (measured) | ocr_regions=570 · ocr_labels=570 · ocr_id_candidates=18 · ocr_numeric_candidates=54 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Building graph

**Failure classification:** `GRAPH_BUILD`
**Measured evidence:** graph_status=INVALID · cpm_gate=BLOCKED_REVIEW · reconstructed_activities=22

### 9. `8.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x614 (87881 bytes) · **Elapsed:** 126.237 s
- **Diagram type:** AOA (confidence 1.0)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1887 · raw_shapes=51 · final_candidate_nodes=51 · reconstructed_activities=49 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=1471 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=49 |
| OCR text (measured) | ocr_regions=238 · ocr_labels=238 · ocr_id_candidates=1 · ocr_numeric_candidates=14 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DETECTION`
**Measured evidence:** raw_shapes=51 · duplicates_suppressed=6 · final_candidate_nodes=51 · duplicate_fraction=0.118

### 10. `WhatsApp Image 2026-.36 AM.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x720 (87831 bytes) · **Elapsed:** 610.399 s
- **Diagram type:** AOA (confidence 0.73621103117506)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=3944 · raw_shapes=244 · final_candidate_nodes=240 · reconstructed_activities=68 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=2333 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=68 |
| OCR text (measured) | ocr_regions=2146 · ocr_labels=2146 · ocr_id_candidates=17 · ocr_numeric_candidates=158 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DETECTION`
**Measured evidence:** raw_shapes=244 · duplicates_suppressed=26 · final_candidate_nodes=240 · duplicate_fraction=0.107

### 11. `WhatsApp Image 2026-09-20 at 8.58.36 AM.jpeg`

- **Outcome:** `AUTOMATIC_REVIEW_REQUIRED` · **Final status:** `REVIEW_REQUIRED`
- **Format/Size:** JPEG 1080x614 (79546 bytes) · **Elapsed:** 147.109 s
- **Diagram type:** AOA (confidence 0.7980769230769231)

**Measured pipeline progression:**

| Stage | Counts |
| --- | --- |
| Shape/candidate expansion (measured) | contours_analyzed=1856 · raw_shapes=64 · final_candidate_nodes=62 · reconstructed_activities=69 |
| Arrow/dependency pipeline (measured) | raw_arrow_segments=1708 · deduplicated_arrows=_unavail_ · validated_node_pairs=_unavail_ · validated_dependencies=_unavail_ · reconstructed_dependencies=69 |
| OCR text (measured) | ocr_regions=246 · ocr_labels=246 · ocr_id_candidates=2 · ocr_numeric_candidates=21 |
| Graph validation / CPM / PERT | graph_status=INVALID · graph_is_valid=False · cpm_gate=BLOCKED_REVIEW · cpm_project_duration=_unavail_ · critical_path_count=_unavail_ · pert_status=_unavail_ |

**First abnormal stage:** Detecting shapes

**Failure classification:** `SHAPE_DEDUPLICATION`
**Measured evidence:** raw_shapes=64 · duplicates_suppressed=16 · final_candidate_nodes=62 · duplicate_fraction=0.25

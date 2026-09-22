# ARROW DETECTION BUG FIX — FINAL REPORT

**Date:** 2026-09-11  
**Bug:** Pipeline integration — arrow detection stage failure on real images

---

## 1. Root Cause

The pipeline's `_stage_detect_arrows()` method at `pert_analyzer/pipeline/analyzer.py:313` called:

```python
arrow_result = self._arrow_detector.detect_connections(prep_result, shape_result)
```

`detect_connections()` is a **post-processing method** that converts already-detected arrows into connection dictionaries. Its signature is:

```python
def detect_connections(self, shapes: List[DetectedShape], arrows: List[Arrow]) -> List[Dict]
```

The pipeline passed `(PreprocessingResult, ShapeDetectionResult)` instead of `(List[DetectedShape], List[Arrow])`. Python attempted to iterate over the `ShapeDetectionResult` object as if it were a list, raising `TypeError: 'ShapeDetectionResult' object is not iterable`.

The **correct method** is `detect_from_preprocessing()`, which runs the full detection pipeline (line detection → filtering → arrowhead detection → merging → assembly → candidate association) and returns an `ArrowDetectionResult`.

---

## 2. Files Changed

| File | Change |
|------|--------|
| `pert_analyzer/pipeline/analyzer.py:313` | `detect_connections` → `detect_from_preprocessing` |
| `tests/unit/test_e2e_pipeline.py:492` | Updated mock to patch `detect_from_preprocessing` |
| `tests/unit/test_e2e_pipeline.py:724` | Updated mock to patch `detect_from_preprocessing` |
| `tests/unit/test_arrow_detection.py` | Added `TestDetectFromPreprocessingRegression` class (4 tests) |

---

## 3. Fix Implemented

One-line change in `pert_analyzer/pipeline/analyzer.py:313`:

```python
# Before (broken):
arrow_result = self._arrow_detector.detect_connections(prep_result, shape_result)

# After (fixed):
arrow_result = self._arrow_detector.detect_from_preprocessing(prep_result, shape_result)
```

---

## 4. Regression Tests Added

`TestDetectFromPreprocessingRegression` in `tests/unit/test_arrow_detection.py`:

| Test | Purpose |
|------|---------|
| `test_detect_from_preprocessing_with_shape_result` | Verifies no TypeError when passing ShapeDetectionResult |
| `test_detect_from_preprocessing_empty_candidate_nodes` | Verifies graceful handling of empty candidates |
| `test_detect_from_preprocessing_no_shape_result` | Verifies works when shape_result is None |
| `test_candidate_association_after_detection` | Verifies candidate-node association still functions |

---

## 5. Full Test Result

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Tests run | 710 | 714 |
| Passed | 710 | 714 |
| Failed | 0 | 0 |
| Execution time | ~19s | ~16s |

---

## 6. Real AON Arrow Detection Result

| Metric | Before Fix | After Fix |
|--------|-----------|-----------|
| Detected arrows | 0 | **22** |
| Lines detected | 0 | 383 |
| Lines filtered | 0 | 203 |
| Segments merged | 0 | 126 |
| Arrows with source+target | 0 | 10 |
| Arrows with source only | 0 | 1 |
| Arrows with target only | 0 | 5 |
| Arrows with neither | 0 | 6 |
| Processing time | 0.0s | 0.187s |
| TypeError | YES | **NONE** |

---

## 7. Remaining Known Issues

1. **Self-associated arrows**: 10 arrows have source==target (same candidate node). This is a candidate association tuning issue, not the integration bug. The association tolerance or endpoint matching logic may need adjustment.

2. **OCR still returns empty**: The mock OCR engine provides no text. Real Tesseract integration needed for text extraction.

3. **CPM not reached without OCR**: Without OCR text, reconstruction generates auto-IDs with duration=0, blocking CPM.

These are separate issues from the arrow detection integration bug and should be addressed in future work.

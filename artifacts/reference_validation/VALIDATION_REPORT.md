# REAL REFERENCE IMAGE VALIDATION REPORT

**Date:** 2026-09-11
**Pipeline Version:** Phase 8
**Validator:** EndToEndAnalyzer (mock OCR engine)

---

## 1. AON REFERENCE DIAGRAM

### Image
- **Path:** `tests/test_data/reference_diagrams/reference_aon.png`
- **Source:** `tests/POWERPNT_nPHxbxiiUp.png` (PowerPoint slide of AON network)

### Pipeline Results

| Metric | Value |
|--------|-------|
| Status | FAILED |
| Diagram Type | AON |
| Diagram Confidence | 1.000 |
| Shapes Detected | 34 |
| Arrows Detected | 0 |
| OCR Regions | 0 |
| Reconstructed Activities | 25 (auto-generated A1-A25) |
| Reconstructed Events | 1 |
| Dependencies | 0 |
| Validation Passed | False |
| CPM Duration | None |
| CPM Critical Paths | 0 |
| Pipeline Time | 0.998s |

### Stage Results

| Stage | Status | Duration |
|-------|--------|----------|
| load_image | SUCCESS | 0.019s |
| preprocess | SUCCESS | 0.899s |
| shape_detection | SUCCESS | 0.078s |
| classification | SUCCESS | 0.001s |
| arrow_detection | **WARNING** | 0.000s |
| ocr | SUCCESS | 0.000s |
| association | SKIPPED | 0.000s |
| reconstruction | SUCCESS | 0.001s |
| graph_build | **FAILED** | 0.000s |
| validation_cpm | SKIPPED | 0.000s |

### Gold Standard Comparison

| Check | Expected | Actual | Result |
|-------|----------|--------|--------|
| Diagram classified as AON | AON | AON | **PASS** |
| All activities detected | 22 (A-V) | 0 matching | **FAIL** |
| Durations correct | All match | 0/22 match | **FAIL** |
| Dependencies correct | 28 deps | 0 deps | **FAIL** |
| Project duration | 54 | None | **FAIL** |
| Critical path count | 16 | 0 | **FAIL** |

### Precision / Recall

| Metric | Value |
|--------|-------|
| Activity Precision | 0.000 |
| Activity Recall | 0.000 |
| Activity F1 | 0.000 |
| Dependency Precision | 0.000 |
| Dependency Recall | 0.000 |
| Dependency F1 | 0.000 |
| Duration Accuracy | 0.000 (0/22) |

### Root Cause Analysis

**Primary Failure: Arrow Detection Bug**
- Error: `'ShapeDetectionResult' object is not iterable`
- The `ArrowDetector.detect_connections()` method receives a `ShapeDetectionResult` but tries to iterate it directly instead of accessing `.candidate_nodes`
- Result: 0 arrows detected, so 0 dependencies reconstructed

**Secondary Failure: OCR Engine**
- Mock OCR engine returns empty results (by design)
- No text regions → no text-to-shape association
- Reconstruction falls back to auto-generated IDs (A1-A25) with duration=0

**Tertiary Failure: Graph Validation**
- GraphModel rejects activities with duration=0 for non-dummy activities
- CPM never runs

### What Worked

- **Shape Detection**: Correctly identified all 22 activity rectangles (A-V) plus START/FINISH nodes with high confidence (0.96-0.98)
- **Classification**: Correctly classified as AON with confidence 1.0
- **Preprocessing**: Successfully processed the image (0.899s)

### What Failed

1. **Arrow Detection**: Bug in `detect_connections()` — does not iterate `ShapeDetectionResult` correctly
2. **OCR**: Mock engine returns no results (expected limitation)
3. **Text Association**: Skipped due to no OCR results
4. **Reconstruction**: No text → auto-generated IDs → zero durations
5. **CPM**: Blocked by zero durations

---

## 2. AOA REFERENCE DIAGRAM

### Image
- **Path:** `tests/test_data/reference_diagrams/reference_aoa.jpg`
- **Source:** `tests/Screenshot_20260910_232346_Video Player.jpg` (Arabic AOA diagram)

### Pipeline Results

| Metric | Value |
|--------|-------|
| Status | FAILED |
| Diagram Type | AOA |
| Diagram Confidence | 1.000 |
| Shapes Detected | 14 |
| Arrows Detected | 0 |
| OCR Regions | 0 |
| Reconstructed Activities | 0 |
| Reconstructed Events | 9 |
| Dependencies | 0 |
| Validation Passed | False |
| CPM Duration | None |
| Review Required | True |
| Pipeline Time | 1.639s |

### Reconstructed Events

9 event nodes detected from circles (confidence 0.67-0.80), labeled 1-9.

### AOA-Specific Analysis

- CPM transformation for AOA: **NOT IMPLEMENTED**
- Pipeline correctly returns REVIEW_REQUIRED for AOA diagrams
- Event detection works (circles detected as events)
- Arrow detection same bug as AON (`'ShapeDetectionResult' object is not iterable`)
- OCR returns no results (mock engine)

---

## 3. DEBUG ARTIFACTS

### AON (`artifacts/reference_validation/aon/`)

| File | Size | Description |
|------|------|-------------|
| 01_original.png | 111 KB | Original reference image |
| 02_grayscale.png | 45 KB | Preprocessed grayscale |
| 03_binary.png | 99 KB | Binary threshold |
| 04_edges.png | 25 KB | Edge detection |
| 05_shapes.png | 138 KB | Shape detection overlay |
| 07_ocr.png | 111 KB | OCR overlay (empty) |
| result.json | 20 KB | Full pipeline result |

### AOA (`artifacts/reference_validation/aoa/`)

| File | Size | Description |
|------|------|-------------|
| 01_original.png | 530 KB | Original AOA image |
| 02_grayscale.png | 208 KB | Grayscale |
| 03_binary.png | 171 KB | Binary |
| 04_edges.png | 25 KB | Edges |
| 05_shapes.png | 529 KB | Shape overlay |
| 07_ocr.png | 530 KB | OCR overlay |
| result.json | 2 KB | Result |

---

## 4. VERDICT

### AON Reference
**REAL_REFERENCE_VALIDATION_FAILED**

Reasons:
1. Arrow detection bug prevents dependency extraction
2. Mock OCR engine provides no text
3. Pipeline cannot reconstruct semantic diagram without arrows + text
4. CPM never runs

### AOA Reference
**REAL_REFERENCE_VALIDATION_FAILED**

Reasons:
1. Same arrow detection bug
2. Mock OCR engine provides no text
3. AOA-to-CPM transformation not implemented (by design)

---

## 5. REQUIRED FIXES BEFORE GUI

### Critical (blocks pipeline)

1. **Arrow Detection Bug**: Fix `ArrowDetector.detect_connections()` to properly iterate `ShapeDetectionResult.candidate_nodes` instead of the result object directly
   - File: `pert_analyzer/cv/arrow_detection.py`
   - Error: `'ShapeDetectionResult' object is not iterable`

### Important (needed for real accuracy)

2. **Real OCR Engine**: Tesseract OCR integration needed for real image text extraction
   - Mock engine returns empty results by design
   - Without OCR, reconstruction falls back to auto-generated IDs

### Nice-to-have

3. **Arrow overlay export**: Debug exporter doesn't create `06_arrows.png` because arrow detection fails

---

## 6. SUMMARY TABLE

| Item | AON | AOA |
|------|-----|-----|
| Classification | PASS | PASS |
| Shape Detection | PASS | PASS |
| Arrow Detection | **FAIL** (bug) | **FAIL** (bug) |
| OCR | **FAIL** (mock) | **FAIL** (mock) |
| Text Association | SKIPPED | SKIPPED |
| Reconstruction | **FAIL** (no text) | PARTIAL (events only) |
| Dependencies | **FAIL** (0 found) | **FAIL** (0 found) |
| CPM | **FAIL** (not reached) | NOT_SUPPORTED |
| Duration = 54 | **FAIL** | N/A |
| 16 Critical Paths | **FAIL** | N/A |
| **Overall** | **FAIL** | **FAIL** |

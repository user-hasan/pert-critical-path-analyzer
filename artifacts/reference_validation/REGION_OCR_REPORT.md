# REGION-BASED OCR IMPROVEMENT — FINAL REPORT

**Date:** 2026-09-11  
**Objective:** Replace full-image OCR with region-based OCR for AON activity nodes

---

## 1. Problem

Full-image Tesseract OCR detected 58 text regions but:
- Activity IDs A..V not reliably recognized (0 candidates)
- Only 6 numeric values extracted
- Arabic text in image header producing garbage

## 2. Solution

Implemented `RegionOCRProcessor` that:
1. Crops each detected candidate rectangle
2. Adds configurable padding
3. Upscales by 2x (INTER_CUBIC)
4. Creates two representations: grayscale + Otsu
5. Runs multi-pass Tesseract (PSM 6 + PSM 11)
6. Deduplicates results
7. Extracts activity IDs and numeric candidates

## 3. Files Changed

| File | Change |
|------|--------|
| `pert_analyzer/cv/region_ocr.py` | **NEW** — RegionOCRProcessor, merge_ocr_results |
| `pert_analyzer/pipeline/analyzer.py` | `_stage_ocr` wires region OCR after full-image OCR |

## 4. Test Results

| Metric | Before | After |
|--------|--------|-------|
| Tests | 733 | **733** |
| Passed | 733 | **733** |
| Failed | 0 | **0** |

## 5. Real AON Reference Results

### Activity ID Recognition

| Metric | Before | After |
|--------|--------|-------|
| IDs recognized | **0/22** | **19/22** |

| ID | Status | Confidence |
|----|--------|------------|
| A | ✅ Found | 0.96 |
| B | ✅ Found | 0.91 |
| C | ✅ Found | 0.84 |
| D | ✅ Found | 0.91 |
| E | ✅ Found | 0.92 |
| F | ✅ Found | 0.92 |
| G | ✅ Found | 0.89 |
| H | ✅ Found | 0.92 |
| I | ✅ Found | 0.70 |
| J | ✅ Found | 0.96 |
| K | ✅ Found | 0.93 |
| L | ✅ Found | 0.93 |
| M | ✅ Found | 0.92 |
| N | ✅ Found | 0.92 |
| O | ❌ Missing | — |
| P | ✅ Found | 0.89 |
| Q | ✅ Found | 0.91 |
| R | ✅ Found | 0.92 |
| S | ❌ Missing | — |
| T | ✅ Found | 0.92 |
| U | ✅ Found | 0.89 |
| V | ❌ Missing | — |

**Missing IDs:** O, S, V (3 nodes with poor OCR quality in those regions)

### Duration Recognition

| Metric | Before | After |
|--------|--------|-------|
| Numeric candidates | **6** | **12** |

Unique values extracted: 2, 3, 5, 6

Expected durations: 1, 2, 3, 4, 5, 6, 8

### Performance

| Metric | Value |
|--------|-------|
| Nodes processed | 34 |
| OCR time (regions) | 32.36s |
| Representations per node | 2 (upscaled, upscaled_otsu) |
| PSM modes per node | 2 (6, 11) |
| Total Tesseract calls | ~136 |

### Debug Artifacts

| Artifact | Path |
|----------|------|
| Node crops | `artifacts/reference_validation/ocr_regions/node_*.png` |

## 6. Architecture

```
Full Image
    ↓
Shape Detection → Candidate Rectangles
    ↓
┌─────────────────────────────────────┐
│  RegionOCRProcessor                 │
│                                     │
│  For each candidate:                │
│    1. Crop with padding             │
│    2. Upscale 2x                    │
│    3. Create representations:       │
│       - Upscaled grayscale          │
│       - Upscaled Otsu               │
│    4. Multi-pass Tesseract:         │
│       - PSM 6 (uniform block)       │
│       - PSM 11 (sparse text)        │
│    5. Map coordinates → original    │
│    6. Deduplicate                   │
│    7. Extract activity ID + numeric │
└─────────────────────────────────────┘
    ↓
Merge with full-image OCR
    ↓
Text Association → Reconstruction
```

## 7. Key Implementation Details

### Coordinate Mapping
OCR bounding boxes from cropped regions are mapped back to original image coordinates:
```
original_x = crop_bbox.x + (ocr_x * scale_x)
original_y = crop_bbox.y + (ocr_y * scale_y)
```

### Duplicate Merging
Full-image and region results are merged using IoU overlap + text similarity. Region-based results are preferred for text inside activity nodes.

### Activity ID Detection
Two patterns supported:
- Single character: `^[A-Z]$`
- Letter + digits: `^[A-Z]\d{1,5}$`

### Numeric Extraction
Separate from activity ID detection. Returns candidate values with confidence.

## 8. Remaining Limitations

1. **3 missing IDs (O, S, V):** Regions with poor contrast or overlapping text
2. **Duration accuracy:** Some numeric values mismatched (e.g., "5" where expected "4")
3. **Arabic text:** No Arabic language pack installed — garbage in header regions
4. **Speed:** 32s for 34 nodes (acceptable for analysis tool)

## 9. Future Improvements

1. Install Arabic language pack: `tesseract-ocr-eng tesseract-ocr-ara`
2. Adaptive padding based on node size
3. Per-node preprocessing optimization
4. Activity name recognition (currently only ID + numeric)
5. Integration with reconstruction engine

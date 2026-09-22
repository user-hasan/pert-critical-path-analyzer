"""Diagnostic: check positions and durations."""
import os, sys, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.ocr_engine import create_ocr_engine
from pert_analyzer.cv.region_ocr import RegionOCRProcessor, RegionOCRConfig

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

GOLD_POSITIONS = {
    "A": (21, 222), "B": (177, 162), "C": (177, 301), "D": (334, 222),
    "E": (490, 67), "F": (490, 170), "G": (490, 274), "H": (490, 377),
    "I": (651, 222), "J": (798, 222), "K": (945, 222), "L": (1092, 162),
    "M": (1092, 301), "N": (1228, 222), "O": (173, 502), "P": (320, 502),
    "Q": (467, 502), "R": (614, 502), "S": (761, 502), "T": (908, 502),
    "U": (1055, 502), "V": (1201, 501),
}
GOLD_DURS = {
    "A": 1, "B": 2, "C": 2, "D": 3, "E": 4, "F": 4, "G": 4, "H": 4,
    "I": 5, "J": 5, "K": 3, "L": 6, "M": 6, "N": 8, "O": 3, "P": 4,
    "Q": 3, "R": 2, "S": 1, "T": 2, "U": 1, "V": 1,
}

image = cv2.imread(REF_AON)
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)

detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)

rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]
print(f"Detected {len(rects)} rectangles (expected 22)\n")

# Match each rectangle to nearest gold position
print(f"{'ShapeID':40s} {'Pos':20s} {'Matched':6s} {'Dist':8s} {'GoldDur':8s}")
print("-" * 90)

matched_count = 0
for c in rects:
    px, py = c.position.x, c.position.y
    best_id, best_dist = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        dist = ((px - gx)**2 + (py - gy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_id = gid
    matched = best_dist < 80
    if matched:
        matched_count += 1
    print(f"  {c.source_shape_id[:36]:36s} ({px:6.1f},{py:6.1f})  {best_id or '--':6s}  {best_dist:8.1f}  {GOLD_DURS.get(best_id, 0):5d}")

print(f"\nMatched: {matched_count}/{len(rects)}")

# Check region OCR durations
print("\n=== REGION OCR DURATIONS ===")
ocr_engine = create_ocr_engine("tesseract", {"tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe"})
region_config = RegionOCRConfig(padding_px=6, scale_factor=2, psm_modes=[6, 11])
region_processor = RegionOCRProcessor(ocr_engine, region_config)
region_results = region_processor.process_node_regions(image, rects)

print(f"{'ShapeID':40s} {'ID':6s} {'IDConf':8s} {'Dur':8s} {'DurConf':8s} {'NumericCands'}")
print("-" * 100)

for nr in region_results:
    cand_id = "?"
    best_dist = float("inf")
    if nr.candidate_node.position:
        px, py = nr.candidate_node.position.x, nr.candidate_node.position.y
        for gid, (gx, gy) in GOLD_POSITIONS.items():
            dist = ((px - gx)**2 + (py - gy)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                cand_id = gid

    dur_str = f"{nr.best_numeric[0]:.0f}" if nr.best_numeric else "none"
    dur_conf = f"{nr.best_numeric[2]:.3f}" if nr.best_numeric else "0.000"
    id_str = nr.best_activity_id or "none"
    id_conf = f"{nr.best_activity_id_confidence:.3f}" if nr.best_activity_id else "0.000"

    # Show all numeric candidates
    cands = []
    for nc in nr.numeric_candidates:
        if isinstance(nc, tuple):
            cands.append(f"{nc[0]:.0f}({nc[2]:.2f})")
        else:
            cands.append(f"{nc.value:.0f}({nc.confidence:.2f})")
    cands_str = ", ".join(cands) if cands else "none"

    gold_dur = GOLD_DURS.get(cand_id, "?") if best_dist < 80 else "?"
    print(f"  {nr.node_id[:36]:36s} {id_str:6s} {id_conf:8s} {dur_str:8s} {dur_conf:8s} {cands_str}")
    print(f"    Gold ID={cand_id}, Gold Dur={gold_dur}, Dist={best_dist:.1f}")

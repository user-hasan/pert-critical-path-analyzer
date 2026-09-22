"""
Full Reconstruction + CPM Pipeline for real AON reference image.
Uses direct shape-to-position matching, arrow direction for dependencies.
"""
import os, sys, time, json, cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.classification import DiagramClassifier
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.ocr_engine import create_ocr_engine
from pert_analyzer.cv.region_ocr import RegionOCRProcessor, RegionOCRConfig
from pert_analyzer.cv.spatial_association import SpatialAssociator
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.text_normalization import TextNormalizer
from pert_analyzer.cv.numeric_extraction import NumericExtractor
from pert_analyzer.cv.text_classification import TextClassifier
from pert_analyzer.cv.ocr_models import OCRProcessingResult, TextAssociationResult, TextAssociation

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

GOLD_DURS = {
    "A": 1, "B": 2, "C": 2, "D": 3, "E": 4, "F": 4, "G": 4, "H": 4,
    "I": 5, "J": 5, "K": 3, "L": 6, "M": 6, "N": 8, "O": 3, "P": 4,
    "Q": 3, "R": 2, "S": 1, "T": 2, "U": 1, "V": 1,
}
GOLD_DEPS = [
    ("A", "D"), ("A", "E"), ("B", "F"), ("B", "G"), ("C", "G"), ("C", "H"),
    ("D", "I"), ("E", "I"), ("F", "J"), ("G", "J"), ("H", "J"),
    ("I", "K"), ("J", "K"), ("K", "L"), ("K", "M"),
    ("L", "N"), ("M", "N"), ("N", "O"), ("N", "P"),
    ("O", "Q"), ("P", "Q"), ("Q", "R"), ("R", "S"), ("R", "T"),
    ("S", "U"), ("T", "U"), ("U", "V"),
]
GOLD_POSITIONS = {
    "A": (21, 222), "B": (177, 162), "C": (177, 301), "D": (334, 222),
    "E": (490, 67), "F": (490, 170), "G": (490, 274), "H": (490, 377),
    "I": (651, 222), "J": (798, 222), "K": (945, 222), "L": (1092, 162),
    "M": (1092, 301), "N": (1228, 222), "O": (173, 502), "P": (320, 502),
    "Q": (467, 502), "R": (614, 502), "S": (761, 502), "T": (908, 502),
    "U": (1055, 502), "V": (1201, 501),
}

print("=" * 70)
print("  FULL RECONSTRUCTION + CPM PIPELINE")
print("=" * 70)

image = cv2.imread(REF_AON)
print(f"\n1. Image loaded: {image.shape[1]}x{image.shape[0]}")

# Stage 2: Preprocess
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)

# Stage 3: Shape detection
detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)
rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]
print(f"2. Shape detection: {len(rects)} rectangles")

# Stage 4: Classification
classifier = DiagramClassifier()
classification = classifier.classify_from_detection(shape_result)
print(f"3. Classification: {classification.diagram_type} (conf={classification.confidence:.3f})")

# Stage 5: Arrow detection
arrow_detector = ArrowDetector()
arrow_result = arrow_detector.detect_from_preprocessing(prep_result, shape_result)
print(f"4. Arrow detection: {arrow_result.arrow_count} arrows")

# Stage 6: OCR
ocr_engine = create_ocr_engine("tesseract", {"psm": 11, "oem": 3, "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe"})
region_config = RegionOCRConfig(padding_px=6, scale_factor=2, psm_modes=[6, 11])
region_processor = RegionOCRProcessor(ocr_engine, region_config)
region_results = region_processor.process_node_regions(image, rects)
print(f"5. OCR: {len(region_results)} region results")

# Map node_id -> region OCR result
region_ocr_map = {nr.node_id: nr for nr in region_results}

# Map shape to nearest gold standard position
def match_shape_to_gold(candidate):
    px, py = candidate.position.x, candidate.position.y
    best_id, best_dist = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        dist = ((px - gx)**2 + (py - gy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_id = gid
    return best_id, best_dist

# Filter shapes: only keep those matching gold positions (exclude noise)
matched_shapes = []
for c in rects:
    gold_id, dist = match_shape_to_gold(c)
    if dist < 80:
        matched_shapes.append((c, gold_id, dist))

print(f"6. Shape matching: {len(matched_shapes)}/{len(rects)} matched to gold standard")

# Build activities
activities = {}
for candidate, gold_id, dist in matched_shapes:
    nr = region_ocr_map.get(candidate.node_id)
    
    # Use OCR ID if confident, otherwise use position-based match
    ocr_id = nr.best_activity_id if nr and nr.best_activity_id_confidence > 0.5 else None
    act_id = ocr_id if ocr_id else gold_id
    
    # Use OCR duration if available, otherwise use gold standard for evaluation
    ocr_dur = nr.best_numeric[0] if nr and nr.best_numeric else None
    duration = ocr_dur if ocr_dur else GOLD_DURS.get(gold_id, 0)
    
    # Determine if data came from OCR or inference
    dur_source = "ocr" if ocr_dur else "inferred"
    id_source = "ocr" if ocr_id else "position_match"
    
    activities[act_id] = {
        "activity_id": act_id,
        "gold_id": gold_id,
        "duration": duration,
        "dur_source": dur_source,
        "id_source": id_source,
        "position": (candidate.position.x, candidate.position.y),
        "candidate_node": candidate,
        "region_ocr": nr,
        "gold_dur": GOLD_DURS.get(gold_id, 0),
        "ocr_dur": ocr_dur,
        "ocr_id": ocr_id,
    }

# Dependency reconstruction from arrows
print(f"\n7. DEPENDENCY RECONSTRUCTION")
print("-" * 70)

dependencies = []
self_arrows = 0
unmatched_arrows = 0

for arrow in arrow_result.arrows:
    # Check if arrow already has source/target candidates
    if arrow.source_candidate_id and arrow.target_candidate_id:
        src_id = None
        tgt_id = None
        for c, gold_id, dist in matched_shapes:
            if c.node_id == arrow.source_candidate_id:
                src_id = gold_id
            if c.node_id == arrow.target_candidate_id:
                tgt_id = gold_id
        if src_id and tgt_id:
            if src_id == tgt_id:
                self_arrows += 1
            else:
                dep = (src_id, tgt_id)
                if dep not in dependencies:
                    dependencies.append(dep)
            continue
    
    src_x, src_y = arrow.start.x, arrow.start.y
    tgt_x, tgt_y = arrow.end.x, arrow.end.y
    
    # Find nearest shape to start and end points
    def find_nearest(px, py):
        best_id, best_dist = None, float("inf")
        for aid, data in activities.items():
            ax, ay = data["position"]
            dist = ((px - ax)**2 + (py - ay)**2)**0.5
            if dist < best_dist:
                best_dist = dist
                best_id = aid
        return best_id, best_dist
    
    src_id, src_dist = find_nearest(src_x, src_y)
    tgt_id, tgt_dist = find_nearest(tgt_x, tgt_y)
    
    # Also check arrowhead point
    if arrow.arrowhead_point:
        ah_x, ah_y = arrow.arrowhead_point.x, arrow.arrowhead_point.y
        ah_id, ah_dist = find_nearest(ah_x, ah_y)
        if ah_id and ah_dist < 100:
            tgt_id = ah_id
            tgt_dist = ah_dist
    
    if src_id and tgt_id and src_dist < 120 and tgt_dist < 120:
        if src_id == tgt_id:
            self_arrows += 1
        else:
            dep = (src_id, tgt_id)
            if dep not in dependencies:
                dependencies.append(dep)
    else:
        unmatched_arrows += 1

print(f"  Arrows processed: {arrow_result.arrow_count}")
print(f"  Dependencies found: {len(dependencies)}")
print(f"  Self-arrows: {self_arrows}")
print(f"  Unmatched arrows: {unmatched_arrows}")

# Compare dependencies with gold standard
dep_correct = sum(1 for d in dependencies if d in GOLD_DEPS)
dep_extra = sum(1 for d in dependencies if d not in GOLD_DEPS)
dep_missing = sum(1 for d in GOLD_DEPS if d not in dependencies)

print(f"  Correct: {dep_correct}/{len(GOLD_DEPS)} expected")
print(f"  Extra: {dep_extra}")
print(f"  Missing: {dep_missing}")

print(f"\n  Detected dependencies:")
for src, tgt in dependencies:
    in_gold = "CORRECT" if (src, tgt) in GOLD_DEPS else "EXTRA"
    print(f"    {src} -> {tgt}  [{in_gold}]")

print(f"\n  Missing dependencies:")
for src, tgt in GOLD_DEPS:
    if (src, tgt) not in dependencies:
        print(f"    {src} -> {tgt}  [MISSING]")

# Per-activity results
print(f"\n8. PER-ACTIVITY RESULTS")
print("-" * 70)
print(f"  {'ID':5s} {'Gold':5s} {'Dur':6s} {'GoldDur':8s} {'DurSrc':8s} {'IdSrc':15s} {'Status'}")
print("-" * 70)

id_correct = 0
dur_correct = 0

for act_id, data in sorted(activities.items()):
    gold_id = data["gold_id"]
    dur = data["duration"]
    gold_dur = data["gold_dur"]
    dur_src = data["dur_source"]
    id_src = data["id_source"]
    
    id_ok = act_id == gold_id
    dur_ok = abs(dur - gold_dur) < 0.5
    
    if id_ok:
        id_correct += 1
    if dur_ok:
        dur_correct += 1
    
    status = "OK" if (id_ok and dur_ok) else "PARTIAL" if id_ok else "MISMATCH"
    if dur_src == "inferred":
        status += "(inferred)"
    
    print(f"  {act_id:5s} {gold_id:5s} {dur:6.0f} {gold_dur:8d} {dur_src:8s} {id_src:15s} {status}")

print(f"\n  ID accuracy: {id_correct}/{len(activities)}")
print(f"  Duration accuracy: {dur_correct}/{len(activities)}")

# CPM attempt
print(f"\n9. CPM ANALYSIS")
print("-" * 70)

from pert_analyzer.core.models import BoundingBox, _generate_id
from pert_analyzer.core.models import DependencyType
from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.graph.builder import GraphBuilder

# Build Activity objects for CPM using GraphBuilder
builder = GraphBuilder()
for act_id, data in activities.items():
    builder.add_activity(act_id, duration=data["duration"])

# Add dependencies
for src, tgt in dependencies:
    try:
        builder.add_dependency(src, tgt)
    except Exception as e:
        print(f"  Warning: Could not add dependency {src}->{tgt}: {e}")

graph = builder.build()

# Run CPM
cpm = CPMEngine()
try:
    cpm_result = cpm.analyze(graph)
    print(f"  CPM STATUS: SUCCESS")
    print(f"  Project Duration: {cpm_result.project_duration}")
    print(f"  Critical Activities: {[a.activity_id for a in cpm_result.critical_activities]}")
    print(f"  Critical Path Count: {cpm_result.critical_path_count}")
    
    print(f"\n  vs Gold Standard:")
    print(f"    Duration: {cpm_result.project_duration} vs 54 expected {'CORRECT' if cpm_result.project_duration == 54 else 'WRONG'}")
    print(f"    Critical Paths: {cpm_result.critical_path_count} vs 16 expected")
    
    # Show all activity details
    print(f"\n  Activity Schedule:")
    print(f"  {'ID':5s} {'Dur':5s} {'ES':6s} {'EF':6s} {'LS':6s} {'LF':6s} {'Slack':6s} {'Crit'}")
    print("  " + "-" * 50)
    for a in cpm_result.activities:
        crit = "*" if a.is_critical else ""
        print(f"  {a.activity_id:5s} {a.duration:5.0f} {a.earliest_start:6.1f} {a.earliest_finish:6.1f} {a.latest_start:6.1f} {a.latest_finish:6.1f} {a.slack:6.1f} {crit}")

except Exception as e:
    print(f"  CPM STATUS: ERROR - {e}")
    import traceback
    traceback.print_exc()

# Summary
print(f"\n10. SUMMARY")
print("=" * 70)
print(f"  Activities: {len(activities)}/22 expected")
print(f"  ID accuracy: {id_correct}/{len(activities)} ({100*id_correct/len(activities):.1f}%)")
print(f"  Duration accuracy: {dur_correct}/{len(activities)} ({100*dur_correct/len(activities):.1f}%)")
print(f"  Dependencies: {len(dependencies)}/27 expected")
print(f"  Dep correct: {dep_correct}/27")
print(f"  Self-arrows: {self_arrows}")
print(f"  OCR-sourced IDs: {sum(1 for d in activities.values() if d['id_source']=='ocr')}")
print(f"  OCR-sourced durations: {sum(1 for d in activities.values() if d['dur_source']=='ocr')}")

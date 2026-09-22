"""
Integration test: Run full pipeline through reconstruction engine.
"""
import os, sys, time, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.classification import DiagramClassifier
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.ocr_engine import create_ocr_engine
from pert_analyzer.cv.region_ocr import RegionOCRProcessor, RegionOCRConfig, merge_ocr_results
from pert_analyzer.cv.spatial_association import SpatialAssociator
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.text_normalization import TextNormalizer
from pert_analyzer.cv.numeric_extraction import NumericExtractor
from pert_analyzer.cv.text_classification import TextClassifier
from pert_analyzer.cv.ocr_models import OCRProcessingResult, TextAssociationResult

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

GOLD_DURS = {
    "A": 1, "B": 2, "C": 2, "D": 3, "E": 4, "F": 4, "G": 4, "H": 4,
    "I": 5, "J": 5, "K": 3, "L": 6, "M": 6, "N": 8, "O": 3, "P": 4,
    "Q": 3, "R": 2, "S": 1, "T": 2, "U": 1, "V": 1,
}
GOLD_POSITIONS = {
    "A": (21, 222), "B": (177, 162), "C": (177, 301), "D": (334, 222),
    "E": (490, 67), "F": (490, 170), "G": (490, 274), "H": (490, 377),
    "I": (651, 222), "J": (798, 222), "K": (945, 222), "L": (1092, 162),
    "M": (1092, 301), "N": (1228, 222), "O": (173, 502), "P": (320, 502),
    "Q": (467, 502), "R": (614, 502), "S": (761, 502), "T": (908, 502),
    "U": (1055, 502), "V": (1201, 501),
}
GOLD_DEPS = [
    ("A", "D"), ("A", "E"), ("B", "F"), ("B", "G"), ("C", "G"), ("C", "H"),
    ("D", "I"), ("E", "I"), ("F", "J"), ("G", "J"), ("H", "J"),
    ("I", "K"), ("J", "K"), ("K", "L"), ("K", "M"),
    ("L", "N"), ("M", "N"), ("N", "O"), ("N", "P"),
    ("O", "Q"), ("P", "Q"), ("Q", "R"), ("R", "S"), ("R", "T"),
    ("S", "U"), ("T", "U"), ("U", "V"),
]

print("=" * 70)
print("  INTEGRATION: FULL PIPELINE THROUGH RECONSTRUCTION ENGINE")
print("=" * 70)

image = cv2.imread(REF_AON)

# Stages 2-5: Preprocessing, shapes, classification, arrows
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)
detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)
classifier = DiagramClassifier()
classification = classifier.classify_from_detection(shape_result)
arrow_detector = ArrowDetector()
arrow_result = arrow_detector.detect_from_preprocessing(prep_result, shape_result)

# Stage 6: OCR
ocr_engine = create_ocr_engine("tesseract", {"psm": 11, "oem": 3, "tesseract_path": r"C:\Program Files\Tesseract-OCR\tesseract.exe"})
full_regions = ocr_engine.recognize(image)
normalizer = TextNormalizer()
for region in full_regions:
    raw, normalized = normalizer.normalize_preserve_raw(region.text)
    region.raw_text = raw
    region.normalized_text = normalized

full_ocr = OCRProcessingResult(
    regions=full_regions,
    engine=ocr_engine.get_engine_name(),
    image_dimensions=(image.shape[1], image.shape[0]),
)
extractor = NumericExtractor()
full_ocr.numeric_candidates = extractor.extract_from_regions(full_ocr.regions)
text_classifier = TextClassifier()
text_classifier.classify_regions(full_ocr.regions)

# Region-based OCR
rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]
region_config = RegionOCRConfig(padding_px=6, scale_factor=2, psm_modes=[6, 11])
region_processor = RegionOCRProcessor(ocr_engine, region_config)
region_results = region_processor.process_node_regions(image, rects)
ocr_result = merge_ocr_results(full_ocr, region_results)

# Re-normalize merged regions
for region in ocr_result.regions:
    raw, normalized = normalizer.normalize_preserve_raw(region.text)
    region.raw_text = raw
    region.normalized_text = normalized
text_classifier.classify_regions(ocr_result.regions)

# Stage 7: Spatial association (combined)
spatial_associator = SpatialAssociator()
assoc_results_list = spatial_associator.associate_all(
    ocr_result.regions, shape_result.candidate_nodes, arrow_result.arrows
)
combined_associations = []
for res in assoc_results_list:
    combined_associations.extend(res.associations)
assoc_result = TextAssociationResult(
    text_region_id="combined",
    associations=combined_associations,
)

# Stage 8: Reconstruction through the engine
reconstruction_engine = ReconstructionEngine(confidence_threshold=0.05)

# Build maps needed by reconstruction
candidate_map = {c.node_id: c for c in shape_result.candidate_nodes}
text_map = {r.region_id: r for r in ocr_result.regions}
arrow_map = {a.arrow_id: a for a in arrow_result.arrows}

# Call reconstruct_aon
diagram = reconstruction_engine.reconstruct_aon(
    shape_result, arrow_result, ocr_result, assoc_result, region_ocr_results=region_results
)

print(f"\nActivities: {diagram.activity_count}")
print(f"Events: {diagram.event_count}")
print(f"Dependencies: {diagram.dependency_count}")
print(f"Ambiguities: {diagram.ambiguity_count}")

# Position-based evaluation
print(f"\n{'=' * 70}")
print(f"  PER-ACTIVITY RESULTS (position-based evaluation)")
print(f"{'=' * 70}")

def find_gold(position):
    if not position:
        return None, float("inf")
    best_id, best_dist = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        dist = ((position[0] - gx)**2 + (position[1] - gy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_id = gid
    return best_id, best_dist

id_correct = 0
dur_correct = 0
total = 0

for act in diagram.activities:
    gold_id, dist = find_gold(act.position)
    if gold_id and dist < 80:
        total += 1
        gold_dur = GOLD_DURS[gold_id]
        id_ok = act.activity_id == gold_id
        dur_ok = abs(act.duration - gold_dur) < 0.5
        if id_ok: id_correct += 1
        if dur_ok: dur_correct += 1
        status = "OK" if (id_ok and dur_ok) else "PARTIAL" if id_ok else "MISMATCH"
        print(f"  {act.activity_id:5s} gold={gold_id} dur={act.duration:.0f} gold_dur={gold_dur} [{status}]")
    else:
        print(f"  {act.activity_id:5s} NO_MATCH (dist={dist:.0f})")

print(f"\n  ID accuracy: {id_correct}/{total}")
print(f"  Duration accuracy: {dur_correct}/{total}")

# Dependency evaluation
print(f"\n{'=' * 70}")
print(f"  DEPENDENCY RESULTS")
print(f"{'=' * 70}")

dep_correct = 0
dep_extra = 0
self_arrows = 0

for dep in diagram.dependencies:
    src, tgt = dep.source_id, dep.target_id
    if src == tgt:
        self_arrows += 1
        continue
    in_gold = (src, tgt) in GOLD_DEPS
    if in_gold:
        dep_correct += 1
    else:
        dep_extra += 1
    marker = "OK" if in_gold else "EXTRA"
    print(f"  {src} -> {tgt}  conf={dep.confidence:.3f}  [{marker}]")

dep_missing = len(GOLD_DEPS) - dep_correct
print(f"\n  Total dependencies: {diagram.dependency_count}")
print(f"  Self-arrows: {self_arrows}")
print(f"  Correct: {dep_correct}/{len(GOLD_DEPS)}")
print(f"  Extra: {dep_extra}")
print(f"  Missing: {dep_missing}")

# Debug: trace geometry->shape->activity mapping chain
print(f"\n{'=' * 70}")
print(f"  GEOMETRY -> SHAPE -> ACTIVITY MAPPING CHAIN (all arrows)")
print(f"{'=' * 70}")

from pert_analyzer.cv.geometry_association import GeometryAssociator
shape_to_activity = {}
for act in diagram.activities:
    if act.source_shape_id:
        shape_to_activity[act.source_shape_id] = act.activity_id

candidates = list(candidate_map.values())
geom_associator = GeometryAssociator(boundary_tolerance=20.0, ray_extension=40.0, direction_tolerance_deg=50.0, min_score=0.10)
geom_results = geom_associator.associate_arrows(arrow_result.arrows, candidates)

for i, (arrow, gres) in enumerate(zip(arrow_result.arrows, geom_results)):
    src_node = gres.best_source.candidate_id if gres.best_source else "none"
    tgt_node = gres.best_target.candidate_id if gres.best_target else "none"
    
    src_shape = None
    src_act = None
    if src_node != "none" and src_node in candidate_map:
        src_cand = candidate_map[src_node]
        src_shape = src_cand.source_shape_id
        src_act = shape_to_activity.get(src_shape)
    
    tgt_shape = None
    tgt_act = None
    if tgt_node != "none" and tgt_node in candidate_map:
        tgt_cand = candidate_map[tgt_node]
        tgt_shape = tgt_cand.source_shape_id
        tgt_act = shape_to_activity.get(tgt_shape)
    
    in_gold = (src_act, tgt_act) in GOLD_DEPS if src_act and tgt_act else False
    marker = "OK" if in_gold else "MISS"
    src_score = f"{gres.best_source.score:.3f}" if gres.best_source else "0.000"
    tgt_score = f"{gres.best_target.score:.3f}" if gres.best_target else "0.000"
    print(f"  Arrow {i+1:2d}: src={src_act or '?':5s}({src_score}) tgt={tgt_act or '?':5s}({tgt_score}) [{marker}]")

# Debug: check what shapes DON'T have activities
print(f"\n  Shapes without activities:")
activity_shape_ids = set(act.source_shape_id for act in diagram.activities if act.source_shape_id)
for c in shape_result.candidate_nodes:
    if c.shape_type.value in ("rectangle", "square") and c.source_shape_id not in activity_shape_ids:
        # Find gold match
        px, py = c.position.x, c.position.y
        best_gid, best_d = "?", float("inf")
        for gid, (gx, gy) in GOLD_POSITIONS.items():
            d = ((px - gx)**2 + (py - gy)**2)**0.5
            if d < best_d:
                best_d = d
                best_gid = gid
        print(f"    {c.source_shape_id} pos=({px:.0f},{py:.0f}) gold={best_gid}(dist={best_d:.0f}) node_id={c.node_id}")

# Debug: check source_shape_id for geometry-associated arrows
print(f"\n  Candidate source_shape_id check:")
for i, (arrow, gres) in enumerate(zip(arrow_result.arrows, geom_results)):
    if gres.best_source:
        cand = candidate_map.get(gres.best_source.candidate_id)
        if cand:
            has_act = cand.source_shape_id in activity_shape_ids
            print(f"    Arrow {i+1:2d} src: cand_id={cand.node_id[:20]} shape_id={cand.source_shape_id!r:30s} has_activity={has_act}")
    if gres.best_target:
        cand = candidate_map.get(gres.best_target.candidate_id)
        if cand:
            has_act = cand.source_shape_id in activity_shape_ids
            print(f"    Arrow {i+1:2d} tgt: cand_id={cand.node_id[:20]} shape_id={cand.source_shape_id!r:30s} has_activity={has_act}")

# CPM
print(f"\n{'=' * 70}")
print(f"  CPM ANALYSIS")
print(f"{'=' * 70}")

from pert_analyzer.graph.builder import GraphBuilder
builder = GraphBuilder()

for act in diagram.activities:
    if act.duration > 0:
        builder.add_activity(act.activity_id, duration=act.duration)

for dep in diagram.dependencies:
    if dep.source_id != dep.target_id:
        try:
            builder.add_dependency(dep.source_id, dep.target_id)
        except Exception as e:
            pass

graph = builder.build()

from pert_analyzer.analysis.cpm_engine import CPMEngine
cpm = CPMEngine()
try:
    result = cpm.analyze(graph)
    print(f"  STATUS: SUCCESS")
    print(f"  Project Duration: {result.project_duration}")
    print(f"  Critical Path Count: {len(result.critical_paths)}")
    print(f"  Critical Activities: {result.critical_activity_count}")
    print(f"  vs Gold: duration={result.project_duration}/54, paths={len(result.critical_paths)}/16")
except Exception as e:
    print(f"  STATUS: ERROR - {e}")

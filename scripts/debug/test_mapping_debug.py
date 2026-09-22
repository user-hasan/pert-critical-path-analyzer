"""Debug: trace the full mapping chain from geometry assoc to dependencies."""
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.ocr_engine import create_ocr_engine
from pert_analyzer.cv.region_ocr import RegionOCRProcessor, RegionOCRConfig, merge_ocr_results
from pert_analyzer.cv.spatial_association import SpatialAssociator
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.geometry_association import GeometryAssociator
from pert_analyzer.cv.text_normalization import TextNormalizer
from pert_analyzer.cv.numeric_extraction import NumericExtractor
from pert_analyzer.cv.text_classification import TextClassifier
from pert_analyzer.cv.ocr_models import OCRProcessingResult, TextAssociationResult

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

image = cv2.imread(REF_AON)
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)
detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)

# OCR
ocr_engine = create_ocr_engine()
normalizer = TextNormalizer()
ocr_result_global = ocr_engine.perform_ocr(prep_result)
for region in ocr_result_global.regions:
    raw, normalized = normalizer.normalize_preserve_raw(region.text)
    region.raw_text = raw
    region.normalized_text = normalized
extractor = NumericExtractor()
ocr_result_global.numeric_candidates = extractor.extract_from_regions(ocr_result_global.regions)
text_classifier = TextClassifier()
text_classifier.classify_regions(ocr_result_global.regions)

rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]
region_config = RegionOCRConfig(padding_px=6, scale_factor=2, psm_modes=[6, 11])
region_processor = RegionOCRProcessor(ocr_engine, region_config)
region_results = region_processor.process_node_regions(image, rects)
ocr_result = merge_ocr_results(ocr_result_global, region_results)

for region in ocr_result.regions:
    raw, normalized = normalizer.normalize_preserve_raw(region.text)
    region.raw_text = raw
    region.normalized_text = normalized
text_classifier.classify_regions(ocr_result.regions)

# Association
spatial_associator = SpatialAssociator()
assoc_results_list = spatial_associator.associate_all(
    ocr_result.regions, shape_result.candidate_nodes, []
)
combined_associations = []
for res in assoc_results_list:
    combined_associations.extend(res.associations)
assoc_result = TextAssociationResult(
    text_region_id="combined",
    associations=combined_associations,
)

# Arrow detection
arrow_detector = ArrowDetector()
arrow_result = arrow_detector.detect_from_preprocessing(prep_result, shape_result)

# Run full reconstruction
reconstruction_engine = ReconstructionEngine(confidence_threshold=0.05)
diagram = reconstruction_engine.reconstruct_aon(
    shape_result, arrow_result, ocr_result, assoc_result, region_ocr_results=region_results
)

# Now trace: what's in the diagram?
print(f"\nActivities ({len(diagram.activities)}):")
for act in diagram.activities:
    print(f"  {act.activity_id:10s} label={act.label:10s} dur={act.duration} shape_id={act.source_shape_id}")

# Build shape_to_activity
shape_to_activity = {}
for act in diagram.activities:
    if act.source_shape_id:
        shape_to_activity[act.source_shape_id] = act.activity_id

print(f"\nshape_to_activity ({len(shape_to_activity)}):")
for sid, aid in shape_to_activity.items():
    cand = None
    for c in shape_result.candidate_nodes:
        if c.source_shape_id == sid:
            cand = c
            break
    pos_str = f"({cand.position.x:.0f},{cand.position.y:.0f})" if cand else "?"
    print(f"  {sid:30s} -> {aid:10s} pos={pos_str}")

# Now trace dependency mapping
candidate_map = {c.node_id: c for c in shape_result.candidate_nodes}
candidates = list(candidate_map.values())
associator = GeometryAssociator(boundary_tolerance=20.0, ray_extension=40.0, direction_tolerance_deg=50.0, min_score=0.10)
assoc_results = associator.associate_arrows(arrow_result.arrows, candidates)

GOLD_POSITIONS = {
    "A": (21, 222), "B": (177, 162), "C": (177, 301), "D": (334, 222),
    "E": (490, 67), "F": (490, 170), "G": (490, 274), "H": (490, 377),
    "I": (651, 222), "J": (798, 222), "K": (945, 222), "L": (1092, 162),
    "M": (1092, 301), "N": (1228, 222), "O": (173, 502), "P": (320, 502),
    "Q": (467, 502), "R": (614, 502), "S": (761, 502), "T": (908, 502),
    "U": (1055, 502), "V": (1201, 501),
}
GOLD_DEPS = set([
    ("A", "D"), ("A", "E"), ("B", "F"), ("B", "G"), ("C", "G"), ("C", "H"),
    ("D", "I"), ("E", "I"), ("F", "J"), ("G", "J"), ("H", "J"),
    ("I", "K"), ("J", "K"), ("K", "L"), ("K", "M"),
    ("L", "N"), ("M", "N"), ("N", "O"), ("N", "P"),
    ("O", "Q"), ("P", "Q"), ("Q", "R"), ("R", "S"), ("R", "T"),
    ("S", "U"), ("T", "U"), ("U", "V"),
])

def match_gold(pos):
    px, py = pos
    best, best_d = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        d = ((px - gx)**2 + (py - gy)**2)**0.5
        if d < best_d:
            best_d = d
            best = gid
    return best, best_d

correct = 0
for i, (arrow, aresult) in enumerate(zip(arrow_result.arrows, assoc_results)):
    src_candidate_id = aresult.best_source.candidate_id if aresult.best_source else "none"
    tgt_candidate_id = aresult.best_target.candidate_id if aresult.best_target else "none"
    
    # Trace source mapping
    src_shape_id = None
    src_activity_id = None
    if src_candidate_id != "none":
        src_cand = candidate_map.get(src_candidate_id)
        if src_cand:
            src_shape_id = src_cand.source_shape_id
            if src_shape_id in shape_to_activity:
                src_activity_id = shape_to_activity[src_shape_id]
    
    # Trace target mapping
    tgt_shape_id = None
    tgt_activity_id = None
    if tgt_candidate_id != "none":
        tgt_cand = candidate_map.get(tgt_candidate_id)
        if tgt_cand:
            tgt_shape_id = tgt_cand.source_shape_id
            if tgt_shape_id in shape_to_activity:
                tgt_activity_id = shape_to_activity[tgt_shape_id]
    
    # Check gold
    is_ok = False
    if src_activity_id and tgt_activity_id:
        if (src_activity_id, tgt_activity_id) in GOLD_DEPS:
            is_ok = True
            correct += 1
    
    status = "OK" if is_ok else "EXTRA"
    if aresult.is_self_association:
        status = "SELF"
    if aresult.rejected:
        status = "REJECTED"
    
    # Gold position mapping
    src_gold = "?"
    tgt_gold = "?"
    if src_candidate_id != "none":
        src_cand = candidate_map.get(src_candidate_id)
        if src_cand:
            src_gold, _ = match_gold((src_cand.position.x, src_cand.position.y))
    if tgt_candidate_id != "none":
        tgt_cand = candidate_map.get(tgt_candidate_id)
        if tgt_cand:
            tgt_gold, _ = match_gold((tgt_cand.position.x, tgt_cand.position.y))
    
    print(f"  Arrow {i+1}: ({arrow.start.x:.0f},{arrow.start.y:.0f})->({arrow.end.x:.0f},{arrow.end.y:.0f})")
    print(f"    Source: gold={src_gold} cand_id={src_candidate_id[:20]:20s} shape={src_shape_id} act={src_activity_id}")
    print(f"    Target: gold={tgt_gold} cand_id={tgt_candidate_id[:20]:20s} shape={tgt_shape_id} act={tgt_activity_id}")
    print(f"    Status: {status}")

print(f"\nCorrect: {correct}/{len(arrow_result.arrows)}")

# Also print what dependencies the engine found
print(f"\n{'='*80}")
print(f"  DEPENDENCIES IN DIAGRAM ({diagram.dependency_count})")
print(f"{'='*80}")
for dep in diagram.dependencies:
    print(f"  {dep.source_activity_id} -> {dep.target_activity_id}  conf={dep.confidence:.3f}")
    is_gold = (dep.source_activity_id, dep.target_activity_id) in GOLD_DEPS
    print(f"    Gold: {'OK' if is_gold else 'WRONG'}")

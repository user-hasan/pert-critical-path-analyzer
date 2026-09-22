"""Debug geometry association."""
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.geometry_association import GeometryAssociator

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

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

image = cv2.imread(REF_AON)
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)
detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)
rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]

# Build position map
def match_shape_to_gold(candidate):
    px, py = candidate.position.x, candidate.position.y
    best_id, best_dist = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        dist = ((px - gx)**2 + (py - gy)**2)**0.5
        if dist < best_dist:
            best_dist = dist
            best_id = gid
    return best_id, best_dist

# Map node_id to gold_id
node_to_gold = {}
for c in rects:
    gold_id, dist = match_shape_to_gold(c)
    if dist < 80:
        node_to_gold[c.node_id] = gold_id

arrow_detector = ArrowDetector()
arrow_result = arrow_detector.detect_from_preprocessing(prep_result, shape_result)

associator = GeometryAssociator(
    boundary_tolerance=20.0,
    ray_extension=40.0,
    direction_tolerance_deg=50.0,
    min_score=0.10,
)

results = associator.associate_arrows(arrow_result.arrows, rects)

print(f"{'#':4s} {'Arrow':30s} {'SrcID':6s} {'SrcScore':9s} {'SrcBnd':8s} {'SrcDir':8s} {'TgtID':6s} {'TgtScore':9s} {'TgtBnd':8s} {'TgtDir':8s} {'Status':20s}")
print("-" * 130)

correct = 0
for i, (arrow, result) in enumerate(zip(arrow_result.arrows, results)):
    src_id = "none"
    src_score = 0
    src_bnd = 0
    src_dir = 0
    if result.best_source:
        src_cand = node_to_gold.get(result.best_source.candidate_id, "?")
        src_id = src_cand
        src_score = result.best_source.score
        src_bnd = result.best_source.boundary_intersection
        src_dir = result.best_source.direction_alignment

    tgt_id = "none"
    tgt_score = 0
    tgt_bnd = 0
    tgt_dir = 0
    if result.best_target:
        tgt_cand = node_to_gold.get(result.best_target.candidate_id, "?")
        tgt_id = tgt_cand
        tgt_score = result.best_target.score
        tgt_bnd = result.best_target.boundary_intersection
        tgt_dir = result.best_target.direction_alignment

    in_gold = (src_id, tgt_id) in GOLD_DEPS if src_id != "none" and tgt_id != "none" else False
    if in_gold:
        correct += 1
    status = "OK" if in_gold else "EXTRA"
    if result.is_self_association:
        status = "SELF"
    if result.rejected:
        status = f"REJECTED:{result.rejection_reason}"

    arrow_str = f"({arrow.start.x:.0f},{arrow.start.y:.0f})->({arrow.end.x:.0f},{arrow.end.y:.0f})"
    print(f"  {i+1:3d} {arrow_str:30s} {src_id:6s} {src_score:9.3f} {src_bnd:8.3f} {src_dir:8.3f} {tgt_id:6s} {tgt_score:9.3f} {tgt_bnd:8.3f} {tgt_dir:8.3f} {status:20s}")

print(f"\nCorrect: {correct}/{len(arrow_result.arrows)}")

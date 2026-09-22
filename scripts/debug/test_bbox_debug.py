"""Debug: check actual bounding boxes near arrow 1 start."""
import os, sys, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pert_analyzer.cv.preprocessing import ImagePreprocessor, PreprocessingConfig
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.geometry_association import GeometryAssociator, CandidateScore
from pert_analyzer.core.models import Point

REF_AON = os.path.join("tests", "test_data", "reference_diagrams", "reference_aon.png")

GOLD_POSITIONS = {
    "A": (21, 222), "B": (177, 162), "C": (177, 301), "D": (334, 222),
    "E": (490, 67), "F": (490, 170), "G": (490, 274), "H": (490, 377),
    "I": (651, 222), "J": (798, 222), "K": (945, 222), "L": (1092, 162),
    "M": (1092, 301), "N": (1228, 222), "O": (173, 502), "P": (320, 502),
    "Q": (467, 502), "R": (614, 502), "S": (761, 502), "T": (908, 502),
    "U": (1055, 502), "V": (1201, 501),
}

image = cv2.imread(REF_AON)
preprocessor = ImagePreprocessor(PreprocessingConfig())
prep_result = preprocessor.process_image(image)
detector = ShapeDetector()
shape_result = detector.detect_from_preprocessing(prep_result)
rects = [c for c in shape_result.candidate_nodes if c.shape_type.value in ("rectangle", "square")]

def match_gold(c):
    px, py = c.position.x, c.position.y
    best, best_d = None, float("inf")
    for gid, (gx, gy) in GOLD_POSITIONS.items():
        d = ((px - gx)**2 + (py - gy)**2)**0.5
        if d < best_d:
            best_d = d
            best = gid
    return best, best_d

# Check arrow 1 start (1095, 300)
start_point = Point(1095, 300)
print("Arrow 1 start: (1095, 300)")
print(f"\nCandidates near arrow 1 start:")
for c in rects:
    gold, dist = match_gold(c)
    bb = c.bounding_box
    cx, cy = c.position.x, c.position.y
    
    # Distance from arrow start to center
    d_center = start_point.distance_to(c.position)
    
    # Is point inside bounding box?
    inside = bb.contains_point(start_point, margin=10)
    
    # Distance to boundary
    dx = max(bb.x - start_point.x, 0, start_point.x - (bb.x + bb.width))
    dy = max(bb.y - start_point.y, 0, start_point.y - (bb.y + bb.height))
    d_boundary = (dx**2 + dy**2)**0.5
    
    if d_center < 200 or inside:
        print(f"  {gold:3s} ({c.node_id[:20]:20s}) center=({cx:.0f},{cy:.0f}) "
              f"bb=({bb.x:.0f},{bb.y:.0f},{bb.width:.0f},{bb.height:.0f}) "
              f"d_center={d_center:.1f} d_boundary={d_boundary:.1f} inside={inside}")

# Now manually run scoring for arrow 1
from pert_analyzer.cv.geometry_association import GeometryAssociator
from pert_analyzer.cv.models import DetectedArrow, ArrowheadType, LineStyle
from pert_analyzer.core.models import BoundingBox
import math

associator = GeometryAssociator(boundary_tolerance=20.0, ray_extension=40.0, direction_tolerance_deg=50.0, min_score=0.10)

arrow = DetectedArrow(
    start=Point(1095, 300),
    end=Point(1341, 300),
    direction_vector=(246, 0),
    angle_deg=0,
    length=246,
    confidence=0.25,
)

print(f"\n\nManual scoring for arrow 1 (1095,300)->(1341,300):")
for c in rects:
    gold, dist = match_gold(c)
    if dist > 200:
        continue
    
    score = associator._score_single_candidate(
        c, Point(1095, 300), (1.0, 0.0), False, arrow
    )
    
    if score.score > 0.1:
        ev = score.evidence
        print(f"  {gold:3s} score={score.score:.3f} "
              f"bnd={score.boundary_intersection:.3f} "
              f"epd={score.endpoint_distance:.3f} "
              f"dir={score.direction_alignment:.3f} "
              f"ang={score.angle_consistency:.3f} "
              f"d_center={ev['dist_to_center']:.1f} "
              f"cp={ev['center_proximity']:.3f} "
              f"ib={ev['inside_bonus']:.3f}")

# PERT & Critical Path Analyzer — Architecture Document

**Version:** 1.0  
**Phase:** 8 — End-to-End Validation & Human-Review Foundation  
**Supervisor:** Dr. Adel Al-Afeery  

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                          │
│  ┌─────────┐  ┌──────────┐  ┌────────────┐  ┌──────────────┐  │
│  │ Dashboard│  │ Analysis │  │  Review    │  │ Visualization│  │
│  │  View    │  │ Workspace│  │  Panel     │  │    Panel     │  │
│  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └──────┬───────┘  │
│       └──────────────┴──────────────┴────────────────┘          │
│                            │                                     │
│                     PySide6 Qt Signals/Slots                     │
├─────────────────────────────┼───────────────────────────────────┤
│                   APPLICATION LAYER                              │
│  ┌──────────────────────────┴────────────────────────────────┐  │
│  │              AnalysisOrchestrator                          │  │
│  │  (Coordinates pipeline stages, manages state, callbacks)  │  │
│  └──────────────────────────┬────────────────────────────────┘  │
├─────────────────────────────┼───────────────────────────────────┤
│                     DOMAIN LAYER                                 │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │   CPM    │  │   PERT   │  │  Graph   │  │  Validation  │   │
│  │  Engine  │  │  Engine  │  │ Builder  │  │    Engine    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                 COMPUTER VISION LAYER                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Image   │  │  Shape   │  │  Arrow   │  │     OCR      │   │
│  │Preprocess│  │ Detection│  │Detection │  │   Engine     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                INFRASTRUCTURE LAYER                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │  Config  │  │ Logging  │  │  SQLite  │  │   Assets     │   │
│  │ Manager  │  │ Manager  │  │  Store   │  │   Manager    │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Layered Architecture

The application follows a strict four-layer architecture:

| Layer | Responsibility | Dependencies |
|-------|---------------|--------------|
| **Presentation** | GUI, visualization, user interaction | Application Layer |
| **Application** | Orchestration, workflow, state management | Domain Layer |
| **Domain** | Business logic, algorithms, graph theory | Core Models (no external deps) |
| **Infrastructure** | CV, OCR, persistence, config | Core Models |

**Dependency Rules:**
- Upper layers depend on lower layers only
- Lower layers NEVER depend on upper layers
- Domain layer has ZERO external framework dependencies
- Infrastructure adapters implement Domain interfaces

---

## 3. Module Responsibilities

### 3.1 Core Models (`pert_analyzer.core.models`)
- Defines all data structures as typed Python dataclasses
- Pure data containers with no business logic
- Used by ALL layers

### 3.2 Core Interfaces (`pert_analyzer.core.interfaces`)
- Abstract base classes defining contracts
- OCR engine interface
- Diagram detector interface
- Graph builder interface
- Analysis engine interface
- Visualization interface

### 3.3 CV Pipeline (`pert_analyzer.cv`)
- **Preprocessor**: Grayscale, threshold, denoise, deskew
- **ShapeDetector**: Circles, rectangles, polygons
- **ArrowDetector**: Lines, arrowheads, connections
- **DiagramClassifier**: AON vs AOA detection
- **SpatialAssociate**: Maps OCR text to visual elements

### 3.4 OCR Engine (`pert_analyzer.ocr`)
- Abstract OCR interface (supports Tesseract, EasyOCR, PaddleOCR)
- Text extraction with bounding boxes
- Confidence scoring
- Language detection (English + Arabic)

### 3.5 Analysis Engine (`pert_analyzer.analysis`)
- **CPMEngine**: Critical Path Method calculations
- **PERTEngine**: PERT calculations (Phase 2+)
- **ProjectAnalyzer**: Orchestrates all analysis

### 3.6 Graph Module (`pert_analyzer.graph`)
- **GraphBuilder**: Constructs NetworkX graph from detected elements
- **GraphValidator**: Checks structural validity
- **GraphVisualizer**: Renders network diagrams

### 3.7 Validation (`pert_analyzer.validation`)
- Structural validation
- Data validation
- Confidence assessment
- Issue reporting

### 3.8 GUI (`pert_analyzer.gui`)
- PySide6 application
- Main window with tabbed interface
- Each major view is a separate widget
- Signal/slot communication with application layer

### 3.9 Persistence (`pert_analyzer.persistence`)
- SQLite database for project storage
- Session management
- History tracking

---

## 4. Data Models

### 4.1 Geometry Models
```python
@dataclass
class Point:
    x: float
    y: float

@dataclass
class BoundingBox:
    x: float
    y: float
    width: float
    height: float
    center: Point  # computed property

@dataclass
class Arrow:
    start: Point
    end: Point
    has_arrowhead: bool
    bounding_box: BoundingBox
    confidence: float
    metadata: dict
```

### 4.2 Visual Element Models
```python
@dataclass
class DetectedShape:
    shape_id: str
    shape_type: str  # "circle", "rectangle", "polygon"
    contour: np.ndarray
    bounding_box: BoundingBox
    centroid: Point
    confidence: float
    color: tuple
    metadata: dict

@dataclass
class OCRResult:
    text: str
    bounding_box: BoundingBox
    confidence: float
    language: str
    raw_data: dict
```

### 4.3 Semantic Models
```python
@dataclass
class Node:
    node_id: str
    label: str
    position: Point
    bounding_box: BoundingBox
    node_type: str  # "event", "activity"
    confidence: float
    metadata: dict

@dataclass
class Activity:
    activity_id: str
    name: str
    duration: float
    optimistic_time: Optional[float] = None
    most_likely_time: Optional[float] = None
    pessimistic_time: Optional[float] = None
    source_node: Optional[str] = None
    target_node: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    confidence: float = 0.0
    is_dummy: bool = False
    metadata: dict = field(default_factory=dict)

@dataclass
class Dependency:
    source: str
    target: str
    activity_id: Optional[str] = None
    dependency_type: str = "finish_to_start"
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
```

### 4.4 Graph Model
```python
@dataclass
class GraphModel:
    diagram_type: str  # "AON" or "AOA"
    nodes: Dict[str, Node]
    activities: Dict[str, Activity]
    dependencies: List[Dependency]
    source_node_id: Optional[str] = None
    sink_node_id: Optional[str] = None
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
```

### 4.5 Analysis Models
```python
@dataclass
class ActivityAnalysis:
    activity_id: str
    early_start: float
    early_finish: float
    late_start: float
    late_finish: float
    total_float: float
    free_float: float
    is_critical: bool

@dataclass
class AnalysisResult:
    project_duration: float
    critical_path: List[str]
    critical_paths: List[List[str]]  # multiple critical paths
    activity_analyses: Dict[str, ActivityAnalysis]
    project_variance: Optional[float] = None
    project_std_dev: Optional[float] = None
    analysis_type: str = "CPM"  # "CPM" or "PERT"
    metadata: dict = field(default_factory=dict)
```

### 4.6 Validation Models
```python
@dataclass
class ValidationIssue:
    issue_id: str
    severity: str  # "error", "warning", "info"
    category: str  # "structural", "data", "confidence"
    message: str
    element_id: Optional[str] = None
    suggestion: Optional[str] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class ValidationResult:
    is_valid: bool
    issues: List[ValidationIssue]
    confidence_score: float
    summary: str
```

### 4.7 Diagram Models
```python
@dataclass
class DiagramAnalysis:
    diagram_type: str  # "AON", "AOA", "unknown"
    detection_confidence: float
    detected_shapes: List[DetectedShape]
    detected_arrows: List[Arrow]
    ocr_results: List[OCRResult]
    preprocessed_image: Optional[np.ndarray] = None
    original_image: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)

@dataclass
class Project:
    project_id: str
    name: str
    description: str
    diagram: DiagramAnalysis
    graph: Optional[GraphModel] = None
    analysis: Optional[AnalysisResult] = None
    validation: Optional[ValidationResult] = None
    created_at: str = ""
    modified_at: str = ""
    status: str = "draft"
    metadata: dict = field(default_factory=dict)
```

---

## 5. Interface Definitions

### 5.1 OCR Engine Interface
```python
class OCREngine(ABC):
    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[OCRResult]: ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]: ...

    @abstractmethod
    def get_confidence_threshold(self) -> float: ...
```

### 5.2 Shape Detector Interface
```python
class ShapeDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedShape]: ...

    @abstractmethod
    def get_supported_shapes(self) -> List[str]: ...
```

### 5.3 Arrow Detector Interface
```python
class ArrowDetector(ABC):
    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Arrow]: ...

    @abstractmethod
    def detect_connections(
        self, shapes: List[DetectedShape], arrows: List[Arrow]
    ) -> List[Dependency]: ...
```

### 5.4 Diagram Classifier Interface
```python
class DiagramClassifier(ABC):
    @abstractmethod
    def classify(
        self, shapes: List[DetectedShape], arrows: List[Arrow]
    ) -> Tuple[str, float]: ...

    @abstractmethod
    def get_supported_types(self) -> List[str]: ...
```

### 5.5 Graph Builder Interface
```python
class GraphBuilder(ABC):
    @abstractmethod
    def build(
        self, diagram: DiagramAnalysis
    ) -> GraphModel: ...

    @abstractmethod
    def validate(self, graph: GraphModel) -> ValidationResult: ...
```

### 5.6 Analysis Engine Interface
```python
class AnalysisEngine(ABC):
    @abstractmethod
    def analyze(self, graph: GraphModel) -> AnalysisResult: ...

    @abstractmethod
    def get_analysis_type(self) -> str: ...
```

### 5.7 Visualizer Interface
```python
class NetworkVisualizer(ABC):
    @abstractmethod
    def visualize(
        self, graph: GraphModel, analysis: Optional[AnalysisResult] = None
    ) -> np.ndarray: ...

    @abstractmethod
    def visualize_interactive(
        self, graph: GraphModel, analysis: Optional[AnalysisResult] = None
    ) -> None: ...
```

---

## 6. Image Processing Pipeline

```
Raw Image (BGR)
    │
    ▼
┌─────────────────┐
│  Color Space     │  Convert to grayscale, HSV as needed
│  Conversion      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Noise           │  Gaussian blur, median filter
│  Reduction       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Contrast        │  CLAHE, histogram equalization
│  Enhancement     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Binarization    │  Adaptive/Otsu thresholding
│                  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Deskew          │  Correct rotation/perspective
│  Correction      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Morphological   │  Close gaps, remove noise
│  Operations      │
└────────┬────────┘
         │
         ▼
Preprocessed Image (Binary/Grayscale)
```

---

## 6.1 Image Preprocessing Implementation (Phase 3)

### Module Structure

```
pert_analyzer/cv/
├── __init__.py          # Public API exports
├── exceptions.py        # CV-specific exception hierarchy
├── models.py            # PreprocessingConfig, PreprocessingResult, CoordinateMapping
└── preprocessing.py     # ImagePreprocessor pipeline
```

### Preprocessing Pipeline

The `ImagePreprocessor` class processes raw images through a configurable pipeline:

```
Raw Image (BGR or Grayscale)
    │
    ▼
┌─────────────────┐
│  Validation      │  Check dimensions, format, bounds
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Grayscale       │  Convert BGR→Grayscale if needed
│  Conversion      │  (preserves grayscale input)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Resize          │  Scale to max dimensions
│  (aspect-ratio   │  or by scale factor
│   preserving)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Noise           │  Gaussian, Median, or
│  Reduction       │  Bilateral filtering
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Contrast        │  CLAHE or Histogram
│  Enhancement     │  Equalization
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Thresholding    │  Global, Otsu, or Adaptive
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Edge            │  Canny or Sobel
│  Detection       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Skew            │  Estimate rotation angle
│  Estimation      │  (Hough line analysis)
└────────┬────────┘
         │
         ▼
Multiple Image Representations + CoordinateMapping
```

### Available Representations

The preprocessing result provides multiple image representations for downstream tasks:

| Representation | Use Case | Description |
|---------------|----------|-------------|
| `original` | Reference, OCR | Unmodified input image |
| `normalized` | General processing | Color-normalized, resized |
| `grayscale` | Shape detection | Single-channel grayscale |
| `denoised` | Pre-detection | Noise-reduced grayscale |
| `contrast_enhanced` | Feature extraction | CLAHE/equalized contrast |
| `binary` | Shape detection | Global/Otsu thresholded |
| `adaptive_binary` | Varying lighting | Locally adaptive threshold |
| `edges` | Line/arrow detection | Canny/Sobel edges |

### Coordinate Transformation

The `CoordinateMapping` class maintains a mapping between original and processed image coordinates:

```python
@dataclass
class CoordinateMapping:
    original_width: int      # Input image width
    original_height: int     # Input image height
    processed_width: int     # Output image width
    processed_height: int    # Output image height
    scale_x: float           # X scale factor
    scale_y: float           # Y scale factor
    offset_x: float          # X translation offset
    offset_y: float          # Y translation offset
    rotation_angle: float    # Applied deskew angle

    def to_original(x, y) -> (x, y)     # Processed → Original
    def to_processed(x, y) -> (x, y)    # Original → Processed
    def scale_bbox(x, y, w, h) -> (x, y, w, h)  # BBox conversion
```

This is essential for converting detected shapes/arrows/OCR back to original image coordinates.

### Configuration

All preprocessing parameters are configurable via `PreprocessingConfig`:

```python
config = PreprocessingConfig(
    # Resize
    resize_strategy=ResizeStrategy.MAX_DIMENSION,
    max_width=2000,
    max_height=2000,

    # Noise reduction
    blur_method=BlurMethod.GAUSSIAN,
    blur_kernel_size=5,

    # Contrast
    contrast_method=ContrastMethod.CLAHE,
    clahe_clip_limit=2.0,

    # Threshold
    threshold_method=ThresholdMethod.ADAPTIVE_GAUSSIAN,
    adaptive_block_size=11,

    # Edges
    edge_method=EdgeMethod.CANNY,
    canny_low_threshold=50,
    canny_high_threshold=150,

    # Deskew
    enable_deskew=True,
    deskew_max_angle=15.0,
)
```

### Error Handling

Domain-specific exceptions under `pert_analyzer.cv.exceptions`:

| Exception | Use Case |
|-----------|----------|
| `CVError` | Base exception for all CV errors |
| `ImageLoadError` | File not found, unreadable, corrupted |
| `ImageValidationError` | Zero dimensions, None input |
| `ImageSizeError` | Too small or too large |
| `ImageFormatError` | Unsupported file format |
| `PreprocessingError` | Pipeline operation failure |
| `InvalidConfigError` | Invalid configuration values |

### Design Principles

1. **Multiple representations**: Different downstream tasks get their optimal input
2. **Coordinate preservation**: All transformations are reversible
3. **Aspect ratio preservation**: Diagrams are never distorted
4. **Original preservation**: Original image is always available
5. **Configurable pipeline**: Every step can be tuned or disabled
6. **Fail-safe**: Invalid configs raise early, unsupported methods log warnings
7. **No detection logic**: Only preprocessing — no contour/shape/OCR detection

---

## 6.2 Shape Detection & Classification Implementation (Phase 4)

### Module Structure

```
pert_analyzer/cv/
├── __init__.py            # Public API exports (all CV components)
├── exceptions.py          # CV-specific exception hierarchy
├── models.py              # ShapeType, ShapeDetectionConfig, CandidateNode, etc.
├── preprocessing.py       # ImagePreprocessor pipeline (Phase 3)
├── shape_detection.py     # ShapeDetector — contour-based shape detection
└── classification.py      # DiagramClassifier — AON/AOA/UNKNOWN classification
```

### Shape Detection Pipeline

The `ShapeDetector` class detects geometric shapes (rectangles, circles, polygons) from preprocessed images:

```
Preprocessed Binary Image
    │
    ▼
┌──────────────────────────┐
│ Contour Detection         │  cv2.findContours on thresholded image
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Border Filtering          │  Remove contours touching image borders
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Shape Classification      │  Rectangle / Circle / Polygon / Unknown
│ • Bounding rect aspect    │  via approxPolyDP + circularity
│ • Circularity score       │
│ • Polygon vertex count    │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Confidence Scoring        │  0.0–1.0 based on shape quality
│ • Rectangularity          │  (how well contour fits shape)
│ • Circularity             │
│ • Area consistency        │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Duplicate Suppression     │  IoU-based overlap removal
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ CandidateNode Generation  │  Rectangles → "activity" role
│                           │  Circles → "event" role
└───────────┬──────────────┘
            │
            ▼
    ShapeDetectionResult
    (shapes, candidates, coordinate_mapping)
```

### Diagram Classification

The `DiagramClassifier` class determines diagram type from shape evidence:

| Evidence Factor | AON Score | AOA Score |
|----------------|-----------|-----------|
| Rectangle count | +rect_ratio × 0.4 | — |
| Circle count | — | +circle_ratio × 0.4 |
| ≥3 rectangles | +0.3 | — |
| ≥3 circles | — | +0.3 |
| Circle > rectangle | -0.2 | — |
| Rectangle > circle | — | -0.2 |
| Size consistency | +0.15 | +0.15 |

Classification result:
- **AON**: Rectangles dominant, confidence > 0.3
- **AOA**: Circles dominant, confidence > 0.3
- **UNKNOWN**: Below threshold or insufficient shapes (< 2)

### Key Data Models

| Model | Purpose |
|-------|---------|
| `ShapeType` | Enum: RECTANGLE, SQUARE, CIRCLE, ELLIPSE, POLYGON, UNKNOWN |
| `ShapeDetectionConfig` | Tunable thresholds (min area, circularity, aspect ratio) |
| `CandidateNode` | Interpreted shape as potential diagram node with role |
| `ShapeDetectionResult` | All detected shapes + candidates + coordinate mapping |
| `DiagramClassificationResult` | Diagram type, confidence, evidence dict, warnings |

### Integration with Preprocessing

```python
# Full pipeline: preprocessing → detection → classification
preprocessor = ImagePreprocessor()
detector = ShapeDetector()
classifier = DiagramClassifier()

prep_result = preprocessor.process_image(raw_image)
det_result = detector.detect_from_preprocessing(prep_result)
classification = classifier.classify_from_detection(det_result)
# classification.diagram_type → "AON" | "AOA" | "UNKNOWN"
# classification.confidence → 0.0–1.0
```

### Coordinate Mapping

When preprocessing resizes the image, `ShapeDetectionResult.coordinate_mapping` maps detection coordinates back to original image space:

```python
det_result = detector.detect_from_preprocessing(prep_result)
# Coordinates in det_result are in original image space
# via coordinate_mapping.to_original_coordinates(x, y)
```

### Design Principles

1. **Contour-based detection**: Uses OpenCV contour analysis — no ML required
2. **Probabilistic classification**: Returns confidence scores, not binary decisions
3. **Evidence-based**: AON/AOA classification based on shape distribution evidence
4. **Coordinate preservation**: Maps all results back to original image coordinates
5. **Fail-safe**: Invalid configs raise early, insufficient shapes return UNKNOWN
6. **Extensible**: New shape types can be added to ShapeType enum

---

## 6.3 Arrow & Connection Detection Implementation (Phase 5)

### Module Structure

```
pert_analyzer/cv/
├── __init__.py            # Public API exports (includes ArrowDetector)
├── exceptions.py          # ArrowDetectionError
├── models.py              # ArrowDetectionConfig, DetectedLineSegment, DetectedArrow, etc.
├── preprocessing.py       # ImagePreprocessor pipeline (Phase 3)
├── shape_detection.py     # ShapeDetector — contour-based shape detection (Phase 4)
├── classification.py      # DiagramClassifier — AON/AOA/UNKNOWN classification (Phase 4)
└── arrow_detection.py     # ArrowDetector — line/arrowhead/direction detection
```

### Arrow Detection Pipeline

The `ArrowDetector` class detects lines, arrowheads, and directionality from preprocessed images. It produces structured evidence (`DetectedArrow`), NOT semantic dependencies.

```
Preprocessed Binary Image + ShapeDetectionResult (CandidateNodes)
    │
    ▼
┌──────────────────────────┐
│ Line Detection            │  cv2.HoughLinesP
│ • Probabilistic Hough     │  (rho=1, theta=π/180, threshold=50)
│   Lines Transform         │
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Border Filtering          │  Remove lines touching image edges
│ • Border margin check     │  (within 5px of image boundary)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Shape Border Filtering    │  Remove lines along shape contours
│ • Contour proximity test  │  (within 10px of shape borders)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Arrowhead Detection       │  For each line endpoint:
│ • Contour near endpoint   │  1. Extract ROI around endpoint
│ • Triangularity check     │  2. Find contours in ROI
│ • Apex scoring            │  3. Compute triangularity
│ • Contour area ratio      │  4. Score apex sharpness
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Direction Inference       │  From source→target endpoint
│ • dx/dy dominant axis     │  Direction enum: RIGHT, LEFT,
│ • Angle calculation       │  UP, DOWN, DIAGONAL
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Segment Merging           │  Collinear segment combination
│ • Angle alignment check   │  (within 10°)
│ • Distance proximity      │  (within 15px)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Node Association          │  Map arrows to CandidateNodes
│ • Bounding box overlap    │  source_node, target_node
│ • Centroid containment    │  (if overlap found)
└───────────┬──────────────┘
            │
            ▼
┌──────────────────────────┐
│ Confidence Scoring        │  0.0–1.0 per arrow
│ • has_arrowhead (+0.3)    │  Combines line quality,
│ • line_length (>200:+0.2) │  arrowhead evidence,
│ • node_association (+0.2) │  and node association
│ • straightness (+0.2)     │
└───────────┬──────────────┘
            │
            ▼
    ArrowDetectionResult
    (arrows, lines, total_lines, total_arrows, coordinate_mapping)
```

### Key Data Models

| Model | Purpose |
|-------|---------|
| `LineStyle` | Enum: SOLID, DASHED, DOTTED, UNKNOWN |
| `ArrowheadType` | Enum: TRIANGLE, V_SHAPE, FILLED, OPEN, NONE, UNKNOWN |
| `ArrowDetectionConfig` | Tunable thresholds (Hough params, merge angle, arrowhead radius) |
| `DetectedLineSegment` | Raw line from HoughLinesP with optional arrowhead evidence |
| `ArrowheadEvidence` | Contour analysis results (type, triangularity, apex_score, area_ratio) |
| `DetectedArrow` | Final arrow with direction, source/target nodes, confidence |
| `ArrowDetectionResult` | All detected lines + arrows + coordinate mapping |

### Arrowhead Detection Algorithm

1. **ROI Extraction**: For each line endpoint, extract a square ROI (size = `arrowhead_radius`)
2. **Contour Analysis**: Find contours within the ROI using `cv2.findContours`
3. **Triangularity Scoring**: Based on vertex count from `cv2.approxPolyDP`:
   - 3 vertices → 0.9 (triangle)
   - 4 vertices → 0.6 (quadrilateral)
   - ≤5 vertices → 0.4 (polygon)
   - >5 vertices → 0.2 (complex shape)
4. **Apex Scoring**: The vertex farthest from the line's endpoint, normalized by ROI size
5. **Area Ratio**: Contour area relative to ROI area (larger = more likely arrowhead)
6. **Classification**: `TRIANGLE` if triangularity > 0.7, `V_SHAPE` if apex > 0.6, else `UNKNOWN`

### Confidence Scoring

```python
confidence = 0.0
if has_arrowhead:
    confidence += 0.3          # Arrowhead evidence present
if line_length > 200:
    confidence += 0.2          # Long line (not noise)
if line_length > 100:
    confidence += 0.1          # Medium line
if source_node and target_node:
    confidence += 0.2          # Both endpoints associated with shapes
elif source_node or target_node:
    confidence += 0.1          # One endpoint associated
if straightness > 0.9:
    confidence += 0.2          # Very straight line
elif straightness > 0.7:
    confidence += 0.1          # Reasonably straight
```

### Coordinate Mapping

When preprocessing resizes the image, `ArrowDetectionResult.coordinate_mapping` maps detection coordinates back to original image space:

```python
arrow_result = arrow_detector.detect_connections(prep_result, shape_result)
# Coordinates in arrow_result are in original image space
# via coordinate_mapping.to_original_coordinates(x, y)
```

### Integration with Shape Detection

```python
# Full pipeline: preprocessing → shape detection → arrow detection
preprocessor = ImagePreprocessor()
shape_detector = ShapeDetector()
arrow_detector = ArrowDetector()

prep_result = preprocessor.process_image(raw_image)
shape_result = shape_detector.detect(prep_result)
arrow_result = arrow_detector.detect_connections(prep_result, shape_result)
# shape_result.candidates → CandidateNodes (potential diagram nodes)
# arrow_result.arrows → DetectedArrows (visual connections, NOT semantic dependencies)
```

### Design Principles

1. **Evidence, not semantics**: Produces `DetectedArrow` with visual properties, NOT `Dependency` objects
2. **Two-stage filtering**: Border lines first, then shape-contour lines
3. **Endpoint analysis**: Arrowhead detection at both line endpoints
4. **Triangularity heuristic**: Vertex-count-based scoring for arrowhead shape
5. **Coordinate preservation**: Maps all results back to original image coordinates
6. **Fail-safe**: Empty images return empty results, no exceptions

### Known Limitations

1. **HoughLinesP sensitivity**: May detect spurious lines from anti-aliasing artifacts
2. **Arrowhead detection**: Filled arrowheads merged with line body are hard to isolate
3. **Dashed lines**: Detected but not split into individual segments
4. **Parallel arrows**: Multiple arrows between same nodes may be merged
5. **Performance**: Contour analysis on large images can be slow (~10s for 1000×1000)

---

## 7. OCR & Text Association Implementation (Phase 6)

### Module Structure

```
pert_analyzer/cv/
├── __init__.py            # Public API exports (includes all OCR components)
├── exceptions.py          # OCREngineError, TextAssociationError
├── models.py              # Preprocessing, Shape, Arrow models (Phases 3-5)
├── ocr_models.py          # OCRTextRegion, NumericCandidate, TextAssociation, etc.
├── text_normalization.py  # TextNormalizer — whitespace, Unicode, OCR artifacts
├── numeric_extraction.py  # NumericExtractor — integers, decimals, Arabic-Indic
├── text_classification.py # TextClassifier — ACTIVITY_ID, LABEL, NUMERIC, etc.
├── ocr_engine.py          # OCREngineBase, MockOCREngine, TesseractOCREngine
├── spatial_association.py # SpatialAssociator — text-to-shape, text-to-arrow
├── text_grouping.py       # TextGrouper — spatial text clustering
├── ocr_debug.py           # OCRDebugRenderer — bounding boxes, associations
├── preprocessing.py       # ImagePreprocessor (Phase 3)
├── shape_detection.py     # ShapeDetector (Phase 4)
├── classification.py      # DiagramClassifier (Phase 4)
└── arrow_detection.py     # ArrowDetector (Phase 5)
```

### OCR Pipeline Architecture

```
ImagePreprocessor → ShapeDetector → ArrowDetector → OCR → SpatialAssociation
       ↓                ↓                ↓           ↓         ↓
PreprocessingResult  ShapeDetection   ArrowResult  OCRResult  AssocResult
                      Result
```

### OCR Engine Abstraction

```
OCREngineBase (ABC)
    ├── MockOCREngine         — For testing, no external dependencies
    └── TesseractOCREngine    — pytesseract adapter (optional dependency)

create_ocr_engine("tesseract") → TesseractOCREngine
create_ocr_engine("mock")      → MockOCREngine
```

The engine is replaceable — swap implementations without changing callers.
Installation requirements are documented and optional.

### OCR Result Model

```
OCRProcessingResult
├── regions: List[OCRTextRegion]
│   ├── text, raw_text, normalized_text
│   ├── bounding_box, center, confidence
│   ├── text_type (NUMERIC, ACTIVITY_ID, LABEL, UNKNOWN)
│   ├── is_numeric, parsed_value
│   └── association (target_id, score, evidence)
├── groups: List[TextGroup]
│   ├── member_region_ids
│   ├── combined_text, combined_bounding_box
│   └── group_confidence
├── numeric_candidates: List[NumericCandidate]
│   ├── value, raw_text, bounding_box
│   └── is_integer, is_negative, decimal_places
└── association_results: List[TextAssociationResult]
    ├── associations (target_id, score, reasons)
    ├── best_association
    └── is_ambiguous
```

### Text Normalization

Conservative, configurable normalization that preserves raw text:

| Step | Description | Default |
|------|-------------|---------|
| Unicode normalization | NFKC form | Enabled |
| Arabic-Indic digits | ٠١٢٣٤٥٦٧٨٩ → 0123456789 | Enabled |
| Extended Arabic-Indic | ۰۱۲۳۴۵۶۷۸۹ → 0123456789 | Enabled |
| Whitespace collapse | Multiple spaces → single space | Enabled |
| Zero-width removal | \u200b, \u200c, \u200d removed | Enabled |
| Character confusion | O↔0, l↔1 (context-dependent) | Disabled |

### Numeric Extraction

Supports:
- Integers: `5`, `10`, `-3`
- Decimals: `3.14`, `7.25`
- Thousands separators: `1,000`
- Arabic-Indic numerals: `١٢٣` → `123`
- PERT-like pairs: `3/6`, `5-10`

Returns structured `NumericCandidate` with bounding box and parse warnings.

### Text Type Classification

| Type | Pattern | Example |
|------|---------|---------|
| `ACTIVITY_ID_CANDIDATE` | Letter+digit patterns | A1, B12, AB3 |
| `TEXT_LABEL_CANDIDATE` | 2+ letter characters | Requirements, Design |
| `NUMERIC_CANDIDATE` | Numeric patterns | 5, 3.14, -7 |
| `UNKNOWN` | No pattern match | (anything else) |

Classification is evidence-based — not every region is forced into a type.

### Spatial Association Strategy

#### AON Association (text inside shapes)

```
┌─────────────────────────┐
│  "A1"                   │  Text "A1" center inside rectangle
│  "5"                    │  → strong containment evidence
└─────────────────────────┘
```

Features: containment, overlap ratio, center distance, relative position.

#### AOA Association (text near arrows)

```
○1 ──── "A", "5" ────→ ○2
```

Features: perpendicular distance, projection onto segment, midpoint proximity.

#### Association Scoring

| Feature | Weight | Description |
|---------|--------|-------------|
| Containment | 0.4 | Text center inside shape |
| Distance | 0.3 | Center-to-center proximity |
| Overlap | 0.2 | Bounding box overlap ratio |
| Position | 0.1 | Centeredness within shape |

Ambiguity detected when top-2 scores differ by < 0.15.

### Text Grouping

Spatially close text regions are grouped into logical clusters:

```python
"A" at (150, 110) + "5" at (150, 140) → TextGroup("A 5")
```

Grouping uses horizontal/vertical proximity thresholds.

### Confidence Handling

Separate confidence scores for each layer:
- **OCR confidence**: Text recognition quality (0.0–1.0)
- **Shape confidence**: Geometric detection quality
- **Arrow confidence**: Line/arrowhead detection quality
- **Association confidence**: Spatial matching quality

These are NOT combined into one unexplained number.

### Offline / Missing Engine Behavior

```python
from pert_analyzer.cv import MockOCREngine, create_ocr_engine

# Import succeeds even without Tesseract installed
engine = create_ocr_engine("mock")  # Always works
engine = create_ocr_engine("tesseract")  # Raises OCREngineError if unavailable
```

### OCR Installation Requirements

```bash
# Tesseract (optional)
pip install pytesseract
# Windows: install from https://github.com/UB-Mannheim/tesseract/wiki
```

### Debug Visualization

`OCRDebugRenderer` draws:
- Bounding boxes colored by text type
- Text labels and confidence scores
- Association lines to shapes/arrows
- Ambiguity indicators (double border)
- Text group outlines

### Known Limitations

1. **Tesseract accuracy**: Diagram text with unusual fonts may produce low confidence
2. **Arabic OCR**: Requires Arabic language pack installation
3. **Filled arrowheads**: Hard to isolate from line body (Phase 5 limitation)
4. **Performance**: OCR can be slow on large images; configurable representation selection
5. **Multi-line text**: Regions are individual words; grouping approximates logical blocks

---

## 7.1 Semantic Diagram Reconstruction Implementation (Phase 7)

### Module Structure

```
pert_analyzer/cv/
├── __init__.py                # Public API exports (includes ReconstructionEngine)
├── exceptions.py              # ReconstructionError added
├── reconstruction_models.py   # EvidenceTrace, ReconstructedActivity, ReconstructedEvent,
│                              # ReconstructedDependency, AmbiguityIssue, ValidationResult,
│                              # ReconstructedDiagram
└── reconstruction.py          # ReconstructionEngine — bridges CV evidence to GraphModel
```

### Architecture Role

```
Shape Detection (Phase 4)  ──┐
Arrow Detection (Phase 5)  ──┼──→  ReconstructionEngine  ──→  GraphModel  ──→  CPMEngine
OCR / Text Association     ──┘    (Phase 7)                  (output)        (next phase)
```

The reconstruction engine is the **bridge** between low-level CV detection and high-level graph modeling. It:
- **Consumes** detection results from Phases 4, 5, 6 (never calls CV operations)
- **Aggregates** evidence from multiple sources with confidence scores
- **Produces** semantic candidates with ambiguity tracking
- **Converts** to `GraphModel` for CPM analysis

### AON Reconstruction Strategy

```
Rectangles/Squares  →  ReconstructedActivity (activity_id, label, duration)
Circles/Ellipses    →  ReconstructedEvent (event_id, label)
Arrows              →  ReconstructedDependency (source_id, target_id)
Text inside shapes  →  Activity ID, label, duration extraction
```

### AOA Reconstruction Strategy

```
Circles/Ellipses    →  ReconstructedEvent (event_id, label)
Arrows              →  ReconstructedActivity (activity_id, label, duration)
Text near arrows    →  Activity ID, label, duration extraction
```

### Evidence Trace Model

Every reconstructed element carries provenance:

```python
EvidenceTrace(
    source_phase="shape_detection",     # Which phase produced this
    source_ids=["shape_1"],             # Specific source IDs
    confidence_contribution=0.85,       # Confidence from this source
    description="Rectangle shape detected",
)
```

### Confidence Aggregation

Multiple evidence sources are combined using weighted average:

```python
# Weighted: max value gets 40% weight, rest distributed
overall = 0.4 * max_confidence + 0.6 * avg(other_confidences)
```

### Validation

The engine validates reconstructed diagrams for:
- Missing durations on non-dummy activities
- Isolated nodes (no dependencies)
- Duplicate activity/event IDs
- Empty diagrams
- Low confidence elements

### Ambiguity Detection

Tracks uncertainties:
- `TEXT_NO_MATCH`: Text region has no matching shape
- `ARROW_NO_SOURCE/TARGET`: Arrow endpoints unresolvable
- `SHAPE_NO_TEXT`: Shape has no associated text
- `DUPLICATE_LABEL`: Multiple elements with same ID
- `MISSING_DURATION`: Activity without duration
- `ISOLATED_NODE`: Node with no dependencies

### GraphModel Conversion

`ReconstructedDiagram.to_graph_model()` converts to the standard `GraphModel`:

```python
diagram = engine.reconstruct_aon(shape_result, arrow_result, ocr_result, assoc_result)
graph_model = diagram.to_graph_model()
# graph_model is a standard GraphModel ready for CPMEngine.analyze()
```

### Design Principles

1. **Evidence-based**: Automatic correctness from any single detection
2. **No re-detection**: Consumes existing results; never calls CV operations
3. **Ambiguity-first**: Explicit tracking of uncertainty, not silent defaults
4. **Dual representation**: Native AON and AOA support
5. **Provenance tracking**: Every element traces back to source evidence
6. **Validation at every step**: Validates before and after reconstruction
7. **Configurable threshold**: Minimum confidence for element inclusion

---

## 7.2 End-to-End Validation & Human-Review Foundation (Phase 8)

### Module Structure

```
pert_analyzer/pipeline/
├── __init__.py          # Public API exports
├── result.py            # PipelineResult, AnalysisStatus, StageTiming, StageResult, ImageMetadata
├── human_review.py      # ReviewIssueType, ReviewIssue, CorrectionAction, HumanReviewItem, HumanReviewResult
├── analyzer.py          # EndToEndAnalyzer — 12-stage pipeline orchestrator
├── review_api.py        # ReviewCorrectionAPI — programmatic corrections + re-analysis
├── debug_export.py      # DebugExporter — intermediate images + result JSON export
└── __main__.py          # CLI entry point (python -m pert_analyzer.pipeline)
```

### Architecture Role

```
Image File
    │
    ▼
EndToEndAnalyzer.analyze()
    │
    ├─ 01. Load Image (cv2.imread)
    ├─ 02. Preprocess (ImagePreprocessor)
    ├─ 03. Shape Detection (ShapeDetector)
    ├─ 04. Classification (DiagramClassifier)
    ├─ 05. Arrow Detection (ArrowDetector)
    ├─ 06. OCR (OCREngine)
    ├─ 07. Text Association (SpatialAssociator)
    ├─ 08. Semantic Reconstruction (ReconstructionEngine)
    ├─ 09. Graph Build (to_graph_model)
    ├─ 10. Validate + CPM (CPMEngine)
    ├─ 11. Human Review Analysis
    └─ 12. Debug Export (optional)
    │
    ▼
PipelineResult
```

### Key Components

**EndToEndAnalyzer** — Orchestrates the complete pipeline with:
- Structured error handling per stage (stage failures don't crash pipeline)
- Timing per stage (start_time, end_time, duration)
- Progress callback support
- Lazy OCR initialization
- AOA detection → returns NOT_SUPPORTED (AOA-to-CPM not yet implemented)

**PipelineResult** — Contains:
- Overall status (SUCCESS, REVIEW_REQUIRED, FAILED, NOT_SUPPORTED)
- Per-stage results with timing
- Diagram metadata (type, confidence, counts)
- CPM results (duration, critical paths)
- Review issues and warnings
- JSON serialization for export

**ReviewCorrectionAPI** — Programmatic correction without re-running CV:
- `correct_activity_duration(id, new_value)` 
- `correct_activity_id(old, new)`
- `correct_dependency_source/target()`
- `remove_false_detection(id)`
- `add_missing_dependency(source, target)`
- `mark_dummy_activity(id)`
- `reanalyze()` — rebuilds graph and re-runs CPM

**DebugExporter** — Writes intermediate artifacts:
- Numbered images (01_original through 07_ocr)
- Result JSON
- Shape/arrow/OCR overlay visualizations

### CLI Usage

```bash
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png"
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --output result.json
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --debug
python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --json
```

### Review Required Logic

Pipeline sets `review_required=True` when:
- Any activity has confidence < 0.5
- Any activity has no OCR text evidence
- Any activity has missing duration (non-dummy)
- Reconstruction ambiguities detected
- AOA diagram type detected (CPM not yet supported for AOA)

---

## 8. Diagram Type Detection Strategy

### AON Detection Criteria:
- Rectangular or rounded-rectangle shapes predominate
- Text inside shapes (activity names/IDs)
- Arrows connect shapes directly
- No circular event nodes
- Activity data (duration) inside nodes

### AOA Detection Criteria:
- Circular shapes (event nodes) predominate
- Text near arrows (activity labels)
- Arrows have labels (ID, duration)
- Possible dummy activities (dashed arrows)
- Nodes represent events (numbered)

### Detection Algorithm:
1. Count shape types (circles vs rectangles)
2. Analyze text placement (inside shapes vs near arrows)
3. Check for dummy activity indicators
4. Compute confidence scores for each type
5. Return classification with confidence

---

## 9. AON Representation Strategy

```
┌──────────────┐
│  Activity A  │
│  Duration: 5 │
│  ID: A       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Activity B  │
│  Duration: 3 │
│  ID: B       │
└──────────────┘

Data Model:
- Nodes represent activities
- Edges represent dependencies
- Each node stores: ID, name, duration
- Each edge stores: dependency type
```

---

## 10. AOA Representation Strategy

```
    ┌───┐
    │ 1 │  (Start Event)
    └─┬─┘
      │ A (5)
      ▼
    ┌───┐
    │ 2 │
    └─┬─┘
      │ B (3)
      ▼
    ┌───┐
    │ 3 │  (End Event)
    └───┘

Data Model:
- Nodes represent events (numbered)
- Edges represent activities (with ID, duration)
- Dummy activities: dashed arrows, zero duration
```

---

## 11. Graph Reconstruction Strategy

### Phase 1: Element Classification
- Group detected shapes by type
- Identify OCR text clusters
- Match text to shapes spatially

### Phase 2: Node Construction
- Create nodes from detected shapes
- Assign labels from associated OCR text
- Determine node type (event/activity)

### Phase 3: Edge Construction
- Trace arrow paths between shapes
- Associate activity labels with arrows
- Identify dummy activities

### Phase 4: Graph Assembly
- Build NetworkX DiGraph
- Add nodes with attributes
- Add edges with attributes
- Identify source/sink nodes

### Phase 5: Graph Validation
- Check for cycles (should be DAG)
- Verify connectivity
- Validate data completeness

---

## 12. Validation Strategy

### Structural Validation:
- [ ] Graph is a DAG (no cycles)
- [ ] Single source node exists
- [ ] Single sink node exists
- [ ] All nodes are reachable from source
- [ ] All nodes can reach sink
- [ ] No disconnected components

### Data Validation:
- [ ] All activities have durations
- [ ] Activity IDs are unique
- [ ] No duplicate connections
- [ ] Durations are positive numbers
- [ ] PERT values are valid (O ≤ M ≤ P)

### Confidence Validation:
- [ ] Detection confidence above threshold
- [ ] OCR confidence above threshold
- [ ] No ambiguous associations
- [ ] All elements have associated data

### Issue Reporting:
- Each issue has severity (error/warning/info)
- Each issue has category and message
- Suggestions provided where possible
- Element IDs linked for UI highlighting

---

## 13. Confidence Scoring Strategy

### Component Confidence:
- **Shape Detection**: Based on contour quality, size consistency
- **Arrow Detection**: Based on line straightness, arrowhead clarity
- **OCR**: Based on engine confidence score
- **Spatial Association**: Based on distance and overlap
- **Diagram Classification**: Based on shape distribution

### Composite Confidence:
```
overall_confidence = (
    shape_confidence * 0.2 +
    arrow_confidence * 0.2 +
    ocr_confidence * 0.3 +
    association_confidence * 0.15 +
    classification_confidence * 0.15
)
```

### Confidence Thresholds:
- **High**: ≥ 0.85 — Auto-accept with optional review
- **Medium**: 0.60 — 0.84 — Require user review
- **Low**: < 0.60 — Require manual correction

---

## 14. CPM/PERT Integration Strategy

### CPM Engine:
```
1. Topological sort of activities
2. Forward pass (ES, EF calculation)
   - ES = max(EF of all predecessors)
   - EF = ES + duration
3. Backward pass (LS, LF calculation)
   - LF = min(LS of all successors)
   - LS = LF - duration
4. Float calculation
   - Total Float = LS - ES = LF - EF
   - Free Float = min(ES of successors) - EF
5. Critical path identification
   - Activities where Total Float = 0
```

### PERT Engine (Future):
```
1. For each activity:
   - TE = (O + 4M + P) / 6
   - Variance = ((P - O) / 6)^2
2. Use TE as duration for CPM calculations
3. Project variance = sum of variances on critical path
4. Project std dev = sqrt(project_variance)
5. Z-score calculations for probability
```

### Integration Point:
- Both engines implement `AnalysisEngine` interface
- Both take `GraphModel` as input
- Both return `AnalysisResult` as output
- Orchestrator selects engine based on data availability

---

## 15. Visualization Strategy

### Static Visualization (matplotlib):
- Network diagram with nodes and edges
- Color coding for critical vs non-critical
- Labels for activities and durations
- ES/EF/LS/LF annotations

### Interactive Visualization (Graphviz/pyqtgraph):
- Zoomable/pannable network
- Hover tooltips with activity details
- Click to select and highlight paths
- Export to PNG/SVG

### Analysis Visualization:
- Gantt chart view
- Timeline view
- Float distribution chart
- Critical path highlight

---

## 16. GUI Architecture

### Main Window Structure:
```
┌─────────────────────────────────────────────────────────┐
│  Menu Bar                                               │
├─────────────────────────────────────────────────────────┤
│  Toolbar (Open, Save, Analyze, Export)                  │
├────────┬────────────────────────────────────────────────┤
│        │                                                │
│  Side  │           Main Content Area                    │
│  Panel │                                                │
│        │  ┌──────────────────────────────────────────┐  │
│  - Tree│  │  Tab: Image | Detection | Graph |        │  │
│  - Prop│  │        Analysis | Validation             │  │
│        │  └──────────────────────────────────────────┘  │
│        │                                                │
├────────┴────────────────────────────────────────────────┤
│  Status Bar (Progress, Confidence, Messages)            │
└─────────────────────────────────────────────────────────┘
```

### View Components:
1. **DashboardView**: Project overview, recent projects
2. **ImageView**: Image upload, preprocessing controls
3. **DetectionView**: Shape/arrow/OCR results with overlays
4. **ReviewView**: Edit detected elements, correct OCR
5. **GraphView**: Network diagram visualization
6. **AnalysisView**: CPM/PERT results, critical path
7. **TableView**: Activity table with all metrics
8. **ValidationView**: Issues list, confidence scores
9. **ReportView**: Export options, summary generation
10. **SettingsView**: OCR engine, thresholds, preferences

### Communication Pattern:
- GUI → Application Layer: Method calls
- Application Layer → GUI: Qt signals
- CV/OCR → Application Layer: Callbacks
- Application Layer → CV/OCR: Method calls

---

## 17. Testing Architecture

### Unit Tests:
- Data model construction and validation
- CPM/PERT calculation correctness
- Graph builder logic
- Validation rules
- Configuration loading

### Integration Tests:
- Pipeline stage connections
- CV → OCR → Graph flow
- Analysis engine integration
- GUI → Application → Domain flow

### CV Component Tests:
- Shape detection on reference images
- Arrow detection accuracy
- OCR text extraction quality
- Diagram classification accuracy

### End-to-End Tests:
- Full pipeline: Image → Analysis
- AON diagram processing
- AOA diagram processing
- Error handling and recovery

### Test Data:
- Reference AON diagrams (clean, noisy, varying styles)
- Reference AOA diagrams (with dummy activities)
- Edge cases (skewed, low-quality, complex layouts)
- Ground truth annotations for validation

---

## 18. Future Extensibility

### Extension Points:
1. **New OCR Engines**: Implement `OCREngine` interface
2. **New Diagram Types**: Add to `DiagramClassifier` and `GraphBuilder`
3. **New Analysis Engines**: Implement `AnalysisEngine` interface
4. **New Visualization**: Implement `NetworkVisualizer` interface
5. **New Export Formats**: Add to export module
6. **New Validation Rules**: Add to validation engine

### Plugin Architecture (Future):
- Load analysis engines dynamically
- Load visualization backends dynamically
- Custom validation rule definitions
- User-defined report templates

### Scalability Considerations:
- Large diagram support (100+ activities)
- Batch processing capability
- Parallel CV processing
- Database optimization for history

---

## Risk Assessment

### Technical Risks:
1. **OCR Accuracy**: Arabic/English mixed text — Mitigation: Multiple OCR engines
2. **Shape Detection**: Varying diagram styles — Mitigation: Adaptive thresholds
3. **Arrow Detection**: Overlapping arrows — Mitigation: Multi-scale detection
4. **Diagram Classification**: Ambiguous layouts — Mitigation: Confidence scoring
5. **Performance**: Large images — Mitigation: Image pyramids, ROI processing

### Mitigation Strategies:
- Human review for low-confidence results
- Multiple detection approaches
- Fallback to manual input
- Extensive test data collection
- Iterative improvement based on real usage

---

*This architecture document serves as the blueprint for the PERT & Critical Path Analyzer implementation.*

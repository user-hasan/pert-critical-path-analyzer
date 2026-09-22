"""
Computer Vision pipeline module.

Contains image preprocessing, shape detection, arrow detection,
diagram classification, OCR, text association, and semantic
reconstruction components.
"""

from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.classification import DiagramClassifier
from pert_analyzer.cv.exceptions import (
    ArrowDetectionError,
    ClassificationError,
    CVError,
    ImageFormatError,
    ImageLoadError,
    ImageSizeError,
    ImageValidationError,
    InvalidConfigError,
    OCREngineError,
    PreprocessingError,
    ReconstructionError,
    ShapeDetectionError,
    TextAssociationError,
)
from pert_analyzer.cv.models import (
    ArrowDetectionConfig,
    ArrowDetectionResult,
    ArrowheadType,
    BlurMethod,
    CandidateNode,
    ContrastMethod,
    CoordinateMapping,
    DiagramClassificationResult,
    DetectedArrow,
    DetectedLineSegment,
    EdgeMethod,
    LineStyle,
    PreprocessingConfig,
    PreprocessingResult,
    ResizeStrategy,
    ShapeDetectionConfig,
    ShapeDetectionResult,
    ShapeType,
    ThresholdMethod,
)
from pert_analyzer.cv.ocr_engine import (
    MockOCREngine,
    OCREngineBase,
    TesseractOCREngine,
    create_ocr_engine,
)
from pert_analyzer.cv.ocr_models import (
    AssociationTargetType,
    NumericCandidate,
    OCRConfig,
    OCRProcessingResult,
    OCRTextRegion,
    TextAssociation,
    TextAssociationResult,
    TextGroup,
    TextType,
)
from pert_analyzer.cv.preprocessing import ImagePreprocessor
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.reconstruction_models import (
    AmbiguityIssue,
    AmbiguityType,
    EvidenceTrace,
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
    ReconstructedEvent,
    ValidationResult,
    ValidationSeverity,
)
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.spatial_association import SpatialAssociator
from pert_analyzer.cv.text_classification import TextClassifier
from pert_analyzer.cv.text_grouping import TextGrouper
from pert_analyzer.cv.text_normalization import TextNormalizer

__all__ = [
    # Exceptions
    "CVError",
    "ArrowDetectionError",
    "ClassificationError",
    "ImageLoadError",
    "ImageValidationError",
    "ImageFormatError",
    "ImageSizeError",
    "InvalidConfigError",
    "OCREngineError",
    "PreprocessingError",
    "ReconstructionError",
    "ShapeDetectionError",
    "TextAssociationError",
    # CV Models
    "ArrowDetectionConfig",
    "ArrowDetectionResult",
    "ArrowheadType",
    "BlurMethod",
    "CandidateNode",
    "ContrastMethod",
    "CoordinateMapping",
    "DiagramClassificationResult",
    "DetectedArrow",
    "DetectedLineSegment",
    "EdgeMethod",
    "LineStyle",
    "PreprocessingConfig",
    "PreprocessingResult",
    "ResizeStrategy",
    "ShapeDetectionConfig",
    "ShapeDetectionResult",
    "ShapeType",
    "ThresholdMethod",
    # OCR Models
    "AssociationTargetType",
    "NumericCandidate",
    "OCRConfig",
    "OCRProcessingResult",
    "OCRTextRegion",
    "TextAssociation",
    "TextAssociationResult",
    "TextGroup",
    "TextType",
    # OCR Engine
    "OCREngineBase",
    "MockOCREngine",
    "TesseractOCREngine",
    "create_ocr_engine",
    # Text Processing
    "TextNormalizer",
    "TextClassifier",
    "TextGrouper",
    "SpatialAssociator",
    # Reconstruction Models
    "AmbiguityIssue",
    "AmbiguityType",
    "EvidenceTrace",
    "ReconstructedActivity",
    "ReconstructedDependency",
    "ReconstructedDiagram",
    "ReconstructedEvent",
    "ValidationResult",
    "ValidationSeverity",
    # Reconstruction Engine
    "ReconstructionEngine",
    # Components
    "ArrowDetector",
    "ImagePreprocessor",
    "ShapeDetector",
    "DiagramClassifier",
]

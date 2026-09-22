"""
OCR and text association data models.

Structured models for OCR results, text regions, numeric candidates,
text classification, spatial association, and text grouping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import BoundingBox, Point, _generate_id


# =============================================================================
# Enums
# =============================================================================


class TextType(Enum):
    """Classification of OCR text content."""

    UNKNOWN = "UNKNOWN"
    ACTIVITY_ID_CANDIDATE = "ACTIVITY_ID_CANDIDATE"
    TEXT_LABEL_CANDIDATE = "TEXT_LABEL_CANDIDATE"
    NUMERIC_CANDIDATE = "NUMERIC_CANDIDATE"
    POSSIBLE_PERT_VALUE = "POSSIBLE_PERT_VALUE"
    DURATION_CANDIDATE = "DURATION_CANDIDATE"
    EVENT_LABEL = "EVENT_LABEL"


class AssociationTargetType(Enum):
    """Type of target for spatial association."""

    NODE = "NODE"
    ARROW = "ARROW"
    NONE = "NONE"


# =============================================================================
# OCR Configuration
# =============================================================================


@dataclass
class OCRConfig:
    """
    Configuration for OCR processing.

    Controls engine selection, preprocessing, confidence thresholds,
    and output preferences.
    """

    # Engine settings
    engine_name: str = "tesseract"
    languages: List[str] = field(default_factory=lambda: ["eng"])

    # Tesseract-specific
    psm: int = 11
    oem: int = 3

    # Preprocessing for OCR
    preferred_representation: str = "contrast_enhanced"
    binarize_for_ocr: bool = True
    invert_if_dark_bg: bool = True
    padding_px: int = 5

    # Confidence thresholds
    min_confidence: float = 0.0
    high_confidence: float = 0.7
    low_confidence: float = 0.3

    # Text filtering
    min_text_length: int = 1
    max_text_length: int = 200
    filter_whitespace_only: bool = True

    # Grouping
    enable_grouping: bool = True
    group_distance_threshold: float = 40.0
    group_vertical_tolerance: float = 20.0

    def validate(self) -> List[str]:
        """Validate configuration and return issues."""
        issues = []
        if self.min_confidence < 0 or self.min_confidence > 1:
            issues.append(f"min_confidence must be in [0, 1], got {self.min_confidence}")
        if self.high_confidence < self.min_confidence:
            issues.append(
                f"high_confidence ({self.high_confidence}) must be >= "
                f"min_confidence ({self.min_confidence})"
            )
        if self.low_confidence < self.min_confidence:
            issues.append(
                f"low_confidence ({self.low_confidence}) must be >= "
                f"min_confidence ({self.min_confidence})"
            )
        if self.group_distance_threshold <= 0:
            issues.append(
                f"group_distance_threshold must be > 0, got {self.group_distance_threshold}"
            )
        return issues


# =============================================================================
# OCR Text Region
# =============================================================================


@dataclass
class OCRTextRegion:
    """
    A single text region detected and recognized by OCR.

    Contains the recognized text, its bounding box, confidence,
    normalization data, and classification evidence.
    """

    region_id: str = field(default_factory=lambda: _generate_id("ocr"))
    text: str = ""
    raw_text: str = ""
    normalized_text: str = ""
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    center: Point = field(default_factory=lambda: Point(0, 0))
    confidence: float = 0.0
    language: str = "en"
    source_engine: str = ""
    word_confidences: List[float] = field(default_factory=list)

    # Classification
    text_type: TextType = TextType.UNKNOWN
    text_type_confidence: float = 0.0
    text_type_evidence: Dict[str, Any] = field(default_factory=dict)

    # Numeric data
    is_numeric: bool = False
    parsed_value: Optional[float] = None
    numeric_parse_warnings: List[str] = field(default_factory=list)

    # Association
    associated_target_id: Optional[str] = None
    associated_target_type: AssociationTargetType = AssociationTargetType.NONE
    association_score: float = 0.0
    association_evidence: Dict[str, Any] = field(default_factory=dict)
    is_ambiguous: bool = False

    # Grouping
    group_id: Optional[str] = None

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Compute center and normalize text after initialization."""
        self.center = self.bounding_box.center
        if not self.raw_text:
            self.raw_text = self.text
        if not self.normalized_text:
            self.normalized_text = self.text.strip()


# =============================================================================
# Numeric Candidate
# =============================================================================


@dataclass
class NumericCandidate:
    """
    A numeric value extracted from OCR text.

    Preserves the original OCR string and provides structured numeric data.
    """

    candidate_id: str = field(default_factory=lambda: _generate_id("num"))
    value: float = 0.0
    raw_text: str = ""
    confidence: float = 0.0
    bounding_box: BoundingBox = field(
        default_factory=lambda: BoundingBox(0, 0, 0, 0)
    )
    source_region_id: str = ""
    parse_warnings: List[str] = field(default_factory=list)
    is_integer: bool = False
    is_negative: bool = False
    decimal_places: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Compute metadata from value."""
        if self.raw_text and not self.parse_warnings:
            self.is_integer = self.value == int(self.value)
            self.is_negative = self.value < 0
            if "." in self.raw_text:
                parts = self.raw_text.split(".")
                self.decimal_places = len(parts[1]) if len(parts) > 1 else 0


# =============================================================================
# Text Association
# =============================================================================


@dataclass
class TextAssociation:
    """
    Association between an OCR text region and a diagram element.

    Represents spatial association evidence, NOT final semantic interpretation.
    """

    association_id: str = field(default_factory=lambda: _generate_id("assoc"))
    text_region_id: str = ""
    candidate_target_id: str = ""
    target_type: AssociationTargetType = AssociationTargetType.NONE
    association_score: float = 0.0
    is_best_candidate: bool = False
    is_ambiguous: bool = False
    evidence: Dict[str, Any] = field(default_factory=dict)
    reasons: List[str] = field(default_factory=list)


@dataclass
class TextAssociationResult:
    """
    Complete result of text association for a single text region.

    Contains all candidate associations, the best candidate, and ambiguity flags.
    """

    text_region_id: str = ""
    associations: List[TextAssociation] = field(default_factory=list)
    best_association: Optional[TextAssociation] = None
    is_ambiguous: bool = False
    has_no_match: bool = True

    @property
    def association_count(self) -> int:
        """Number of candidate associations."""
        return len(self.associations)

    @property
    def has_associations(self) -> bool:
        """Check if any associations were found."""
        return len(self.associations) > 0


# =============================================================================
# Text Group
# =============================================================================


@dataclass
class TextGroup:
    """
    A logical grouping of related OCR text regions.

    Groups text regions that are spatially close and likely represent
    the same diagram element (e.g., activity ID + duration in one box).
    """

    group_id: str = field(default_factory=lambda: _generate_id("tgroup"))
    member_region_ids: List[str] = field(default_factory=list)
    combined_text: str = ""
    normalized_combined_text: str = ""
    combined_bounding_box: Optional[BoundingBox] = None
    group_confidence: float = 0.0
    member_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> Optional[Point]:
        """Get center of combined bounding box."""
        if self.combined_bounding_box:
            return self.combined_bounding_box.center
        return None


# =============================================================================
# OCR Processing Result
# =============================================================================


@dataclass
class OCRProcessingResult:
    """
    Complete result of OCR processing on an image.

    Contains all detected text regions, groups, numeric candidates,
    warnings, and metadata.
    """

    regions: List[OCRTextRegion] = field(default_factory=list)
    groups: List[TextGroup] = field(default_factory=list)
    numeric_candidates: List[NumericCandidate] = field(default_factory=list)
    association_results: List[TextAssociationResult] = field(
        default_factory=list
    )
    engine: str = ""
    image_dimensions: Tuple[int, int] = (0, 0)
    coordinate_mapping: Optional[Any] = None
    source_representation: str = ""
    processing_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def region_count(self) -> int:
        """Number of detected text regions."""
        return len(self.regions)

    @property
    def group_count(self) -> int:
        """Number of text groups."""
        return len(self.groups)

    @property
    def numeric_count(self) -> int:
        """Number of numeric candidates."""
        return len(self.numeric_candidates)

    @property
    def average_confidence(self) -> float:
        """Average OCR confidence across all regions."""
        if not self.regions:
            return 0.0
        return sum(r.confidence for r in self.regions) / len(self.regions)

    @property
    def high_confidence_count(self) -> int:
        """Count of regions with high confidence (>= 0.7)."""
        return sum(1 for r in self.regions if r.confidence >= 0.7)

    def get_regions_by_type(self, text_type: TextType) -> List[OCRTextRegion]:
        """Get text regions of a specific type."""
        return [r for r in self.regions if r.text_type == text_type]

    def get_numeric_regions(self) -> List[OCRTextRegion]:
        """Get text regions classified as numeric."""
        return [r for r in self.regions if r.text_type == TextType.NUMERIC_CANDIDATE]

    def get_labeled_regions(self) -> List[OCRTextRegion]:
        """Get text regions classified as labels."""
        return [r for r in self.regions if r.text_type == TextType.TEXT_LABEL_CANDIDATE]

    def get_id_candidates(self) -> List[OCRTextRegion]:
        """Get text regions classified as activity ID candidates."""
        return [r for r in self.regions if r.text_type == TextType.ACTIVITY_ID_CANDIDATE]

    def to_original_coordinates(
        self, x: float, y: float
    ) -> Tuple[float, float]:
        """Convert detection coordinates to original image coordinates."""
        if self.coordinate_mapping is None:
            return (x, y)
        return self.coordinate_mapping.to_original(x, y)

    def get_region_by_id(self, region_id: str) -> Optional[OCRTextRegion]:
        """Get a text region by its ID."""
        for region in self.regions:
            if region.region_id == region_id:
                return region
        return None

    def get_group_by_id(self, group_id: str) -> Optional[TextGroup]:
        """Get a text group by its ID."""
        for group in self.groups:
            if group.group_id == group_id:
                return group
        return None

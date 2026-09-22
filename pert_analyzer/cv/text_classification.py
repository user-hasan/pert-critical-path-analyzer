"""
Text type classification for OCR results.

Classifies OCR text regions into candidate types based on content patterns,
spatial context, and heuristic rules. Evidence-based — does not force every
region into a type.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pert_analyzer.cv.ocr_models import (
    NumericCandidate,
    OCRTextRegion,
    TextType,
)
from pert_analyzer.cv.numeric_extraction import NumericExtractor


# Pattern definitions for classification
ACTIVITY_ID_PATTERNS = [
    re.compile(r"^[A-Za-z]\d{1,4}$"),          # A1, B12, c3
    re.compile(r"^[A-Za-z]{1,3}\d{1,4}$"),     # AB1, A12
    re.compile(r"^\d{1,4}[A-Za-z]$"),           # 1A, 12B
    re.compile(r"^[A-Z]\d{1,2}$"),              # A1, B12
    re.compile(r"^(?:act|activity|task|job)\s*\d+$", re.IGNORECASE),  # Act 1, Task 5
]

PERT_VALUE_PATTERNS = [
    re.compile(r"^\d+(?:\.\d+)?$"),  # Integer or decimal
    re.compile(r"^-?\d+(?:\.\d+)?$"),  # With optional negative
]


@dataclass
class TextClassificationConfig:
    """Configuration for text type classification."""

    # Pattern matching
    enable_pattern_matching: bool = True
    enable_numeric_detection: bool = True
    enable_spatial_hints: bool = True

    # Confidence thresholds
    high_classification_confidence: float = 0.8
    medium_classification_confidence: float = 0.5
    low_classification_confidence: float = 0.3

    # Activity ID patterns
    activity_id_patterns: List[str] = field(default_factory=lambda: [
        r"^[A-Za-z]\d{1,4}$",
        r"^[A-Za-z]{1,3}\d{1,4}$",
        r"^\d{1,4}[A-Za-z]$",
    ])

    # Label heuristics
    min_label_length: int = 2
    max_label_length: int = 50


class TextClassifier:
    """
    Classifies OCR text regions into candidate types.

    Uses pattern matching, numeric detection, and spatial context
    to produce evidence-based classifications.
    """

    def __init__(self, config: Optional[TextClassificationConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or TextClassificationConfig()
        self._numeric_extractor = NumericExtractor()
        self._compiled_patterns = self._compile_patterns()

    def _compile_patterns(self) -> List[re.Pattern]:
        """Compile activity ID patterns."""
        patterns = []
        for p in self.config.activity_id_patterns:
            patterns.append(re.compile(p))
        return patterns

    def classify(
        self,
        region: OCRTextRegion,
        sibling_regions: Optional[List[OCRTextRegion]] = None,
    ) -> TextType:
        """
        Classify a text region into a candidate type.

        Args:
            region: The text region to classify.
            sibling_regions: Other text regions in the same context (e.g., same shape).

        Returns:
            TextType classification.
        """
        text = region.normalized_text or region.text
        if not text or not text.strip():
            return TextType.UNKNOWN

        text = text.strip()

        # Check for numeric first
        if self.config.enable_numeric_detection:
            if self._is_numeric_candidate(text, region):
                return TextType.NUMERIC_CANDIDATE

        # Check for activity ID
        if self.config.enable_pattern_matching:
            if self._is_activity_id_candidate(text):
                return TextType.ACTIVITY_ID_CANDIDATE

        # Check for text label
        if self._is_text_label_candidate(text):
            return TextType.TEXT_LABEL_CANDIDATE

        return TextType.UNKNOWN

    def classify_with_confidence(
        self,
        region: OCRTextRegion,
        sibling_regions: Optional[List[OCRTextRegion]] = None,
    ) -> Tuple[TextType, float, Dict[str, Any]]:
        """
        Classify a text region with confidence and evidence.

        Args:
            region: The text region to classify.
            sibling_regions: Other text regions in the same context.

        Returns:
            Tuple of (TextType, confidence, evidence_dict).
        """
        text = region.normalized_text or region.text
        if not text or not text.strip():
            return (TextType.UNKNOWN, 0.0, {"reason": "empty text"})

        text = text.strip()
        evidence: Dict[str, Any] = {"text": text}

        # Check for numeric
        if self.config.enable_numeric_detection:
            if self._is_numeric_candidate(text, region):
                confidence = self._compute_numeric_confidence(text, region)
                evidence["reason"] = "numeric pattern"
                evidence["is_integer"] = text.replace(".", "").replace("-", "").isdigit()
                return (TextType.NUMERIC_CANDIDATE, confidence, evidence)

        # Check for activity ID
        if self.config.enable_pattern_matching:
            confidence = self._compute_activity_id_confidence(text)
            if confidence > 0:
                evidence["reason"] = "activity ID pattern"
                evidence["pattern_match"] = True
                return (TextType.ACTIVITY_ID_CANDIDATE, confidence, evidence)

        # Check for text label
        if self._is_text_label_candidate(text):
            confidence = self._compute_label_confidence(text, sibling_regions)
            evidence["reason"] = "text label heuristic"
            return (TextType.TEXT_LABEL_CANDIDATE, confidence, evidence)

        return (TextType.UNKNOWN, 0.0, {"reason": "no matching pattern"})

    def _is_numeric_candidate(self, text: str, region: OCRTextRegion) -> bool:
        """Check if text is a numeric candidate."""
        # Direct check
        if self._numeric_extractor.is_numeric(text):
            return True

        # Check OCR region's parsed value
        if region.is_numeric and region.parsed_value is not None:
            return True

        return False

    def _is_activity_id_candidate(self, text: str) -> bool:
        """Check if text matches activity ID patterns."""
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                return True
        return False

    def _is_text_label_candidate(self, text: str) -> bool:
        """Check if text is likely a label."""
        if len(text) < self.config.min_label_length:
            return False
        if len(text) > self.config.max_label_length:
            return False
        # Must contain at least one letter
        if not any(c.isalpha() for c in text):
            return False
        return True

    def _compute_numeric_confidence(
        self, text: str, region: OCRTextRegion
    ) -> float:
        """Compute confidence for numeric classification."""
        base_confidence = region.confidence if region.confidence > 0 else 0.8

        # Boost if text is purely digits
        clean = text.replace(".", "").replace("-", "").replace("+", "")
        if clean.isdigit():
            base_confidence = min(base_confidence + 0.1, 1.0)

        return base_confidence

    def _compute_activity_id_confidence(self, text: str) -> float:
        """Compute confidence for activity ID classification."""
        # Exact pattern match gives high confidence
        for pattern in self._compiled_patterns:
            if pattern.match(text):
                return self.config.high_classification_confidence
        return 0.0

    def _compute_label_confidence(
        self,
        text: str,
        sibling_regions: Optional[List[OCRTextRegion]] = None,
    ) -> float:
        """Compute confidence for text label classification."""
        confidence = self.config.medium_classification_confidence

        # Longer labels are more likely to be real labels
        if len(text) >= 5:
            confidence += 0.1

        # If siblings exist and some are numeric, boost label confidence
        if sibling_regions:
            numeric_siblings = sum(
                1 for s in sibling_regions
                if s.text_type == TextType.NUMERIC_CANDIDATE
            )
            if numeric_siblings > 0:
                confidence += 0.15

        return min(confidence, 1.0)

    def classify_region(
        self,
        region: OCRTextRegion,
        sibling_regions: Optional[List[OCRTextRegion]] = None,
    ) -> OCRTextRegion:
        """
        Classify a text region and update its text_type fields in place.

        Args:
            region: The text region to classify (modified in place).
            sibling_regions: Other text regions in the same context.

        Returns:
            The same region with updated classification fields.
        """
        text_type, confidence, evidence = self.classify_with_confidence(
            region, sibling_regions
        )
        region.text_type = text_type
        region.text_type_confidence = confidence
        region.text_type_evidence = evidence
        return region

    def classify_regions(
        self, regions: List[OCRTextRegion]
    ) -> List[OCRTextRegion]:
        """
        Classify multiple text regions.

        Args:
            regions: List of text regions to classify.

        Returns:
            The same list with updated classification fields.
        """
        for region in regions:
            self.classify_region(region, sibling_regions=regions)
        return regions

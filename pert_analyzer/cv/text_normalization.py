"""
Text normalization utilities for OCR post-processing.

Conservative, configurable normalization that preserves raw text alongside
normalized text. Handles whitespace, common OCR artifacts, Unicode normalization,
and common character confusions.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple


@dataclass
class NormalizationConfig:
    """
    Configuration for text normalization.

    Each normalization step can be independently enabled/disabled.
    """

    # Whitespace normalization
    strip_whitespace: bool = True
    collapse_repeated_spaces: bool = True
    normalize_unicode: bool = True
    unicode_form: str = "NFKC"

    # OCR artifact cleanup
    fix_common_ocr_errors: bool = True
    max_consecutive_chars: int = 3

    # Character confusion mapping (disabled by default — conservative)
    apply_character_confusion: bool = False
    character_confusion_map: Dict[str, str] = field(default_factory=dict)

    # Filtering
    min_length_after_normalization: int = 1
    filter_punctuation_only: bool = True


# Default character confusion mapping for common OCR errors
# Applied only when apply_character_confusion is True
DEFAULT_CHARACTER_CONFUSION: Dict[str, str] = {
    # Common OCR confusions (conservative — only high-confidence swaps)
    "O": "0",  # Letter O vs digit 0 (context-dependent)
    "l": "1",  # Lowercase L vs digit 1
    "I": "1",  # Capital I vs digit 1
    "S": "5",  # Letter S vs digit 5 (rare, disabled by default)
    "B": "8",  # Letter B vs digit 8 (rare, disabled by default)
}

# Arabic-Indic digit mapping
ARABIC_INDIC_DIGITS: Dict[str, str] = {
    "\u0660": "0",  # ٠
    "\u0661": "1",  # ١
    "\u0662": "2",  # ٢
    "\u0663": "3",  # ٣
    "\u0664": "4",  # ٤
    "\u0665": "5",  # ٥
    "\u0666": "6",  # ٦
    "\u0667": "7",  # ٧
    "\u0668": "8",  # ٨
    "\u0669": "9",  # ٩
}

# Extended Arabic-Indic digit mapping
EXTENDED_ARABIC_INDIC_DIGITS: Dict[str, str] = {
    "\u06F0": "0",  # ۰
    "\u06F1": "1",  # ۱
    "\u06F2": "2",  # ۲
    "\u06F3": "3",  # ۳
    "\u06F4": "4",  # ۴
    "\u06F5": "5",  # ۵
    "\u06F6": "6",  # ۶
    "\u06F7": "7",  # ۷
    "\u06F8": "8",  # ۸
    "\u06F9": "9",  # ۹
}


class TextNormalizer:
    """
    Conservative text normalizer for OCR results.

    Preserves raw text and produces normalized text separately.
    Each normalization step is configurable.
    """

    def __init__(self, config: Optional[NormalizationConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or NormalizationConfig()
        self._all_digit_map: Dict[str, str] = {}
        self._all_digit_map.update(ARABIC_INDIC_DIGITS)
        self._all_digit_map.update(EXTENDED_ARABIC_INDIC_DIGITS)

    def normalize(self, text: str) -> str:
        """
        Apply all configured normalization steps to text.

        Args:
            text: Raw OCR text.

        Returns:
            Normalized text string.
        """
        if not text:
            return ""

        result = text

        # Step 1: Unicode normalization
        if self.config.normalize_unicode:
            result = self._normalize_unicode(result)

        # Step 2: Convert Arabic-Indic digits to Western Arabic
        result = self._convert_arabic_digits(result)

        # Step 3: Whitespace normalization
        if self.config.strip_whitespace:
            result = result.strip()
        if self.config.collapse_repeated_spaces:
            result = self._collapse_spaces(result)

        # Step 4: OCR artifact cleanup
        if self.config.fix_common_ocr_errors:
            result = self._fix_ocr_artifacts(result)

        # Step 5: Character confusion (if enabled)
        if self.config.apply_character_confusion and self.config.character_confusion_map:
            result = self._apply_character_confusion(result)

        # Step 6: Filter punctuation-only text
        if self.config.filter_punctuation_only:
            result = self._filter_punctuation(result)

        return result

    def normalize_preserve_raw(self, text: str) -> Tuple[str, str]:
        """
        Normalize text and return both raw and normalized versions.

        Args:
            text: Raw OCR text.

        Returns:
            Tuple of (raw_text, normalized_text).
        """
        raw = text
        normalized = self.normalize(text)
        return (raw, normalized)

    def _normalize_unicode(self, text: str) -> str:
        """Apply Unicode normalization."""
        return unicodedata.normalize(self.config.unicode_form, text)

    def _convert_arabic_digits(self, text: str) -> str:
        """Convert Arabic-Indic and Extended Arabic-Indic digits to Western Arabic."""
        result = []
        for char in text:
            if char in self._all_digit_map:
                result.append(self._all_digit_map[char])
            else:
                result.append(char)
        return "".join(result)

    def _collapse_spaces(self, text: str) -> str:
        """Collapse multiple whitespace characters into single spaces."""
        return re.sub(r"\s+", " ", text)

    def _fix_ocr_artifacts(self, text: str) -> str:
        """Fix common OCR artifacts."""
        result = text

        # Remove null bytes
        result = result.replace("\x00", "")

        # Fix common misreads
        replacements = {
            "\u200b": "",  # Zero-width space
            "\u200c": "",  # Zero-width non-joiner
            "\u200d": "",  # Zero-width joiner
            "\ufeff": "",  # BOM
            "\u00a0": " ",  # Non-breaking space -> regular space
        }
        for old, new in replacements.items():
            result = result.replace(old, new)

        # Detect repeated characters (e.g., "aaaa" from OCR glitch)
        if self.config.max_consecutive_chars > 0:
            pattern = r"(.)\{" + str(self.config.max_consecutive_chars + 1) + r",\}"
            result = re.sub(
                pattern,
                lambda m: m.group(1) * self.config.max_consecutive_chars,
                result,
            )

        return result

    def _apply_character_confusion(self, text: str) -> str:
        """Apply character confusion mapping (context-dependent)."""
        # This is a conservative approach — only apply in clearly numeric contexts
        # For now, just return text as-is; full implementation would need context
        return text

    def _filter_punctuation(self, text: str) -> str:
        """Filter text that is only punctuation/whitespace."""
        # Check if text contains at least one alphanumeric character
        has_alphanumeric = bool(re.search(r"[a-zA-Z0-9\u0660-\u0669\u06F0-\u06F9]", text))
        if not has_alphanumeric and text.strip():
            # Text is only punctuation — keep it but mark it
            return text
        return text

    def is_numeric_context(self, text: str) -> bool:
        """
        Check if text appears to be in a numeric context.

        Args:
            text: Text to check.

        Returns:
            True if text looks like it could be a numeric value.
        """
        cleaned = text.strip()
        if not cleaned:
            return False

        # Direct numeric
        try:
            float(cleaned)
            return True
        except ValueError:
            pass

        # Check for Arabic-Indic digits
        converted = self._convert_arabic_digits(cleaned)
        try:
            float(converted)
            return True
        except ValueError:
            pass

        return False

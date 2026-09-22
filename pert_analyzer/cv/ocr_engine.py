"""
OCR engine abstraction and adapter pattern.

Provides engine-agnostic OCR interface with pluggable backends.
Supports Tesseract (when available) and a mock engine for testing.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

from pert_analyzer.core.models import BoundingBox
from pert_analyzer.cv.ocr_models import OCRConfig, OCRTextRegion, OCRProcessingResult

logger = logging.getLogger(__name__)


class OCREngineBase(ABC):
    """
    Abstract base class for OCR engines.

    All OCR backends must implement this interface.
    The engine is replaceable — swap implementations without changing callers.
    """

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize the OCR engine.

        Args:
            config: Optional engine-specific configuration.

        Returns:
            True if initialization succeeded.

        Raises:
            OCREngineError: If initialization fails critically.
        """
        ...

    @abstractmethod
    def recognize(
        self,
        image: np.ndarray,
        config: Optional[OCRConfig] = None,
    ) -> List[OCRTextRegion]:
        """
        Recognize text in an image.

        Args:
            image: Input image (grayscale or color, numpy array).
            config: Optional OCR configuration overrides.

        Returns:
            List of recognized text regions with bounding boxes and confidence.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the engine is available and initialized."""
        ...

    @abstractmethod
    def get_engine_name(self) -> str:
        """Return the name of this engine."""
        ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        ...


class OCREngineError(Exception):
    """Raised when an OCR engine operation fails."""

    def __init__(self, engine: str, operation: str, reason: str):
        self.engine = engine
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"OCR engine '{engine}' failed during '{operation}': {reason}"
        )


class MockOCREngine(OCREngineBase):
    """
    Mock OCR engine for testing.

    Returns configurable text regions without requiring any external
    OCR installation. Useful for unit tests and development.
    """

    def __init__(self):
        """Initialize mock engine."""
        self._initialized = False
        self._mock_results: List[OCRTextRegion] = []
        self._languages = ["eng"]

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize mock engine (always succeeds)."""
        self._initialized = True
        if config:
            if "results" in config:
                self._mock_results = config["results"]
            if "languages" in config:
                self._languages = config["languages"]
        return True

    def set_mock_results(self, results: List[OCRTextRegion]) -> None:
        """Set mock results to return."""
        self._mock_results = results

    def recognize(
        self,
        image: np.ndarray,
        config: Optional[OCRConfig] = None,
    ) -> List[OCRTextRegion]:
        """
        Return mock OCR results.

        Args:
            image: Input image (ignored in mock).
            config: Optional configuration (ignored in mock).

        Returns:
            Pre-configured mock text regions.
        """
        if not self._initialized:
            raise OCREngineError("mock", "recognize", "engine not initialized")
        return list(self._mock_results)

    def is_available(self) -> bool:
        """Mock engine is always available when initialized."""
        return self._initialized

    def get_engine_name(self) -> str:
        """Return mock engine name."""
        return "mock"

    def get_supported_languages(self) -> List[str]:
        """Return mock supported languages."""
        return list(self._languages)


class TesseractOCREngine(OCREngineBase):
    """
    Tesseract OCR engine adapter.

    Wraps pytesseract for OCR operations. Requires:
    - pytesseract Python package
    - Tesseract OCR executable installed on the system

    Installation:
        pip install pytesseract
        # On Windows: install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki
    """

    def __init__(self):
        """Initialize Tesseract engine."""
        self._initialized = False
        self._pytesseract: Any = None
        self._tesseract_path: Optional[str] = None
        self._languages = ["eng"]
        self._psm = 11
        self._oem = 3

    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        Initialize Tesseract engine.

        Args:
            config: Optional configuration.
                - tesseract_path: Path to tesseract executable.
                - languages: List of language codes (e.g. ["eng", "ara"]).
                - psm: Page segmentation mode (default 11 = sparse text).
                - oem: OCR engine mode (default 3 = default).

        Returns:
            True if initialization succeeded.

        Raises:
            OCREngineError: If pytesseract is not installed or Tesseract not found.
        """
        try:
            import pytesseract
            self._pytesseract = pytesseract
        except ImportError:
            raise OCREngineError(
                "tesseract",
                "initialize",
                "pytesseract not installed. Install with: pip install pytesseract"
            )

        if config:
            self._tesseract_path = config.get("tesseract_path")
            if self._tesseract_path:
                pytesseract.pytesseract.tesseract_cmd = self._tesseract_path
            self._languages = config.get("languages", ["eng"])
            self._psm = config.get("psm", 11)
            self._oem = config.get("oem", 3)

        # Test that tesseract is actually available
        try:
            pytesseract.get_tesseract_version()
            self._initialized = True
            return True
        except Exception as e:
            raise OCREngineError(
                "tesseract",
                "initialize",
                f"Tesseract executable not found: {e}"
            )

    def recognize(
        self,
        image: np.ndarray,
        config: Optional[OCRConfig] = None,
    ) -> List[OCRTextRegion]:
        """
        Recognize text using Tesseract.

        Args:
            image: Input image (grayscale or color).
            config: Optional OCR configuration overrides.

        Returns:
            List of recognized text regions.
        """
        if not self._initialized or self._pytesseract is None:
            raise OCREngineError("tesseract", "recognize", "engine not initialized")

        pytesseract = self._pytesseract
        lang = "+".join(self._languages if config is None else config.languages)

        # Determine PSM/OEM from config or instance defaults
        psm = self._psm
        oem = self._oem
        if config and hasattr(config, "psm"):
            psm = config.psm
        if config and hasattr(config, "oem"):
            oem = config.oem

        tesseract_config = f"--oem {oem} --psm {psm}"

        try:
            # Get detailed data with bounding boxes
            data = pytesseract.image_to_data(
                image,
                lang=lang,
                output_type=pytesseract.Output.DICT,
                config=tesseract_config,
            )
        except Exception as e:
            raise OCREngineError("tesseract", "recognize", str(e))

        regions = []
        n_boxes = len(data["text"])

        for i in range(n_boxes):
            text = data["text"][i].strip()
            conf_raw = data["conf"][i]

            # Skip empty results
            if not text:
                continue

            # Handle confidence: Tesseract returns -1 for unprocessed, 0-100 for valid
            try:
                conf_val = float(conf_raw)
            except (ValueError, TypeError):
                continue
            if conf_val < 0:
                continue
            conf = conf_val / 100.0  # Normalize to 0.0-1.0

            # Skip very low confidence results
            if config and conf < config.min_confidence:
                continue

            x = int(data["left"][i])
            y = int(data["top"][i])
            w = int(data["width"][i])
            h = int(data["height"][i])

            # Skip invalid bounding boxes
            if w <= 0 or h <= 0:
                continue

            bbox = BoundingBox(x=x, y=y, width=w, height=h)

            region = OCRTextRegion(
                text=text,
                raw_text=text,
                normalized_text=text.strip(),
                bounding_box=bbox,
                confidence=conf,
                language=lang,
                source_engine="tesseract",
                word_confidences=[conf],
            )
            regions.append(region)

        return regions

    def is_available(self) -> bool:
        """Check if Tesseract is available."""
        return self._initialized and self._pytesseract is not None

    def get_engine_name(self) -> str:
        """Return engine name."""
        return "tesseract"

    def get_supported_languages(self) -> List[str]:
        """Return supported languages."""
        return list(self._languages)


def create_ocr_engine(
    engine_name: str = "mock",
    config: Optional[Dict[str, Any]] = None,
) -> OCREngineBase:
    """
    Factory function to create OCR engines.

    Args:
        engine_name: Name of the engine ("tesseract" or "mock").
        config: Optional engine configuration.
            For tesseract:
                - tesseract_path: Path to tesseract executable.
                - languages: List of language codes (default ["eng"]).
                - psm: Page segmentation mode (default 11).
                - oem: OCR engine mode (default 3).

    Returns:
        Initialized OCR engine.

    Raises:
        OCREngineError: If engine creation or initialization fails.
    """
    if engine_name == "tesseract":
        engine = TesseractOCREngine()
    elif engine_name == "mock":
        engine = MockOCREngine()
    else:
        raise OCREngineError(engine_name, "create", f"Unknown engine: {engine_name}")

    engine.initialize(config)
    return engine

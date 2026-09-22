"""
OCR engine abstraction module.

Provides a unified interface for multiple OCR backends
(Tesseract, EasyOCR, PaddleOCR).
"""

from pert_analyzer.core.interfaces import OCREngine

__all__ = ["OCREngine"]

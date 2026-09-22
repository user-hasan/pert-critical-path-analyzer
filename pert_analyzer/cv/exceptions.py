"""
Domain-specific exceptions for Computer Vision operations.

These exceptions represent error conditions specific to image loading,
validation, and preprocessing in the CV pipeline.
"""

from __future__ import annotations

from typing import List, Optional


class CVError(Exception):
    """Base exception for all CV-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class ImageLoadError(CVError):
    """Raised when an image cannot be loaded from a path."""

    def __init__(self, path: str, reason: str = "file not found"):
        self.path = path
        self.reason = reason
        super().__init__(
            f"Failed to load image from '{path}': {reason}",
            details={"path": path, "reason": reason},
        )


class ImageValidationError(CVError):
    """Raised when an image fails validation checks."""

    def __init__(self, message: str, checks_failed: Optional[List[str]] = None):
        self.checks_failed = checks_failed or []
        super().__init__(
            message,
            details={"checks_failed": self.checks_failed},
        )


class ImageFormatError(CVError):
    """Raised when an image format is unsupported or invalid."""

    def __init__(self, format_name: str, supported_formats: Optional[List[str]] = None):
        self.format_name = format_name
        self.supported_formats = supported_formats or []
        super().__init__(
            f"Unsupported image format: '{format_name}'",
            details={
                "format": format_name,
                "supported": self.supported_formats,
            },
        )


class ImageSizeError(CVError):
    """Raised when an image has invalid dimensions."""

    def __init__(self, width: int, height: int, reason: str = "too small"):
        self.width = width
        self.height = height
        self.reason = reason
        super().__init__(
            f"Invalid image size: {width}x{height} ({reason})",
            details={"width": width, "height": height, "reason": reason},
        )


class PreprocessingError(CVError):
    """Raised when an image preprocessing operation fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Preprocessing failed during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )


class InvalidConfigError(CVError):
    """Raised when preprocessing configuration is invalid."""

    def __init__(self, parameter: str, value: any, reason: str = "invalid value"):
        self.parameter = parameter
        self.value = value
        self.reason = reason
        super().__init__(
            f"Invalid configuration for '{parameter}': {value} ({reason})",
            details={"parameter": parameter, "value": str(value), "reason": reason},
        )


class ShapeDetectionError(CVError):
    """Raised when shape detection fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Shape detection failed during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )


class ClassificationError(CVError):
    """Raised when diagram classification fails."""

    def __init__(self, reason: str, shapes_count: int = 0):
        self.reason = reason
        self.shapes_count = shapes_count
        super().__init__(
            f"Diagram classification failed: {reason} "
            f"(analyzed {shapes_count} shapes)",
            details={"reason": reason, "shapes_count": shapes_count},
        )


class ArrowDetectionError(CVError):
    """Raised when arrow detection fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Arrow detection failed during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )


class OCREngineError(CVError):
    """Raised when an OCR engine operation fails."""

    def __init__(self, engine: str, operation: str, reason: str):
        self.engine = engine
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"OCR engine '{engine}' failed during '{operation}': {reason}",
            details={"engine": engine, "operation": operation, "reason": reason},
        )


class TextAssociationError(CVError):
    """Raised when text association fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Text association failed during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )


class ReconstructionError(CVError):
    """Raised when diagram reconstruction fails."""

    def __init__(self, operation: str, reason: str):
        self.operation = operation
        self.reason = reason
        super().__init__(
            f"Reconstruction failed during '{operation}': {reason}",
            details={"operation": operation, "reason": reason},
        )

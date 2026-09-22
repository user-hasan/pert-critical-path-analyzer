"""
Abstract interfaces for the PERT & Critical Path Analyzer.

These interfaces define contracts between modules, enabling:
- Independent development and testing
- Swappable implementations
- Clear dependency boundaries
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pert_analyzer.core.models import (
    Activity,
    AnalysisResult,
    Arrow,
    DiagramAnalysis,
    DetectedShape,
    GraphModel,
    Node,
    OCRResult,
    ValidationResult,
)


class OCREngine(ABC):
    """Abstract interface for OCR engines."""

    @abstractmethod
    def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the OCR engine. Returns True if successful."""
        ...

    @abstractmethod
    def extract_text(self, image: np.ndarray) -> List[OCRResult]:
        """Extract text regions from an image."""
        ...

    @abstractmethod
    def extract_text_from_region(
        self, image: np.ndarray, region: Any
    ) -> List[OCRResult]:
        """Extract text from a specific region (bounding box or contour)."""
        ...

    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Return list of supported language codes."""
        ...

    @abstractmethod
    def get_confidence_threshold(self) -> float:
        """Return the minimum confidence threshold for results."""
        ...

    @abstractmethod
    def set_confidence_threshold(self, threshold: float) -> None:
        """Set the minimum confidence threshold for results."""
        ...

    @abstractmethod
    def is_initialized(self) -> bool:
        """Check if the engine is ready for use."""
        ...


class ShapeDetector(ABC):
    """Abstract interface for shape detection."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[DetectedShape]:
        """Detect shapes in the given image."""
        ...

    @abstractmethod
    def detect_in_region(
        self, image: np.ndarray, region: Any
    ) -> List[DetectedShape]:
        """Detect shapes within a specific region."""
        ...

    @abstractmethod
    def get_supported_shapes(self) -> List[str]:
        """Return list of supported shape types."""
        ...

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set detection parameters."""
        ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current detection parameters."""
        ...


class ArrowDetector(ABC):
    """Abstract interface for arrow/line detection."""

    @abstractmethod
    def detect(self, image: np.ndarray) -> List[Arrow]:
        """Detect arrows and lines in the image."""
        ...

    @abstractmethod
    def detect_connections(
        self,
        shapes: List[DetectedShape],
        arrows: List[Arrow],
    ) -> List[Dict[str, Any]]:
        """Detect connections between shapes via arrows."""
        ...

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set detection parameters."""
        ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current detection parameters."""
        ...


class DiagramClassifier(ABC):
    """Abstract interface for diagram type classification."""

    @abstractmethod
    def classify(
        self,
        shapes: List[DetectedShape],
        arrows: List[Arrow],
        ocr_results: Optional[List[OCRResult]] = None,
    ) -> Tuple[str, float]:
        """Classify the diagram type. Returns (type, confidence)."""
        ...

    @abstractmethod
    def get_supported_types(self) -> List[str]:
        """Return list of supported diagram types."""
        ...

    @abstractmethod
    def get_confidence(self) -> float:
        """Return the confidence of the last classification."""
        ...


class GraphBuilder(ABC):
    """Abstract interface for graph construction."""

    @abstractmethod
    def build_from_diagram(
        self, diagram: DiagramAnalysis
    ) -> GraphModel:
        """Build a graph model from diagram analysis results."""
        ...

    @abstractmethod
    def build_aon(
        self,
        nodes: List[Node],
        activities: List[Activity],
        dependencies: List[Dict[str, Any]],
    ) -> GraphModel:
        """Build an AON graph from explicit elements."""
        ...

    @abstractmethod
    def build_aoa(
        self,
        nodes: List[Node],
        activities: List[Activity],
        dependencies: List[Dict[str, Any]],
    ) -> GraphModel:
        """Build an AOA graph from explicit elements."""
        ...

    @abstractmethod
    def validate_graph(self, graph: GraphModel) -> ValidationResult:
        """Validate the structure of a graph."""
        ...


class AnalysisEngine(ABC):
    """Abstract interface for project analysis engines."""

    @abstractmethod
    def analyze(self, graph: GraphModel) -> AnalysisResult:
        """Perform analysis on the project graph."""
        ...

    @abstractmethod
    def get_analysis_type(self) -> str:
        """Return the type of analysis (e.g., 'CPM', 'PERT')."""
        ...

    @abstractmethod
    def can_analyze(self, graph: GraphModel) -> bool:
        """Check if this engine can analyze the given graph."""
        ...

    @abstractmethod
    def get_critical_path(self, graph: GraphModel) -> List[str]:
        """Extract the critical path from the graph."""
        ...

    @abstractmethod
    def get_all_critical_paths(self, graph: GraphModel) -> List[List[str]]:
        """Extract all critical paths from the graph."""
        ...


class NetworkVisualizer(ABC):
    """Abstract interface for network visualization."""

    @abstractmethod
    def render_static(
        self,
        graph: GraphModel,
        analysis: Optional[AnalysisResult] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Render a static image of the network."""
        ...

    @abstractmethod
    def render_to_file(
        self,
        graph: GraphModel,
        output_path: str,
        analysis: Optional[AnalysisResult] = None,
        **kwargs: Any,
    ) -> bool:
        """Render and save the network to a file."""
        ...

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Return list of supported output formats."""
        ...


class ImagePreprocessor(ABC):
    """Abstract interface for image preprocessing."""

    @abstractmethod
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess an image for analysis."""
        ...

    @abstractmethod
    def set_parameters(self, params: Dict[str, Any]) -> None:
        """Set preprocessing parameters."""
        ...

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get current preprocessing parameters."""
        ...


class PipelineStage(ABC):
    """Abstract interface for a pipeline processing stage."""

    @abstractmethod
    def process(self, data: Any) -> Any:
        """Process input data and return results."""
        ...

    @abstractmethod
    def get_stage_name(self) -> str:
        """Return the name of this pipeline stage."""
        ...

    @abstractmethod
    def get_stage_description(self) -> str:
        """Return a description of what this stage does."""
        ...

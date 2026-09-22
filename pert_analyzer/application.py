"""
Application orchestrator for the PERT & Critical Path Analyzer.

Coordinates pipeline stages, manages project state, and provides
the main interface between GUI and domain logic.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional

from pert_analyzer.config.manager import AppConfig, get_config
from pert_analyzer.core.models import (
    AnalysisResult,
    DiagramAnalysis,
    GraphModel,
    Project,
    ValidationResult,
)

logger = logging.getLogger(__name__)


class AnalysisOrchestrator:
    """
    Orchestrates the complete analysis pipeline.

    This class coordinates:
    - Image preprocessing
    - CV detection (shapes, arrows)
    - OCR text extraction
    - Diagram classification
    - Graph reconstruction
    - Validation
    - CPM/PERT analysis
    - Visualization
    """

    def __init__(self, config: Optional[AppConfig] = None):
        """Initialize the orchestrator with configuration."""
        self._config = config or get_config()
        self._project: Optional[Project] = None
        self._callbacks: Dict[str, List[Callable]] = {}

        # Pipeline components (initialized lazily)
        self._preprocessor = None
        self._shape_detector = None
        self._arrow_detector = None
        self._ocr_engine = None
        self._classifier = None
        self._graph_builder = None
        self._analysis_engine = None
        self._validator = None
        self._visualizer = None

        logger.info("AnalysisOrchestrator initialized")

    @property
    def project(self) -> Optional[Project]:
        """Get the current project."""
        return self._project

    @property
    def is_project_loaded(self) -> bool:
        """Check if a project is currently loaded."""
        return self._project is not None

    def create_project(
        self, name: str = "Untitled Project", description: str = ""
    ) -> Project:
        """Create a new project."""
        self._project = Project(name=name, description=description)
        logger.info("Created project: %s", name)
        self._emit("project_created", self._project)
        return self._project

    def load_project(self, project: Project) -> None:
        """Load an existing project."""
        self._project = project
        logger.info("Loaded project: %s", project.name)
        self._emit("project_loaded", self._project)

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register a callback for an event."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _emit(self, event: str, data: Any = None) -> None:
        """Emit an event to registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(data)
            except Exception as e:
                logger.error("Callback error for event %s: %s", event, e)

    def analyze_image(
        self,
        image_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
    ) -> Optional[Project]:
        """
        Run the complete analysis pipeline on an image.

        Args:
            image_path: Path to the input image.
            progress_callback: Optional callback(stage_name, progress_0_to_1).

        Returns:
            The updated project with analysis results, or None on failure.
        """
        if not self._project:
            self.create_project()

        start_time = time.time()

        try:
            # Stage 1: Load image
            self._report_progress(progress_callback, "Loading image", 0.0)
            image = self._load_image(image_path)
            if image is None:
                logger.error("Failed to load image: %s", image_path)
                return None

            # Initialize diagram analysis
            self._project.diagram = DiagramAnalysis(
                original_image=image,
                image_path=image_path,
            )

            # Stage 2: Preprocess image
            self._report_progress(progress_callback, "Preprocessing image", 0.1)
            preprocessed = self._preprocess_image(image)
            self._project.diagram.preprocessed_image = preprocessed

            # Stage 3: Detect shapes
            self._report_progress(progress_callback, "Detecting shapes", 0.2)
            shapes = self._detect_shapes(preprocessed)
            self._project.diagram.detected_shapes = shapes

            # Stage 4: Detect arrows
            self._report_progress(progress_callback, "Detecting arrows", 0.3)
            arrows = self._detect_arrows(preprocessed)
            self._project.diagram.detected_arrows = arrows

            # Stage 5: OCR
            self._report_progress(progress_callback, "Extracting text", 0.4)
            ocr_results = self._extract_text(preprocessed)
            self._project.diagram.ocr_results = ocr_results

            # Stage 6: Classify diagram type
            self._report_progress(progress_callback, "Classifying diagram", 0.5)
            diagram_type, confidence = self._classify_diagram(
                shapes, arrows, ocr_results
            )
            self._project.diagram.diagram_type = diagram_type
            self._project.diagram.detection_confidence = confidence

            # Stage 7: Build graph
            self._report_progress(progress_callback, "Building graph", 0.6)
            graph = self._build_graph(self._project.diagram)
            self._project.graph = graph

            # Stage 8: Validate
            self._report_progress(progress_callback, "Validating graph", 0.7)
            validation = self._validate_graph(graph)
            self._project.validation = validation

            # Stage 9: Analyze (if valid)
            if validation.is_valid:
                self._report_progress(
                    progress_callback, "Running analysis", 0.8
                )
                analysis = self._run_analysis(graph)
                self._project.analysis = analysis
            else:
                logger.warning(
                    "Graph validation failed with %d errors",
                    validation.error_count,
                )

            # Stage 10: Complete
            self._report_progress(progress_callback, "Complete", 1.0)
            self._project.status = "review"
            self._project.update_modified()

            elapsed = time.time() - start_time
            self._project.metadata["processing_time"] = elapsed
            logger.info(
                "Analysis completed in %.2f seconds", elapsed
            )

            self._emit("analysis_complete", self._project)
            return self._project

        except Exception as e:
            logger.error("Analysis failed: %s", e, exc_info=True)
            if self._project:
                self._project.status = "error"
                self._project.metadata["error"] = str(e)
            self._emit("analysis_error", str(e))
            return None

    def _report_progress(
        self,
        callback: Optional[Callable[[str, float], None]],
        stage: str,
        progress: float,
    ) -> None:
        """Report progress to the callback."""
        if callback:
            try:
                callback(stage, progress)
            except Exception:
                pass
        logger.debug("Progress: %s (%.0f%%)", stage, progress * 100)

    def _load_image(self, path: str) -> Any:
        """Load an image from path."""
        try:
            import cv2

            image = cv2.imread(path)
            if image is None:
                logger.error("cv2.imread returned None for: %s", path)
            return image
        except ImportError:
            logger.error("OpenCV not installed. Cannot load images.")
            return None

    def _preprocess_image(self, image: Any) -> Any:
        """Preprocess the image for analysis."""
        # Placeholder: will use ImagePreprocessor in Phase 3
        logger.info("Image preprocessing placeholder (Phase 3)")
        return image

    def _detect_shapes(self, image: Any) -> list:
        """Detect shapes in the image."""
        # Placeholder: will use ShapeDetector in Phase 4
        logger.info("Shape detection placeholder (Phase 4)")
        return []

    def _detect_arrows(self, image: Any) -> list:
        """Detect arrows in the image."""
        # Placeholder: will use ArrowDetector in Phase 5
        logger.info("Arrow detection placeholder (Phase 5)")
        return []

    def _extract_text(self, image: Any) -> list:
        """Extract text from the image."""
        # Placeholder: will use OCREngine in Phase 6
        logger.info("OCR extraction placeholder (Phase 6)")
        return []

    def _classify_diagram(
        self, shapes: list, arrows: list, ocr_results: list
    ) -> tuple:
        """Classify the diagram type."""
        # Placeholder: will use DiagramClassifier in Phase 7
        logger.info("Diagram classification placeholder (Phase 7)")
        return ("unknown", 0.0)

    def _build_graph(self, diagram: DiagramAnalysis) -> GraphModel:
        """Build a graph from diagram analysis."""
        # Placeholder: will use GraphBuilder in Phase 7
        logger.info("Graph building placeholder (Phase 7)")
        return GraphModel()

    def _validate_graph(self, graph: GraphModel) -> ValidationResult:
        """Validate the graph structure."""
        # Placeholder: will use validation engine in Phase 8
        logger.info("Graph validation placeholder (Phase 8)")
        return ValidationResult(is_valid=True, summary="Placeholder validation")

    def _run_analysis(self, graph: GraphModel) -> AnalysisResult:
        """Run CPM/PERT analysis on the graph."""
        # Placeholder: will use AnalysisEngine in Phase 1
        logger.info("Analysis engine placeholder (Phase 1)")
        return AnalysisResult()

    def get_project_summary(self) -> Dict[str, Any]:
        """Get a summary of the current project."""
        if not self._project:
            return {"status": "no_project"}

        summary = {
            "name": self._project.name,
            "status": self._project.status,
            "diagram_type": "unknown",
            "node_count": 0,
            "activity_count": 0,
            "dependency_count": 0,
            "has_analysis": self._project.analysis is not None,
            "is_valid": (
                self._project.validation.is_valid
                if self._project.validation
                else False
            ),
        }

        if self._project.diagram:
            summary["diagram_type"] = self._project.diagram.diagram_type
            summary["shape_count"] = self._project.diagram.shape_count
            summary["arrow_count"] = self._project.diagram.arrow_count
            summary["ocr_count"] = self._project.diagram.ocr_result_count

        if self._project.graph:
            summary["node_count"] = self._project.graph.node_count
            summary["activity_count"] = self._project.graph.activity_count
            summary["dependency_count"] = self._project.graph.dependency_count

        return summary

    def reset(self) -> None:
        """Reset the orchestrator and clear the current project."""
        self._project = None
        logger.info("Orchestrator reset")
        self._emit("project_reset")

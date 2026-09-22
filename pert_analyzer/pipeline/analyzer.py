"""
End-to-end analysis pipeline orchestrator.

Coordinates all pipeline stages from image loading through CPM analysis.
Preserves intermediate results and provides structured error handling.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Callable, List, Optional

from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.core.models import DiagramType, GraphModel
from pert_analyzer.cv.arrow_detection import ArrowDetector
from pert_analyzer.cv.classification import DiagramClassifier
from pert_analyzer.cv.exceptions import CVError
from pert_analyzer.cv.models import ArrowDetectionResult, ShapeDetectionResult
from pert_analyzer.cv.ocr_engine import OCREngineError, create_ocr_engine
from pert_analyzer.cv.ocr_models import OCRProcessingResult
from pert_analyzer.cv.preprocessing import ImagePreprocessor
from pert_analyzer.cv.reconstruction import ReconstructionEngine
from pert_analyzer.cv.reconstruction_models import ReconstructedDiagram
from pert_analyzer.cv.shape_detection import ShapeDetector
from pert_analyzer.cv.spatial_association import SpatialAssociator
from pert_analyzer.graph.builder import GraphBuilder
from pert_analyzer.pipeline.human_review import (
    HumanReviewResult,
    ReviewIssue,
    ReviewIssueType,
)
from pert_analyzer.pipeline.progress import StageProgress
from pert_analyzer.pipeline.result import (
    AnalysisStatus,
    ImageMetadata,
    PipelineResult,
)

logger = logging.getLogger(__name__)


class EndToEndAnalyzer:
    """
    End-to-end image analysis pipeline.

    Coordinates preprocessing, shape detection, arrow detection,
    OCR, reconstruction, validation, and CPM analysis.

    Usage:
        analyzer = EndToEndAnalyzer()
        result = analyzer.analyze("path/to/diagram.png")
        print(result.to_summary_string())
    """

    def __init__(
        self,
        ocr_engine_name: str = "tesseract",
        export_debug: bool = False,
        debug_output_dir: str = "analysis_output",
        ocr_languages: Optional[list] = None,
        tesseract_path: Optional[str] = None,
        ocr_psm: int = 11,
        ocr_oem: int = 3,
    ):
        """
        Initialize the analyzer.

        Args:
            ocr_engine_name: OCR engine to use ("mock", "tesseract").
            export_debug: Whether to export debug artifacts.
            debug_output_dir: Directory for debug artifacts.
            ocr_languages: Tesseract language codes (default ["eng"]).
            tesseract_path: Optional path to tesseract executable.
            ocr_psm: Tesseract page segmentation mode (default 11 = sparse text).
            ocr_oem: Tesseract OCR engine mode (default 3).
        """
        self.ocr_engine_name = ocr_engine_name
        self.export_debug = export_debug
        self.debug_output_dir = debug_output_dir
        self._ocr_languages = ocr_languages or ["eng"]
        self._ocr_psm = ocr_psm
        self._ocr_oem = ocr_oem

        # Auto-detect Tesseract path if not provided
        if tesseract_path:
            self._tesseract_path = tesseract_path
        else:
            self._tesseract_path = self._detect_tesseract_path()

        # Pipeline components
        self._preprocessor = ImagePreprocessor()
        self._shape_detector = ShapeDetector()
        self._classifier = DiagramClassifier()
        self._arrow_detector = ArrowDetector()
        self._ocr_engine = None
        self._spatial_associator = SpatialAssociator()
        self._reconstruction_engine = ReconstructionEngine()
        self._cpm_engine = CPMEngine()

        # Lazy OCR init
        self._ocr_initialized = False

    @staticmethod
    def _detect_tesseract_path() -> Optional[str]:
        """Auto-detect Tesseract executable path.

        Checks common installation locations on Windows/Linux/macOS.
        Returns the path if found, None otherwise.
        """
        import os
        import shutil

        # Check if tesseract is in PATH
        in_path = shutil.which("tesseract")
        if in_path:
            return in_path

        # Common Windows installation paths
        windows_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            os.path.expanduser(r"~\AppData\Local\Tesseract-OCR\tesseract.exe"),
        ]

        for path in windows_paths:
            if os.path.isfile(path):
                return path

        # Common Linux paths
        linux_paths = [
            "/usr/bin/tesseract",
            "/usr/local/bin/tesseract",
            "/snap/bin/tesseract",
        ]

        for path in linux_paths:
            if os.path.isfile(path):
                return path

        return None

    def _ensure_ocr(self) -> None:
        """Initialize OCR engine if not already done."""
        if self._ocr_initialized:
            return
        try:
            ocr_config = {
                "languages": self._ocr_languages,
                "psm": self._ocr_psm,
                "oem": self._ocr_oem,
            }
            if self._tesseract_path:
                ocr_config["tesseract_path"] = self._tesseract_path
            self._ocr_engine = create_ocr_engine(self.ocr_engine_name, ocr_config)
            self._ocr_initialized = True
        except (OCREngineError, Exception) as e:
            logger.warning("OCR engine '%s' unavailable: %s", self.ocr_engine_name, e)
            self._ocr_engine = None
            self._ocr_initialized = True

    def analyze(
        self,
        image_path: str,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        stage_callback: Optional[Callable[[StageProgress], None]] = None,
    ) -> PipelineResult:
        """
        Run the complete analysis pipeline on an image.

        Args:
            image_path: Path to the diagram image.
            progress_callback: Optional callback(stage_name, progress_0_to_1).
            stage_callback: Optional structured callback receiving a
                ``StageProgress`` event for each real stage transition.
                Rich (but real) metrics are attached where a stage result
                is already available; plain ``progress_callback`` callers
                never see those extras, so their values are unchanged.

        Returns:
            PipelineResult with all intermediate results and status.
        """
        result = PipelineResult()
        pipeline_start = time.time()

        def report(
            stage: str,
            progress: float,
            *,
            state: str = "RUNNING",
            message: str = "",
            metrics: Optional[dict] = None,
        ) -> None:
            if progress_callback:
                try:
                    progress_callback(stage, progress)
                except Exception:
                    pass
            if stage_callback:
                try:
                    stage_callback(
                        StageProgress(
                            stage_id=stage,
                            state=state,
                            progress=progress,
                            message=message,
                            metrics=dict(metrics or {}),
                        )
                    )
                except Exception:
                    pass

        # Stage 1: Load and validate image
        report("Loading image", 0.0)
        image = self._stage_load_image(image_path, result)
        if image is None:
            result.status = AnalysisStatus.FAILED
            result.total_time = time.time() - pipeline_start
            return result

        # Stage 2: Preprocess
        report("Preprocessing", 0.1)
        prep_result = self._stage_preprocess(image, result)
        if prep_result is None:
            result.status = AnalysisStatus.FAILED
            result.total_time = time.time() - pipeline_start
            return result

        # Stage 3: Detect shapes
        report("Detecting shapes", 0.2)
        shape_result = self._stage_detect_shapes(prep_result, result)
        report(
            "Detecting shapes",
            0.2,
            state="COMPLETED",
            metrics={"shapes": result.shape_count},
        )

        # Stage 4: Classify diagram
        report("Classifying diagram", 0.3)
        self._stage_classify(shape_result, result)

        # Stage 5: Detect arrows
        report("Detecting arrows", 0.4)
        arrow_result = self._stage_detect_arrows(prep_result, shape_result, result)
        report(
            "Detecting arrows",
            0.4,
            state="COMPLETED",
            metrics={"arrows": result.arrow_count},
        )

        # Stage 6: OCR
        report("Extracting text (OCR)", 0.5)
        ocr_result = self._stage_ocr(image, result)
        report(
            "Extracting text (OCR)",
            0.5,
            state="COMPLETED",
            metrics={"ocr": result.ocr_region_count},
        )

        # Stage 7: Associate text
        report("Associating text", 0.6)
        assoc_result = self._stage_associate_text(
            shape_result, arrow_result, ocr_result, result
        )

        # Stage 8: Reconstruct semantics
        report("Reconstructing diagram", 0.7)
        reconstruction = self._stage_reconstruct(
            shape_result, arrow_result, ocr_result, assoc_result, result
        )
        report(
            "Reconstructing diagram",
            0.7,
            state="COMPLETED",
            metrics={"activities": result.reconstructed_activity_count},
        )

        # Stage 9: Convert to GraphModel
        report("Building graph", 0.8)
        graph = self._stage_build_graph(reconstruction, result)

        # Stage 10: Validate and run CPM
        report("Validating and analyzing", 0.9)
        self._stage_validate_and_cpm(graph, reconstruction, result)

        # Stage 11: Human review analysis
        self._stage_review_analysis(reconstruction, result)

        # Final status classification (never overwritten after this point)
        result.status = self._classify_final_status(result)

        # Export debug artifacts if requested
        if self.export_debug:
            self._export_debug_artifacts(image_path, result)

        result.total_time = time.time() - pipeline_start
        report("Complete", 1.0)

        logger.info(
            "Pipeline complete: status=%s, duration=%.3fs",
            result.status.value,
            result.total_time,
        )

        return result

    # =========================================================================
    # Pipeline Stages
    # =========================================================================

    def _stage_load_image(
        self, image_path: str, result: PipelineResult
    ) -> Optional[Any]:
        """Stage 1: Load and validate the input image."""
        stage_start = time.time()
        try:
            if not os.path.exists(image_path):
                result.add_error(f"Image file not found: {image_path}")
                result.set_stage("load_image", "FAILED", error=f"File not found: {image_path}")
                return None

            import cv2

            image = cv2.imread(image_path)
            if image is None:
                result.add_error(f"Failed to load image: {image_path}")
                result.set_stage("load_image", "FAILED", error="cv2.imread returned None")
                return None

            result.image_metadata = ImageMetadata.from_image(image, image_path)
            result.set_stage("load_image", "SUCCESS")
            result._original_image = image
            return image

        except ImportError:
            result.add_error("OpenCV (cv2) is not installed")
            result.set_stage("load_image", "FAILED", error="OpenCV not installed")
            return None
        except Exception as e:
            result.add_error(f"Error loading image: {e}")
            result.set_stage("load_image", "FAILED", error=str(e))
            return None
        finally:
            result.stages["load_image"].timing.end_time = time.time()
            result.stages["load_image"].timing.start_time = stage_start

    def _stage_preprocess(
        self, image: Any, result: PipelineResult
    ) -> Optional[Any]:
        """Stage 2: Preprocess the image."""
        stage_start = time.time()
        try:
            prep_result = self._preprocessor.process_image(image)
            result._preprocessing_result = prep_result
            result.set_stage("preprocess", "SUCCESS")
            return prep_result
        except CVError as e:
            result.add_warning(f"Preprocessing warning: {e}")
            result.set_stage("preprocess", "WARNING", warning=str(e))
            return None
        except Exception as e:
            result.add_error(f"Preprocessing failed: {e}")
            result.set_stage("preprocess", "FAILED", error=str(e))
            return None
        finally:
            result.stages["preprocess"].timing.end_time = time.time()
            result.stages["preprocess"].timing.start_time = stage_start

    def _stage_detect_shapes(
        self, prep_result: Any, result: PipelineResult
    ) -> Optional[ShapeDetectionResult]:
        """Stage 3: Detect shapes in the preprocessed image."""
        stage_start = time.time()
        try:
            shape_result = self._shape_detector.detect_from_preprocessing(prep_result)
            result._shape_result = shape_result
            result.shape_count = len(shape_result.candidate_nodes)
            result.set_stage("shape_detection", "SUCCESS")
            return shape_result
        except Exception as e:
            result.add_warning(f"Shape detection failed: {e}")
            result.set_stage("shape_detection", "WARNING", warning=str(e))
            return None
        finally:
            result.stages["shape_detection"].timing.end_time = time.time()
            result.stages["shape_detection"].timing.start_time = stage_start

    def _stage_classify(
        self,
        shape_result: Optional[ShapeDetectionResult],
        result: PipelineResult,
    ) -> None:
        """Stage 4: Classify the diagram type."""
        stage_start = time.time()
        try:
            if shape_result is None:
                result.diagram_type = "UNKNOWN"
                result.set_stage("classification", "SKIPPED", warning="No shapes to classify")
                return

            classification = self._classifier.classify_from_detection(shape_result)
            result._classification_result = classification
            result.diagram_type = classification.diagram_type
            result.diagram_confidence = classification.confidence
            result.set_stage("classification", "SUCCESS")
        except Exception as e:
            result.diagram_type = "UNKNOWN"
            result.set_stage("classification", "WARNING", warning=str(e))
        finally:
            result.stages["classification"].timing.end_time = time.time()
            result.stages["classification"].timing.start_time = stage_start

    def _stage_detect_arrows(
        self,
        prep_result: Any,
        shape_result: Optional[ShapeDetectionResult],
        result: PipelineResult,
    ) -> Optional[ArrowDetectionResult]:
        """Stage 5: Detect arrows."""
        stage_start = time.time()
        try:
            if prep_result is None or shape_result is None:
                result.set_stage("arrow_detection", "SKIPPED", warning="Missing preprocessing or shapes")
                return None

            arrow_result = self._arrow_detector.detect_from_preprocessing(prep_result, shape_result)
            result._arrow_result = arrow_result
            result.arrow_count = arrow_result.arrow_count
            result.set_stage("arrow_detection", "SUCCESS")
            return arrow_result
        except Exception as e:
            result.add_warning(f"Arrow detection failed: {e}")
            result.set_stage("arrow_detection", "WARNING", warning=str(e))
            return None
        finally:
            result.stages["arrow_detection"].timing.end_time = time.time()
            result.stages["arrow_detection"].timing.start_time = stage_start

    def _stage_ocr(
        self, image: Any, result: PipelineResult
    ) -> Optional[OCRProcessingResult]:
        """Stage 6: Run full-image OCR + region-based node OCR, merge results."""
        stage_start = time.time()
        try:
            self._ensure_ocr()
            if self._ocr_engine is None:
                result.add_warning(
                    "OCR_ENGINE_UNAVAILABLE: Tesseract OCR is not installed or "
                    "its executable path is not configured."
                )
                result.set_stage("ocr", "SKIPPED", warning="OCR_ENGINE_UNAVAILABLE")
                return None

            import numpy as np

            if not isinstance(image, np.ndarray):
                result.set_stage("ocr", "WARNING", warning="Image is not a numpy array")
                return None

            # --- Full-image OCR (catches START, FINISH, titles, non-node text) ---
            full_regions = self._ocr_engine.recognize(image)
            ocr_result = OCRProcessingResult(
                regions=full_regions,
                engine=self._ocr_engine.get_engine_name(),
                image_dimensions=(image.shape[1], image.shape[0]),
            )

            # Text normalization
            from pert_analyzer.cv.text_normalization import TextNormalizer
            normalizer = TextNormalizer()
            for region in ocr_result.regions:
                raw, normalized = normalizer.normalize_preserve_raw(region.text)
                region.raw_text = raw
                region.normalized_text = normalized

            # Numeric extraction
            from pert_analyzer.cv.numeric_extraction import NumericExtractor
            extractor = NumericExtractor()
            ocr_result.numeric_candidates = extractor.extract_from_regions(ocr_result.regions)
            for region in ocr_result.regions:
                if normalizer.is_numeric_context(region.text):
                    region.is_numeric = True
                    candidates = extractor.extract_from_region(region)
                    if candidates:
                        region.parsed_value = candidates[0].value

            # Text classification
            from pert_analyzer.cv.text_classification import TextClassifier
            classifier = TextClassifier()
            classifier.classify_regions(ocr_result.regions)

            # --- Region-based OCR for detected candidate nodes ---
            shape_result = getattr(result, '_shape_result', None)
            region_ocr_results = None
            if shape_result and shape_result.candidate_nodes:
                from pert_analyzer.cv.region_ocr import RegionOCRProcessor, merge_ocr_results
                region_processor = RegionOCRProcessor(self._ocr_engine)
                region_results = region_processor.process_node_regions(image, shape_result.candidate_nodes)
                region_ocr_results = region_results
                ocr_result = merge_ocr_results(ocr_result, region_results)

                # Re-normalize and classify merged regions
                for region in ocr_result.regions:
                    raw, normalized = normalizer.normalize_preserve_raw(region.text)
                    region.raw_text = raw
                    region.normalized_text = normalized
                classifier.classify_regions(ocr_result.regions)

                result.ocr_region_count = ocr_result.region_count

            result._ocr_result = ocr_result
            result._region_ocr_results = region_ocr_results
            result.ocr_region_count = ocr_result.region_count
            result.set_stage("ocr", "SUCCESS")
            return ocr_result
        except OCREngineError as e:
            result.add_warning(f"OCR failed: {e}")
            result.set_stage("ocr", "WARNING", warning=str(e))
            return None
        except Exception as e:
            result.add_warning(f"OCR error: {e}")
            result.set_stage("ocr", "WARNING", warning=str(e))
            return None
        finally:
            result.stages["ocr"].timing.end_time = time.time()
            result.stages["ocr"].timing.start_time = stage_start

    def _stage_associate_text(
        self,
        shape_result: Optional[ShapeDetectionResult],
        arrow_result: Optional[ArrowDetectionResult],
        ocr_result: Optional[OCRProcessingResult],
        result: PipelineResult,
    ) -> Optional[Any]:
        """Stage 7: Associate text with shapes and arrows."""
        stage_start = time.time()
        try:
            if ocr_result is None or not ocr_result.regions:
                result.set_stage("association", "SKIPPED", warning="No OCR results")
                return None

            text_regions = ocr_result.regions
            candidates = shape_result.candidate_nodes if shape_result else []
            arrows = arrow_result.arrows if arrow_result else []

            assoc_results = self._spatial_associator.associate_all(
                text_regions, candidates, arrows
            )
            result._association_results = assoc_results
            result.set_stage("association", "SUCCESS")
            return assoc_results
        except Exception as e:
            result.add_warning(f"Text association failed: {e}")
            result.set_stage("association", "WARNING", warning=str(e))
            return None
        finally:
            result.stages["association"].timing.end_time = time.time()
            result.stages["association"].timing.start_time = stage_start

    def _stage_reconstruct(
        self,
        shape_result: Optional[ShapeDetectionResult],
        arrow_result: Optional[ArrowDetectionResult],
        ocr_result: Optional[OCRProcessingResult],
        assoc_result: Optional[Any],
        result: PipelineResult,
    ) -> Optional[ReconstructedDiagram]:
        """Stage 8: Reconstruct semantic diagram."""
        stage_start = time.time()
        try:
            if shape_result is None:
                result.add_error("Cannot reconstruct: no shapes detected")
                result.set_stage("reconstruction", "FAILED", error="No shapes detected")
                return None

            diagram_type = result.diagram_type.upper()

            # Combine List[TextAssociationResult] into a single TextAssociationResult
            combined_assoc = assoc_result
            if isinstance(assoc_result, list) and assoc_result:
                from pert_analyzer.cv.ocr_models import TextAssociation
                all_associations = []
                for r in assoc_result:
                    if hasattr(r, 'associations'):
                        all_associations.extend(r.associations)
                all_associations.sort(key=lambda a: a.association_score, reverse=True)
                if all_associations:
                    all_associations[0].is_best_candidate = True
                combined_assoc = type(assoc_result[0])(
                    text_region_id="combined",
                    associations=all_associations,
                    best_association=all_associations[0] if all_associations else None,
                    is_ambiguous=len(all_associations) > 1,
                    has_no_match=len(all_associations) == 0,
                )

            if diagram_type == "AOA":
                reconstruction = self._reconstruction_engine.reconstruct_aoa(
                    shape_result, arrow_result, ocr_result, combined_assoc
                )
            else:
                region_ocr_results = getattr(result, '_region_ocr_results', None)
                reconstruction = self._reconstruction_engine.reconstruct_aon(
                    shape_result, arrow_result, ocr_result, combined_assoc,
                    region_ocr_results=region_ocr_results,
                )

            result._reconstruction = reconstruction
            result.reconstructed_activity_count = reconstruction.activity_count
            result.reconstructed_event_count = reconstruction.event_count
            result.dependency_count = reconstruction.dependency_count
            result.set_stage("reconstruction", "SUCCESS")
            return reconstruction
        except Exception as e:
            result.add_error(f"Reconstruction failed: {e}")
            result.set_stage("reconstruction", "FAILED", error=str(e))
            return None
        finally:
            result.stages["reconstruction"].timing.end_time = time.time()
            result.stages["reconstruction"].timing.start_time = stage_start

    def _stage_build_graph(
        self,
        reconstruction: Optional[ReconstructedDiagram],
        result: PipelineResult,
    ) -> Optional[GraphModel]:
        """Stage 9: Convert reconstruction to GraphModel."""
        stage_start = time.time()
        try:
            if reconstruction is None:
                result.set_stage("graph_build", "SKIPPED", warning="No reconstruction")
                return None

            blockers = self._collect_graph_blockers(reconstruction)
            if blockers:
                preview = ", ".join(blockers[:3])
                if len(blockers) > 3:
                    preview += f" (+{len(blockers) - 3} more)"
                result.review_required = True
                result.set_stage(
                    "graph_build",
                    "REVIEW_REQUIRED",
                    warning=preview,
                )
                return None

            graph = reconstruction.to_graph_model()
            result._graph_model = graph
            result.set_stage("graph_build", "SUCCESS")
            return graph
        except Exception as e:
            result.add_error(f"Graph build failed: {e}")
            result.set_stage("graph_build", "FAILED", error=str(e))
            return None
        finally:
            result.stages["graph_build"].timing.end_time = time.time()
            result.stages["graph_build"].timing.start_time = stage_start

    def _collect_graph_blockers(
        self, reconstruction: Optional[ReconstructedDiagram]
    ) -> List[str]:
        """List reconstruction defects that defer graph construction to review.

        These are human-reviewable problems (invalid/zero durations, missing
        or duplicate activity IDs) that would otherwise crash graph building.
        They surface as REVIEW_REQUIRED rather than fatal pipeline errors.
        """
        blockers: List[str] = []
        if reconstruction is None:
            return blockers

        seen_ids = set()
        for act in reconstruction.activities:
            aid = (act.activity_id or "").strip()
            if not aid:
                blockers.append(
                    f"Activity at geometric node "
                    f"'{act.geometric_node_id or '?'}' is missing an identifier; "
                    "a corrected ID is required before CPM can run."
                )
            else:
                if aid in seen_ids:
                    blockers.append(
                        f"Duplicate activity identifier '{aid}'; a unique ID is "
                        "required before CPM can run."
                    )
                seen_ids.add(aid)

            if not act.is_dummy and act.duration is not None and act.duration <= 0:
                blockers.append(
                    f"Activity '{aid or act.geometric_node_id or '?'}' has "
                    f"invalid duration {act.duration}; a valid duration is "
                    "required before CPM can run."
                )
        return blockers

    def _classify_final_status(self, result: PipelineResult) -> AnalysisStatus:
        """Determine the final pipeline status from stage outcomes.

        Rules:
          - Fatal errors (load, preprocessing, catastrophic exceptions)
            -> FAILED.
          - Any reconstruction defect flagged for review, or any stage in
            REVIEW_REQUIRED -> REVIEW_REQUIRED.
          - A validated graph with CPM results -> SUCCESS.
          - Otherwise -> FAILED (never silently succeed).
        """
        if result.errors:
            return AnalysisStatus.FAILED
        if result.review_required or any(
            stage.status == "REVIEW_REQUIRED" for stage in result.stages.values()
        ):
            return AnalysisStatus.REVIEW_REQUIRED
        if result.validation_passed:
            return AnalysisStatus.SUCCESS
        return AnalysisStatus.FAILED

    def _stage_validate_and_cpm(
        self,
        graph: Optional[GraphModel],
        reconstruction: Optional[ReconstructedDiagram],
        result: PipelineResult,
    ) -> None:
        """Stage 10: Validate graph and run CPM."""
        stage_start = time.time()
        try:
            if graph is None:
                result.set_stage("validation_cpm", "SKIPPED", warning="No graph")
                return

            # Check diagram type — AOA CPM not yet supported
            if graph.diagram_type == DiagramType.AOA:
                result.review_required = True
                result.review_issues.append({
                    "type": "NOT_SUPPORTED_FOR_CPM",
                    "description": "AOA reconstruction succeeded but CPM transformation is not yet supported for AOA representation",
                })
                result.set_stage("validation_cpm", "REVIEW_REQUIRED")
                return

            # Validate the graph
            if not graph.activities:
                result.validation_passed = False
                result.validation_errors.append("No activities in graph")
                result.set_stage("validation_cpm", "FAILED", error="No activities")
                return

            # Check for missing durations
            missing_duration = [
                aid for aid, act in graph.activities.items()
                if act.duration <= 0 and not act.is_dummy
            ]
            if missing_duration:
                result.validation_warnings.append(
                    f"Activities missing duration: {missing_duration}"
                )

            # Run CPM
            cpm_result = self._cpm_engine.analyze(graph)
            result._cpm_result = cpm_result
            result.cpm_project_duration = cpm_result.project_duration
            result.cpm_critical_path = cpm_result.critical_path
            result.cpm_critical_paths = cpm_result.critical_paths
            result.cpm_critical_activity_count = sum(
                1 for aa in cpm_result.activity_analyses.values()
                if aa.is_critical
            )
            result.validation_passed = True
            result.set_stage("validation_cpm", "SUCCESS")

        except Exception as e:
            result.validation_passed = False
            result.validation_errors.append(str(e))
            result.set_stage("validation_cpm", "FAILED", error=str(e))
        finally:
            result.stages["validation_cpm"].timing.end_time = time.time()
            result.stages["validation_cpm"].timing.start_time = stage_start

    def _stage_review_analysis(
        self,
        reconstruction: Optional[ReconstructedDiagram],
        result: PipelineResult,
    ) -> None:
        """Analyze reconstruction for human review requirements."""
        if reconstruction is None:
            return

        issues = []

        # Check for low confidence activities
        for act in reconstruction.activities:
            if act.confidence < 0.5:
                issues.append({
                    "type": ReviewIssueType.LOW_SHAPE_CONFIDENCE.value,
                    "description": f"Activity '{act.activity_id}' has low confidence ({act.confidence:.3f})",
                    "element_id": act.activity_id,
                    "element_type": "activity",
                })

            if not act.source_text_region_ids:
                issues.append({
                    "type": ReviewIssueType.UNCERTAIN_ACTIVITY_ID.value,
                    "description": f"Activity '{act.activity_id}' has no OCR text evidence",
                    "element_id": act.activity_id,
                    "element_type": "activity",
                })

        # Check for uncertain durations
        for act in reconstruction.activities:
            if not act.is_dummy and act.duration <= 0:
                issues.append({
                    "type": ReviewIssueType.MISSING_DURATION.value,
                    "description": f"Activity '{act.activity_id}' has no duration",
                    "element_id": act.activity_id,
                    "element_type": "activity",
                })

        # Check reconstruction ambiguities
        for amb in reconstruction.ambiguities:
            issues.append({
                "type": amb.ambiguity_type.value,
                "description": amb.description,
                "element_id": ", ".join(amb.involved_ids) if amb.involved_ids else "",
                "element_type": "ambiguous",
            })

        if issues:
            result.review_required = True
            result.review_issues.extend(issues)

    # =========================================================================
    # Debug Export
    # =========================================================================

    def _export_debug_artifacts(
        self, image_path: str, result: PipelineResult
    ) -> None:
        """Export debug images and intermediate results."""
        try:
            from pert_analyzer.pipeline.debug_export import DebugExporter

            exporter = DebugExporter(self.debug_output_dir)
            exporter.export_all(image_path, result)
        except Exception as e:
            logger.warning("Debug export failed: %s", e)

"""
Debug artifact export for pipeline diagnostics.

Exports intermediate images, detection overlays, and pipeline
results for debugging CV failures.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Optional

logger = logging.getLogger(__name__)


class DebugExporter:
    """
    Exports debug artifacts from the analysis pipeline.

    Creates numbered intermediate images and a result JSON file
    for diagnosing CV failures.
    """

    def __init__(self, output_dir: str = "analysis_output"):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory to write debug artifacts.
        """
        self.output_dir = output_dir

    def export_all(
        self, image_path: str, result: Any
    ) -> str:
        """
        Export all debug artifacts.

        Args:
            image_path: Original image path.
            result: PipelineResult with intermediate data.

        Returns:
            Path to the output directory.
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Export original image
        self._export_image("01_original.png", result._original_image)

        # Export preprocessed image
        if result._preprocessing_result is not None:
            self._export_preprocessed(result._preprocessing_result)

        # Export shape detection overlay
        if result._shape_result is not None and result._original_image is not None:
            self._export_shape_overlay(result._original_image, result._shape_result)

        # Export arrow detection overlay
        if result._arrow_result is not None and result._original_image is not None:
            self._export_arrow_overlay(result._original_image, result._arrow_result)

        # Export OCR overlay
        if result._ocr_result is not None and result._original_image is not None:
            self._export_ocr_overlay(result._original_image, result._ocr_result)

        # Export result JSON
        self._export_result_json(result)

        logger.info("Debug artifacts exported to: %s", self.output_dir)
        return self.output_dir

    def _export_image(self, filename: str, image: Any) -> None:
        """Export a single image."""
        if image is None:
            return
        try:
            import cv2

            filepath = os.path.join(self.output_dir, filename)
            cv2.imwrite(filepath, image)
        except Exception as e:
            logger.warning("Failed to export %s: %s", filename, e)

    def _export_preprocessed(self, prep_result: Any) -> None:
        """Export preprocessed image representations."""
        try:
            import cv2

            representations = {
                "02_grayscale.png": "grayscale",
                "03_binary.png": "binary",
                "04_edges.png": "edges",
            }

            for filename, repr_name in representations.items():
                img = prep_result.get_representation(repr_name)
                if img is not None:
                    filepath = os.path.join(self.output_dir, filename)
                    cv2.imwrite(filepath, img)
        except Exception as e:
            logger.warning("Failed to export preprocessed images: %s", e)

    def _export_shape_overlay(
        self, image: Any, shape_result: Any
    ) -> None:
        """Export image with shape detection overlay."""
        try:
            import cv2
            import numpy as np

            overlay = image.copy()
            for candidate in shape_result.candidate_nodes:
                bbox = candidate.bounding_box
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                color = (0, 255, 0) if "RECT" in str(candidate.shape_type).upper() else (255, 0, 0)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 2)
                label = f"{candidate.shape_type.value}: {candidate.confidence:.2f}"
                cv2.putText(overlay, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            filepath = os.path.join(self.output_dir, "05_shapes.png")
            cv2.imwrite(filepath, overlay)
        except Exception as e:
            logger.warning("Failed to export shape overlay: %s", e)

    def _export_arrow_overlay(
        self, image: Any, arrow_result: Any
    ) -> None:
        """Export image with arrow detection overlay."""
        try:
            import cv2

            overlay = image.copy()
            for arrow in arrow_result.arrows:
                start = (int(arrow.start.x), int(arrow.start.y))
                end = (int(arrow.end.x), int(arrow.end.y))
                cv2.arrowedLine(overlay, start, end, (0, 0, 255), 2, tipLength=0.3)
                mid = arrow.midpoint
                label = f"{arrow.confidence:.2f}"
                cv2.putText(overlay, label, (int(mid.x), int(mid.y)),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            filepath = os.path.join(self.output_dir, "06_arrows.png")
            cv2.imwrite(filepath, overlay)
        except Exception as e:
            logger.warning("Failed to export arrow overlay: %s", e)

    def _export_ocr_overlay(
        self, image: Any, ocr_result: Any
    ) -> None:
        """Export image with OCR text overlay."""
        try:
            import cv2

            overlay = image.copy()
            for region in ocr_result.regions:
                bbox = region.bounding_box
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 255, 0), 1)
                text = f"{region.raw_text} ({region.confidence:.2f})"
                cv2.putText(overlay, text, (x, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

            filepath = os.path.join(self.output_dir, "07_ocr.png")
            cv2.imwrite(filepath, overlay)
        except Exception as e:
            logger.warning("Failed to export OCR overlay: %s", e)

    def _export_result_json(self, result: Any) -> None:
        """Export the pipeline result as JSON."""
        try:
            filepath = os.path.join(self.output_dir, "result.json")
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(result.to_json(indent=2))
        except Exception as e:
            logger.warning("Failed to export result JSON: %s", e)

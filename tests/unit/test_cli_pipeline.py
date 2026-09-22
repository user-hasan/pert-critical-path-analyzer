"""
Tests for the CLI entry point.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult


# =============================================================================
# CLI main() Tests
# =============================================================================


class TestCLIMain:
    def test_no_command_returns_1(self) -> None:
        from pert_analyzer.pipeline.__main__ import main

        with patch("sys.argv", ["pert_analyzer.pipeline"]):
            exit_code = main()
            assert exit_code == 1


# =============================================================================
# CLI analyze-image Tests
# =============================================================================


class TestCLIAnalyzeImage:
    def test_nonexistent_file_returns_1(self) -> None:
        from pert_analyzer.pipeline.__main__ import main

        with patch("sys.argv", ["pert_analyzer.pipeline", "analyze-image", "/tmp/nonexistent_xyz.png"]):
            exit_code = main()
            assert exit_code == 1

    def test_success_returns_0(self) -> None:
        from pert_analyzer.pipeline.__main__ import run_analyze_image

        fd, path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.imwrite(path, img)
            except ImportError:
                pass

            args = MagicMock()
            args.image_path = path
            args.output = None
            args.debug = False
            args.debug_dir = "analysis_output"
            args.ocr_engine = "mock"
            args.json = False

            with patch("pert_analyzer.pipeline.analyzer.EndToEndAnalyzer") as MockAnalyzer:
                mock_result = PipelineResult()
                mock_result.status = AnalysisStatus.SUCCESS
                mock_result.diagram_type = "AON"
                mock_result.diagram_confidence = 0.9
                mock_result.shape_count = 5
                mock_result.arrow_count = 4
                mock_result.reconstructed_activity_count = 3
                mock_result.cpm_project_duration = 12.0
                mock_result.cpm_critical_path = ["A", "B", "C"]
                mock_result.critical_paths = [["A", "B", "C"]]
                mock_result.cpm_critical_activity_count = 3
                mock_result.validation_passed = True
                mock_result.review_required = False
                mock_result.review_issues = []
                mock_result.warnings = []
                mock_result.errors = []
                mock_result.total_time = 0.5

                mock_instance = MockAnalyzer.return_value
                mock_instance.analyze.return_value = mock_result

                exit_code = run_analyze_image(args)
                assert exit_code == 0
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_output_file_written(self) -> None:
        from pert_analyzer.pipeline.__main__ import run_analyze_image

        fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        fd, out_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(out_path)

        try:
            img = np.zeros((100, 100, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.imwrite(img_path, img)
            except ImportError:
                pass

            args = MagicMock()
            args.image_path = img_path
            args.output = out_path
            args.debug = False
            args.debug_dir = "analysis_output"
            args.ocr_engine = "mock"
            args.json = False

            with patch("pert_analyzer.pipeline.analyzer.EndToEndAnalyzer") as MockAnalyzer:
                mock_result = PipelineResult()
                mock_result.status = AnalysisStatus.SUCCESS
                mock_result.diagram_type = "AON"
                mock_result.diagram_confidence = 0.9
                mock_result.shape_count = 5
                mock_result.arrow_count = 4
                mock_result.reconstructed_activity_count = 3
                mock_result.cpm_project_duration = 12.0
                mock_result.cpm_critical_path = ["A", "B", "C"]
                mock_result.critical_paths = [["A", "B", "C"]]
                mock_result.cpm_critical_activity_count = 3
                mock_result.validation_passed = True
                mock_result.review_required = False
                mock_result.review_issues = []
                mock_result.warnings = []
                mock_result.errors = []
                mock_result.total_time = 0.5

                MockAnalyzer.return_value.analyze.return_value = mock_result

                exit_code = run_analyze_image(args)
                assert exit_code == 0
                assert os.path.exists(out_path)

                with open(out_path, "r") as f:
                    data = json.load(f)
                assert data["status"] == "SUCCESS"
        finally:
            if os.path.exists(img_path):
                os.unlink(img_path)
            if os.path.exists(out_path):
                os.unlink(out_path)

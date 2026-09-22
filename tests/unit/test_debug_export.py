"""
Tests for the DebugExporter.

Verifies that debug artifacts are correctly exported to disk.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

import numpy as np
import pytest

from pert_analyzer.pipeline.debug_export import DebugExporter
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult


@pytest.fixture
def debug_output_dir() -> str:
    """Create a temporary directory for debug output."""
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def pipeline_result_with_stages() -> PipelineResult:
    """Create a PipelineResult with some stage data."""
    result = PipelineResult()
    result.status = AnalysisStatus.SUCCESS
    result.diagram_type = "AON"
    result.diagram_confidence = 0.9
    result.shape_count = 3
    result.arrow_count = 2

    # Store mock images
    result._original_image = np.zeros((200, 400, 3), dtype=np.uint8)

    mock_prep = type("MockPrep", (), {
        "get_representation": lambda self, name: np.zeros((200, 400, 3), dtype=np.uint8)
    })()
    result._preprocessing_result = mock_prep

    return result


# =============================================================================
# DebugExporter Tests
# =============================================================================


class TestDebugExporterInit:
    def test_default_output_dir(self) -> None:
        exporter = DebugExporter()
        assert exporter.output_dir == "analysis_output"

    def test_custom_output_dir(self) -> None:
        exporter = DebugExporter(output_dir="/tmp/custom_debug")
        assert exporter.output_dir == "/tmp/custom_debug"


class TestDebugExporterExportAll:
    def test_creates_output_directory(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        result._original_image = np.zeros((100, 100, 3), dtype=np.uint8)

        output = exporter.export_all("/fake/path.png", result)
        assert os.path.isdir(output)

    def test_exports_original_image(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result._original_image = img

        exporter.export_all("/fake/path.png", result)
        assert os.path.exists(os.path.join(debug_output_dir, "01_original.png"))

    def test_exports_result_json(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS
        result.diagram_type = "AON"
        result._original_image = np.zeros((100, 200, 3), dtype=np.uint8)

        exporter.export_all("/fake/path.png", result)
        json_path = os.path.join(debug_output_dir, "result.json")
        assert os.path.exists(json_path)

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "SUCCESS"
        assert data["diagram_type"] == "AON"

    def test_handles_none_image(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        result._original_image = None

        # Should not raise
        exporter.export_all("/fake/path.png", result)
        assert not os.path.exists(os.path.join(debug_output_dir, "01_original.png"))

    def test_exports_preprocessed_representations(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        result._original_image = np.zeros((100, 200, 3), dtype=np.uint8)

        class MockPrep:
            def get_representation(self, name):
                return np.zeros((100, 200, 3), dtype=np.uint8)

        result._preprocessing_result = MockPrep()

        exporter.export_all("/fake/path.png", result)
        # Should try to write grayscale, binary, edges
        expected_files = ["02_grayscale.png", "03_binary.png", "04_edges.png"]
        for fname in expected_files:
            fpath = os.path.join(debug_output_dir, fname)
            assert os.path.exists(fpath), f"Expected {fname} to exist"


class TestDebugExporterExportSingle:
    def test_export_image_creates_file(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        img = np.zeros((50, 50, 3), dtype=np.uint8)

        exporter._export_image("test_image.png", img)
        assert os.path.exists(os.path.join(debug_output_dir, "test_image.png"))

    def test_export_image_skips_none(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)

        exporter._export_image("test_image.png", None)
        assert not os.path.exists(os.path.join(debug_output_dir, "test_image.png"))

    def test_export_result_json_writes_valid_json(self, debug_output_dir: str) -> None:
        exporter = DebugExporter(output_dir=debug_output_dir)
        result = PipelineResult()
        result.status = AnalysisStatus.SUCCESS

        exporter._export_result_json(result)
        json_path = os.path.join(debug_output_dir, "result.json")

        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert data["status"] == "SUCCESS"

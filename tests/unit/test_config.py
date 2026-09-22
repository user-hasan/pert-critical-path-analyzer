"""
Tests for configuration manager.
"""

import json
import os
import tempfile

import pytest

from pert_analyzer.config.manager import (
    AppConfig,
    ConfigManager,
    CVConfig,
    get_config,
)


class TestConfigManager:
    """Tests for ConfigManager."""

    def test_default_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            mgr = ConfigManager(config_path)
            assert mgr.config.ocr.engine == "tesseract"
            assert mgr.config.cv.blur_kernel_size == 5

    def test_load_existing_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {
                "ocr": {"engine": "easyocr", "confidence_threshold": 0.7},
                "cv": {"preprocessing": {"blur_kernel_size": 3}},
            }
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            mgr = ConfigManager(config_path)
            assert mgr.config.ocr.engine == "easyocr"
            assert mgr.config.ocr.confidence_threshold == 0.7
            assert mgr.config.cv.blur_kernel_size == 3

    def test_get_with_dot_notation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            config_data = {"ocr": {"engine": "tesseract"}}
            with open(config_path, "w") as f:
                json.dump(config_data, f)

            mgr = ConfigManager(config_path)
            assert mgr.get("ocr.engine") == "tesseract"
            assert mgr.get("ocr.nonexistent", "default") == "default"

    def test_set_with_dot_notation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            mgr = ConfigManager(config_path)
            mgr.set("ocr.engine", "easyocr")
            assert mgr.get("ocr.engine") == "easyocr"

    def test_save_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            mgr = ConfigManager(config_path)
            mgr.set("ocr.engine", "paddleocr")
            result = mgr.save(config_path)
            assert result

            mgr2 = ConfigManager(config_path)
            assert mgr2.get("ocr.engine") == "paddleocr"

    def test_reset_to_defaults(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "config.json")
            mgr = ConfigManager(config_path)
            mgr.set("ocr.engine", "custom")
            mgr.reset_to_defaults()
            assert mgr.config.ocr.engine == "tesseract"


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_default_values(self):
        config = AppConfig()
        assert config.app["name"] == "PERT & Critical Path Analyzer"

    def test_cv_config_defaults(self):
        config = CVConfig()
        assert config.threshold_method == "adaptive"
        assert config.min_shape_area == 500

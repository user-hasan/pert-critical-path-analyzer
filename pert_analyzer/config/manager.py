"""
Configuration management for the PERT & Critical Path Analyzer.

Loads settings from config files and provides typed access to configuration values.
Separates configuration from source code.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Default configuration values
DEFAULT_CONFIG = {
    "app": {
        "name": "PERT & Critical Path Analyzer",
        "version": "0.1.0",
        "language": "en",
    },
    "ocr": {
        "engine": "tesseract",
        "languages": ["eng"],
        "confidence_threshold": 0.5,
        "tesseract_path": None,
        "psm": 11,
        "oem": 3,
    },
    "cv": {
        "preprocessing": {
            "blur_kernel_size": 5,
            "threshold_method": "adaptive",
            "threshold_block_size": 11,
            "threshold_c": 2,
            "denoise_strength": 10,
            "deskew": True,
        },
        "shape_detection": {
            "min_area": 500,
            "max_area": 100000,
            "min_circularity": 0.5,
            "approximation_accuracy": 0.04,
        },
        "arrow_detection": {
            "min_line_length": 30,
            "max_line_gap": 10,
            "arrowhead_size": 10,
        },
        "diagram_classification": {
            "min_shapes_for_classification": 3,
            "confidence_threshold": 0.6,
        },
    },
    "analysis": {
        "engine": "cpm",
        "pert_enabled": False,
        "what_if_enabled": False,
    },
    "gui": {
        "theme": "light",
        "window_width": 1400,
        "window_height": 900,
        "sidebar_width": 280,
        "show_grid": True,
        "auto_zoom": True,
    },
    "persistence": {
        "database_path": "data/projects.db",
        "auto_save_interval": 300,
        "max_recent_projects": 10,
    },
    "logging": {
        "level": "INFO",
        "file": "logs/analyzer.log",
        "max_bytes": 10485760,
        "backup_count": 5,
    },
}


@dataclass
class OCRConfig:
    """OCR engine configuration."""

    engine: str = "tesseract"
    languages: list = field(default_factory=lambda: ["eng"])
    confidence_threshold: float = 0.5
    tesseract_path: Optional[str] = None
    psm: int = 11
    oem: int = 3


@dataclass
class CVConfig:
    """Computer Vision configuration."""

    blur_kernel_size: int = 5
    threshold_method: str = "adaptive"
    threshold_block_size: int = 11
    threshold_c: int = 2
    denoise_strength: int = 10
    deskew: bool = True
    min_shape_area: int = 500
    max_shape_area: int = 100000
    min_circularity: float = 0.5
    approximation_accuracy: float = 0.04
    min_line_length: int = 30
    max_line_gap: int = 10
    arrowhead_size: int = 10


@dataclass
class AnalysisConfig:
    """Analysis engine configuration."""

    engine: str = "cpm"
    pert_enabled: bool = False
    what_if_enabled: bool = False


@dataclass
class GUIConfig:
    """GUI configuration."""

    theme: str = "light"
    window_width: int = 1400
    window_height: int = 900
    sidebar_width: int = 280
    show_grid: bool = True
    auto_zoom: bool = True


@dataclass
class AppConfig:
    """Top-level application configuration."""

    app: Dict[str, Any] = field(default_factory=lambda: DEFAULT_CONFIG["app"])
    ocr: OCRConfig = field(default_factory=OCRConfig)
    cv: CVConfig = field(default_factory=CVConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    gui: GUIConfig = field(default_factory=GUIConfig)
    persistence: Dict[str, Any] = field(
        default_factory=lambda: DEFAULT_CONFIG["persistence"]
    )
    logging: Dict[str, Any] = field(
        default_factory=lambda: DEFAULT_CONFIG["logging"]
    )


class ConfigManager:
    """Manages application configuration with file-based persistence."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize the configuration manager."""
        self._config_path = config_path or self._find_config_file()
        self._config: AppConfig = AppConfig()
        self._raw_config: Dict[str, Any] = {}
        self._load()

    def _find_config_file(self) -> str:
        """Find the configuration file in standard locations."""
        possible_paths = [
            "config.json",
            "config/config.json",
            os.path.expanduser("~/.pert_analyzer/config.json"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                return path
        return "config.json"

    def _load(self) -> None:
        """Load configuration from file."""
        import copy
        if os.path.exists(self._config_path):
            try:
                with open(self._config_path, "r", encoding="utf-8") as f:
                    self._raw_config = json.load(f)
                self._apply_config(self._raw_config)
                logger.info("Configuration loaded from %s", self._config_path)
            except Exception as e:
                logger.warning("Failed to load config: %s. Using defaults.", e)
                self._raw_config = copy.deepcopy(DEFAULT_CONFIG)
                self._apply_config(self._raw_config)
        else:
            logger.info("No config file found. Using defaults.")
            self._raw_config = copy.deepcopy(DEFAULT_CONFIG)
            self._apply_config(self._raw_config)

    def _apply_config(self, config: Dict[str, Any]) -> None:
        """Apply configuration values to the AppConfig dataclass."""
        if "app" in config:
            self._config.app = config["app"]
        if "ocr" in config:
            ocr = config["ocr"]
            self._config.ocr = OCRConfig(
                engine=ocr.get("engine", "tesseract"),
                languages=ocr.get("languages", ["eng"]),
                confidence_threshold=ocr.get("confidence_threshold", 0.5),
                tesseract_path=ocr.get("tesseract_path"),
                psm=ocr.get("psm", 11),
                oem=ocr.get("oem", 3),
            )
        if "cv" in config:
            cv = config["cv"]
            pre = cv.get("preprocessing", {})
            shape = cv.get("shape_detection", {})
            arrow = cv.get("arrow_detection", {})
            self._config.cv = CVConfig(
                blur_kernel_size=pre.get("blur_kernel_size", 5),
                threshold_method=pre.get("threshold_method", "adaptive"),
                threshold_block_size=pre.get("threshold_block_size", 11),
                threshold_c=pre.get("threshold_c", 2),
                denoise_strength=pre.get("denoise_strength", 10),
                deskew=pre.get("deskew", True),
                min_shape_area=shape.get("min_area", 500),
                max_shape_area=shape.get("max_area", 100000),
                min_circularity=shape.get("min_circularity", 0.5),
                approximation_accuracy=shape.get("approximation_accuracy", 0.04),
                min_line_length=arrow.get("min_line_length", 30),
                max_line_gap=arrow.get("max_line_gap", 10),
                arrowhead_size=arrow.get("arrowhead_size", 10),
            )
        if "analysis" in config:
            a = config["analysis"]
            self._config.analysis = AnalysisConfig(
                engine=a.get("engine", "cpm"),
                pert_enabled=a.get("pert_enabled", False),
                what_if_enabled=a.get("what_if_enabled", False),
            )
        if "gui" in config:
            g = config["gui"]
            self._config.gui = GUIConfig(
                theme=g.get("theme", "light"),
                window_width=g.get("window_width", 1400),
                window_height=g.get("window_height", 900),
                sidebar_width=g.get("sidebar_width", 280),
                show_grid=g.get("show_grid", True),
                auto_zoom=g.get("auto_zoom", True),
            )
        if "persistence" in config:
            self._config.persistence = config["persistence"]
        if "logging" in config:
            self._config.logging = config["logging"]

    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        return self._config

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get a configuration value using dot notation (e.g., 'ocr.engine')."""
        keys = key_path.split(".")
        value = self._raw_config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set(self, key_path: str, value: Any) -> None:
        """Set a configuration value using dot notation."""
        keys = key_path.split(".")
        target = self._raw_config
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value
        self._apply_config(self._raw_config)

    def save(self, path: Optional[str] = None) -> bool:
        """Save configuration to file."""
        save_path = path or self._config_path
        try:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(self._raw_config, f, indent=2, ensure_ascii=False)
            logger.info("Configuration saved to %s", save_path)
            return True
        except Exception as e:
            logger.error("Failed to save config: %s", e)
            return False

    def reset_to_defaults(self) -> None:
        """Reset configuration to defaults."""
        import copy
        self._raw_config = copy.deepcopy(DEFAULT_CONFIG)
        self._apply_config(self._raw_config)
        logger.info("Configuration reset to defaults")


# Singleton instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get or create the global configuration manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> AppConfig:
    """Get the current application configuration."""
    return get_config_manager().config

"""
Main entry point for the PERT & Critical Path Analyzer.

This module provides the application entry point for both
GUI and CLI modes.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from pert_analyzer import __version__
from pert_analyzer.config.logging_config import setup_logging
from pert_analyzer.config.manager import get_config_manager


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pert_analyzer",
        description="PERT & Critical Path Analyzer - "
        "Analyze project network diagrams from images",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run in command-line interface mode",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to the diagram image to analyze (CLI mode)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file",
    )
    return parser.parse_args()


def run_cli(args: argparse.Namespace) -> int:
    """Run the application in CLI mode."""
    logger = logging.getLogger("pert_analyzer.cli")
    logger.info("Starting CLI mode")

    if not args.image:
        logger.error("No image specified. Use --image <path>")
        return 1

    image_path = Path(args.image)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        return 1

    from pert_analyzer.application import AnalysisOrchestrator

    orchestrator = AnalysisOrchestrator()
    orchestrator.create_project(name=image_path.stem)

    def progress_report(stage: str, progress: float) -> None:
        bar_len = 30
        filled = int(bar_len * progress)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r[{bar}] {progress*100:5.1f}% - {stage}", end="", flush=True)

    print(f"Analyzing: {image_path}")
    project = orchestrator.analyze_image(
        str(image_path), progress_callback=progress_report
    )
    print()

    if project and project.analysis:
        print("\n=== Analysis Results ===")
        print(f"Project Duration: {project.analysis.project_duration}")
        print(f"Critical Path: {' -> '.join(project.analysis.critical_path)}")
        print(f"Total Activities: {project.graph.activity_count if project.graph else 0}")
        print(f"Critical Activities: {project.analysis.critical_activity_count}")
        return 0
    else:
        print("\nAnalysis failed. Check logs for details.")
        return 1


def run_gui(args: argparse.Namespace) -> int:
    """Run the application in GUI mode."""
    try:
        from PySide6.QtWidgets import QApplication

        from pert_analyzer.gui.main_window import MainWindow

        app = QApplication(sys.argv)
        app.setApplicationName("PERT & Critical Path Analyzer")
        app.setApplicationVersion(__version__)

        window = MainWindow()
        window.show()

        return app.exec()
    except ImportError:
        print(
            "PySide6 is not installed. Install it with:\n"
            "  pip install PySide6"
        )
        return 1


def main() -> int:
    """Main entry point."""
    args = parse_arguments()

    # Setup logging
    config_mgr = get_config_manager(args.config)
    log_config = config_mgr.config.logging
    setup_logging(
        level=args.log_level or log_config.get("level", "INFO"),
        log_file=args.log_file or log_config.get("file"),
    )

    logger = logging.getLogger("pert_analyzer")
    logger.info("PERT & Critical Path Analyzer v%s", __version__)

    if args.cli:
        return run_cli(args)
    else:
        return run_gui(args)


if __name__ == "__main__":
    sys.exit(main())

"""
CLI entry point for the pipeline module.

Provides command-line interface for running the complete
analysis pipeline against diagram images.

Usage:
    python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png"
    python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --output result.json
    python -m pert_analyzer.pipeline analyze-image "path/to/diagram.png" --debug
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="pert_analyzer.pipeline",
        description="PERT & Critical Path Analyzer — End-to-End Pipeline",
    )
    subparsers = parser.add_subparsers(dest="command")

    # analyze-image command
    analyze_parser = subparsers.add_parser(
        "analyze-image",
        help="Analyze a diagram image through the complete pipeline",
    )
    analyze_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the diagram image",
    )
    analyze_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to export JSON result",
    )
    analyze_parser.add_argument(
        "--debug",
        action="store_true",
        help="Export debug artifacts to analysis_output/",
    )
    analyze_parser.add_argument(
        "--debug-dir",
        type=str,
        default="analysis_output",
        help="Directory for debug artifacts (default: analysis_output)",
    )
    analyze_parser.add_argument(
        "--ocr-engine",
        type=str,
        default="tesseract",
        choices=["mock", "tesseract"],
        help="OCR engine to use (default: tesseract)",
    )
    analyze_parser.add_argument(
        "--ocr-lang",
        type=str,
        default="eng",
        help="Tesseract language codes, e.g. 'eng' or 'eng+ara' (default: eng)",
    )
    analyze_parser.add_argument(
        "--tesseract-path",
        type=str,
        default=None,
        help="Path to tesseract executable (optional, auto-detected by default)",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON to stdout",
    )

    # reviews command
    review_parser = subparsers.add_parser(
        "reviews",
        help="Analyze an image and list human-review items",
    )
    review_parser.add_argument(
        "image_path",
        type=str,
        help="Path to the diagram image",
    )
    review_parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Path to export the review session JSON",
    )
    review_parser.add_argument(
        "--json",
        action="store_true",
        help="Output review items as JSON to stdout",
    )
    review_parser.add_argument(
        "--tesseract-path",
        type=str,
        default=None,
        help="Path to tesseract executable (optional, auto-detected by default)",
    )

    # apply-reviews command
    apply_parser = subparsers.add_parser(
        "apply-reviews",
        help="Load a review session and apply the recorded decisions",
    )
    apply_parser.add_argument(
        "session_json",
        type=str,
        help="Path to the review session JSON (from 'reviews --output')",
    )
    apply_parser.add_argument(
        "--image", type=str, default=None,
        help="Path to the source diagram image (needed for the reconstruction)",
    )
    apply_parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path to export the ReviewedGraphCandidate JSON",
    )

    return parser.parse_args()


def run_analyze_image(args: argparse.Namespace) -> int:
    """Run the analyze-image command."""
    from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    print("-" * 50)
    print("PERT & Critical Path Analyzer")
    print("-" * 50)
    print(f"\nInput: {image_path.name}")
    print(f"Analyzing...")

    analyzer = EndToEndAnalyzer(
        ocr_engine_name=args.ocr_engine,
        export_debug=args.debug,
        debug_output_dir=args.debug_dir,
        ocr_languages=args.ocr_lang.split("+"),
        tesseract_path=args.tesseract_path,
    )

    def progress(stage: str, pct: float) -> None:
        bar_len = 30
        filled = int(bar_len * pct)
        bar = "=" * filled + "-" * (bar_len - filled)
        print(f"\r  [{bar}] {pct*100:5.1f}% - {stage:<25}", end="", flush=True)

    result = analyzer.analyze(str(image_path), progress_callback=progress)
    print()

    if args.json:
        print(result.to_json())
    else:
        print(result.to_summary_string())

    # Export JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(result.to_json(indent=2))
        print(f"\nResult exported to: {output_path}")

    if args.debug:
        print(f"Debug artifacts exported to: {args.debug_dir}")

    return 0 if result.status.value in ("SUCCESS", "REVIEW_REQUIRED") else 1


def run_reviews(args: argparse.Namespace) -> int:
    """Run the reviews command: analyze image + list review items."""
    from pert_analyzer.pipeline.review_api import ReviewWorkflow

    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {image_path}", file=sys.stderr)
        return 1

    workflow = ReviewWorkflow.analyze(
        str(image_path),
        source_image_id=image_path.name,
        tesseract_path=args.tesseract_path,
    )
    session = workflow.get_review_items()

    if args.json:
        print(session.to_dict())
    else:
        print("-" * 50)
        print("Human Review Items")
        print("-" * 50)
        print(f"Source image : {session.source_image_id}")
        print(f"Pending activities   : {session.pending_activity_count}")
        print(f"Pending dependencies : {session.pending_dependency_count}")
        print(f"Pending durations    : {session.pending_duration_count}")
        print(f"Ambiguities          : {len(session.ambiguities)}")
        print(f"Decisions            : {len(session.decisions)}")
        for item in session.activities:
            print(f"  [activity] node={item.geometric_node_id} "
                  f"current={item.current_activity_id!r} reason={item.reason!r}")
        for item in session.dependencies:
            print(f"  [dependency] arrow={item.arrow_id} "
                  f"{item.current_source_id}->{item.current_target_id} "
                  f"(dir={item.proposed_direction}) conf={item.confidence:.3f}")
        for item in session.durations:
            print(f"  [duration] node={item.geometric_node_id} "
                  f"current={item.current_duration} reason={item.reason!r}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            import json

            f.write(json.dumps(session.to_dict(), indent=2))
        print(f"\nReview session exported to: {output_path}")

    return 0


def run_apply_reviews(args: argparse.Namespace) -> int:
    """Run the apply-reviews command: apply decisions and show the gate."""
    import json

    from pert_analyzer.pipeline.analyzer import EndToEndAnalyzer
    from pert_analyzer.pipeline.human_review import (
        ReviewSession,
        apply_review_decisions,
        build_review_summary,
    )

    session_path = Path(args.session_json)
    if not session_path.exists():
        print(f"Error: Session file not found: {session_path}", file=sys.stderr)
        return 1

    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    reconstruction = None
    if args.image:
        if not Path(args.image).exists():
            print(f"Error: Image file not found: {args.image}", file=sys.stderr)
            return 1
        result = EndToEndAnalyzer().analyze(args.image)
        reconstruction = result._reconstruction

    session = ReviewSession.from_dict(data, reconstruction=reconstruction)
    candidate = apply_review_decisions(
        reconstruction if reconstruction is not None else session.reconstruction,
        session,
    )
    summary = build_review_summary(candidate)

    print("-" * 50)
    print("Applied Review Decisions")
    print("-" * 50)
    print(summary)
    if candidate.validation:
        for err in candidate.validation.errors:
            print(f"  [validation] {err.code}: {err.message}")
    print(f"CPM duration : {candidate.cpm_project_duration}")
    print(f"Critical paths : {candidate.critical_path_count}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(json.dumps(candidate.to_dict(), indent=2))
        print(f"\nCandidate exported to: {output_path}")

    return 0 if candidate.cpm_gate.value == "RUNNABLE" else 1


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.command == "analyze-image":
        return run_analyze_image(args)
    elif args.command == "reviews":
        return run_reviews(args)
    elif args.command == "apply-reviews":
        return run_apply_reviews(args)
    else:
        print("Usage: python -m pert_analyzer.pipeline analyze-image <image_path>")
        print("       python -m pert_analyzer.pipeline analyze-image <image_path> --output result.json")
        print("       python -m pert_analyzer.pipeline reviews <image_path>")
        print("       python -m pert_analyzer.pipeline apply-reviews <session.json> --image <image_path>")
        return 1


if __name__ == "__main__":
    sys.exit(main())

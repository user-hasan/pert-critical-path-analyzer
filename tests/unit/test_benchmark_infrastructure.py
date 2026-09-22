"""
Deterministic unit tests for the generalization-benchmark infrastructure.

These tests never run the real CV/OCR pipeline: pipeline "runs" are
synthetic ``ReviewWorkflow``-like objects built from real pipeline DTOs
plus tiny cv2-generated images for scanner/runner paths.  The real
corpus is executed manually (never inside pytest).
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from pert_analyzer.benchmark import (
    DatasetScanner,
    FailureClass,
    GroundTruthError,
    analyze_single,
    build_markdown,
    classify_first_abnormal_stage,
    classify_outcome,
    discover_images,
    extract_stage_metrics,
    extract_stage_timings,
    load_ground_truth,
    relative_image_path,
    run_benchmark,
    summarize_bottlenecks,
    write_reports,
)
from pert_analyzer.benchmark.metrics import UNAVAILABLE
from pert_analyzer.cv.models import ArrowDetectionResult, CandidateNode, ShapeDetectionResult
from pert_analyzer.cv.ocr_models import OCRProcessingResult, OCRTextRegion, TextType
from pert_analyzer.cv.reconstruction_models import (
    ReconstructedActivity,
    ReconstructedDependency,
    ReconstructedDiagram,
)
from pert_analyzer.cv.validated_dependency import DependencyValidationReport
from pert_analyzer.core.models import Dependency
from pert_analyzer.pipeline.human_review import CpmGateStatus
from pert_analyzer.pipeline.result import AnalysisStatus, PipelineResult


# ---------------------------------------------------------------------------
# Synthetic workflow helpers (no real CV ever runs).
# ---------------------------------------------------------------------------


def _make_shape_result(
    detected: int = 80,
    candidates: int = 74,
    duplicates: int = 0,
) -> ShapeDetectionResult:
    return ShapeDetectionResult(
        detected_shapes=[object() for _ in range(detected)],
        candidate_nodes=[CandidateNode() for _ in range(candidates)],
        contours_analyzed=90,
        shapes_filtered=6,
        duplicates_suppressed=duplicates,
    )


def _make_arrow_result(detected: int = 28, raw: int = 40) -> ArrowDetectionResult:
    return ArrowDetectionResult(
        arrows=[object() for _ in range(detected)],
        raw_line_segments=[object() for _ in range(raw)],
        lines_detected=raw,
        lines_filtered=0,
        segments_merged=0,
        arrows_detected=detected,
    )


def _make_ocr_result() -> OCRProcessingResult:
    regions = [
        OCRTextRegion(text=f"A{i}", text_type=TextType.ACTIVITY_ID_CANDIDATE)
        for i in range(4)
    ] + [
        OCRTextRegion(text="2", text_type=TextType.NUMERIC_CANDIDATE),
        OCRTextRegion(text="5", text_type=TextType.DURATION_CANDIDATE),
        OCRTextRegion(text="label", text_type=TextType.TEXT_LABEL_CANDIDATE),
        OCRTextRegion(text="", text_type=TextType.UNKNOWN),
    ]
    return OCRProcessingResult(regions=regions)


def _make_validation_report(
    raw: int = 28,
    dedup: int = 28,
    accepted: int = 20,
    review: int = 6,
    rejected: int = 2,
) -> DependencyValidationReport:
    return DependencyValidationReport(
        raw_arrow_count=raw,
        deduplicated_arrow_count=dedup,
        accepted_count=accepted,
        review_count=review,
        rejected_count=rejected,
        self_loop_count=0,
        duplicate_edge_count=0,
    )


def _make_reconstruction(
    activity_count: int = 74,
    dependency_count: int = 20,
    report: DependencyValidationReport | None = None,
) -> ReconstructedDiagram:
    activities = [
        ReconstructedActivity(
            activity_id=f"n{i}", duration=1.0, geometric_node_id=f"g{i}"
        )
        for i in range(activity_count)
    ]
    dependencies = [
        ReconstructedDependency(source_id=f"n{i}", target_id=f"n{i + 1}")
        for i in range(dependency_count)
    ]
    metadata = {}
    if report is not None:
        metadata["dependency_validation"] = report
    return ReconstructedDiagram(
        activities=activities, dependencies=dependencies, metadata=metadata
    )


def _make_fake_workflow(
    status=AnalysisStatus.REVIEW_REQUIRED,
    activity_count: int = 74,
    dependency_count: int = 20,
    cpm_duration: float | None = None,
    diagram_type="AON",
    with_report: bool = True,
) -> SimpleNamespace:
    report = _make_validation_report() if with_report else None
    recon = _make_reconstruction(activity_count, dependency_count, report)
    result = PipelineResult()
    result._shape_result = _make_shape_result()
    result._arrow_result = _make_arrow_result()
    result._ocr_result = _make_ocr_result()
    result._reconstruction = recon
    result.diagram_type = diagram_type
    result.status = status
    result.reconstructed_activity_count = activity_count
    result.dependency_count = dependency_count

    session = SimpleNamespace(
        activities=[object() for _ in range(3)],
        dependencies=[object() for _ in range(4)],
        durations=[object() for _ in range(2)],
    )

    cpm = None
    gate = CpmGateStatus.BLOCKED_REVIEW
    if cpm_duration is not None:
        cpm = SimpleNamespace(project_duration=cpm_duration, critical_paths=[])
        gate = CpmGateStatus.RUNNABLE
    candidate = SimpleNamespace(
        validation=SimpleNamespace(
            status="VALID" if cpm_duration is not None else "INVALID",
            is_valid=cpm_duration is not None,
            error_count=0 if cpm_duration is not None else 1,
        ),
        cpm_gate=gate,
        cpm=cpm,
        pure_critical_paths=[] if cpm_duration is not None else None,
        graph=SimpleNamespace(
            activities={f"n{i}": object() for i in range(activity_count)},
            dependencies=[
                Dependency(source=f"n{i}", target=f"n{i + 1}")
                for i in range(dependency_count)
            ],
        ),
    )
    return SimpleNamespace(
        pipeline_result=result,
        review_session=session,
        apply=lambda: candidate,
    )


def _write_png(path: Path, size: int = 12) -> None:
    image = np.zeros((size, size, 3), dtype=np.uint8)
    image[:, 0] = 255
    assert cv2.imwrite(str(path), image)


def _fake_analyze_fn(workflow_factory=_make_fake_workflow):
    def _analyze(path: str):
        return workflow_factory()
    return _analyze


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def test_dataset_scanner_recursive_filtering(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "sub1").mkdir(parents=True)
    (root / "sub2").mkdir()
    images = ["a.png", "b.JPG", "c.jpeg", "d.bmp", "e.tif", "f.tiff"]
    for index, name in enumerate(images):
        _write_png(root / "sub1" / name if index % 2 else root / name)
    (root / "notes.json").write_text("{}", encoding="utf-8")
    (root / "sub2" / "readme.txt").write_text("x", encoding="utf-8")
    (root / "sub2" / "data.csv").write_text("a,b", encoding="utf-8")

    found = DatasetScanner(root).discover()
    names = {p.name for p in found}
    assert names == set(images)
    assert all(p.is_file() for p in found)


def test_discover_images_deterministic_order(tmp_path: Path):
    root = tmp_path / "dataset"
    (root / "b").mkdir(parents=True)
    _write_png(root / "Z.png")
    _write_png(root / "a.png")
    _write_png(root / "b" / "A.png")
    _write_png(root / "b" / "mid.JPG")

    found = discover_images(root)
    rels = [p.relative_to(root).as_posix() for p in found]
    assert rels == sorted(rels, key=str.lower)
    assert rels[0] == "a.png"  # case-folded: "a" < "b" < "z"
    assert "b/A.png" in rels


def test_discover_images_missing_root_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        discover_images(tmp_path / "does-not-exist")


def test_relative_image_path_raises_outside_root(tmp_path: Path):
    root = tmp_path / "dataset"
    root.mkdir()
    outside = tmp_path / "outside.png"
    with pytest.raises(ValueError):
        relative_image_path(root, outside)


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def test_ground_truth_dict_form(tmp_path: Path):
    image = tmp_path / "diagram.png"
    _write_png(image)
    (tmp_path / "diagram.json").write_text(
        json.dumps(
            {
                "diagram_type": "aon",
                "activities": {
                    "A": {"duration": 2.0, "position": [3, 4]},
                    "B": {"duration": 5.0},
                },
                "dependencies": [["A", "B"]],
            }
        ),
        encoding="utf-8",
    )
    gt = load_ground_truth(image)
    assert gt is not None
    assert gt.diagram_type == "AON"
    assert gt.activity_count == 2
    assert gt.dependencies == [("A", "B")]
    assert gt.durations == {"A": 2.0, "B": 5.0}  # enrichment from activity entries
    assert gt.positions == {"A": (3.0, 4.0)}  # enrichment from activity entries


def test_ground_truth_list_form_and_missing(tmp_path: Path):
    image = tmp_path / "diagram.png"
    _write_png(image)
    (tmp_path / "diagram.json").write_text(
        json.dumps(
            {
                "activities": [
                    {"id": "A", "duration": 2.0, "position": [3, 4]},
                    {"id": "B"},
                ],
                "dependencies": [{"source": "A", "target": "B"}],
                "durations": {"A": 2.0, "B": 5.0},
            }
        ),
        encoding="utf-8",
    )
    gt = load_ground_truth(image)
    assert gt.activity_count == 2
    assert gt.durations == {"A": 2.0, "B": 5.0}

    other = tmp_path / "no-gt.png"
    _write_png(other)
    assert load_ground_truth(other) is None


def test_ground_truth_malformed_raises(tmp_path: Path):
    image = tmp_path / "diagram.png"
    _write_png(image)
    (tmp_path / "diagram.json").write_text("{ not json", encoding="utf-8")
    with pytest.raises(GroundTruthError):
        load_ground_truth(image)


# ---------------------------------------------------------------------------
# Metric extraction mapping (the canonical table)
# ---------------------------------------------------------------------------


def test_extract_stage_metrics_full_mapping():
    workflow = _make_fake_workflow()
    metrics = extract_stage_metrics(workflow)

    assert metrics["raw_shapes"] == 80
    assert metrics["contours_analyzed"] == 90
    assert metrics["shapes_filtered"] == 6
    assert metrics["duplicates_suppressed"] == 0
    assert metrics["final_candidate_nodes"] == 74

    assert metrics["reconstructed_activities"] == 74
    assert metrics["reconstructed_events"] == 0
    assert metrics["reconstructed_dependencies"] == 20
    assert metrics["raw_arrow_candidates"] == 28
    assert metrics["deduplicated_arrows"] == 28
    assert metrics["validated_node_pairs"] == 28
    assert metrics["validated_dependencies"] == 20
    assert metrics["review_candidate_dependencies"] == 6
    assert metrics["rejected_dependencies"] == 2

    assert metrics["ocr_regions"] == 8
    assert metrics["ocr_labels"] == 7
    assert metrics["ocr_id_candidates"] == 4
    assert metrics["ocr_numeric_candidates"] == 2

    assert metrics["review_items"] == 9  # 3 + 4 + 2
    assert metrics["graph_status"] == "INVALID"
    assert metrics["graph_is_valid"] is False
    assert metrics["cpm_gate"] == "BLOCKED_REVIEW"
    assert metrics["cpm_project_duration"] == UNAVAILABLE
    assert metrics["pert_status"] == UNAVAILABLE


def test_extract_stage_metrics_presentation_roundtrips():
    workflow = _make_fake_workflow(cpm_duration=54.0)
    metrics = extract_stage_metrics(workflow)
    assert metrics["graph_status"] == "VALID"
    assert metrics["graph_is_valid"] is True
    assert metrics["cpm_gate"] == "RUNNABLE"
    assert metrics["cpm_project_duration"] == 54.0


def test_extract_stage_metrics_missing_stages_unavailable():
    result = PipelineResult()
    workflow = SimpleNamespace(
        pipeline_result=result,
        review_session=None,
        apply=lambda: None,
    )
    metrics = extract_stage_metrics(workflow)
    for key in (
        "raw_shapes",
        "final_candidate_nodes",
        "reconstructed_activities",
        "ocr_regions",
        "review_items",
        "graph_status",
    ):
        assert metrics[key] == UNAVAILABLE
    assert metrics["pert_status"] == UNAVAILABLE


def test_extract_stage_timings_missing_is_unavailable():
    timings = extract_stage_timings(PipelineResult())
    assert timings["Detecting shapes"] == UNAVAILABLE


# ---------------------------------------------------------------------------
# Outcome + first-abnormal-stage classification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status,expected",
    [
        (AnalysisStatus.SUCCESS, "AUTOMATIC_SUCCESS"),
        (AnalysisStatus.REVIEW_REQUIRED, "AUTOMATIC_REVIEW_REQUIRED"),
        (AnalysisStatus.FAILED, "FATAL_FAILURE"),
    ],
)
def test_classify_outcome(status, expected):
    assert classify_outcome(_make_fake_workflow(status=status)) == expected


def test_classify_first_abnormal_stage_shape_detection():
    metrics = {
        "final_candidate_nodes": 74,
        "raw_shapes": 80,
        "duplicates_suppressed": 0,
        "reconstructed_activities": 74,
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage == "Detecting shapes"
    assert cls == FailureClass.SHAPE_DETECTION
    assert evidence["final_candidate_nodes"] == 74
    assert evidence["duplicates_suppressed"] == 0


def test_classify_first_abnormal_stage_dedup_effective_but_large():
    metrics = {
        "final_candidate_nodes": 60,
        "raw_shapes": 300,
        "duplicates_suppressed": 240,
        "reconstructed_activities": 60,
    }
    stage, cls, _ = classify_first_abnormal_stage(metrics)
    assert stage == "Detecting shapes"
    assert cls == FailureClass.SHAPE_DEDUPLICATION


def test_classify_first_abnormal_stage_healthy_is_unknown():
    metrics = {
        "final_candidate_nodes": 8,
        "raw_shapes": 10,
        "duplicates_suppressed": 2,
        "reconstructed_activities": 8,
        "arrows_detected": 7,
        "deduplicated_arrows": 7,
        "validated_dependencies": 6,
        "reconstructed_dependencies": 6,
        "ocr_regions": 20,
        "ocr_labels": 16,
        "graph_is_valid": True,
        "graph_status": "VALID",
        "cpm_gate": "RUNNABLE",
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage is None
    assert cls == FailureClass.UNKNOWN
    assert evidence == {}


def test_classify_first_abnormal_stage_no_dependencies():
    metrics = {
        "final_candidate_nodes": 8,
        "raw_shapes": 10,
        "duplicates_suppressed": 2,
        "reconstructed_activities": 8,
        "arrows_detected": 0,
        "deduplicated_arrows": 0,
        "validated_dependencies": 0,
        "reconstructed_dependencies": 0,
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage == "Detecting arrows"
    assert cls == FailureClass.ARROW_DETECTION
    assert evidence["arrows_detected"] == 0


def test_classify_first_abnormal_stage_aoa_arrow_inflation():
    metrics = {
        "diagram_type": "AOA",
        "final_candidate_nodes": 15,
        "raw_shapes": 15,
        "duplicates_suppressed": 0,
        "reconstructed_activities": 74,
        "arrows_detected": 74,
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage == "Detecting arrows"
    assert cls == FailureClass.ARROW_DETECTION
    assert evidence["arrows_detected"] == 74


def test_classify_first_abnormal_stage_aon_node_inflation():
    metrics = {
        "diagram_type": "AON",
        "final_candidate_nodes": 15,
        "raw_shapes": 15,
        "duplicates_suppressed": 0,
        "reconstructed_activities": 74,
        "arrows_detected": 74,
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage == "Reconstructing diagram"
    assert cls == FailureClass.NODE_RECONSTRUCTION
    assert evidence["final_candidate_nodes"] == 15


def test_classify_first_abnormal_stage_aoa_shape_precedence():
    metrics = {
        "diagram_type": "AOA",
        "final_candidate_nodes": 240,
        "raw_shapes": 244,
        "duplicates_suppressed": 4,
        "reconstructed_activities": 235,
        "arrows_detected": 235,
    }
    stage, cls, evidence = classify_first_abnormal_stage(metrics)
    assert stage == "Detecting shapes"
    assert cls == FailureClass.SHAPE_DETECTION
    assert evidence["final_candidate_nodes"] == 240


def test_summarize_bottlenecks_ranking():
    results = [
        {"failure_classification": {"failure_class": FailureClass.SHAPE_DETECTION}},
        {"failure_classification": {"failure_class": FailureClass.SHAPE_DETECTION}},
        {"failure_classification": {"failure_class": FailureClass.OCR}},
        {"failure_classification": {"failure_class": FailureClass.UNKNOWN}},
    ]
    ranked = summarize_bottlenecks(results)
    assert ranked[0] == {
        "failure_class": FailureClass.SHAPE_DETECTION,
        "image_count": 2,
    }
    assert ranked[-1]["failure_class"] == FailureClass.UNKNOWN


# ---------------------------------------------------------------------------
# Runner (synthetic analyzer, tiny real images)
# ---------------------------------------------------------------------------


def test_analyze_single_with_fake_analyzer(tmp_path: Path):
    image = tmp_path / "diagram.png"
    _write_png(image)
    run = analyze_single(image, dataset_root=tmp_path, analyze_fn=_fake_analyze_fn())
    assert run.relative_path == "diagram.png"
    assert run.width == 12 and run.height == 12
    assert run.file_format == "PNG"
    assert run.outcome == "AUTOMATIC_REVIEW_REQUIRED"
    assert run.final_status == "REVIEW_REQUIRED"
    assert run.stage_metrics["final_candidate_nodes"] == 74
    assert run.failure_classification["failure_class"] == FailureClass.SHAPE_DETECTION
    assert run.activity_ids[:2] == ["n0", "n1"]
    assert len(run.dependency_edges) == 20
    assert run.extra == {}


def test_analyze_single_skips_gt_when_disabled(tmp_path: Path):
    image = tmp_path / "diagram.png"
    _write_png(image)
    (tmp_path / "diagram.json").write_text(
        json.dumps({"activities": {"A": {"duration": 1.0}}}), encoding="utf-8"
    )
    run = analyze_single(
        image,
        dataset_root=tmp_path,
        analyze_fn=_fake_analyze_fn(),
        load_gt=False,
    )
    assert run.ground_truth is None
    assert run.gt_compare is None


def test_run_benchmark_incremental_json(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        _write_png(dataset / name)
    out = tmp_path / "partial.json"

    summary = run_benchmark(
        dataset, analyze_fn=_fake_analyze_fn(), output_path=str(out)
    )
    assert summary["images_discovered"] == 3
    assert summary["images_analyzed"] == 3
    assert summary["summary"]["automatic_review_required"] == 3

    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["partial"] is True
    assert data["results_count"] == 3
    assert {r["relative_path"] for r in data["results"]} == {
        "a.png",
        "b.png",
        "c.png",
    }


def test_run_benchmark_tolerates_failing_images(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_png(dataset / "ok.png")
    _write_png(dataset / "boom.png")

    def _flaky_analyze(path: str):
        if Path(path).name == "boom.png":
            raise RuntimeError("simulated OCR crash")
        return _make_fake_workflow()

    summary = run_benchmark(dataset, analyze_fn=_flaky_analyze)
    assert summary["images_analyzed"] == 2
    assert summary["summary"]["fatal_failures"] == 1
    assert summary["summary"]["automatic_review_required"] == 1

    by_rel = {r["relative_path"]: r for r in summary["images"]}
    assert "simulated OCR crash" in by_rel["boom.png"]["error"]
    assert by_rel["boom.png"]["outcome"] == "FATAL_FAILURE"
    assert by_rel["ok.png"]["stage_metrics"]["final_candidate_nodes"] == 74


def test_run_benchmark_unreadable_image_is_failure(tmp_path: Path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    bad = dataset / "corrupt.png"
    bad.write_bytes(b"not an image at all")
    summary = run_benchmark(dataset, analyze_fn=_fake_analyze_fn())
    assert summary["images_analyzed"] == 1
    assert summary["summary"]["fatal_failures"] == 1
    assert "Unreadable" in summary["images"][0]["error"]


def test_no_ground_truth_in_dataset(tmp_path: Path):
    """The unit-TEST corpus has no GT; run_benchmark must report it."""
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_png(dataset / "x.png")
    summary = run_benchmark(dataset, analyze_fn=_fake_analyze_fn())
    assert summary["summary"]["without_ground_truth"] == 1
    assert summary["summary"]["with_ground_truth"] == 0


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _sample_summary(tmp_path: Path) -> dict:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_png(dataset / "sample.png")
    return run_benchmark(dataset, analyze_fn=_fake_analyze_fn())


def test_build_markdown_renders_structure(tmp_path: Path):
    summary = _sample_summary(tmp_path)
    md = build_markdown(summary)
    assert "## Summary" in md
    assert "## Per-image results" in md
    assert "Bottleneck summary" in md
    assert "sample.png" in md
    assert FailureClass.SHAPE_DETECTION in md
    assert "AUTOMATIC_REVIEW_REQUIRED" in md


def test_write_reports_creates_md_and_json(tmp_path: Path):
    summary = _sample_summary(tmp_path)
    md = tmp_path / "out" / "GENERALIZATION_BENCHMARK.md"
    jf = tmp_path / "out" / "GENERALIZATION_BENCHMARK.json"
    write_reports(summary, md, jf)

    assert md.is_file()
    assert jf.is_file()
    assert "images_discovered" in json.loads(jf.read_text(encoding="utf-8"))
    text = md.read_text(encoding="utf-8")
    assert "Generalization Benchmark" in text
    assert "sample.png" in text


def test_build_markdown_handles_unavailable_metrics(tmp_path: Path):
    summary = _sample_summary(tmp_path)
    for entry in summary["images"]:
        entry["stage_metrics"]["cpm_project_duration"] = UNAVAILABLE
        entry["stage_metrics"]["pert_status"] = UNAVAILABLE
    md = build_markdown(summary)
    assert "_unavail_" in md
    assert not md.startswith("Traceback")
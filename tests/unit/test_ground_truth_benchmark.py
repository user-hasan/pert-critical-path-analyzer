"""
Deterministic unit tests for the v1.0 ground-truth annotation + accuracy layer.

These tests are fully synthetic: they never execute the real CV/OCR
pipeline and never run real corpus images (those runs are manual, not
part of pytest).  They cover the annotation schema/validator, GT
discovery and optional behavior, geometric node matching, dependency
and direction comparison, duration comparison, precision/recall/F1,
and uncertain-item exclusion.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from pert_analyzer.benchmark.accuracy import (
    compare_aon,
    compare_dependencies,
    compare_durations,
    match_nodes,
    precision_recall_f1,
)
from pert_analyzer.benchmark.annotations import (
    SCHEMA_VERSION,
    AnnotationError,
    annotation_path_for_image,
    default_ground_truth_dir,
    from_dict,
    load_annotation,
    validate_annotation_dict,
)
from pert_analyzer.benchmark.discovery import discover_images

AON_DICT = {
    "schema_version": SCHEMA_VERSION,
    "image": "1.png",
    "diagram_type": "AON",
    "annotation_status": "COMPLETE",
    "activities": [
        {"id": "A", "duration": 5.0, "position": [100.0, 100.0],
         "bounding_box": [80.0, 80.0, 40.0, 40.0], "successors": ["B"]},
        {"id": "B", "duration": 3.0, "position": [240.0, 100.0],
         "bounding_box": [220.0, 80.0, 40.0, 40.0], "predecessors": ["A"]},
    ],
    "events": [],
    "dependencies": [{"source": "A", "target": "B"}],
}

AOA_DICT = {
    "schema_version": SCHEMA_VERSION,
    "image": "3.jpeg",
    "diagram_type": "AOA",
    "annotation_status": "COMPLETE",
    "activities": [
        {"id": "a1", "predecessors": ["S"], "successors": ["1"]},
        {"id": "a2", "predecessors": ["1"], "successors": ["F"]},
    ],
    "events": [
        {"id": "S", "position": [100.0, 100.0], "bounding_box": [65.0, 65.0, 70.0, 70.0]},
        {"id": "1", "position": [300.0, 100.0], "bounding_box": [265.0, 65.0, 70.0, 70.0]},
        {"id": "F", "position": [500.0, 100.0], "bounding_box": [465.0, 65.0, 70.0, 70.0]},
    ],
    "dependencies": [],
}


def _geoms(ids, xys, boxes):
    return {
        i: ({"position": [x, y], "bbox": list(b)} if b else {"position": [x, y]})
        for i, (x, y), b in zip(ids, xys, boxes)
    }


class _Item:
    def __init__(self, id_, geom):
        self.id = id_
        self.geom = geom


def _id(item):
    return item.id


def _geom_of(item):
    return dict(item.geom)


# ---------------------------------------------------------------------------
# 1. Schema validation
# ---------------------------------------------------------------------------


def test_schema_validation_rejects_missing_version():
    bad = {k: v for k, v in AON_DICT.items() if k != "schema_version"}
    with pytest.raises(AnnotationError):
        from_dict(bad)


def test_schema_validation_rejects_bad_diagram_type_and_status():
    bad_type = dict(AON_DICT, diagram_type="TREE")
    with pytest.raises(AnnotationError):
        from_dict(bad_type)
    bad_status = dict(AON_DICT, annotation_status="MAYBE")
    with pytest.raises(AnnotationError):
        from_dict(bad_status)


def test_aon_validation_passes():
    report = validate_annotation_dict(dict(AON_DICT))
    assert report.is_valid
    assert report.errors == []


def test_aoa_validation_passes():
    report = validate_annotation_dict(dict(AOA_DICT))
    assert report.is_valid
    assert report.errors == []


# ---------------------------------------------------------------------------
# 2. Duplicate ids / invalid deps / missing durations
# ---------------------------------------------------------------------------


def test_duplicate_activity_ids_detected():
    dup = {
        **AON_DICT,
        "activities": AON_DICT["activities"] + [{"id": "A"}],
    }
    report = validate_annotation_dict(dup)
    assert not report.is_valid
    assert any("duplicate" in i.message.lower() for i in report.errors)


def test_invalid_dependency_reference_detected():
    broken = {
        **AON_DICT,
        "dependencies": [{"source": "A", "target": "Z"}],
    }
    report = validate_annotation_dict(broken)
    assert not report.is_valid


def test_missing_durations_warns_but_is_valid():
    nodur = {
        **AON_DICT,
        "activities": [
            {k: v for k, v in a.items() if k != "duration"}
            for a in AON_DICT["activities"]
        ],
    }
    report = validate_annotation_dict(nodur)
    assert report.is_valid
    assert report.warnings


# ---------------------------------------------------------------------------
# 3. Ground truth discovery + optional behavior
# ---------------------------------------------------------------------------


def test_ground_truth_discovery_mapping(tmp_path):
    dataset = tmp_path / "Imag PERT"
    dataset.mkdir()
    (dataset / "1.png").write_bytes(b"\x89PNG")
    assert default_ground_truth_dir(dataset) == tmp_path / "ground_truth"
    path = annotation_path_for_image(dataset.parent / "ground_truth", dataset / "1.png")
    assert path == tmp_path / "ground_truth" / "1.json"
    assert discover_images(dataset) == [dataset / "1.png"]


def test_load_annotation_optional_behavior(tmp_path):
    dataset = tmp_path / "Imag PERT"
    dataset.mkdir()
    gt = tmp_path / "ground_truth"
    gt.mkdir()
    image = dataset / "1.png"
    image.write_bytes(b"\x89PNG")
    # No annotation present -> None.
    assert load_annotation(image, ground_truth_dir=gt) is None
    # Legacy file (no schema_version) is ignored by the v1.0 loader.
    (gt / "1.json").write_text(json.dumps({"counts": {"activities": 22}}), encoding="utf-8")
    assert load_annotation(image, ground_truth_dir=gt) is None
    # v1.0 file loads.
    (gt / "1.json").write_text(json.dumps(AON_DICT), encoding="utf-8")
    loaded = load_annotation(image, ground_truth_dir=gt)
    assert loaded is not None and loaded.diagram_type == "AON"


# ---------------------------------------------------------------------------
# 4. Node matching (geometric, one-to-one)
# ---------------------------------------------------------------------------


def test_match_nodes_outer_perfect_and_one_to_one():
    gt = _geoms(["A", "B", "C"],
                [(100.0, 100.0), (240.0, 100.0), (400.0, 300.0)],
                [(80.0, 80.0, 40.0, 40.0), (220.0, 80.0, 40.0, 40.0), (380.0, 280.0, 40.0, 40.0)])
    det = [_Item("d1", {"position": [100.0, 100.0], "bbox": [80.0, 80.0, 40.0, 40.0]}),
           _Item("d2", {"position": [240.0, 100.0], "bbox": [220.0, 80.0, 40.0, 40.0]}),
           _Item("d3", {"position": [400.0, 300.0], "bbox": [380.0, 280.0, 40.0, 40.0]}),
           _Item("dX", {"position": [900.0, 900.0], "bbox": [880.0, 880.0, 40.0, 40.0]})]
    matched = match_nodes(gt, det, _id, _geom_of)
    assert matched["countable"]
    assert len(matched["matched"]) == 3
    # One-to-one: no det id used twice.
    assert len(set(matched["matched"].values())) == 3
    assert matched["det_unmatched"] == ["dX"]


def test_match_nodes_center_fallback_without_bbox():
    gt = {"A": {"position": [100.0, 100.0]}}
    det = [_Item("d1", {"position": [101.0, 100.0]})]
    matched = match_nodes(gt, det, _id, _geom_of, center_tolerance=40.0)
    assert matched["countable"]
    assert matched["matched"] == {"A": "d1"}


# ---------------------------------------------------------------------------
# 5. Dependency matching, direction, durations, P/R/F1
# ---------------------------------------------------------------------------


def test_dependency_directed_precision_recall():
    gt = [("A", "B"), ("B", "C")]
    det = [("A", "B"), ("C", "D")]
    out = compare_dependencies(det, gt, {})
    directed = out["directed"]
    assert directed["precision"] == pytest.approx(0.5)  # 1 TP / 2 detected
    assert directed["recall"] == pytest.approx(0.5)    # 1 TP / 2 expected
    assert out["direction"]["correct_pairs"] == 1


def test_dependency_reversed_pair_tracks_direction_wrong():
    gt = [("A", "B")]
    det = [("B", "A")]
    out = compare_dependencies(det, gt, {})
    assert out["directed"]["precision"] == 0.0  # wrong direction = directed FP
    assert out["undirected_pair"]["precision"] == 1.0  # pair correct
    assert out["direction"]["correct_pairs"] == 0
    assert out["direction"]["reversed_pairs"] == 1
    assert out["direction"]["direction_accuracy"] == 0.0


def test_dependency_self_loop_does_not_crash_direction():
    """A self-loop (X -> X) collapses to a 1-element frozenset; direction
    comparison must stay well-defined instead of failing to unpack."""
    gt = [("A", "B"), ("B", "B")]
    det = [("A", "B"), ("B", "B")]
    out = compare_dependencies(det, gt, {})
    assert out["directed"]["precision"] == 1.0
    assert out["directed"]["recall"] == 1.0
    assert out["direction"]["correct_pairs"] == 2
    assert out["direction"]["reversed_pairs"] == 0


def test_duration_comparison_exact_absolute_relative():
    out = compare_durations({"A": "d1", "B": "d2"}, {"A": 10.0, "B": 20.0}, {"d1": 10.0, "d2": 22.0})
    assert out["compared"] == 2
    assert out["exact_match_accuracy"] == pytest.approx(0.5)
    assert out["mean_abs_error"] == pytest.approx(1.0)
    assert out["mean_relative_error"] == pytest.approx(0.05)  # (0.0 + 0.1) / 2


def test_duration_never_rounds_silently():
    out = compare_durations({"A": "d1"}, {"A": 1.3}, {"d1": 1.300001})
    assert out["exact_matches"] == 0  # diff beyond the 1e-9 tolerance is non-exact
    assert out["mean_abs_error"] == 0.0  # 1e-6 rounds to 4 decimals


def test_aon_id_accuracy_keys_by_activity_id_not_geometric_id():
    """OCR id accuracy must read the label the matcher used (activity_id),
    even when the semantic and geometric ids differ."""
    ann = from_dict(AON_DICT)
    reconstruction = SimpleNamespace(activities=[
        SimpleNamespace(
            activity_id="act_1", geometric_node_id="SHAPE_001",
            semantic_activity_id="A", duration=5.0,
            position=(100.0, 100.0),
            bounding_box=SimpleNamespace(x=80, y=80, width=40, height=40),
        ),
        SimpleNamespace(
            activity_id="act_2", geometric_node_id="SHAPE_002",
            semantic_activity_id="B", duration=3.0,
            position=(240.0, 100.0),
            bounding_box=SimpleNamespace(x=220, y=80, width=40, height=40),
        ),
    ])
    out = compare_aon(ann, reconstruction, [["A", "B"]])
    assert out["node_matching"]["matched"] == 2
    ids = out["ids"]
    assert ids["compared"] == 2
    assert ids["id_accuracy"] == 1.0  # 2/2 matched with correct semantic labels


def test_precision_recall_f1_values():
    prf = precision_recall_f1(tp=6, fp=2, fn=2)
    assert prf["precision"] == pytest.approx(0.75)
    assert prf["recall"] == pytest.approx(0.75)
    assert prf["f1"] == pytest.approx(0.75)
    assert prf["true_positives"] == 6


def test_f1_harmonic_mean():
    prf = precision_recall_f1(tp=8, fp=2, fn=4)
    assert prf["precision"] == pytest.approx(0.8)
    assert prf["recall"] == pytest.approx(0.6667, abs=1e-4)
    assert prf["f1"] == pytest.approx(2 * 0.8 * (2 / 3) / (0.8 + 2 / 3), abs=1e-4)


# ---------------------------------------------------------------------------
# 6. Uncertain exclusion
# ---------------------------------------------------------------------------


def test_uncertain_annotation_excludes_uncertain_items():
    ann = from_dict({
        **AON_DICT,
        "uncertain": {"activities": ["B"], "dependencies": [["A", "B"]]},
    })
    assert [a.id for a in ann.comparable_activities()] == ["A"]
    assert ann.comparable_dependencies() == []
    assert len(ann.uncertain_activities) == 1
    assert len(ann.uncertain_dependencies) == 1
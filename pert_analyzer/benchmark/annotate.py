"""
Command-line helper for v1.0 ground-truth annotations.

Usage::

    python -m pert_analyzer.benchmark.annotate <dataset> list
    python -m pert_analyzer.benchmark.annotate <dataset> show <image>
    python -m pert_analyzer.benchmark.annotate <dataset> init <image> --type AON|AOA
    python -m pert_analyzer.benchmark.annotate <dataset> validate [--strict]
    python -m pert_analyzer.benchmark.annotate <dataset> export-draft
    python -m pert_analyzer.benchmark.annotate <dataset> set-status <image> <COMPLETE|PARTIAL|UNCERTAIN>
    python -m pert_analyzer.benchmark.annotate <dataset> set-duration <image> <activity-id> <duration>
    python -m pert_analyzer.benchmark.annotate <dataset> set-expected <image> --activities N --events N --dependencies N
    python -m pert_analyzer.benchmark.annotate <dataset> add-activity <image> <id> [--duration D]
    python -m pert_analyzer.benchmark.annotate <dataset> add-event <image> <id>
    python -m pert_analyzer.benchmark.annotate <dataset> add-dependency <image> <source> <target>

It never touches the dataset or the production pipeline: it only reads
images to compute annotation paths and writes ``<gt_dir>/<image stem>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from pert_analyzer.benchmark.annotations import (
    ANNOTATION_STATUSES,
    SCHEMA_VERSION,
    SUPPORTED_DIAGRAM_TYPES,
    AnnotationError,
    GroundTruthValidator,
    annotation_path_for_image,
    default_ground_truth_dir,
    load_annotation,
    validate_annotation_dict,
)
from pert_analyzer.benchmark.discovery import discover_images


def _resolve_gt_dir(dataset: str | Path, gt_dir: str | Path | None) -> Path:
    if gt_dir is not None:
        return Path(gt_dir)
    return default_ground_truth_dir(dataset)


def _resolve_image(dataset: str | Path, image: str) -> Path:
    candidate = Path(dataset) / image
    if candidate.is_file():
        return candidate
    if Path(image).is_file():
        return Path(image)
    return candidate


def _cmd_list(dataset: str | Path, gt_dir: str | Path | None) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    images = discover_images(dataset)
    print(f"Dataset: {Path(dataset).resolve()}")
    print(f"Ground-truth dir: {gt}")
    print(f"Images found: {len(images)}")
    print("")
    header = f"{'Image':<60} {'Status':<12} {'Type':<6} State"
    print(header)
    print("-" * len(header))
    for image in images:
        path = annotation_path_for_image(gt, image)
        if path.is_file():
            try:
                ann = load_annotation(image, ground_truth_dir=gt)
                if ann is None:
                    state = "legacy (no schema_version)"
                    status = "-"
                    dtype = "-"
                else:
                    state = "v1.0"
                    status = ann.annotation_status
                    dtype = ann.diagram_type
            except AnnotationError as exc:
                state = f"ERROR: {exc}"
                status = "-"
                dtype = "-"
            print(f"{image.name:<60} {status:<12} {dtype:<6} {state}")
        else:
            print(f"{image.name:<60} {'-':<12} {'-':<6} missing")
    return 0


def _cmd_show(dataset: str | Path, gt_dir: str | Path | None, image: str) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    source = img
    path = annotation_path_for_image(gt, source)
    if not path.is_file():
        print(f"No annotation for {image!r} (expected {path})")
        return 1
    print(path)
    print(json.dumps(json.loads(path.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
    return 0


def _load_or_new(path: Path) -> Dict[str, Any]:
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                return raw
        except (OSError, json.JSONDecodeError):
            pass
    return {"schema_version": SCHEMA_VERSION}


def _cmd_init(
    dataset: str | Path,
    gt_dir: str | Path | None,
    image: str,
    diagram_type: str,
    status: str,
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    if not img.is_file():
        print(f"Image not found: {img}")
        return 1
    path = annotation_path_for_image(gt, img)
    if path.is_file():
        print(f"Annotation already exists: {path}")
        return 1
    gt.mkdir(parents=True, exist_ok=True)
    template = {
        "schema_version": SCHEMA_VERSION,
        "image": image,
        "diagram_type": diagram_type.upper(),
        "annotation_status": status.upper(),
        "activities": [],
        "events": [],
        "dependencies": [],
        "notes": "auto-created template; populate by hand",
    }
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Created: {path}")
    return 0


def _cmd_validate(dataset: str | Path, gt_dir: str | Path | None, strict: bool) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    images = discover_images(dataset)
    errors = 0
    warnings = 0
    found = 0
    print(f"Validating annotations in {gt} ...")
    print("")
    for image in images:
        path = annotation_path_for_image(gt, image)
        if not path.is_file():
            continue
        found += 1
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[{image.name}] UNREADABLE: {exc}")
            errors += 1
            continue
        if "schema_version" not in raw:
            print(f"[{image.name}] legacy file (no schema_version); skipped")
            continue
        try:
            report = validate_annotation_dict(raw)
        except AnnotationError as exc:
            print(f"[{image.name}] INVALID: {exc}")
            errors += 1
            continue
        for issue in report.issues:
            if issue.severity == "ERROR":
                errors += 1
                print(f"[{image.name}] ERROR {issue.code}: {issue.message}")
            else:
                warnings += 1
                print(f"[{image.name}] WARN  {issue.code}: {issue.message}")
        if report.issues:
            print(f"[{image.name}] -> {len(report.issues)} issue(s)")
        else:
            print(f"[{image.name}] OK")
    print("")
    print(f"Annotations found: {found} | errors: {errors} | warnings: {warnings}")
    if strict and warnings:
        print("Strict mode: warnings treated as failure.")
        return 1
    return 1 if errors else 0


def _write(raw: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(raw, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cmd_set_status(
    dataset: str | Path, gt_dir: str | Path | None, image: str, status: str
) -> int:
    status = status.upper()
    if status not in ANNOTATION_STATUSES:
        print(f"Invalid status {status!r}; expected one of {ANNOTATION_STATUSES}")
        return 1
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    if not path.is_file():
        print(f"No annotation: {path}")
        return 1
    raw = _load_or_new(path)
    raw["annotation_status"] = status
    _write(raw, path)
    print(f"{path.name}: annotation_status = {status}")
    return 0


def _cmd_set_duration(
    dataset: str | Path,
    gt_dir: str | Path | None,
    image: str,
    activity: str,
    duration: float,
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    raw = _load_or_new(path)
    for entry in raw.get("activities") or []:
        if entry.get("id") == activity:
            entry["duration"] = float(duration)
            _write(raw, path)
            print(f"{path.name}: activity {activity!r} duration = {duration}")
            return 0
    print(f"Activity {activity!r} not found in {path.name}")
    return 1


def _cmd_set_expected(
    dataset: str | Path,
    gt_dir: str | Path | None,
    image: str,
    activities: int | None,
    events: int | None,
    dependencies: int | None,
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    raw = _load_or_new(path)
    for key, value in (("expected_activities", activities),
                       ("expected_events", events),
                       ("expected_dependencies", dependencies)):
        if value is not None:
            raw[key] = int(value)
    _write(raw, path)
    print(f"{path.name}: expected counts updated -> {raw.get('expected_activities')} acts / "
          f"{raw.get('expected_events')} evts / {raw.get('expected_dependencies')} deps")
    return 0


def _cmd_add_activity(
    dataset: str | Path,
    gt_dir: str | Path | None,
    image: str,
    activity: str,
    duration: float | None,
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    raw = _load_or_new(path)
    entries = raw.setdefault("activities", [])
    if any(e.get("id") == activity for e in entries):
        print(f"Activity {activity!r} already present")
        return 1
    entry: Dict[str, Any] = {"id": activity}
    if duration is not None:
        entry["duration"] = float(duration)
    entries.append(entry)
    _write(raw, path)
    print(f"{path.name}: added activity {activity!r}")
    return 0


def _cmd_add_event(
    dataset: str | Path, gt_dir: str | Path | None, image: str, event: str
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    raw = _load_or_new(path)
    events = raw.setdefault("events", [])
    if any(e.get("id") == event for e in events):
        print(f"Event {event!r} already present")
        return 1
    events.append({"id": event})
    _write(raw, path)
    print(f"{path.name}: added event {event!r}")
    return 0


def _geom_blob(position, bbox) -> Dict[str, Any]:
    """Serialize detector geometry into the annotation schema's units."""
    geom: Dict[str, Any] = {}
    if position is not None:
        geom["position"] = [float(position[0]), float(position[1])]
    if bbox is not None:
        if hasattr(bbox, "x"):
            geom["bbox"] = [
                float(bbox.x), float(bbox.y),
                float(bbox.width), float(bbox.height),
            ]
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            geom["bbox"] = [float(v) for v in bbox[:4]]
    return geom


def _diagram_type_of(result: Any) -> str:
    value = getattr(result, "diagram_type", None)
    if hasattr(value, "value"):
        value = value.value
    return str(value or "UNKNOWN").upper()


def _event_geometry(events) -> Dict[str, Any]:
    """event_id -> (position, radius) for the draft arrow association."""
    positions, radii = {}, {}
    for evt in events:
        if evt.position is None:
            continue
        positions[evt.event_id] = (
            float(evt.position[0]), float(evt.position[1]))
        bb = getattr(evt, "bounding_box", None)
        radius = 0.0
        if bb is not None:
            radius = (float(bb.width) + float(bb.height)) / 4.0
        radii[evt.event_id] = radius
    return positions, radii


def _unique_draft_id(raw_id: str, seen: Dict[str, int]) -> str:
    """Return a unique id, appending _2, _3... when a raw id repeats (OCR)."""
    if raw_id not in seen:
        seen[raw_id] = 1
        return raw_id
    seen[raw_id] += 1
    return f"{raw_id}_{seen[raw_id]}"


def _draft_for(
    image: Path,
    workflow: Any,
    dtype: str,
) -> Dict[str, Any]:
    """Build a v1.0 annotation DRAFT from real detector output.

    The draft is explicitly marked UNCERTAIN and carries a note that the
    user must verify ids against the image, add durations and fix the
    dependency list before setting ``annotation_status`` to COMPLETE.
    """
    result = getattr(workflow, "pipeline_result", None) or getattr(
        workflow, "result", None
    )
    reconstruction = getattr(result, "_reconstruction", None)
    notes = (
        "DRAFT seeded from detector output. Verify every id against the "
        "image (correct OCR errors), add durations, add/remove dependencies, "
        "then set annotation_status=COMPLETE for benchmark inclusion."
    )
    raw: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "image": image.name,
        "diagram_type": dtype,
        "annotation_status": "UNCERTAIN",
        "activities": [],
        "events": [],
        "dependencies": [],
        "notes": notes,
    }
    if reconstruction is None:
        return raw

    if dtype == "AON":
        acts = list(getattr(reconstruction, "activities", None) or [])
        seen: Dict[str, int] = {}
        id_of: Dict[str, str] = {}
        for idx, act in enumerate(acts):
            base_key = act.activity_id or act.geometric_node_id or f"ACT_{idx}"
            raw_id = (
                act.semantic_activity_id
                or act.activity_id
                or f"INFERRED_{idx}"
            )
            chosen = _unique_draft_id(str(raw_id), seen)
            id_of[str(base_key)] = chosen
            entry: Dict[str, Any] = {"id": chosen}
            if act.duration is not None and float(act.duration) > 0.0:
                entry["duration"] = float(act.duration)
            entry.update(_geom_blob(act.position, act.bounding_box))
            raw["activities"].append(entry)

        # Seed dependencies from the reviewed candidate graph (if available),
        # remapping graph node ids back to the (deduplicated) draft ids.
        apply = getattr(workflow, "apply", None)
        if callable(apply):
            try:
                candidate = apply()
                graph = getattr(candidate, "graph", None)
                for dep in getattr(graph, "dependencies", None) or []:
                    source = getattr(dep, "source", None) or getattr(
                        dep, "source_id", None
                    )
                    target = getattr(dep, "target", None) or getattr(
                        dep, "target_id", None
                    )
                    if source is None or target is None:
                        continue
                    src = id_of.get(str(source), str(source))
                    tgt = id_of.get(str(target), str(target))
                    raw["dependencies"].append(
                        {"source": src, "target": tgt}
                    )
            except Exception:
                pass
    else:
        events = list(getattr(reconstruction, "events", None) or [])
        seen = {}
        id_of = {}
        for evt in events:
            chosen = _unique_draft_id(str(evt.event_id), seen)
            id_of[str(evt.event_id)] = chosen
            entry = {"id": chosen}
            entry.update(_geom_blob(evt.position, evt.bounding_box))
            raw["events"].append(entry)

        positions, radii = _event_geometry(events)
        arrow_result = getattr(result, "_arrow_result", None)
        if positions:
            # Seed event-pair dependencies using the same association the
            # accuracy comparator uses (nearest event per arrow endpoint).
            from pert_analyzer.benchmark.accuracy import _associate_arrow_events

            arrows = list(getattr(arrow_result, "arrows", None) or [])
            for assoc in _associate_arrow_events(
                arrows, events, positions, radii
            ):
                pair = assoc.get("pair")
                if pair:
                    raw["dependencies"].append(
                        {
                            "source": id_of.get(str(pair[0]), str(pair[0])),
                            "target": id_of.get(str(pair[1]), str(pair[1])),
                        }
                    )
    return raw


def _cmd_export_draft(dataset: str | Path, gt_dir: str | Path | None) -> int:
    """Run the real pipeline on every image and seed an annotation draft."""
    from pert_analyzer.pipeline.review_api import ReviewWorkflow

    gt = _resolve_gt_dir(dataset, gt_dir)
    images = discover_images(dataset)
    print(f"Drafting annotations from detector output into {gt}")
    print(f"Images: {len(images)}")
    print("")
    ok, skipped = 0, 0
    for image in images:
        try:
            workflow = ReviewWorkflow.analyze(
                str(image), source_image_id=image.name, stage_callback=None
            )
            result = getattr(workflow, "pipeline_result", None) or getattr(
                workflow, "result", None
            )
        except Exception as exc:
            print(f"[{image.name}] ANALYSIS FAILED: {exc}")
            skipped += 1
            continue
        dtype = _diagram_type_of(result)
        if dtype not in SUPPORTED_DIAGRAM_TYPES or result is None:
            print(f"[{image.name}] SKIPPED (no recognised diagram: {dtype})")
            skipped += 1
            continue
        raw = _draft_for(image, workflow, dtype)
        if dtype == "AON":
            if not raw["activities"]:
                print(f"[{image.name}] SKIPPED (0 detected activities)")
                skipped += 1
                continue
        elif not raw["events"]:
            print(f"[{image.name}] SKIPPED (0 detected events)")
            skipped += 1
            continue
        path = annotation_path_for_image(gt, image)
        _write(raw, path)
        print(
            f"[{image.name}] {dtype}: {len(raw['activities'])} acts / "
            f"{len(raw['events'])} evts / {len(raw['dependencies'])} deps "
            f"-> {path}"
        )
        ok += 1
    print("")
    print(f"Drafted: {ok} | skipped: {skipped}")
    return 0


def _cmd_add_dependency(
    dataset: str | Path,
    gt_dir: str | Path | None,
    image: str,
    source: str,
    target: str,
) -> int:
    gt = _resolve_gt_dir(dataset, gt_dir)
    img = _resolve_image(dataset, image)
    path = annotation_path_for_image(gt, img)
    raw = _load_or_new(path)
    deps = raw.setdefault("dependencies", [])
    if any(d.get("source") == source and d.get("target") == target for d in deps):
        print(f"Dependency {source} -> {target} already present")
        return 1
    deps.append({"source": source, "target": target})
    _write(raw, path)
    print(f"{path.name}: added dependency {source} -> {target}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m pert_analyzer.benchmark.annotate",
        description="Manage v1.0 ground-truth annotations for the benchmark corpus.",
    )
    parser.add_argument("dataset", help="Path to the benchmark dataset (folder of images).")
    parser.add_argument("--gt-dir", default=None, help="Ground-truth directory (default: sibling 'ground_truth').")
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="List images and their annotation state.")
    p_list.set_defaults(func=_cmd_list)

    p_show = sub.add_parser("show", help="Show one annotation JSON.")
    p_show.add_argument("image")
    p_show.set_defaults(func=_cmd_show)

    p_init = sub.add_parser("init", help="Create an annotation template.")
    p_init.add_argument("image")
    p_init.add_argument("--type", choices=SUPPORTED_DIAGRAM_TYPES, required=True)
    p_init.add_argument("--status", choices=ANNOTATION_STATUSES, default="UNCERTAIN")
    p_init.set_defaults(func=_cmd_init)

    p_val = sub.add_parser("validate", help="Validate all annotations.")
    p_val.add_argument("--strict", action="store_true")
    p_val.set_defaults(func=_cmd_validate)

    p_draft = sub.add_parser(
        "export-draft",
        help="Run the pipeline and seed annotation drafts from detector output."
        " Drafts are marked UNCERTAIN until hand-verified.",
    )
    p_draft.set_defaults(func=_cmd_export_draft)

    p_status = sub.add_parser("set-status", help="Set annotation_status.")
    p_status.add_argument("image")
    p_status.add_argument("status", choices=ANNOTATION_STATUSES)
    p_status.set_defaults(func=_cmd_set_status)

    p_dur = sub.add_parser("set-duration", help="Set an activity duration.")
    p_dur.add_argument("image")
    p_dur.add_argument("activity")
    p_dur.add_argument("duration", type=float)
    p_dur.set_defaults(func=_cmd_set_duration)

    p_exp = sub.add_parser("set-expected", help="Set expected counts.")
    p_exp.add_argument("image")
    p_exp.add_argument("--activities", type=int, default=None)
    p_exp.add_argument("--events", type=int, default=None)
    p_exp.add_argument("--dependencies", type=int, default=None)
    p_exp.set_defaults(func=_cmd_set_expected)

    p_a = sub.add_parser("add-activity", help="Add an activity entry.")
    p_a.add_argument("image")
    p_a.add_argument("id")
    p_a.add_argument("--duration", type=float, default=None)
    p_a.set_defaults(func=_cmd_add_activity)

    p_e = sub.add_parser("add-event", help="Add an event entry.")
    p_e.add_argument("image")
    p_e.add_argument("id")
    p_e.set_defaults(func=_cmd_add_event)

    p_d = sub.add_parser("add-dependency", help="Add a dependency edge.")
    p_d.add_argument("image")
    p_d.add_argument("source")
    p_d.add_argument("target")
    p_d.set_defaults(func=_cmd_add_dependency)

    return parser


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    def _dispatch(**kwargs) -> int:
        return args.func(args.dataset, args.gt_dir, **kwargs)

    if args.command == "list":
        return _cmd_list(args.dataset, args.gt_dir)
    if args.command == "validate":
        return _cmd_validate(args.dataset, args.gt_dir, args.strict)
    if args.command == "export-draft":
        return _cmd_export_draft(args.dataset, args.gt_dir)
    if args.command == "init":
        return _cmd_init(args.dataset, args.gt_dir, args.image, args.type, args.status)
    if args.command == "show":
        return _cmd_show(args.dataset, args.gt_dir, args.image)
    if args.command == "set-status":
        return _cmd_set_status(args.dataset, args.gt_dir, args.image, args.status)
    if args.command == "set-duration":
        return _cmd_set_duration(args.dataset, args.gt_dir, args.image, args.activity, args.duration)
    if args.command == "set-expected":
        return _cmd_set_expected(args.dataset, args.gt_dir, args.image,
                                 args.activities, args.events, args.dependencies)
    if args.command == "add-activity":
        return _cmd_add_activity(args.dataset, args.gt_dir, args.image, args.id, args.duration)
    if args.command == "add-event":
        return _cmd_add_event(args.dataset, args.gt_dir, args.image, args.id)
    if args.command == "add-dependency":
        return _cmd_add_dependency(args.dataset, args.gt_dir, args.image, args.source, args.target)
    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    sys.exit(main())

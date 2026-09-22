"""
Ground-truth annotation schema, discovery, loading and validation (v1.0).

The corpus images themselves are never modified.  A dedicated folder
(``tests/test_data/ground_truth/``) holds one JSON file per diagram,
named ``<image stem>.json`` (e.g. ``1.png`` -> ``1.json``).  The mapping
is deterministic and documented in the annotation header.

Schema v1.0 (full form)::

    {
      "schema_version": "1.0",
      "annotation_status": "COMPLETE | PARTIAL | UNCERTAIN",
      "image": "8.jpeg",
      "diagram_type": "AON | AOA",
      "activities": [
        {
          "id": "A",
          "duration": 2,
          "position": [x, y],          # optional
          "bbox": [x, y, w, h],        # optional
          "predecessors": ["B"],       # optional, AON
          "successors": ["C"],         # optional, AON
          "dummy": false               # optional, AOA
        }
      ],
      "events": [                      # AOA only
        {"id": "1", "position": [x, y], "bbox": [x, y, w, h]}
      ],
      "dependencies": [["A", "B"], ...],          # AON: activity -> activity
                                                    # AOA: event  -> event (optional override)
      "expected_activities": 22,       # optional manual counts
      "expected_events": 30,
      "expected_dependencies": 28,
      "uncertain": {                   # items to keep OUT of metric math
        "activities": ["X"],
        "events": ["e3"],
        "dependencies": [["P", "Q"]],
        "notes": "handwritten text not readable"
      },
      "notes": "free text"
    }

Rules
-----
- ``diagram_type`` selects the AON or AOA comparison path.
- Activity/event/dependency ids must be unique and self-referential
  references must be resolvable.
- AON: activities are nodes; dependencies are directed precedence pairs
  between activity ids.
- AOA: activities/arrows are edges between ``source_event``/``target_event``
  event ids (stored as ``predecessors=[source_event]`` and
  ``successors=[target_event]`` or the explicit ``dependencies`` list).
- Unknown truth must be marked ``UNCERTAIN`` (whole annotation or specific
  items), never fabricated.  Items listed under ``uncertain`` are excluded
  from metric calculations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = "1.0"
SUPPORTED_DIAGRAM_TYPES = ("AON", "AOA")
ANNOTATION_STATUSES = ("COMPLETE", "PARTIAL", "UNCERTAIN")

# Default ground-truth directory: sibling of the dataset root named
# ``ground_truth`` (``tests/test_data/Imag PERT`` -> ``tests/test_data/ground_truth``).
def default_ground_truth_dir(dataset_root: str | Path) -> Path:
    return Path(dataset_root).resolve().parent / "ground_truth"


def annotation_path_for_image(
    ground_truth_dir: str | Path, image_path: str | Path
) -> Path:
    """Return the deterministic annotation JSON path for an image.

    Mapping rule: ``<ground_truth_dir>/<image stem>.json``.  Stems are
    unique across the corpus (documented; no collisions exist).
    """
    return Path(ground_truth_dir) / (Path(image_path).stem + ".json")


class AnnotationError(Exception):
    """Raised when an annotation file cannot be loaded or is invalid."""


class AnnotationIssue:
    """A single validation finding with severity."""

    __slots__ = ("severity", "code", "message")

    def __init__(self, severity: str, code: str, message: str):
        self.severity = severity  # "ERROR" | "WARNING"
        self.code = code
        self.message = message

    def to_dict(self) -> Dict[str, str]:
        return {"severity": self.severity, "code": self.code, "message": self.message}

    def __repr__(self) -> str:  # pragma: no cover - debug only
        return f"{self.severity}[{self.code}]: {self.message}"


@dataclass
class ValidationReport:
    """Result of validating one annotation."""

    issues: List[AnnotationIssue] = field(default_factory=list)

    @property
    def errors(self) -> List[AnnotationIssue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[AnnotationIssue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_dict() for i in self.issues],
        }


@dataclass
class AnnotatedActivity:
    """One annotated activity (AON node or AOA arrow)."""

    id: str
    duration: Optional[float] = None
    position: Optional[Tuple[float, float]] = None
    bbox: Optional[List[float]] = None
    predecessors: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    dummy: bool = False
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def source_event(self) -> Optional[str]:
        vals = [v for v in self.predecessors if isinstance(v, str)]
        return vals[0] if vals else None

    @property
    def target_event(self) -> Optional[str]:
        vals = [v for v in self.successors if isinstance(v, str)]
        return vals[0] if vals else None


@dataclass
class AnnotatedEvent:
    """One annotated event node (AOA circle)."""

    id: str
    position: Optional[Tuple[float, float]] = None
    bbox: Optional[List[float]] = None
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruthAnnotation:
    """Parsed v1.0 annotation for one image."""

    image: str = ""
    schema_version: str = SCHEMA_VERSION
    annotation_status: str = "UNCERTAIN"
    diagram_type: str = "UNKNOWN"
    activities: List[AnnotatedActivity] = field(default_factory=list)
    events: List[AnnotatedEvent] = field(default_factory=list)
    dependencies: List[Tuple[str, str]] = field(default_factory=list)
    expected_activities: Optional[int] = None
    expected_events: Optional[int] = None
    expected_dependencies: Optional[int] = None
    uncertain: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    # -- Derived sets -------------------------------------------------------
    @property
    def uncertain_activities(self) -> set:
        return {str(x) for x in (self.uncertain.get("activities") or [])}

    @property
    def uncertain_events(self) -> set:
        return {str(x) for x in (self.uncertain.get("events") or [])}

    @property
    def uncertain_dependencies(self) -> set:
        return {
            tuple(str(x) for x in d)
            for d in (self.uncertain.get("dependencies") or [])
        }

    def comparable_activities(self) -> List[AnnotatedActivity]:
        """Activities that are safe to include in metric math."""
        bad = self.uncertain_activities
        return [a for a in self.activities if a.id not in bad]

    def comparable_events(self) -> List[AnnotatedEvent]:
        bad = self.uncertain_events
        return [e for e in self.events if e.id not in bad]

    def comparable_dependencies(self) -> List[Tuple[str, str]]:
        """Normalized dependency list minus uncertain pairs (AON path).

        For AOA the canonical edge set is derived from comparable
        activities' event transitions unless an explicit ``dependencies``
        list is provided (then that list is used after filtering).
        """
        bad = self.uncertain_dependencies
        deps = self.dependencies or self._derived_aoa_deps()
        out = [d for d in deps if (d[0], d[1]) not in bad]
        # de-duplicate while preserving order
        seen = set()
        result = []
        for d in out:
            if d not in seen:
                seen.add(d)
                result.append(d)
        return result

    def _derived_aoa_deps(self) -> List[Tuple[str, str]]:
        if self.diagram_type != "AOA":
            return []
        out: List[Tuple[str, str]] = []
        for a in self.comparable_activities():
            a_uncertain = a.id in self.uncertain_activities
            if a_uncertain:
                continue
            src = a.source_event
            tgt = a.target_event
            if src and tgt:
                out.append((src, tgt))
        return out

    def activity_count(self) -> int:
        return len(self.comparable_activities())

    def event_count(self) -> int:
        return len(self.comparable_events())

    def dependency_count(self) -> int:
        return len(self.comparable_dependencies())

    def duration_map(self) -> Dict[str, float]:
        """id -> duration for comparable activities that carry one."""
        bad = self.uncertain_activities
        return {
            a.id: float(a.duration)
            for a in self.activities
            if a.id not in bad and a.duration is not None
        }

    def activity_geometries(self) -> Dict[str, Dict[str, Any]]:
        bad = self.uncertain_activities
        out: Dict[str, Dict[str, Any]] = {}
        for a in self.activities:
            if a.id in bad:
                continue
            entry: Dict[str, Any] = {}
            if a.position is not None:
                entry["position"] = a.position
            if a.bbox is not None:
                entry["bbox"] = a.bbox
            out[a.id] = entry
        return out

    def event_geometries(self) -> Dict[str, Dict[str, Any]]:
        bad = self.uncertain_events
        out: Dict[str, Dict[str, Any]] = {}
        for e in self.events:
            if e.id in bad:
                continue
            entry: Dict[str, Any] = {}
            if e.position is not None:
                entry["position"] = e.position
            if e.bbox is not None:
                entry["bbox"] = e.bbox
            out[e.id] = entry
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "annotation_status": self.annotation_status,
            "image": self.image,
            "diagram_type": self.diagram_type,
            "expected_activities": self.expected_activities,
            "expected_events": self.expected_events,
            "expected_dependencies": self.expected_dependencies,
            "activity_count": self.activity_count(),
            "event_count": self.event_count(),
            "dependency_count": self.dependency_count(),
            "uncertain_counts": {
                "activities": len(self.uncertain_activities),
                "events": len(self.uncertain_events),
                "dependencies": len(self.uncertain_dependencies),
            },
            "source_path": self.source_path,
        }


# =============================================================================
# Parsing
# =============================================================================


def _parse_bbox(value: Any, what: str) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) not in (4, 2):
        raise AnnotationError(
            f"{what} bounding box must be [x, y, w, h] or [x, y]: {value!r}"
        )
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        raise AnnotationError(f"{what} bounding box must be numeric: {value!r}")
    if len(nums) == 2:
        nums = nums + [0.0, 0.0]
    if nums[2] < 0 or nums[3] < 0:
        raise AnnotationError(f"{what} bounding box has negative size: {value!r}")
    return nums


def _parse_position(value: Any, what: str) -> Tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise AnnotationError(f"{what} position must be [x, y]: {value!r}")
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, ValueError):
        raise AnnotationError(f"{what} position must be numeric: {value!r}")


def _parse_activity(raw: Dict[str, Any]) -> AnnotatedActivity:
    aid = raw.get("id")
    if not isinstance(aid, str) or not aid.strip():
        raise AnnotationError(f"Activity without a string id: {raw!r}")
    duration = raw.get("duration")
    if duration is not None:
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise AnnotationError(f"Activity {aid!r} has non-numeric duration")
    position = None
    if "position" in raw and raw["position"] is not None:
        position = _parse_position(raw["position"], f"activity {aid!r}")
    bbox = None
    if "bbox" in raw and raw["bbox"] is not None:
        bbox = _parse_bbox(raw["bbox"], f"activity {aid!r}")
    elif "bounding_box" in raw and raw["bounding_box"] is not None:
        bbox = _parse_bbox(raw["bounding_box"], f"activity {aid!r}")
    preds = [str(x) for x in (raw.get("predecessors") or []) if isinstance(x, str)]
    succs = [str(x) for x in (raw.get("successors") or []) if isinstance(x, str)]
    dummy = bool(raw.get("dummy", False))
    return AnnotatedActivity(
        id=aid,
        duration=duration,
        position=position,
        bbox=bbox,
        predecessors=preds,
        successors=succs,
        dummy=dummy,
        raw=raw,
    )


def _parse_event(raw: Dict[str, Any]) -> AnnotatedEvent:
    eid = raw.get("id")
    if not isinstance(eid, str) or not eid.strip():
        raise AnnotationError(f"Event without a string id: {raw!r}")
    position = None
    if "position" in raw and raw["position"] is not None:
        position = _parse_position(raw["position"], f"event {eid!r}")
    bbox = None
    if "bbox" in raw and raw["bbox"] is not None:
        bbox = _parse_bbox(raw["bbox"], f"event {eid!r}")
    elif "bounding_box" in raw and raw["bounding_box"] is not None:
        bbox = _parse_bbox(raw["bounding_box"], f"event {eid!r}")
    return AnnotatedEvent(id=eid, position=position, bbox=bbox, raw=raw)


def _parse_dependencies(raw: Any) -> List[Tuple[str, str]]:
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise AnnotationError("dependencies must be a list")
    out: List[Tuple[str, str]] = []
    for entry in raw:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and all(isinstance(x, str) for x in entry)
        ):
            out.append((entry[0], entry[1]))
        elif isinstance(entry, dict):
            src = entry.get("source") or entry.get("source_id")
            tgt = entry.get("target") or entry.get("target_id")
            if isinstance(src, str) and isinstance(tgt, str):
                out.append((src, tgt))
            else:
                raise AnnotationError(f"Invalid dependency entry: {entry!r}")
        else:
            raise AnnotationError(f"Invalid dependency entry: {entry!r}")
    return out


def _parse_uncertain(raw: Any) -> Dict[str, Any]:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise AnnotationError("uncertain must be an object")
    out = dict(raw)
    for key, value in list(out.items()):
        if key in ("activities", "events"):
            out[key] = [str(x) for x in (value or [])]
        elif key == "dependencies":
            out[key] = [
                tuple(str(x) for x in d)
                for d in (value or [])
                if isinstance(d, (list, tuple)) and len(d) == 2
            ]
    return out


def from_dict(raw: Dict[str, Any], source_path: Optional[str] = None) -> GroundTruthAnnotation:
    """Parse a v1.0 annotation dict into a structured object."""
    if not isinstance(raw, dict):
        raise AnnotationError("annotation must be a JSON object")
    schema = raw.get("schema_version")
    if not isinstance(schema, str):
        raise AnnotationError("missing schema_version")
    if schema != SCHEMA_VERSION:
        raise AnnotationError(
            f"unsupported schema_version {schema!r} (expected {SCHEMA_VERSION!r})"
        )
    diagram_type = str(raw.get("diagram_type") or "UNKNOWN").upper()
    if diagram_type not in SUPPORTED_DIAGRAM_TYPES:
        raise AnnotationError(
            f"unsupported diagram_type {diagram_type!r} "
            f"(expected one of {SUPPORTED_DIAGRAM_TYPES})"
        )
    status = str(raw.get("annotation_status") or "UNCERTAIN").upper()
    if status not in ANNOTATION_STATUSES:
        raise AnnotationError(
            f"unsupported annotation_status {status!r} "
            f"(expected one of {ANNOTATION_STATUSES})"
        )

    activities: List[AnnotatedActivity] = []
    acts_raw = raw.get("activities")
    if acts_raw is not None:
        if not isinstance(acts_raw, list):
            raise AnnotationError("activities must be a list")
        for entry in acts_raw:
            if isinstance(entry, str):
                entry = {"id": entry}
            if not isinstance(entry, dict):
                raise AnnotationError(f"Invalid activity entry: {entry!r}")
            activities.append(_parse_activity(entry))

    events: List[AnnotatedEvent] = []
    evts_raw = raw.get("events")
    if evts_raw is not None:
        if not isinstance(evts_raw, list):
            raise AnnotationError("events must be a list")
        for entry in evts_raw:
            if isinstance(entry, str):
                entry = {"id": entry}
            if not isinstance(entry, dict):
                raise AnnotationError(f"Invalid event entry: {entry!r}")
            events.append(_parse_event(entry))

    dependencies = _parse_dependencies(raw.get("dependencies"))

    def _opt_int(key: str) -> Optional[int]:
        value = raw.get(key)
        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            raise AnnotationError(f"{key} must be an integer, got {value!r}")

    return GroundTruthAnnotation(
        image=str(raw.get("image") or ""),
        schema_version=SCHEMA_VERSION,
        annotation_status=status,
        diagram_type=diagram_type,
        activities=activities,
        events=events,
        dependencies=dependencies,
        expected_activities=_opt_int("expected_activities"),
        expected_events=_opt_int("expected_events"),
        expected_dependencies=_opt_int("expected_dependencies"),
        uncertain=_parse_uncertain(raw.get("uncertain")),
        notes=str(raw.get("notes") or ""),
        raw=raw,
        source_path=source_path,
    )


def load_annotation(
    image_path: str | Path,
    ground_truth_dir: str | Path | None = None,
) -> Optional[GroundTruthAnnotation]:
    """Load the v1.0 annotation for ``image_path``.

    Lookup order: explicit ``ground_truth_dir`` first; otherwise the
    default sibling ``ground_truth`` folder; finally a same-basename
    ``.json`` next to the image (legacy).  Returns ``None`` when no
    v1.0 annotation exists, raises :class:`AnnotationError` on invalid
    content that *is* schema-versioned.
    """
    image = Path(image_path)
    candidates: List[Path] = []
    if ground_truth_dir is not None:
        candidates.append(annotation_path_for_image(ground_truth_dir, image))
    else:
        # The default ground truth folder is a sibling of the *dataset
        # root* (the folder that contains the images), not of the image.
        candidates.append(
            annotation_path_for_image(
                default_ground_truth_dir(image.parent), image
            )
        )
    if image.with_suffix(".json").is_file():
        candidates.append(image.with_suffix(".json"))

    for path in candidates:
        if not path.is_file():
            continue
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise AnnotationError(f"Unreadable annotation file {path}: {exc}")
        if not isinstance(raw, dict):
            raise AnnotationError(f"Annotation file {path} must contain a JSON object")
        # A file without schema_version is the legacy Mode-B ground truth:
        # not a v1.0 annotation, so ignore it here (the legacy loader
        # handles it separately).
        if "schema_version" not in raw:
            continue
        return from_dict(raw, source_path=str(path))
    return None


# =============================================================================
# Validation
# =============================================================================


class GroundTruthValidator:
    """Validates annotation structure, references and constraints."""

    def validate(self, annotation: GroundTruthAnnotation) -> ValidationReport:
        """Validate a parsed annotation; returns issues (never raises)."""
        report = ValidationReport()
        if not annotation.schema_version:
            report.issues.append(
                AnnotationIssue("ERROR", "schema_missing", "missing schema_version")
            )
        elif annotation.schema_version != SCHEMA_VERSION:
            report.issues.append(
                AnnotationIssue(
                    "ERROR",
                    "schema_unsupported",
                    f"unsupported schema_version {annotation.schema_version!r}",
                )
            )
        if annotation.diagram_type not in SUPPORTED_DIAGRAM_TYPES:
            report.issues.append(
                AnnotationIssue(
                    "ERROR",
                    "diagram_type",
                    f"unsupported diagram_type {annotation.diagram_type!r}",
                )
            )
        if annotation.annotation_status not in ANNOTATION_STATUSES:
            report.issues.append(
                AnnotationIssue(
                    "ERROR",
                    "annotation_status",
                    f"unsupported annotation_status {annotation.annotation_status!r}",
                )
            )

        # Duplicate ids.
        activity_ids = [a.id for a in annotation.activities]
        event_ids = [e.id for e in annotation.events]
        seen_acts = set()
        for aid in activity_ids:
            if aid in seen_acts:
                report.issues.append(
                    AnnotationIssue("ERROR", "duplicate_activity_id", f"duplicate activity id {aid!r}")
                )
            seen_acts.add(aid)
        seen_evts = set()
        for eid in event_ids:
            if eid in seen_evts:
                report.issues.append(
                    AnnotationIssue("ERROR", "duplicate_event_id", f"duplicate event id {eid!r}")
                )
            seen_evts.add(eid)

        # Cross-id namespace collisions.
        overlap = seen_acts & seen_evts
        for oid in sorted(overlap):
            report.issues.append(
                AnnotationIssue(
                    "WARNING",
                    "id_namespace_collision",
                    f"id {oid!r} used by both an activity and an event",
                )
            )

        # References.
        dep_ref_idspace = event_ids if annotation.diagram_type == "AOA" else activity_ids
        dep_ref_kind = "event" if annotation.diagram_type == "AOA" else "activity"
        self._validate_refs(
            report,
            dep_ref_kind,
            dep_ref_idspace,
            annotation.dependencies and [d[0] for d in annotation.dependencies] or [],
        )
        self._validate_refs(
            report,
            dep_ref_kind,
            dep_ref_idspace,
            annotation.dependencies and [d[1] for d in annotation.dependencies] or [],
        )
        node_ids = set(activity_ids) | set(event_ids)
        for act in annotation.activities:
            for pred in act.predecessors:
                if pred not in node_ids:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "invalid_reference",
                            f"activity {act.id!r} predecessor {pred!r} does not exist",
                        )
                    )
            for succ in act.successors:
                if succ not in node_ids:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "invalid_reference",
                            f"activity {act.id!r} successor {succ!r} does not exist",
                        )
                    )

        # AOA: activities are arrows -> exactly one source and one target event.
        if annotation.diagram_type == "AOA":
            for act in annotation.activities:
                src = act.source_event
                tgt = act.target_event
                if src is None:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "missing_aoa_source",
                            f"AOA activity {act.id!r} has no source_event",
                        )
                    )
                elif src not in event_ids:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "invalid_aoa_source",
                            f"AOA activity {act.id!r} source_event {src!r} is not an event",
                        )
                    )
                if tgt is None:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "missing_aoa_target",
                            f"AOA activity {act.id!r} has no target_event",
                        )
                    )
                elif tgt not in event_ids:
                    report.issues.append(
                        AnnotationIssue(
                            "ERROR",
                            "invalid_aoa_target",
                            f"AOA activity {act.id!r} target_event {tgt!r} is not an event",
                        )
                    )
                if not act.dummy and act.duration is None:
                    report.issues.append(
                        AnnotationIssue(
                            "WARNING",
                            "missing_duration",
                            f"AOA activity {act.id!r} has no duration",
                        )
                    )

        if annotation.diagram_type == "AON":
            for act in annotation.activities:
                if not act.dummy and act.duration is None:
                    report.issues.append(
                        AnnotationIssue(
                            "WARNING",
                            "missing_duration",
                            f"AON activity {act.id!r} has no duration",
                        )
                    )

        # Bounding boxes sanity.
        for act in annotation.activities:
            self._check_bbox(report, f"activity {act.id!r}", act.bbox)
        for evt in annotation.events:
            self._check_bbox(report, f"event {evt.id!r}", evt.bbox)

        # Expected counts must be non-negative when provided.
        for key in ("expected_activities", "expected_events", "expected_dependencies"):
            value = getattr(annotation, key)
            if value is not None and value < 0:
                report.issues.append(
                    AnnotationIssue(
                        "ERROR", "negative_expected", f"{key} must be >= 0"
                    )
                )

        # Uncertain items must reference existing ids/pairs.
        valid_ids = node_ids
        for key in ("activities", "events"):
            for uid in annotation.uncertain.get(key) or []:
                if key == "activities" and uid not in seen_acts:
                    report.issues.append(
                        AnnotationIssue(
                            "WARNING",
                            "uncertain_unknown_reference",
                            f"uncertain {key} item {uid!r} is not annotated",
                        )
                    )
                if key == "events" and uid not in seen_evts:
                    report.issues.append(
                        AnnotationIssue(
                            "WARNING",
                            "uncertain_unknown_reference",
                            f"uncertain {key} item {uid!r} is not annotated",
                        )
                    )
        self._validate_refs(
            report,
            "id",
            sorted(valid_ids),
            [d[0] for d in (annotation.uncertain.get("dependencies") or [])],
        )
        self._validate_refs(
            report,
            "id",
            sorted(valid_ids),
            [d[1] for d in (annotation.uncertain.get("dependencies") or [])],
        )

        return report

    @staticmethod
    def _validate_refs(
        report: ValidationReport, kind: str, known: List[str], refs: List[str]
    ) -> None:
        base = set(known)
        for ref in refs:
            if ref not in base:
                report.issues.append(
                    AnnotationIssue(
                        "ERROR",
                        "invalid_dependency_ref",
                        f"dependency/reference {ref!r} does not match any {kind} id",
                    )
                )

    @staticmethod
    def _check_bbox(report: ValidationReport, what: str, bbox: Any) -> None:
        if bbox is None:
            return
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            report.issues.append(
                AnnotationIssue(
                    "ERROR", "malformed_bbox", f"{what} has malformed bounding box {bbox!r}"
                )
            )
            return
        x, y, w, h = bbox
        for v in (x, y, w, h):
            if not isinstance(v, (int, float)):
                report.issues.append(
                    AnnotationIssue(
                        "ERROR", "malformed_bbox", f"{what} bounding box must be numeric"
                    )
                )
                return


def validate_annotation_dict(raw: Dict[str, Any]) -> ValidationReport:
    """Validate a raw annotation dict (used by the annotate CLI)."""
    annotation = from_dict(raw)
    return GroundTruthValidator().validate(annotation)
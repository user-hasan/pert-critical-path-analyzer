"""
Optional ground-truth loading for the benchmark (Mode B evaluation).

Ground truth lives next to an image as ``<basename>.json`` and may
describe any subset of:

    {
      "diagram_type": "AON",
      "activities": {"A": {"duration": 2.0, "position": [x, y],
                            "bounding_box": [x, y, w, h]}, ...},
      "dependencies": [["A", "B"], ...],
      "durations": {"A": 2.0},
      "positions": {"A": [x, y]},
      "bounding_boxes": {"A": [x, y, w, h]}
    }

``activities`` may also be a list of dicts (``{"id": ..., ..}``).
Only the fields actually present are compared (partial ground truth).
A malformed or unparseable file never aborts the benchmark: it is
reported as ``GT_INVALID`` and the image stays diagnostic-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class GroundTruthError(Exception):
    """Raised when a ground-truth file cannot be parsed."""


@dataclass
class GroundTruth:
    """Parsed ground truth for one image; any field may be unknown."""

    diagram_type: Optional[str] = None
    activities: Optional[Dict[str, Dict[str, Any]]] = None
    dependencies: Optional[List[Tuple[str, str]]] = None
    durations: Optional[Dict[str, float]] = None
    positions: Optional[Dict[str, Tuple[float, float]]] = None
    bounding_boxes: Optional[Dict[str, List[float]]] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    source_path: Optional[str] = None

    @property
    def activity_count(self) -> Optional[int]:
        if self.activities is None:
            return None
        return len(self.activities)

    @property
    def dependency_count(self) -> Optional[int]:
        if self.dependencies is None:
            return None
        return len(self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "diagram_type": self.diagram_type,
            "activity_count": self.activity_count,
            "dependency_count": self.dependency_count,
            "has_durations": self.durations is not None,
            "has_positions": self.positions is not None,
            "has_bounding_boxes": self.bounding_boxes is not None,
            "source_path": self.source_path,
        }


def _parse_dependencies(raw: Any) -> List[Tuple[str, str]]:
    deps: List[Tuple[str, str]] = []
    if not isinstance(raw, list):
        raise GroundTruthError("dependencies must be a list")
    for entry in raw:
        if (
            isinstance(entry, (list, tuple))
            and len(entry) == 2
            and all(isinstance(x, str) for x in entry)
        ):
            deps.append((entry[0], entry[1]))
        elif isinstance(entry, dict):
            src = entry.get("source") or entry.get("source_id")
            tgt = entry.get("target") or entry.get("target_id")
            if isinstance(src, str) and isinstance(tgt, str):
                deps.append((src, tgt))
            else:
                raise GroundTruthError(f"Invalid dependency entry: {entry!r}")
        else:
            raise GroundTruthError(f"Invalid dependency entry: {entry!r}")
    return deps


def _parse_activities(raw: Any) -> Dict[str, Dict[str, Any]]:
    if raw is None:
        return None  # type: ignore[return-value]
    activities: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw, dict):
        for aid, info in raw.items():
            if not isinstance(info, dict):
                info = {}
            activities[str(aid)] = dict(info)
        return activities
    if isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                raise GroundTruthError(f"Invalid activity entry: {entry!r}")
            aid = entry.get("id") or entry.get("activity_id")
            if not isinstance(aid, str):
                raise GroundTruthError(f"Activity entry has no string id: {entry!r}")
            activities[aid] = dict(entry)
        return activities
    raise GroundTruthError("activities must be a dict or a list")


def _parse_float_map(raw: Any, field_name: str) -> Optional[Dict[str, float]]:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return {str(k): float(v) for k, v in raw.items()}
    if isinstance(raw, list):
        out: Dict[str, float] = {}
        for entry in raw:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[0], str)
            ):
                out[entry[0]] = float(entry[1])
            elif isinstance(entry, dict) and (
                entry.get("id") or entry.get("activity_id")
            ):
                aid = entry.get("id") or entry.get("activity_id")
                val = entry.get("duration")
                if val is not None:
                    out[str(aid)] = float(val)
            else:
                raise GroundTruthError(
                    f"Invalid {field_name} entry: {entry!r}"
                )
        return out or None
    raise GroundTruthError(f"{field_name} must be a dict or a list")


def _parse_position_map(raw: Any) -> Optional[Dict[str, Tuple[float, float]]]:
    if not isinstance(raw, dict):
        raise GroundTruthError("positions must be a dict")
    out: Dict[str, Tuple[float, float]] = {}
    for aid, pos in raw.items():
        if not (isinstance(pos, (list, tuple)) and len(pos) >= 2):
            raise GroundTruthError(f"Invalid position for {aid!r}: {pos!r}")
        out[str(aid)] = (float(pos[0]), float(pos[1]))
    return out or None


def _parse_bbox_map(raw: Any) -> Optional[Dict[str, List[float]]]:
    if not isinstance(raw, dict):
        raise GroundTruthError("bounding_boxes must be a dict")
    out: Dict[str, List[float]] = {}
    for aid, bb in raw.items():
        if not (isinstance(bb, (list, tuple)) and len(bb) >= 2):
            raise GroundTruthError(f"Invalid bounding_box for {aid!r}: {bb!r}")
        out[str(aid)] = [float(v) for v in bb]
    return out or None


def load_ground_truth(image_path: str | Path) -> Optional[GroundTruth]:
    """Load ground truth for ``image_path`` when a same-basename .json exists.

    Returns ``None`` when no ground-truth file is present (diagnostic-only
    / Mode A).  Raises :class:`GroundTruthError` on malformed content.
    """
    image = Path(image_path)
    gt_path = image.with_suffix(".json")
    if not gt_path.is_file():
        return None

    try:
        raw = json.loads(gt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundTruthError(
            f"Unreadable ground-truth file {gt_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise GroundTruthError(
            f"Ground-truth file {gt_path} must contain a JSON object"
        )

    activities = _parse_activities(raw.get("activities"))
    dependencies = _parse_dependencies(raw.get("dependencies"))

    try:
        durations = _parse_float_map(raw.get("durations"), "durations")
    except GroundTruthError:
        durations = None
    positions = None
    if "positions" in raw:
        try:
            positions = _parse_position_map(raw["positions"])
        except GroundTruthError:
            positions = None
    bounding_boxes = None
    if "bounding_boxes" in raw:
        try:
            bounding_boxes = _parse_bbox_map(raw["bounding_boxes"])
        except GroundTruthError:
            bounding_boxes = None

    diagram_type = raw.get("diagram_type")
    if isinstance(diagram_type, str):
        diagram_type = diagram_type.upper()

    # Fall back to per-activity embedded fields when the top-level maps
    # are absent (positions/bounding boxes are optional enrichment).
    if positions is None and activities:
        embedded: Dict[str, Tuple[float, float]] = {}
        for aid, info in activities.items():
            pos = info.get("position")
            if isinstance(pos, (list, tuple)) and len(pos) >= 2:
                embedded[aid] = (float(pos[0]), float(pos[1]))
        positions = embedded or None
    if bounding_boxes is None and activities:
        embedded_bb: Dict[str, List[float]] = {}
        for aid, info in activities.items():
            bb = info.get("bounding_box") or info.get("bbox")
            if isinstance(bb, (list, tuple)) and len(bb) >= 2:
                embedded_bb[aid] = [float(v) for v in bb]
        bounding_boxes = embedded_bb or None
    if durations is None and activities:
        embedded_dur: Dict[str, float] = {}
        for aid, info in activities.items():
            dur = info.get("duration")
            if dur is not None:
                try:
                    embedded_dur[aid] = float(dur)
                except (TypeError, ValueError):
                    continue
        durations = embedded_dur or None

    return GroundTruth(
        diagram_type=diagram_type,
        activities=activities,
        dependencies=dependencies,
        durations=durations,
        positions=positions,
        bounding_boxes=bounding_boxes,
        raw=raw,
        source_path=str(gt_path),
    )
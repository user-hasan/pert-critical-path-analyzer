"""
Geometry mapping for the image context overlay.

Given a review item and the underlying reconstruction, produces
highlight rectangles that can be drawn non-destructively on top of the
original image.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HighlightRect:
    """A non-destructive overlay rectangle to be drawn on the image."""

    x: float
    y: float
    w: float
    h: float
    color: str = "accent"
    label: str = ""
    kind: str = "box"


@dataclass
class HighlightLine:
    """A non-destructive line drawn between two points."""

    x1: float
    y1: float
    x2: float
    y2: float
    color: str = "accent"
    label: str = ""


Highlight = Any


def _find_activity(
    reconstruction: Any,
    node_id: Optional[str],
    activity_id: Optional[str],
) -> Any:
    """Find a ReconstructedActivity by geometric node id or activity id."""
    if reconstruction is None:
        return None
    for act in getattr(reconstruction, "activities", []) or []:
        if node_id and getattr(act, "source_node_id", None) == node_id:
            return act
        if node_id and getattr(act, "geometric_node_id", None) == node_id:
            return act
    if activity_id:
        for act in getattr(reconstruction, "activities", []) or []:
            if getattr(act, "activity_id", None) == activity_id:
                return act
    return None


def _box_for_activity(act: Any, pad: float = 6.0) -> Optional[Tuple[float, float, float, float]]:
    """Extract a (x, y, w, h) bounding box from a reconstructed activity."""
    bb = getattr(act, "bounding_box", None)
    if bb is not None:
        x = getattr(bb, "x", 0.0)
        y = getattr(bb, "y", 0.0)
        w = getattr(bb, "width", 0.0)
        h = getattr(bb, "height", 0.0)
        if w > 0 and h > 0:
            return (x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    pos = getattr(act, "position", None)
    if pos is not None:
        px = getattr(pos, "x", float(pos[0] if not hasattr(pos, "x") else 0))
        py = getattr(pos, "y", float(pos[1] if not hasattr(pos, "y") else 0))
        if not (isinstance(px, (int, float)) and math.isfinite(px)):
            return None
        return (px - 16, py - 16, 32, 32)
    return None


def _center_of(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
    return (box[0] + box[2] / 2, box[1] + box[3] / 2)


def activity_highlights(
    review_item: Any,
    reconstruction: Any,
) -> List[HighlightRect]:
    """Highlight the bounding box for an activity/duration review item."""
    node_id = getattr(review_item, "geometric_node_id", None)
    act = _find_activity(reconstruction, node_id, getattr(review_item, "activity_id", None))
    if act is None:
        return []
    box = _box_for_activity(act)
    if box is None:
        return []
    label = getattr(act, "activity_id", node_id or "")
    return [HighlightRect(box[0], box[1], box[2], box[3], color="accent", label=label)]


def dependency_highlights(
    review_item: Any,
    reconstruction: Any,
) -> List[Highlight]:
    """Highlight source and target boxes plus a connecting arrow line."""
    src_id = getattr(review_item, "source_node_id", None)
    tgt_id = getattr(review_item, "target_node_id", None)
    src_act = _find_activity(reconstruction, src_id, getattr(review_item, "current_source_id", None))
    tgt_act = _find_activity(reconstruction, tgt_id, getattr(review_item, "current_target_id", None))

    highlights: List[Highlight] = []
    src_box = _box_for_activity(src_act, pad=4.0) if src_act is not None else None
    tgt_box = _box_for_activity(tgt_act, pad=4.0) if tgt_act is not None else None

    if src_box is not None:
        highlights.append(HighlightRect(src_box[0], src_box[1], src_box[2], src_box[3], color="success", label="source"))
    if tgt_box is not None:
        highlights.append(HighlightRect(tgt_box[0], tgt_box[1], tgt_box[2], tgt_box[3], color="danger", label="target"))

    if src_box is not None and tgt_box is not None:
        cx1, cy1 = _center_of(src_box)
        cx2, cy2 = _center_of(tgt_box)
        highlights.append(HighlightLine(cx1, cy1, cx2, cy2, color="accent", label="arrow"))

    return highlights


def highlights_for_item(
    review_item: Any,
    category: Any,
    reconstruction: Any,
) -> List[Highlight]:
    """
    Determine highlight geometry for a review item.

    category is a ReviewCategory enum (imported at call site to avoid circular).
    """
    from pert_analyzer.gui.reviews.categories import ReviewCategory

    if reconstruction is None:
        return []
    cat = category if isinstance(category, ReviewCategory) else ReviewCategory(category)
    if cat == ReviewCategory.ACTIVITIES:
        return activity_highlights(review_item, reconstruction)
    if cat == ReviewCategory.DEPENDENCIES:
        return dependency_highlights(review_item, reconstruction)
    if cat == ReviewCategory.DURATIONS:
        return activity_highlights(review_item, reconstruction)
    return []

"""
Review categories and helpers for filtering review items from a ReviewSession.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional


class ReviewCategory(Enum):
    ACTIVITIES = "ACTIVITIES"
    DEPENDENCIES = "DEPENDENCIES"
    DURATIONS = "DURATIONS"


CATEGORIES = tuple(ReviewCategory)


def pending_count(review_session: Any, category: ReviewCategory) -> int:
    """Return the number of pending items for the given category."""
    if review_session is None:
        return 0
    items = pending_items(review_session, category)
    return len(items)


def pending_items(review_session: Any, category: ReviewCategory) -> List[Any]:
    """Return the list of pending review items for the given category."""
    if review_session is None:
        return []
    if category == ReviewCategory.ACTIVITIES:
        return [
            a
            for a in getattr(review_session, "activities", []) or []
            if getattr(a, "status", None) is not None
            and getattr(a.status, "value", str(a.status)) == "PENDING"
        ]
    if category == ReviewCategory.DEPENDENCIES:
        return [
            d
            for d in getattr(review_session, "dependencies", []) or []
            if getattr(d, "status", None) is not None
            and getattr(d.status, "value", str(d.status)) == "PENDING"
        ]
    if category == ReviewCategory.DURATIONS:
        return [
            d
            for d in getattr(review_session, "durations", []) or []
            if getattr(d, "status", None) is not None
            and getattr(d.status, "value", str(d.status)) == "PENDING"
        ]
    return []


def all_pending_count(review_session: Any) -> int:
    """Total pending review items across all categories."""
    if review_session is None:
        return 0
    return sum(
        pending_count(review_session, cat)
        for cat in CATEGORIES
    )


def item_display_summary(item: Any, category: ReviewCategory) -> dict[str, str]:
    """Extract a concise display summary from a pending review item."""
    if category == ReviewCategory.ACTIVITIES:
        gid = getattr(item, "geometric_node_id", "?")
        aid = getattr(item, "current_activity_id", "?")
        conf = getattr(item, "confidence", 0.0)
        status = getattr(getattr(item, "status", None), "value", "?")
        return {"id": gid, "label": aid, "confidence": f"{conf:.0%}", "status": status}
    if category == ReviewCategory.DEPENDENCIES:
        src = getattr(item, "current_source_id", "?")
        tgt = getattr(item, "current_target_id", "?")
        conf = getattr(item, "confidence", 0.0)
        return {"id": f"{src} \u2192 {tgt}", "label": src, "confidence": f"{conf:.0%}", "status": "PENDING"}
    if category == ReviewCategory.DURATIONS:
        aid = getattr(item, "activity_id", getattr(item, "geometric_node_id", "?"))
        dur = getattr(item, "current_duration", None)
        dur_str = f"{dur:g}" if dur is not None else "N/A"
        conf = getattr(item, "confidence", 0.0)
        return {"id": aid, "label": f"Duration {dur_str}", "confidence": f"{conf:.0%}", "status": "PENDING"}
    return {}


def item_key(item: Any, category: ReviewCategory) -> str:
    """Unique key for a review item used for identification and lookup."""
    if category == ReviewCategory.ACTIVITIES:
        return f"act:{getattr(item, 'geometric_node_id', '?')}"
    if category == ReviewCategory.DEPENDENCIES:
        return f"dep:{getattr(item, 'arrow_id', '?')}"
    if category == ReviewCategory.DURATIONS:
        return f"dur:{getattr(item, 'geometric_node_id', getattr(item, 'activity_id', '?'))}"
    return f"unknown:{id(item)}"

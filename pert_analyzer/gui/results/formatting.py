"""
Number and duration formatting for the Results dashboard.

Internal precision is always preserved; only display strings are formatted.
Missing values render as "Unavailable" rather than being computed locally.
"""

from __future__ import annotations

from typing import Any


def format_number(value: Any) -> str:
    """Format a numeric value without inventing data.

    Uses :g so integer-valued floats (54.0) render cleanly (54) while
    fractional values (54.5) keep their precision.
    """
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):g}"
    except (TypeError, ValueError):
        return "Unavailable"


def format_duration(value: Any) -> str:
    """Format a project/activity duration as '54 days' / '54.5 days'."""
    if value is None:
        return "Unavailable"
    try:
        return f"{float(value):g} days"
    except (TypeError, ValueError):
        return "Unavailable"


def format_days(value: Any) -> str:
    """Format a timing value (ES/EF/LS/LF) without the 'days' suffix."""
    return format_number(value)
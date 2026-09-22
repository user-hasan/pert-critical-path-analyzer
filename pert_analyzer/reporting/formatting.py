"""
Qt-free deterministic formatting helpers for reports.

These small helpers centralise numeric/date formatting so every exporter
(and the dashboard) renders the same strings. Nothing here parses or
recomputes analysis data.
"""

from __future__ import annotations

from typing import Optional


def fmt_number(value: Optional[float], nd: int = 2) -> str:
    """Format a number with ``nd`` decimals, or an em-dash for None."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.{nd}f}"
    except (TypeError, ValueError):
        return "—"


def fmt_duration(value: Optional[float]) -> str:
    """Format a project/activity duration value."""
    return fmt_number(value, nd=2)


def fmt_float(value: Optional[float], nd: int = 3) -> str:
    """Format a float with ``nd`` decimals (e.g. variance / std-dev)."""
    return fmt_number(value, nd=nd)


def fmt_percent(probability: Optional[float]) -> str:
    """Format a probability (0-1) as a percentage string."""
    if probability is None:
        return "—"
    try:
        return f"{float(probability) * 100.0:,.2f}%"
    except (TypeError, ValueError):
        return "—"


def fmt_yes_no(value: Optional[bool]) -> str:
    return "Yes" if value else "No"


def short_path(path: Optional[list]) -> str:
    """Render a critical path as a compact arrow-joined string."""
    if not path:
        return "—"
    return " → ".join(str(step) for step in path)

"""
Reusable helper functions that produce QSS snippets for common UI components.

Usage in widget code::

    from pert_analyzer.gui.themes.components import badge_qss, card_qss

    widget.setStyleSheet(badge_qss("success"))
"""

from __future__ import annotations

from pert_analyzer.gui.themes.palette import (
    ACCENT,
    BG,
    BORDER,
    DANGER,
    ELEVATED,
    INFO,
    SUCCESS,
    SURFACE,
    SURFACE_LIGHT,
    TEXT,
    TEXT_MUTED,
    TEXT_SECONDARY,
    WARNING,
)
from pert_analyzer.gui.themes.spacing import RADIUS_LG, RADIUS_MD, RADIUS_SM

# ── Badge semantic → color map ─────────────────────────────────────
_BADGE_COLORS: dict[str, str] = {
    "VALID": SUCCESS,
    "READY": SUCCESS,
    "OK": SUCCESS,
    "ACCEPTED": SUCCESS,
    "SUCCESS": SUCCESS,
    "INVALID": DANGER,
    "BLOCKED": DANGER,
    "ERROR": DANGER,
    "REJECTED": DANGER,
    "CRITICAL": DANGER,
    "WARNING": WARNING,
    "REVIEW_REQUIRED": WARNING,
    "PENDING": WARNING,
    "INFO": INFO,
    "CPM": ACCENT,
    "PERT": ACCENT,
    "NON_CRITICAL": TEXT_SECONDARY,
    "CORRECTED": ACCENT,
}


def badge_color(status: str) -> str:
    """Return the semantic colour hex for *status*."""
    return _BADGE_COLORS.get(status.upper().replace(" ", "_"), TEXT_MUTED)


def badge_qss(status: str) -> str:
    """Return a QSS snippet for a badge QLabel showing *status*."""
    fg = badge_color(status)
    return (
        f"color: {fg}; background-color: {fg}22; "
        f"border: 1px solid {fg}44; border-radius: {RADIUS_SM}px; "
        f"padding: 2px 8px; font-size: 11px; font-weight: 600;"
    )


def card_qss() -> str:
    """Return a QSS snippet for a section / KPI card."""
    return (
        f"background-color: {SURFACE}; border: 1px solid {BORDER}; "
        f"border-radius: {RADIUS_LG}px; padding: {RADIUS_MD}px;"
    )


def elevated_card_qss() -> str:
    """Return a QSS snippet for a slightly elevated card."""
    return (
        f"background-color: {ELEVATED}; border: 1px solid {BORDER}; "
        f"border-radius: {RADIUS_LG}px; padding: {RADIUS_MD}px;"
    )


def flat_surface_qss() -> str:
    """Return a QSS snippet for a flat surface (e.g. list backgrounds)."""
    return (
        f"background-color: {SURFACE_LIGHT}; border: 1px solid {BORDER}; "
        f"border-radius: {RADIUS_MD}px;"
    )


def primary_button_qss() -> str:
    """QSS for a primary action button."""
    return (
        f"background-color: {ACCENT}; color: #ffffff; border: none; "
        f"border-radius: {RADIUS_SM}px; padding: 8px 20px; "
        f"min-height: 28px; font-weight: 600;"
    )


def danger_button_qss() -> str:
    """QSS for a danger action button."""
    return (
        f"background-color: {DANGER}; color: #ffffff; border: none; "
        f"border-radius: {RADIUS_SM}px; padding: 8px 20px; "
        f"min-height: 28px; font-weight: 600;"
    )


def ghost_button_qss() -> str:
    """QSS for a ghost / transparent button."""
    return (
        f"background-color: transparent; color: {TEXT}; border: 1px solid {BORDER}; "
        f"border-radius: {RADIUS_SM}px; padding: 6px 16px;"
    )


def status_banner_qss(status: str) -> str:
    """QSS for a validation / status banner."""
    color_map = {"VALID": SUCCESS, "INVALID": DANGER, "WARNING": WARNING}
    fg = color_map.get(status.upper(), TEXT_SECONDARY)
    return (
        f"background-color: {fg}18; border: 1px solid {fg}55; "
        f"border-radius: {RADIUS_MD}px; color: {fg}; padding: 10px 16px;"
    )

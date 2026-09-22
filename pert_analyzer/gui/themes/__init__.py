"""
GUI themes package — centralized design system.

Import colours / tokens from the sub-modules, or import the convenience
re-exports below::

    from pert_analyzer.gui.themes import COLORS, TITLE_FONT, apply_theme
"""

from __future__ import annotations

from pert_analyzer.gui.themes.palette import COLORS, NETWORK_COLORS
from pert_analyzer.gui.themes.style import apply_theme
from pert_analyzer.gui.themes.typography import (
    DISPLAY_FONT,
    LABEL_FONT,
    MUTED_FONT,
    STATUS_FONT,
    SUBTITLE_FONT,
    TITLE_FONT,
)

__all__ = [
    "COLORS",
    "NETWORK_COLORS",
    "DISPLAY_FONT",
    "LABEL_FONT",
    "MUTED_FONT",
    "STATUS_FONT",
    "SUBTITLE_FONT",
    "TITLE_FONT",
    "apply_theme",
]

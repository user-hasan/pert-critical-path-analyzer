"""
Results dashboard presentation layer (Phase 4).

Pure presentation: extracts a snapshot from the backend CPM result /
GraphModel and renders it. Never recomputes ES/EF/LS/LF, floats, critical
paths, or project duration.
"""

from __future__ import annotations

__all__: list[str] = []
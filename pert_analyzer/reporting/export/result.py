"""
Structured export outcome (Qt-free).

Every exporter returns an :class:`ExportResult` so the GUI can surface
success/failure without raising and without parsing strings. Results are
deterministic given the same inputs: ``output_path``, ``rows_written``,
``warnings`` and ``metadata`` are fully determined by the exporter inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ExportResult:
    """Outcome of a single export operation (returns, never raises)."""

    ok: bool
    export_format: str
    output_path: str
    rows_written: int = 0
    filename: str = ""
    project_name: str = ""
    created_at: str = ""
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def message(self) -> str:
        """Human summary of the outcome."""
        if self.ok:
            return f"Exported {self.export_format} to {self.output_path}"
        return f"{self.export_format} export failed: {self.output_path}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "format": self.export_format,
            "output_path": self.output_path,
            "rows_written": self.rows_written,
            "filename": self.filename,
            "project_name": self.project_name,
            "created_at": self.created_at,
            "warnings": list(self.warnings),
            "metadata": dict(self.metadata),
        }


def ok_result(
    *,
    export_format: str,
    output_path: str,
    rows_written: int = 0,
    filename: str = "",
    project_name: str = "",
    created_at: str = "",
    warnings: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    """Build a successful :class:`ExportResult`."""
    return ExportResult(
        ok=True,
        export_format=export_format,
        output_path=output_path,
        rows_written=rows_written,
        filename=filename,
        project_name=project_name,
        created_at=created_at,
        warnings=list(warnings or []),
        metadata=dict(metadata or {}),
    )


def fail_result(
    *,
    export_format: str,
    output_path: str,
    reason: str = "",
    metadata: Optional[Dict[str, Any]] = None,
) -> ExportResult:
    """Build a failed :class:`ExportResult` carrying the failure reason."""
    warnings = [reason] if reason else []
    if reason:
        metadata = dict(metadata or {})
        metadata["error"] = reason
    return ExportResult(
        ok=False,
        export_format=export_format,
        output_path=output_path,
        rows_written=0,
        filename="",
        project_name="",
        created_at="",
        warnings=warnings,
        metadata=dict(metadata or {}),
    )

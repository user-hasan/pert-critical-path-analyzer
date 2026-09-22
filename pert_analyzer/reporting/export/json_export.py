"""JSON export (Phase 6): stdlib-only, deterministic ordering."""

from __future__ import annotations

import json
import os
from typing import Any


def export_json(report: Any, output_path: str, *, indent: int = 2) -> Any:
    """Serialize ``report`` to ``output_path`` as deterministic JSON."""
    from pert_analyzer.reporting.export.result import ExportResult

    try:
        payload = report.to_dict()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent,
                      sort_keys=False)
            handle.write("\n")
        return ExportResult(
            ok=True,
            export_format="json",
            output_path=output_path,
            rows_written=_rows(payload),
            filename=os.path.basename(output_path),
            created_at=report.metadata.created_at,
            project_name=report.metadata.project_name,
            metadata={"indent": indent, "encoding": "utf-8"},
        )
    except Exception as exc:  # pragma: no cover
        return ExportResult(
            ok=False, export_format="json", output_path=output_path,
            warnings=[f"JSON export failed: {exc}"],
        )


def _rows(payload: Any) -> int:
    rows = 0
    if isinstance(payload, dict):
        for key, value in payload.items():
            if isinstance(value, (list, tuple)) and value:
                rows += len(value)
            elif isinstance(value, dict):
                rows += _rows(value)
    return rows

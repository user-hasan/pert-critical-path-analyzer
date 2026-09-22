"""CSV export (Phase 6): deterministic, stdlib-only exporter.

Serializes the authoritative :class:`ProjectReport` to a single CSV file
using only the Python standard library (``csv``). Column ordering, row
ordering, and cell formatting are stable across runs so the output is
deterministic and safe to snapshot in tests. Never raises; always returns
a structured :class:`ExportResult`.
"""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

from pert_analyzer.reporting.export.result import (
    ExportResult,
    fail_result,
    ok_result,
)

HEADER = ("section", "activity", "name", "value")

SECTION_TITLES = {
    "summarysection": "Summary",
    "activitiessection": "Activities",
    "dependenciessection": "Dependencies",
    "criticalpathssection": "Critical Paths",
    "cpmsection": "CPM",
    "pertsection": "PERT",
    "validationsection": "Validation",
    "reviewsection": "Review",
}

SUMMARY_KEYS = (
    "project_duration",
    "critical_path_count",
    "critical_activity_count",
    "activity_count",
    "dependency_count",
)


def export_csv(
    report: Any,
    output_path: str,
    *,
    base_dir: Optional[str] = None,
    filename: Optional[str] = None,
    **kwargs: Any,
) -> ExportResult:
    """Write ``report`` to ``output_path`` as a deterministic CSV file.

    Returns a structured :class:`ExportResult` and never raises.
    """
    try:
        if base_dir:
            output_path = os.path.join(base_dir, filename or os.path.basename(output_path))
        _write_csv(report, output_path)
        return ok_result(
            export_format="csv",
            output_path=output_path,
            rows_written=_count_rows(report),
            filename=os.path.basename(output_path),
            project_name=_project_name(report),
            created_at=_created_at(report),
            metadata={"exporter": "csv", "deterministic": True},
        )
    except Exception as exc:  # pragma: no cover
        return fail_result(
            export_format="csv",
            output_path=output_path,
            reason=f"CSV export failed: {exc}",
        )


def _write_csv(report: Any, output_path: str) -> None:
    rows = _flatten_rows(report)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        writer.writerows(rows)


def _flatten_rows(report: Any) -> List[List[str]]:
    lines: List[List[str]] = []
    if report is None:
        return lines
    sections = _iter_sections(report)
    for index, section in enumerate(sections):
        title = _section_title(section, index)
        if title == "Summary":
            for key in SUMMARY_KEYS:
                value = _first_of(section, key)
                if value is not None and value != "":
                    lines.append([title, "", key, _value_to_cell(value)])
            continue
        if title == "Critical Paths":
            paths = getattr(section, "critical_paths", None) or []
            for idx, path in enumerate(paths, start=1):
                lines.append([title, f"Path {idx}", "",
                              " -> ".join(str(n) for n in path)])
            continue
        for row in _iter_rows(section):
            lines.append(_render_row(title, row))
    return lines


def _count_rows(report: Any) -> int:
    return len(_flatten_rows(report))


def _project_name(report: Any) -> str:
    metadata = getattr(report, "metadata", None)
    if metadata is None:
        return ""
    return _first_of(metadata, "project_name", "name")


def _created_at(report: Any) -> str:
    metadata = getattr(report, "metadata", None)
    if metadata is None:
        return ""
    return _first_of(metadata, "created_at", "generated_at")


def _first_of(obj: Any, *names: str) -> str:
    for name in names:
        value = getattr(obj, name, None)
        if value is None and isinstance(obj, dict):
            value = obj.get(name)
        if value not in (None, ""):
            return str(value)
    return ""


def _iter_sections(report: Any):
    if callable(getattr(report, "sections", None)):
        return list(report.sections())
    if isinstance(getattr(report, "sections", None), (list, tuple)):
        return list(report.sections)
    return []


def _section_title(section: Any, index: int = 0) -> str:
    del index
    title = getattr(section, "title", None)
    if title not in (None, ""):
        return str(title)
    return SECTION_TITLES.get(
        type(section).__name__.lower(), type(section).__name__
    )


def _iter_rows(section: Any):
    if callable(getattr(section, "rows", None)):
        return list(section.rows())
    if isinstance(getattr(section, "rows", None), (list, tuple)):
        return list(section.rows)
    return []


def _render_row(title: str, row: Any) -> List[str]:
    """Serialize one report row into the (section, activity, name, value) shape."""
    if isinstance(row, dict):
        pid = row.get("activity_id") or row.get("id") or ""
        name = row.get("name", row.get("key", "")) or ""
        value = row.get(
            "duration",
            row.get("value", row.get("expected_time", row.get("count"))),
        )
        return [title, str(pid), _text(name), _value_to_cell(value)]
    pid = getattr(row, "activity_id", None)
    if pid is not None:
        name = _text(getattr(row, "name", ""))
        value = getattr(
            row, "duration",
            getattr(row, "expected_time", getattr(row, "value", None)),
        )
        return [title, str(pid), name, _value_to_cell(value)]
    source = getattr(row, "source", None)
    target = getattr(row, "target", None)
    if source is not None and target is not None:
        return [title, _text(source), "", _text(target)]
    path = getattr(row, "path", None)
    if isinstance(path, (list, tuple)):
        return [title, "", "", " -> ".join(_text(n) for n in path)]
    key = _text(getattr(row, "key", getattr(row, "name", "")))
    value = getattr(row, "value", None)
    if value is None:
        value = getattr(row, "float", getattr(row, "count", None))
    return [title, key, "", _value_to_cell(value)]


def _text(value: Any) -> str:
    return str(value) if value is not None else ""


def _value_to_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}".rstrip("0").rstrip(".")
    return str(value)

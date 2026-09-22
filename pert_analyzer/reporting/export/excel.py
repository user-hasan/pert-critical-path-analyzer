"""Excel export (Phase 6): Qt-free, deterministic via xlsxwriter.

Writes a workbook whose sheet layout mirrors the authoritative report:
Summary, Activities, Dependencies, Critical Paths, CPM, PERT, Validation,
Review. Values are the authoritative report rows (no recalculation, no
formulas). Sheet names and cell ordering are deterministic.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

try:  # pragma: no cover
    import xlsxwriter as _xlsxwriter
except Exception:  # pragma: no cover
    _xlsxwriter = None  # type: ignore [assignment, misc]


def export_excel(
    report: Any,
    output_path: str,
    *,
    sheet_name: str = "Summary",
) -> Any:
    """Write ``report`` to an .xlsx workbook and return an ExportResult."""
    from pert_analyzer.reporting.export.result import ExportResult

    if _xlsxwriter is None:
        return ExportResult(
            ok=False, export_format="excel", output_path=output_path,
            warnings=["xlsxwriter is not installed; Excel export unavailable"],
        )

    warnings: List[str] = []
    try:
        rows, sheets = _build_workbook(report, output_path, sheet_name, warnings)
        return ExportResult(
            ok=True,
            export_format="excel",
            output_path=output_path,
            rows_written=rows,
            filename=os.path.basename(output_path),
            created_at=getattr(report.metadata, "created_at", ""),
            project_name=getattr(report.metadata, "project_name", ""),
            warnings=warnings,
            metadata={"sheets": list(sheets), "workbook": True},
        )
    except Exception as exc:  # pragma: no cover
        return ExportResult(
            ok=False, export_format="excel", output_path=output_path,
            warnings=[f"Excel export failed: {exc}"],
        )


def _build_workbook(report: Any, output_path: str, sheet_name: str,
                    warnings: List[str]) -> tuple:
    workbook = _xlsxwriter.Workbook(output_path, {"constant_memory": True})
    try:
        header_fmt = workbook.add_format({"bold": True, "bg_color": "#2C5282",
                                          "font_color": "#FFFFFF"})
        sheets: List[str] = []
        _write_summary_sheet(workbook, report, sheet_name, header_fmt)
        sheets.append(sheet_name)
        for section in report.sections():
            if not getattr(section, "is_available", False):
                continue
            _write_section_sheet(workbook, section, header_fmt)
            sheets.append(section.title)
        rows = getattr(report.metadata, "row_count", 0) or 0
    finally:
        workbook.close()
    return rows, sheets


def _write_summary_sheet(workbook: Any, report: Any, sheet_name: str,
                         header_fmt: Any) -> None:
    sheet = workbook.add_worksheet(sheet_name)
    meta = report.metadata
    sheet.write_string(0, 0, "Project Name", header_fmt)
    sheet.write_string(0, 1, meta.project_name)
    sheet.write_string(1, 0, "Report ID")
    sheet.write_string(1, 1, getattr(meta, "report_id", ""))
    sheet.write_string(2, 0, "Generated At")
    sheet.write_string(2, 1, getattr(meta, "created_at", ""))
    sheet.write_string(3, 0, "Analysis Type")
    sheet.write_string(3, 1, getattr(meta, "analysis_type", ""))


def _write_section_sheet(workbook: Any, section: Any, header_fmt: Any) -> None:
    sheet = workbook.add_worksheet(getattr(section, "title", "Section"))
    if not getattr(section, "rows", None):
        sheet.write_string(0, 0, "No data available", header_fmt)
        return
    header = _header(section)
    for col_i, name in enumerate(header):
        sheet.write_string(0, col_i, name, header_fmt)
    for row_i, row in enumerate(section.rows, start=1):
        for col_i, value in enumerate(_row_values(row)):
            sheet.write_string(row_i, col_i, _cell_text(value))


def _header(section: Any) -> List[str]:
    headers = getattr(section, "headers", None)
    if headers:
        return [str(h) for h in headers]
    row = section.rows[0]
    return [str(i) for i in range(len(_row_values(row)))]


def _row_values(row: Any) -> List[Any]:
    values = getattr(row, "values", None)
    if values is not None:
        return list(values)
    fields = getattr(row, "fields", None) or []
    return [getattr(row, f, None) for f in fields]


def _cell_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)

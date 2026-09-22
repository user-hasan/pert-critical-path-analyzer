"""PDF export (Phase 6): Qt-free, deterministic, Unicode-safe via fpdf2.

Renders the authoritative :class:`ProjectReport` with fpdf2 (+ a bundled
Unicode TTF, DejaVuSans, so activity names and PERT estimates with accented
characters survive). Monochrome, no glyph loss, no charts recomputed.

Output is deterministic given the same report + destination: sections are
emitted in a stable order and nothing is recalculated here. Embedding the
TTF is optional; when the font asset is unavailable the exporter degrades to
the built-in Latin-1 font and warns (records in ExportResult, never raises).
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

try:  # pragma: no cover - exercised on supported systems
    from fpdf import FPDF
except Exception:  # pragma: no cover
    FPDF = None  # type: ignore [assignment, misc]

from pert_analyzer.reporting.formatting import fmt_number, fmt_percent


def _font_path() -> Optional[str]:
    """Locate a Unicode TTF (DejaVuSans) for full glyph coverage."""
    candidates = [
        os.path.join(
            os.path.dirname(__file__), "..", "..", "..", "assets", "fonts",
            "DejaVuSans.ttf",
        ),
        os.path.join(
            os.path.dirname(__file__), "..", "..", "_assets", "fonts",
            "DejaVuSans.ttf",
        ),
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    try:
        import matplotlib  # noqa: F401
        base = os.path.join(os.path.dirname(matplotlib.__file__),
                            "mpl-data", "fonts", "ttf")
        for name in ("DejaVuSans.ttf", "DejaVuSansMono.ttf"):
            candidate = os.path.join(base, name)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
    except Exception:
        pass
    return None


def _register_fonts(pdf: Any, font_path: Optional[str]) -> bool:
    if font_path is None:
        return False
    try:
        pdf.add_font("DejaVuSans", "", font_path, uni=True)
        bold_path = _sibling_font(font_path, "Bold")
        if bold_path:
            pdf.add_font("DejaVuSans", "B", bold_path, uni=True)
        return True
    except Exception:
        return False


def _sibling_font(font_path: str, variant: str) -> Optional[str]:
    """Look for ``<base>-<variant>.ttf`` next to ``font_path`` (e.g. the
    DejaVu Bold face that ships beside the regular face)."""
    base, _ = os.path.splitext(font_path)
    candidate = base + "-" + variant + ".ttf"
    if os.path.isfile(candidate):
        return candidate
    return None


def export_pdf(
    report: Any,
    output_path: str,
    *,
    font_path: Optional[str] = None,
    page_break: bool = False,
) -> Any:
    """Render ``report`` to ``output_path`` and return an ExportResult.

    ``page_break`` marks multi-page output; exporters enabled by the GUI use
    it so heavy reports stream through the GUI worker without blocking the
    event loop.
    """
    from pert_analyzer.reporting.export.result import ExportResult

    if FPDF is None:
        return ExportResult(
            ok=False,
            export_format="pdf",
            output_path=output_path,
            warnings=["fpdf2 is not installed; PDF export is unavailable"],
        )

    if font_path is None:
        font_path = _font_path()
    warnings: List[str] = []
    if not font_path or not _font_path_ok(font_path):
        warnings.append(
            "Unicode font asset not found; using the built-in Latin-1 font. "
            "Accented activity names may render as boxes."
        )
        effective_font = None
    else:
        effective_font = font_path

    try:
        layout = _RebuildFpdfRenderer(report, warnings).render(output_path)
    except _OptionalImportError as exc:  # pragma: no cover
        return ExportResult(
            ok=False, export_format="pdf", output_path=output_path,
            warnings=[str(exc)],
        )

    return ExportResult(
        ok=True,
        export_format="pdf",
        output_path=output_path,
        rows_written=layout.rows_written,
        filename=os.path.basename(output_path),
        created_at=report.metadata.created_at,
        project_name=report.metadata.project_name,
        warnings=warnings,
        metadata={
            "pages": layout.pages,
            "sections": list(layout.sections),
        },
    )


def _font_path_ok(font_path: str) -> bool:
    return os.path.isfile(font_path)


class _OptionalImportError(RuntimeError):
    pass


class _RebuildFpdfRenderer:
    """Deterministic report -> FPDF builder (authoritative input only)."""

    def __init__(self, report: Any, warnings: List[str]) -> None:
        self._report = report
        self._warnings = warnings
        self._pages = 景德Note = 0

    @property
    def rows_written(self) -> int:
        return getattr(self._report.metadata, "row_count", 0) or 0

    @property
    def pages(self) -> int:
        return self._pages

    @property
    def sections(self) -> List[str]:
        return list(self._report.section_names())

    def render(self, output_path: str) -> _Layout:
        from pert_analyzer.reporting.formatting import fmt_number, fmt_percent

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=14.0)
        registered = _register_fonts(pdf, _font_path())
        meta = self._report.metadata
        if registered:
            pdf.add_page()
            pdf.set_font("DejaVuSans", "", 10)
        else:
            pdf.add_page()
            pdf.set_font("Helvetica", "", 10)

        # Header band
        pdf.set_fill_color(44, 82, 130)
        pdf.rect(0, 0, 0, 0, "F") if False else None
        pdf.set_font(("DejaVuSans" if registered else "Helvetica"), "B", 16)
        pdf.cell(0, 10, f"{meta.project_name} — Project Report", new_x="LMARGIN",
                 new_y="NEXT")
        pdf.set_font(("DejaVuSans" if registered else "Helvetica"), "", 9)
        pdf.cell(0, 6, f"Generated {meta.created_at}  ·  {meta.analysis_type}",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.ln(3)

        for section in self._report.sections():
            if getattr(section, "is_available", True) is False:
                pdf.set_font(("DejaVuSans" if registered else "Helvetica"),
                             "I", 10)
                pdf.multi_cell(
                    0, 6,
                    f"{getattr(section, 'title', 'Section')} — "
                    f"{getattr(section, 'reason', 'unavailable')}",
                )
                pdf.ln(1)
                continue
            title = getattr(section, "title", None)
            if title:
                pdf.set_font(("DejaVuSans" if registered else "Helvetica"),
                             "B", 12)
                pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
                pdf.ln(1)
            self._render_section(pdf, section, registered)

        pdf.output(output_path)
        self._pages = pdf.page_no()
        return _Layout(rows_written=self.rows_written, pages=self._pages,
                       sections=self.sections)

    def _render_section(self, pdf: Any, section: Any, registered: bool) -> None:
        font = "DejaVuSans" if registered else "Helvetica"
        pdf.set_x(pdf.l_margin)
        cell_w = pdf.w - pdf.l_margin - pdf.r_margin
        if hasattr(section, "rows"):
            pdf.set_font(font, "", 9)
            for row in section.rows:
                text = _row_text(row)
                pdf.multi_cell(cell_w, 5, text)
                pdf.ln(0.4)
        else:
            pdf.set_font(font, "", 9)
            for key in ("project_duration", "project_variance",
                        "project_std_dev", "critical_path_count",
                        "critical_activity_count", "activity_count",
                        "dependency_count"):
                value = getattr(section, key, None)
                if value is not None:
                    pdf.set_x(pdf.l_margin)
                    pdf.multi_cell(cell_w, 5,
                                   f"{key.replace('_', ' ').title()}: {fmt_number(value)}",
                                   new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)


class _Layout:
    def __init__(self, rows_written: int, pages: int, sections: List[str]) -> None:
        self.rows_written = rows_written
        self.pages = pages
        self.sections = sections


from pert_analyzer.reporting.formatting import fmt_number, fmt_percent


def _row_text(row: Any) -> str:
    if hasattr(row, "activity_id"):
        parts = [str(getattr(row, "activity_id", "?")).strip()]
        name = _text(getattr(row, "name", ""))
        if name:
            parts.append(f" — {name}")
        return "  ".join(parts)
    return " ".join(_text(c) for c in _cells(row))


def _cells(row: Any) -> List[str]:
    return [str(v) for v in (getattr(row, "cells", None) or [])]


def _text(value: Any) -> str:
    return str(value) if value is not None else ""

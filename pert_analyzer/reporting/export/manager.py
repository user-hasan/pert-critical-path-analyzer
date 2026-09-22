"""ExportManager (Phase 6): coordinated PDF/Excel/JSON export.

Collects the authoritative :class:`ProjectReport` and writes PDF, Excel, and
JSON in one deterministic pass is left to callers; this manager is the safe
entry point that runs any single exporter and returns a structured
:class:`ExportResult` (never raises). The GUI calls this from a worker
thread so heavy PDF/Excel work never blocks the event loop.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional


class ExportManager:
    """High-level export entry point for a completed ProjectReport."""

    _FORMATS = ("pdf", "excel", "json", "csv")

    def __init__(self, *, exporter_factories: Optional[Dict[str, Any]] = None,
                 logger: Optional[Any] = None) -> None:
        self._factories = dict(exporter_factories or {})
        self._logger = logger

    @property
    def supported_formats(self) -> List[str]:
        return list(self._FORMATS)

    def export(
        self,
        report: Any,
        *,
        output_format: str,
        output_path: Optional[str] = None,
        base_dir: Optional[str] = None,
        filename: Optional[str] = None,
        **kwargs: Any,
    ) -> Any:
        """Export ``report`` in ``output_format`` and return an ExportResult.

        ``output_path`` wins; otherwise ``base_dir``/``filename`` are used
        to derive one. Exporters return (never raise); library availability
        is surfaced as a failed ExportResult with a clear warning.
        """
        from pert_analyzer.reporting.export.result import (
            ExportResult,
            fail_result,
        )

        if output_format not in self._FORMATS:
            return fail_result(
                export_format=output_format,
                output_path=output_path or "",
                reason=(
                    f"Unsupported export format {output_format!r}; "
                    f"choose from {', '.join(self._FORMATS)}"
                ),
                metadata={"supported_formats": list(self._FORMATS)},
            )

        if output_path is None:
            if not base_dir:
                return fail_result(
                    export_format=output_format,
                    output_path="",
                    reason="No output path or base directory provided",
                )
            resolved = self._resolve_path(base_dir, filename, output_format,
                                          report)
            output_path = resolved

        exporter = self._factories.get(output_format)
        if exporter is None:
            exporter = self._default_exporter(output_format)
        try:
            return exporter(report, output_path, **kwargs)
        except Exception as exc:  # pragma: no cover
            self._log("warning", "Export %s failed", output_format, exc=exc)
            return ExportResult(
                ok=False,
                export_format=output_format,
                output_path=output_path,
                warnings=[f"Export failed: {exc}"],
            )

    def _resolve_path(self, base_dir: str, filename: Optional[str],
                      output_format: str, report: Any) -> str:
        extension = {"pdf": "pdf", "excel": "xlsx", "json": "json", "csv": "csv"}[
            output_format
        ]
        name = filename
        if not name:
            stem = (getattr(getattr(report, "metadata", None), "project_name",
                            None) or "project").strip()
            stem = "".join(c for c in stem if c.isalnum() or c in "-_")
            stem = stem or "project"
            report_id = getattr(getattr(report, "metadata", None), "report_id",
                                "") or ""
            name = f"{stem}_{report_id}.{extension}" if report_id else f"{stem}.{extension}"
        if not name.lower().endswith(f".{extension}"):
            name = f"{name}.{extension}"
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, name)

    def _default_exporter(self, output_format: str) -> Any:
        if output_format == "pdf":
            from pert_analyzer.reporting.export.pdf import export_pdf
            return export_pdf
        if output_format == "excel":
            from pert_analyzer.reporting.export.excel import export_excel
            return export_excel
        if output_format == "json":
            from pert_analyzer.reporting.export.json_export import export_json
            return export_json
        if output_format == "csv":
            from pert_analyzer.reporting.export.csv_export import export_csv
            return export_csv
        raise ValueError(f"Unable to build exporter for {output_format!r}")

    def _log(self, level: str, message: str, *args: Any, **kwargs: Any) -> None:
        if self._logger is None:
            return
        handler = getattr(self._logger, level, None)
        if callable(handler):
            kwargs.pop("exc", None)
            handler(message, *args)

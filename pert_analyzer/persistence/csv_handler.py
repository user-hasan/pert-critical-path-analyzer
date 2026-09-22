"""
CSV import/export for AON activity data.

Provides simple CSV handling for AON-style project data where
each row represents an activity with its dependencies.

CSV Format:
    activity_id,name,duration,predecessors,optimistic_time,most_likely_time,pessimistic_time

Example:
    A,Design,5,,3,5,8
    B,Build,3,A,2,3,5
    C,Test,4,B,3,4,6
    D,Deploy,2,B C,1,2,3

Multiple predecessors are separated by spaces.
"""

from __future__ import annotations

import csv
import io
import logging
import os
from typing import Any, Dict, List, Optional, TextIO, Union

from pert_analyzer.core.models import (
    Activity,
    DiagramType,
    GraphModel,
)
from pert_analyzer.graph.builder import GraphBuilder
from pert_analyzer.graph.exceptions import (
    InvalidGraphDataError,
    InvalidDurationError,
)

logger = logging.getLogger(__name__)

# CSV column headers
CSV_HEADERS = [
    "activity_id",
    "name",
    "duration",
    "predecessors",
    "optimistic_time",
    "most_likely_time",
    "pessimistic_time",
]


def import_csv(
    file_path: Union[str, os.PathLike],
    duration_column: str = "duration",
) -> GraphModel:
    """
    Import a project from a CSV file.

    Args:
        file_path: Path to the CSV file.
        duration_column: Name of the duration column.

    Returns:
        A GraphModel with the imported data.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidGraphDataError: If the CSV structure is invalid.
    """
    with open(file_path, "r", encoding="utf-8-sig") as f:
        return import_csv_from_file(f, duration_column)


def import_csv_from_string(
    csv_string: str,
    duration_column: str = "duration",
) -> GraphModel:
    """
    Import a project from a CSV string.

    Args:
        csv_string: CSV content as a string.
        duration_column: Name of the duration column.

    Returns:
        A GraphModel with the imported data.
    """
    with io.StringIO(csv_string) as f:
        return import_csv_from_file(f, duration_column)


def import_csv_from_file(
    file_obj: TextIO,
    duration_column: str = "duration",
) -> GraphModel:
    """
    Import a project from a file-like object.

    Args:
        file_obj: File-like object with CSV content.
        duration_column: Name of the duration column.

    Returns:
        A GraphModel with the imported data.

    Raises:
        InvalidGraphDataError: If the CSV structure is invalid.
    """
    reader = csv.DictReader(file_obj)

    if reader.fieldnames is None:
        raise InvalidGraphDataError("CSV file is empty")

    # Normalize column names
    fieldnames = [f.strip().lower() for f in reader.fieldnames]

    if "activity_id" not in fieldnames:
        raise InvalidGraphDataError(
            "CSV missing required column 'activity_id'",
            "activity_id",
        )

    # Find the duration column (check common names)
    dur_col = None
    for col_name in ["duration", "dur", "time", duration_column.lower()]:
        if col_name in fieldnames:
            dur_col = col_name
            break

    if dur_col is None:
        raise InvalidGraphDataError(
            "CSV missing duration column (expected 'duration', 'dur', or 'time')",
            "duration",
        )

    builder = GraphBuilder(diagram_type=DiagramType.AON)

    # First pass: add activities
    activities_added = []
    for row_num, row in enumerate(reader, start=2):
        # Normalize keys
        normalized_row = {}
        for k, v in row.items():
            key = k.strip().lower() if k else ""
            value = v.strip() if v else ""
            normalized_row[key] = value

        activity_id = normalized_row.get("activity_id", "")
        if not activity_id:
            raise InvalidGraphDataError(
                f"Row {row_num}: missing activity_id",
                "activity_id",
            )

        name = normalized_row.get("name", activity_id)

        # Parse duration
        duration_str = normalized_row.get(dur_col, "")
        try:
            duration = float(duration_str) if duration_str else None
        except ValueError:
            raise InvalidGraphDataError(
                f"Row {row_num}: invalid duration '{duration_str}'",
                "duration",
            )

        # Parse PERT times
        optimistic = _parse_optional_float(normalized_row.get("optimistic_time", ""))
        most_likely = _parse_optional_float(normalized_row.get("most_likely_time", ""))
        pessimistic = _parse_optional_float(normalized_row.get("pessimistic_time", ""))

        # Parse predecessors
        predecessors_str = normalized_row.get("predecessors", "")
        predecessors = _parse_predecessors(predecessors_str)

        builder.add_activity(
            activity_id=activity_id,
            duration=duration,
            name=name,
            optimistic_time=optimistic,
            most_likely_time=most_likely,
            pessimistic_time=pessimistic,
        )

        activities_added.append((activity_id, predecessors))

    # Second pass: add dependencies
    for activity_id, predecessors in activities_added:
        for pred_id in predecessors:
            builder.add_dependency(pred_id, activity_id)

    return builder.build()


def _parse_predecessors(predecessors_str: str) -> List[str]:
    """
    Parse a predecessors string into a list of activity IDs.

    Supports space-separated, comma-separated, or mixed formats.

    Args:
        predecessors_str: String of predecessor IDs.

    Returns:
        List of predecessor activity IDs.
    """
    if not predecessors_str:
        return []

    # Replace commas with spaces and split
    import re
    ids = re.split(r"[,\s]+", predecessors_str.strip())
    return [pid for pid in ids if pid]


def _parse_optional_float(value: str) -> Optional[float]:
    """Parse a string to float, returning None if empty."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def export_csv(
    graph: GraphModel,
    file_path: Union[str, os.PathLike],
) -> None:
    """
    Export a GraphModel to a CSV file.

    Args:
        graph: The GraphModel to export.
        file_path: Path to the output CSV file.
    """
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        export_csv_to_file(graph, f)


def export_csv_to_string(graph: GraphModel) -> str:
    """
    Export a GraphModel to a CSV string.

    Args:
        graph: The GraphModel to export.

    Returns:
        CSV content as a string.
    """
    output = io.StringIO()
    export_csv_to_file(graph, output)
    return output.getvalue()


def export_csv_to_file(
    graph: GraphModel,
    file_obj: TextIO,
) -> None:
    """
    Export a GraphModel to a file-like object.

    Args:
        graph: The GraphModel to export.
        file_obj: File-like object to write to.
    """
    writer = csv.DictWriter(file_obj, fieldnames=CSV_HEADERS)
    writer.writeheader()

    # Build predecessor map
    pred_map: Dict[str, List[str]] = {}
    for dep in graph.dependencies:
        if dep.target not in pred_map:
            pred_map[dep.target] = []
        pred_map[dep.target].append(dep.source)

    for aid, activity in graph.activities.items():
        predecessors = " ".join(sorted(pred_map.get(aid, [])))

        row: Dict[str, Any] = {
            "activity_id": activity.activity_id,
            "name": activity.name,
            "duration": activity.duration,
            "predecessors": predecessors,
            "optimistic_time": activity.optimistic_time or "",
            "most_likely_time": activity.most_likely_time or "",
            "pessimistic_time": activity.pessimistic_time or "",
        }
        writer.writerow(row)

    logger.info(
        "Exported %d activities to CSV",
        len(graph.activities),
    )

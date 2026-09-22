"""
Domain-specific exceptions for the CPM/PERT analysis engine.

These exceptions represent error conditions specific to project
network analysis, providing structured error information rather
than generic exceptions or print statements.
"""

from __future__ import annotations

from typing import List, Optional


class AnalysisError(Exception):
    """Base exception for all analysis errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class CyclicGraphError(AnalysisError):
    """Raised when the activity network contains a cycle."""

    def __init__(self, cycle_path: Optional[List[str]] = None):
        self.cycle_path = cycle_path or []
        path_str = " -> ".join(self.cycle_path) if self.cycle_path else "detected"
        super().__init__(
            f"Cycle detected in activity network: {path_str}",
            details={"cycle_path": self.cycle_path},
        )


class EmptyGraphError(AnalysisError):
    """Raised when the activity network has no activities."""

    def __init__(self):
        super().__init__("Activity network is empty — no activities to analyze")


class DuplicateActivityError(AnalysisError):
    """Raised when duplicate activity IDs are found."""

    def __init__(self, activity_ids: List[str]):
        self.activity_ids = activity_ids
        super().__init__(
            f"Duplicate activity IDs found: {', '.join(activity_ids)}",
            details={"duplicate_ids": activity_ids},
        )


class MissingActivityError(AnalysisError):
    """Raised when a dependency references a nonexistent activity."""

    def __init__(self, missing_ids: List[str], reference_context: str = ""):
        self.missing_ids = missing_ids
        context = f" in {reference_context}" if reference_context else ""
        super().__init__(
            f"Dependencies reference nonexistent activities: "
            f"{', '.join(missing_ids)}{context}",
            details={"missing_ids": missing_ids, "context": reference_context},
        )


class InvalidDurationError(AnalysisError):
    """Raised when an activity has an invalid duration."""

    def __init__(self, activity_id: str, duration: float):
        self.activity_id = activity_id
        self.duration = duration
        super().__init__(
            f"Activity '{activity_id}' has invalid duration: {duration}",
            details={"activity_id": activity_id, "duration": duration},
        )


class MissingDurationError(AnalysisError):
    """Raised when an activity is missing a required duration."""

    def __init__(self, activity_ids: List[str]):
        self.activity_ids = activity_ids
        super().__init__(
            f"Activities missing duration: {', '.join(activity_ids)}",
            details={"activity_ids": activity_ids},
        )


class DisconnectedGraphError(AnalysisError):
    """Raised when the graph has disconnected components."""

    def __init__(self, component_count: int, unreachable_ids: Optional[List[str]] = None):
        self.component_count = component_count
        self.unreachable_ids = unreachable_ids or []
        super().__init__(
            f"Graph has {component_count} disconnected components",
            details={
                "component_count": component_count,
                "unreachable_ids": self.unreachable_ids,
            },
        )


class MultipleSourceNodesError(AnalysisError):
    """Raised when the graph has multiple source nodes (for networks requiring a single start)."""

    def __init__(self, source_ids: List[str]):
        self.source_ids = source_ids
        super().__init__(
            f"Multiple source nodes found: {', '.join(source_ids)}",
            details={"source_ids": source_ids},
        )


class MultipleSinkNodesError(AnalysisError):
    """Raised when the graph has multiple sink nodes (for networks requiring a single end)."""

    def __init__(self, sink_ids: List[str]):
        self.sink_ids = sink_ids
        super().__init__(
            f"Multiple sink nodes found: {', '.join(sink_ids)}",
            details={"sink_ids": sink_ids},
        )


class InvalidDependencyError(AnalysisError):
    """Raised when a dependency has an invalid type or structure."""

    def __init__(self, dependency_id: str, reason: str):
        self.dependency_id = dependency_id
        super().__init__(
            f"Invalid dependency '{dependency_id}': {reason}",
            details={"dependency_id": dependency_id, "reason": reason},
        )

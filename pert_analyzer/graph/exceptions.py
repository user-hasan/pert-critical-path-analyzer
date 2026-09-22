"""
Domain-specific exceptions for graph construction and validation.

These exceptions represent error conditions specific to building
and manipulating project network graphs.
"""

from __future__ import annotations

from typing import List, Optional


class GraphError(Exception):
    """Base exception for all graph-related errors."""

    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} [{detail_str}]"
        return self.message


class DuplicateActivityError(GraphError):
    """Raised when attempting to add an activity with an existing ID."""

    def __init__(self, activity_id: str):
        self.activity_id = activity_id
        super().__init__(
            f"Activity with ID '{activity_id}' already exists",
            details={"activity_id": activity_id},
        )


class DuplicateNodeError(GraphError):
    """Raised when attempting to add a node with an existing ID."""

    def __init__(self, node_id: str):
        self.node_id = node_id
        super().__init__(
            f"Node with ID '{node_id}' already exists",
            details={"node_id": node_id},
        )


class DuplicateDependencyError(GraphError):
    """Raised when attempting to add a duplicate dependency."""

    def __init__(self, source: str, target: str):
        self.source = source
        self.target = target
        super().__init__(
            f"Dependency from '{source}' to '{target}' already exists",
            details={"source": source, "target": target},
        )


class EmptyGraphError(GraphError):
    """Raised when a graph has no activities or nodes."""

    def __init__(self):
        super().__init__("Graph is empty — no activities or nodes to process")


class SelfDependencyError(GraphError):
    """Raised when an activity depends on itself."""

    def __init__(self, activity_id: str):
        self.activity_id = activity_id
        super().__init__(
            f"Activity '{activity_id}' has a self-dependency",
            details={"activity_id": activity_id},
        )


class InvalidDurationError(GraphError):
    """Raised when an activity has an invalid duration."""

    def __init__(self, activity_id: str, duration: float, reason: str = "negative"):
        self.activity_id = activity_id
        self.duration = duration
        self.reason = reason
        super().__init__(
            f"Activity '{activity_id}' has invalid duration: {duration} ({reason})",
            details={"activity_id": activity_id, "duration": duration, "reason": reason},
        )


class MissingDependencyRefError(GraphError):
    """Raised when a dependency references a nonexistent activity or node."""

    def __init__(self, missing_ids: List[str], context: str = ""):
        self.missing_ids = missing_ids
        self.context = context
        ctx = f" in {context}" if context else ""
        super().__init__(
            f"Dependencies reference nonexistent elements: {', '.join(missing_ids)}{ctx}",
            details={"missing_ids": missing_ids, "context": context},
        )


class InvalidGraphDataError(GraphError):
    """Raised when imported graph data is malformed or incomplete."""

    def __init__(self, message: str, field_name: str = ""):
        self.field_name = field_name
        super().__init__(
            message,
            details={"field": field_name} if field_name else {},
        )


class AOAConversionError(GraphError):
    """Raised when AOA-to-AON conversion fails."""

    def __init__(self, reason: str):
        super().__init__(
            f"AOA conversion failed: {reason}",
            details={"reason": reason},
        )

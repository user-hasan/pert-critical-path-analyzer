"""
GUI-only presentation mapping for backend GraphValidationIssue objects.

The Validation Center never re-implements graph validation, CPM, or
graph-model logic. This module only maps existing backend findings to
display metadata (severity, title, explanation, recommended action,
source) and to review-center targets, so the page can offer "open this
in Review" for the affected item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional, Tuple

from pert_analyzer.gui.reviews.categories import ReviewCategory


class ValidationSeverity(Enum):
    """Display severity for a presented validation issue."""

    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class IssueMeta:
    """Presentation metadata for one validation issue code."""

    title: str
    explanation: str
    action: str
    source: str


_ISSUE_META: dict[str, IssueMeta] = {
    "empty_graph": IssueMeta(
        "Graph is empty",
        "No activities were present in the reviewed graph.",
        "Fix the diagram so that activities are reconstructed, then re-analyze.",
        "graph structure",
    ),
    "missing_activity_id": IssueMeta(
        "Activity with empty ID",
        "An activity has no usable semantic ID.",
        "Open the activity in the Review Center and correct its ID.",
        "graph structure",
    ),
    "duplicate_activity_id": IssueMeta(
        "Duplicate activity IDs",
        "Two activities resolve to the same semantic ID.",
        "Open the conflicting activities in the Review Center and correct their IDs.",
        "review application",
    ),
    "missing_duration": IssueMeta(
        "Missing or invalid duration",
        "An activity has no positive duration, which blocks CPM.",
        "Open the activity in the Review Center and correct its duration.",
        "graph structure",
    ),
    "invalid_duration": IssueMeta(
        "Invalid duration",
        "An activity duration is not positive and finite.",
        "Open the activity in the Review Center and correct its duration.",
        "graph structure",
    ),
    "unknown_dependency_reference": IssueMeta(
        "Unknown dependency reference",
        "A dependency references an activity that does not exist in the graph.",
        "Correct the dependency endpoints in the Review Center.",
        "graph structure",
    ),
    "unknown_dependency_source": IssueMeta(
        "Unknown dependency source",
        "The dependency source does not reference a known activity.",
        "Correct the dependency source in the Review Center.",
        "graph structure",
    ),
    "unknown_dependency_target": IssueMeta(
        "Unknown dependency target",
        "The dependency target does not reference a known activity.",
        "Correct the dependency target in the Review Center.",
        "graph structure",
    ),
    "self_loop": IssueMeta(
        "Self-loop dependency",
        "A dependency points from an activity back to itself.",
        "Correct the dependency endpoints in the Review Center.",
        "graph structure",
    ),
    "duplicate_edge": IssueMeta(
        "Duplicate dependency",
        "The same dependency edge appears more than once.",
        "Review the affected dependency in the Review Center.",
        "graph structure",
    ),
    "cycle": IssueMeta(
        "Cycle detected",
        "The network contains a cycle; CPM cannot run on a cyclic graph.",
        "Re-examine dependency directions in the Review Center and remove the cycle.",
        "graph structure",
    ),
    "disconnected_components": IssueMeta(
        "Disconnected components",
        "The graph has more than one connected component; some activities are unreachable from the network start.",
        "Always allow all components to be reachable from the project start.",
        "graph structure",
    ),
    "disconnected_component": IssueMeta(
        "Disconnected component",
        "The graph has more than one connected component; some activities are unreachable from the network start.",
        "Always allow all components to be reachable from the project start.",
        "graph structure",
    ),
    "isolated_activity": IssueMeta(
        "Isolated activity",
        "An activity has no dependencies and is disconnected from the network.",
        "Connect the activity or remove it from the graph in the Review Center.",
        "graph structure",
    ),
}

_GENERIC = IssueMeta(
    "Validation issue",
    "The validation engine reported a problem with the reviewed graph.",
    "Resolve the reported problem, then revalidate and re-run CPM.",
    "validation",
)

_ALIASES: dict[str, str] = {
    "unknown_dependency_reference": "unknown_dependency_ref",
    "disconnected_component": "disconnected_components",
}

_DURATION_CODES = {"missing_duration", "invalid_duration"}
_ACTIVITY_CODES = {"missing_activity_id", "duplicate_activity_id"}
_DEPENDENCY_CODES = {
    "unknown_dependency_source",
    "unknown_dependency_target",
    "unknown_dependency_reference",
    "self_loop",
    "duplicate_edge",
}


def issue_meta(code: str) -> IssueMeta:
    """Presentation metadata for an issue code (aliases normalized)."""
    normalized = _ALIASES.get(code, code)
    return _ISSUE_META.get(normalized, _GENERIC)


def _present(issue: Any, severity: ValidationSeverity) -> "PresentedIssue":
    code = str(getattr(issue, "code", "unknown"))
    meta = issue_meta(code)
    return PresentedIssue(
        code=code,
        message=str(getattr(issue, "message", "") or ""),
        elements=list(getattr(issue, "elements", []) or []),
        severity=severity,
        title=meta.title,
        explanation=meta.explanation,
        action=meta.action,
        source=meta.source,
        issue=issue,
    )


def present_issues(validation: Any) -> List["PresentedIssue"]:
    """Wrap backend errors and warnings as presentation objects.

    Severity is derived from the backend's own result lists: the errors
    list is blocking, the warnings list is non-blocking. No validation
    logic is duplicated here.
    """
    if validation is None:
        return []
    issues: List[PresentedIssue] = []
    for issue in getattr(validation, "errors", []) or []:
        issues.append(_present(issue, ValidationSeverity.ERROR))
    for issue in getattr(validation, "warnings", []) or []:
        issues.append(_present(issue, ValidationSeverity.WARNING))
    return issues


def blocking_issues(validation: Any) -> List["PresentedIssue"]:
    """Blocking (ERROR) issues only."""
    return [p for p in present_issues(validation) if p.severity == ValidationSeverity.ERROR]


def review_target(issue: Any, review_session: Any) -> Optional[Tuple[ReviewCategory, str]]:
    """Map a validation issue to a specific Review Center item key.

    Returns ``(ReviewCategory, item_key)`` when a matching review item
    exists, otherwise None. Keys use the same format as
    ``categories.item_key`` so ReviewPage.focus_item() can select them.
    """
    if review_session is None or issue is None:
        return None
    raw = str(getattr(issue, "code", "") or "")
    if not raw:
        return None
    elements = list(getattr(issue, "elements", []) or [])

    if raw in _DURATION_CODES:
        key = _duration_key(review_session, elements)
        if key is not None:
            return ReviewCategory.DURATIONS, key
    elif raw in _ACTIVITY_CODES:
        key = _activity_key(review_session, elements)
        if key is not None:
            return ReviewCategory.ACTIVITIES, key
    elif raw in _DEPENDENCY_CODES:
        key = _dependency_key(review_session, elements)
        if key is not None:
            return ReviewCategory.DEPENDENCIES, key
    return None


def _duration_key(review_session: Any, elements: List[Any]) -> Optional[str]:
    for element in elements:
        element = str(element)
        for item in getattr(review_session, "durations", []) or []:
            ids = {
                str(getattr(item, "activity_id", "") or ""),
                str(getattr(item, "geometric_node_id", "") or ""),
            }
            if element in ids:
                node = getattr(item, "geometric_node_id", None) or getattr(item, "activity_id", None)
                if node is not None:
                    return f"dur:{node}"
    return None


def _activity_key(review_session: Any, elements: List[Any]) -> Optional[str]:
    for element in elements:
        element = str(element)
        for item in getattr(review_session, "activities", []) or []:
            ids = {
                str(getattr(item, "current_activity_id", "") or ""),
                str(getattr(item, "proposed_activity_id", "") or ""),
                str(getattr(item, "corrected_activity_id", "") or ""),
            }
            if element in ids:
                node = getattr(item, "geometric_node_id", None)
                if node is not None:
                    return f"act:{node}"
    return None


def _dependency_key(review_session: Any, elements: List[Any]) -> Optional[str]:
    for element in elements:
        element = str(element)
        for item in getattr(review_session, "dependencies", []) or []:
            ids = {
                str(getattr(item, "current_source_id", "") or ""),
                str(getattr(item, "current_target_id", "") or ""),
                str(getattr(item, "corrected_source_id", "") or ""),
                str(getattr(item, "corrected_target_id", "") or ""),
            }
            if element in ids:
                arrow = getattr(item, "arrow_id", None)
                if arrow is not None:
                    return f"dep:{arrow}"
    return None


@dataclass
class PresentedIssue:
    """A validation issue ready for GUI presentation."""

    code: str
    message: str
    elements: List[Any] = field(default_factory=list)
    severity: ValidationSeverity = ValidationSeverity.ERROR
    title: str = "Validation issue"
    explanation: str = ""
    action: str = ""
    source: str = "validation"
    issue: Any = None
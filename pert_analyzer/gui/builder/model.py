"""
Manual Network Builder data model.

The model is the single source of truth for the manually-built network.
It owns activities, dependencies, and delegates to existing infrastructure
for validation and CPM analysis.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from PySide6.QtCore import QObject, Signal

from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DependencyType,
    DiagramType,
    GraphModel,
)
from pert_analyzer.pipeline.human_review import (
    CpmGateStatus,
    GraphValidationResult,
    GraphValidationIssue,
    GraphStatus,
    ReviewSession,
    ReviewedGraphCandidate,
    validate_graph_structure,
)

logger = logging.getLogger(__name__)


@dataclass
class ActivityEntry:
    """Lightweight activity record for the builder table."""

    activity_id: str
    duration: float = 0.0
    predecessors: str = ""
    name: str = ""


@dataclass
class RelationshipEntry:
    """A single directed edge in the network."""

    source: str
    target: str


class ManualNetworkModel(QObject):
    """
    Data model for the manual network builder.

    Signals:
        activity_added:   (activity_id)
        activity_removed: (activity_id)
        activity_changed: (activity_id)
        relationship_added:   (source, target)
        relationship_removed: (source, target)
        model_changed:    () — emitted on any structural change
        validation_changed: () — emitted when validation state changes
        cpm_calculated:   () — emitted when CPM is calculated
        cpm_invalidated:  () — emitted when CPM results go stale
        selection_changed: (item_type, item_id) — 'activity'|'relationship', id
    """

    activity_added = Signal(str)
    activity_removed = Signal(str)
    activity_changed = Signal(str)
    relationship_added = Signal(str, str)
    relationship_removed = Signal(str, str)
    model_changed = Signal()
    validation_changed = Signal()
    cpm_calculated = Signal()
    cpm_invalidated = Signal()
    selection_changed = Signal(str, str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._activities: Dict[str, ActivityEntry] = {}
        self._relationships: List[RelationshipEntry] = []
        self._validation: Optional[GraphValidationResult] = None
        self._cpm_result: Optional[Any] = None
        self._cpm_stale: bool = True
        self._selected_activity: Optional[str] = None
        self._selected_relationship: Optional[Tuple[str, str]] = None

    # ── Activity operations ──────────────────────────────────

    def add_activity(self, activity_id: str, duration: float = 0.0,
                     predecessors: str = "", name: str = "") -> Optional[str]:
        """
        Add an activity. Returns an error string or None on success.
        """
        aid = activity_id.strip()
        if not aid:
            return "Activity ID cannot be empty."
        if aid in self._activities:
            return f'Activity ID "{aid}" already exists.'

        try:
            d = float(duration)
        except (ValueError, TypeError):
            return f"Duration must be a numeric value."
        if d < 0:
            return f"Duration must be a non-negative value."

        # Validate predecessors reference existing activities before adding
        pred_err = self._validate_predecessors(aid, predecessors.strip())
        if pred_err:
            return pred_err

        entry = ActivityEntry(activity_id=aid, duration=d,
                              predecessors=predecessors.strip(), name=name.strip())
        self._activities[aid] = entry

        # Build relationships from the predecessor string
        self._rebuild_relationships_for(aid)

        self.activity_added.emit(aid)
        self._on_structural_change()
        return None

    def update_activity(self, activity_id: str, duration: Optional[float] = None,
                        predecessors: Optional[str] = None) -> Optional[str]:
        """Update an existing activity. Returns error string or None."""
        if activity_id not in self._activities:
            return f'Activity "{activity_id}" does not exist.'

        entry = self._activities[activity_id]

        if duration is not None:
            try:
                d = float(duration)
            except (ValueError, TypeError):
                return "Duration must be a numeric value."
            if d < 0:
                return "Duration must be a non-negative value."
            entry.duration = d

        if predecessors is not None:
            entry.predecessors = predecessors.strip()
            pred_err = self._validate_predecessors(activity_id, entry.predecessors)
            if pred_err:
                return pred_err

        # Rebuild relationships from predecessors
        self._rebuild_relationships_for(activity_id)
        self.activity_changed.emit(activity_id)
        self._on_structural_change()
        return None

    def remove_activity(self, activity_id: str) -> Optional[str]:
        """Remove an activity and its associated relationships."""
        if activity_id not in self._activities:
            return f'Activity "{activity_id}" does not exist.'

        del self._activities[activity_id]

        removed = [
            (r.source, r.target) for r in self._relationships
            if r.source == activity_id or r.target == activity_id
        ]
        self._relationships = [
            r for r in self._relationships
            if r.source != activity_id and r.target != activity_id
        ]

        for src, tgt in removed:
            self.relationship_removed.emit(src, tgt)

        self.activity_removed.emit(activity_id)
        if self._selected_activity == activity_id:
            self._selected_activity = None
            self.selection_changed.emit("", "")
        self._on_structural_change()
        return None

    def rename_activity(self, old_id: str, new_id: str) -> Optional[str]:
        """Rename an activity ID throughout the model."""
        new_id = new_id.strip()
        if not new_id:
            return "Activity ID cannot be empty."
        if new_id == old_id:
            return None
        if new_id in self._activities:
            return f'Activity ID "{new_id}" already exists.'
        if old_id not in self._activities:
            return f'Activity "{old_id}" does not exist.'

        entry = self._activities.pop(old_id)
        entry.activity_id = new_id
        self._activities[new_id] = entry

        for r in self._relationships:
            if r.source == old_id:
                r.source = new_id
            if r.target == old_id:
                r.target = new_id

        self._on_structural_change()
        return None

    def get_activity(self, activity_id: str) -> Optional[ActivityEntry]:
        return self._activities.get(activity_id)

    def get_all_activities(self) -> List[ActivityEntry]:
        return list(self._activities.values())

    def get_activity_ids(self) -> List[str]:
        return list(self._activities.keys())

    def get_predecessors(self, activity_id: str) -> List[str]:
        return [r.source for r in self._relationships if r.target == activity_id]

    def get_successors(self, activity_id: str) -> List[str]:
        return [r.target for r in self._relationships if r.source == activity_id]

    # ── Relationship operations ──────────────────────────────

    def add_relationship(self, source: str, target: str) -> Optional[str]:
        """Add a relationship. Returns error string or None."""
        if source not in self._activities:
            return f'Activity "{source}" does not exist.'
        if target not in self._activities:
            return f'Activity "{target}" does not exist.'
        if source == target:
            return f"{source} \u2192 {target} is not allowed (self-loop)."

        for r in self._relationships:
            if r.source == source and r.target == target:
                return f"Relationship {source} \u2192 {target} already exists."

        self._relationships.append(RelationshipEntry(source=source, target=target))
        self.relationship_added.emit(source, target)
        self._on_structural_change()
        return None

    def remove_relationship(self, source: str, target: str) -> Optional[str]:
        """Remove a relationship."""
        before = len(self._relationships)
        self._relationships = [
            r for r in self._relationships
            if not (r.source == source and r.target == target)
        ]
        if len(self._relationships) < before:
            self.relationship_removed.emit(source, target)
            if self._selected_relationship == (source, target):
                self._selected_relationship = None
                self.selection_changed.emit("", "")
            self._on_structural_change()
            return None
        return f"Relationship {source} \u2192 {target} not found."

    def get_all_relationships(self) -> List[RelationshipEntry]:
        return list(self._relationships)

    def has_relationship(self, source: str, target: str) -> bool:
        return any(r.source == source and r.target == target
                   for r in self._relationships)

    # ── Selection ────────────────────────────────────────────

    def select_activity(self, activity_id: Optional[str]) -> None:
        self._selected_activity = activity_id
        self._selected_relationship = None
        self.selection_changed.emit("activity", activity_id or "")

    def select_relationship(self, source: Optional[str],
                            target: Optional[str]) -> None:
        self._selected_activity = None
        self._selected_relationship = (source, target) if source and target else None
        self.selection_changed.emit(
            "relationship", f"{source}\u2192{target}" if source and target else ""
        )

    @property
    def selected_activity(self) -> Optional[str]:
        return self._selected_activity

    @property
    def selected_relationship(self) -> Optional[Tuple[str, str]]:
        return self._selected_relationship

    # ── Validation ───────────────────────────────────────────

    def validate(self) -> GraphValidationResult:
        """Run real validation using the existing infrastructure."""
        activity_ids = list(self._activities.keys())
        durations = {aid: e.duration for aid, e in self._activities.items()}
        edges = [(r.source, r.target) for r in self._relationships]
        is_dummy = {aid: False for aid in activity_ids}

        self._validation = validate_graph_structure(
            activity_ids=activity_ids,
            durations=durations,
            edges=edges,
            is_dummy=is_dummy,
        )
        self.validation_changed.emit()
        return self._validation

    @property
    def validation_result(self) -> Optional[GraphValidationResult]:
        return self._validation

    @property
    def is_valid(self) -> bool:
        if self._validation is None:
            return False
        return self._validation.status == GraphStatus.VALID

    @property
    def status_text(self) -> str:
        if not self._activities:
            return "No activities defined."
        if self._validation is None:
            return "Not validated."
        if self._validation.status == GraphStatus.INVALID:
            n = len(self._validation.errors)
            return f"{n} issue{'s' if n != 1 else ''} requires attention."
        return "Network valid."

    # ── Graph building ───────────────────────────────────────

    def build_graph_model(self) -> GraphModel:
        """Build a GraphModel from the current data for CPM / Results."""
        graph = GraphModel(diagram_type=DiagramType.AON)
        for aid, entry in self._activities.items():
            activity = Activity(
                activity_id=aid,
                name=entry.name or aid,
                duration=entry.duration,
            )
            graph.activities[aid] = activity
        for r in self._relationships:
            dep = Dependency(
                source=r.source,
                target=r.target,
                dependency_type=DependencyType.FINISH_TO_START.value,
            )
            graph.dependencies.append(dep)
        return graph

    # ── CPM ──────────────────────────────────────────────────

    def calculate_cpm(self) -> Optional[str]:
        """Run CPM on the current graph. Returns error string or None."""
        self.validate()
        if not self.is_valid:
            first_err = (self._validation.errors[0].message
                         if self._validation and self._validation.errors
                         else "Graph is not valid.")
            return first_err

        graph = self.build_graph_model()
        engine = CPMEngine()
        try:
            self._cpm_result = engine.analyze(graph)
        except Exception as exc:
            return f"CPM analysis failed: {exc}"

        self._cpm_stale = False
        self.cpm_calculated.emit()
        return None

    def invalidate_cpm(self) -> None:
        """Mark CPM results as stale."""
        if not self._cpm_stale and self._cpm_result is not None:
            self._cpm_stale = True
            self.cpm_invalidated.emit()

    @property
    def cpm_result(self) -> Optional[Any]:
        return self._cpm_result

    @property
    def cpm_is_stale(self) -> bool:
        return self._cpm_stale

    # ── Candidate creation ───────────────────────────────────

    def create_candidate(self) -> Optional[ReviewedGraphCandidate]:
        """Build the ReviewedGraphCandidate that feeds the Results dashboard."""
        self.validate()
        graph = self.build_graph_model()

        cpm_result = None
        cpm_gate = CpmGateStatus.BLOCKED_REVIEW
        if self.is_valid:
            try:
                engine = CPMEngine()
                cpm_result = engine.analyze(graph)
                cpm_gate = CpmGateStatus.RUNNABLE
                self._cpm_result = cpm_result
                self._cpm_stale = False
            except Exception:
                cpm_gate = CpmGateStatus.BLOCKED_REVIEW

        session = ReviewSession(source_image_id="manual_network_builder")
        candidate = ReviewedGraphCandidate(
            review_session=session,
            graph=graph,
            validation=self._validation,
            cpm=cpm_result,
            cpm_gate=cpm_gate,
            pure_critical_paths=(
                cpm_result.critical_paths if cpm_result else []
            ),
        )
        return candidate

    # ── Summary properties ───────────────────────────────────

    @property
    def activity_count(self) -> int:
        return len(self._activities)

    @property
    def dependency_count(self) -> int:
        return len(self._relationships)

    @property
    def has_activities(self) -> bool:
        return bool(self._activities)

    @property
    def next_activity_id(self) -> str:
        """Generate the next available single-letter activity ID."""
        used = set(self._activities.keys())
        for letter in [chr(c) for c in range(ord('A'), ord('Z') + 1)]:
            if letter not in used:
                return letter
        for i in range(1, 1000):
            candidate = f"A{i}"
            if candidate not in used:
                return candidate
        return "N1"

    # ── Internal ─────────────────────────────────────────────

    def _on_structural_change(self) -> None:
        self.model_changed.emit()
        self.invalidate_cpm()
        self.validate()

    def _rebuild_relationships_for(self, activity_id: str) -> None:
        """Rebuild relationships from an activity's predecessor string."""
        entry = self._activities.get(activity_id)
        if entry is None:
            return

        # Remove existing relationships targeting this activity
        self._relationships = [
            r for r in self._relationships if r.target != activity_id
        ]

        # Parse predecessors and add relationships
        if entry.predecessors and entry.predecessors.strip() not in ("", "-", "\u2014"):
            for token in entry.predecessors.split(","):
                pred = token.strip().upper()
                if pred and pred in self._activities:
                    if not self.has_relationship(pred, activity_id):
                        self._relationships.append(
                            RelationshipEntry(source=pred, target=activity_id)
                        )

    def _validate_predecessors(self, activity_id: str,
                               predecessors_str: str) -> Optional[str]:
        """Validate that all predecessors exist. Returns first error or None."""
        if not predecessors_str or predecessors_str.strip() in ("", "-", "\u2014"):
            return None
        for token in predecessors_str.split(","):
            pred = token.strip().upper()
            if pred and pred not in self._activities:
                return f'Activity "{pred}" does not exist.'
        return None

    def clear(self) -> None:
        """Reset the model completely."""
        self._activities.clear()
        self._relationships.clear()
        self._validation = None
        self._cpm_result = None
        self._cpm_stale = True
        self._selected_activity = None
        self._selected_relationship = None
        self.model_changed.emit()
        self.validation_changed.emit()

    def load_example(self) -> None:
        """Load the reference AON example (A-V) for demonstration."""
        self.clear()
        activities = [
            ("A", 1, ""), ("B", 2, "A"), ("C", 2, "A"),
            ("D", 3, "B, C"), ("E", 4, "D"), ("F", 4, "D"),
            ("G", 4, "D"), ("H", 4, "D"),
            ("I", 5, "E, F, G, H"), ("J", 5, "I"), ("K", 3, "J"),
            ("L", 6, "K"), ("M", 6, "K"),
            ("N", 8, "L, M"), ("O", 3, "N"), ("P", 4, "O"),
            ("Q", 3, "P"), ("R", 2, "Q"), ("S", 1, "R"),
            ("T", 2, "S"), ("U", 1, "T"), ("V", 1, "U"),
        ]
        for aid, dur, pred in activities:
            self.add_activity(aid, duration=dur, predecessors=pred)

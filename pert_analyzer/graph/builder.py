"""
GraphBuilder for constructing GraphModel instances from structured data.

Supports both AON (Activity-on-Node) and AOA (Activity-on-Arrow) inputs.
Provides validation during construction and clean API for building graphs.

Usage:
    builder = GraphBuilder()
    builder.add_activity("A", duration=5)
    builder.add_activity("B", duration=3)
    builder.add_dependency("A", "B")
    graph = builder.build()
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DependencyType,
    DiagramType,
    GraphModel,
    Node,
)
from pert_analyzer.graph.exceptions import (
    DuplicateActivityError,
    DuplicateDependencyError,
    DuplicateNodeError,
    EmptyGraphError,
    InvalidDurationError,
    MissingDependencyRefError,
    SelfDependencyError,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a GraphModel from structured input data.

    This builder provides a fluent API for constructing project networks
    from manually entered data, CSV imports, JSON loads, or programmatic
    construction.

    The builder validates data during construction and raises domain-specific
    exceptions for invalid inputs.

    Usage:
        builder = GraphBuilder()
        builder.add_activity("A", duration=5, name="Design")
        builder.add_activity("B", duration=3, name="Build")
        builder.add_dependency("A", "B")
        graph = builder.build()
    """

    def __init__(self, diagram_type: DiagramType = DiagramType.AON):
        """
        Initialize the builder.

        Args:
            diagram_type: The type of diagram to build (AON or AOA).
        """
        self._graph = GraphModel(diagram_type=diagram_type)
        self._allow_self_dependency = False

    @property
    def graph(self) -> GraphModel:
        """Get the current graph being built."""
        return self._graph

    def allow_self_dependency(self, allow: bool = True) -> GraphBuilder:
        """
        Enable or disable self-dependencies (useful for AOA dummy activities).

        Returns:
            self, for method chaining.
        """
        self._allow_self_dependency = allow
        return self

    def add_activity(
        self,
        activity_id: str,
        duration: float = 0.0,
        name: str = "",
        is_dummy: bool = False,
        source_node: Optional[str] = None,
        target_node: Optional[str] = None,
        optimistic_time: Optional[float] = None,
        most_likely_time: Optional[float] = None,
        pessimistic_time: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphBuilder:
        """
        Add an activity to the graph.

        Args:
            activity_id: Unique identifier for the activity.
            duration: Duration of the activity (must be >= 0).
            name: Human-readable name.
            is_dummy: Whether this is a dummy activity (zero duration).
            source_node: Source node ID (for AOA representation).
            target_node: Target node ID (for AOA representation).
            optimistic_time: PERT optimistic time.
            most_likely_time: PERT most likely time.
            pessimistic_time: PERT pessimistic time.
            metadata: Additional metadata.

        Returns:
            self, for method chaining.

        Raises:
            DuplicateActivityError: If activity_id already exists.
            InvalidDurationError: If duration is negative.
        """
        if activity_id in self._graph.activities:
            raise DuplicateActivityError(activity_id)

        if duration is not None and duration < 0:
            raise InvalidDurationError(activity_id, duration, "negative")

        if duration is not None and duration == 0 and not is_dummy:
            raise InvalidDurationError(activity_id, duration, "zero duration for non-dummy activity")

        activity = Activity(
            activity_id=activity_id,
            name=name or activity_id,
            duration=duration or 0.0,
            is_dummy=is_dummy,
            source_node=source_node,
            target_node=target_node,
            optimistic_time=optimistic_time,
            most_likely_time=most_likely_time,
            pessimistic_time=pessimistic_time,
            metadata=metadata or {},
        )

        self._graph.activities[activity_id] = activity
        logger.debug("Added activity '%s' with duration %.2f", activity_id, duration)
        return self

    def add_node(
        self,
        node_id: str,
        label: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphBuilder:
        """
        Add a node/event to the graph (primarily for AOA representation).

        Args:
            node_id: Unique identifier for the node.
            label: Human-readable label.
            metadata: Additional metadata.

        Returns:
            self, for method chaining.

        Raises:
            DuplicateNodeError: If node_id already exists.
        """
        if node_id in self._graph.nodes:
            raise DuplicateNodeError(node_id)

        node = Node(
            node_id=node_id,
            label=label or node_id,
            metadata=metadata or {},
        )

        self._graph.nodes[node_id] = node
        logger.debug("Added node '%s'", node_id)
        return self

    def add_dependency(
        self,
        source: str,
        target: str,
        dependency_type: DependencyType = DependencyType.FINISH_TO_START,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GraphBuilder:
        """
        Add a dependency between two activities (AON) or nodes (AOA).

        Args:
            source: Source activity/node ID.
            target: Target activity/node ID.
            dependency_type: Type of precedence relationship.
            metadata: Additional metadata.

        Returns:
            self, for method chaining.

        Raises:
            SelfDependencyError: If source == target and self-dependencies not allowed.
            DuplicateDependencyError: If the dependency already exists.
            MissingDependencyRefError: If source or target doesn't exist.
        """
        if source == target and not self._allow_self_dependency:
            raise SelfDependencyError(source)

        # Check for duplicate
        for dep in self._graph.dependencies:
            if dep.source == source and dep.target == target:
                raise DuplicateDependencyError(source, target)

        # Validate references exist
        missing = []
        if self._graph.diagram_type == DiagramType.AON:
            if source not in self._graph.activities:
                missing.append(source)
            if target not in self._graph.activities:
                missing.append(target)
        else:  # AOA
            if source not in self._graph.nodes:
                missing.append(source)
            if target not in self._graph.nodes:
                missing.append(target)

        if missing:
            raise MissingDependencyRefError(missing, "graph construction")

        dependency = Dependency(
            source=source,
            target=target,
            dependency_type=dependency_type.value,
            metadata=metadata or {},
        )

        self._graph.dependencies.append(dependency)
        logger.debug("Added dependency '%s' -> '%s'", source, target)
        return self

    def build(self) -> GraphModel:
        """
        Build and return the GraphModel.

        Returns:
            A validated GraphModel.

        Raises:
            EmptyGraphError: If the graph has no activities (for AON) or
                no nodes (for AOA).
        """
        if self._graph.diagram_type == DiagramType.AON:
            if not self._graph.activities:
                raise EmptyGraphError()
        elif self._graph.diagram_type == DiagramType.AOA:
            if not self._graph.nodes:
                raise EmptyGraphError()

        logger.info(
            "Built %s graph: %d activities, %d nodes, %d dependencies",
            self._graph.diagram_type.value,
            len(self._graph.activities),
            len(self._graph.nodes),
            len(self._graph.dependencies),
        )

        return self._graph

    def reset(self) -> GraphBuilder:
        """
        Reset the builder to start fresh.

        Returns:
            self, for method chaining.
        """
        diagram_type = self._graph.diagram_type
        self._graph = GraphModel(diagram_type=diagram_type)
        return self


def build_aon_graph(
    activities: List[Dict[str, Any]],
    dependencies: List[Tuple[str, str]],
    project_name: str = "",
) -> GraphModel:
    """
    Build an AON graph from structured data.

    This is a convenience function for creating AON graphs from
    dictionaries and tuples.

    Args:
        activities: List of activity dictionaries with keys:
            - activity_id (required)
            - duration (required)
            - name (optional)
            - is_dummy (optional)
            - optimistic_time (optional)
            - most_likely_time (optional)
            - pessimistic_time (optional)
        dependencies: List of (source_id, target_id) tuples.
        project_name: Optional project name for metadata.

    Returns:
        A validated GraphModel.

    Example:
        activities = [
            {"activity_id": "A", "duration": 5, "name": "Design"},
            {"activity_id": "B", "duration": 3, "name": "Build"},
        ]
        deps = [("A", "B")]
        graph = build_aon_graph(activities, deps)
    """
    builder = GraphBuilder(diagram_type=DiagramType.AON)

    if project_name:
        builder.graph.metadata["project_name"] = project_name

    for act_data in activities:
        builder.add_activity(
            activity_id=act_data["activity_id"],
            duration=act_data.get("duration", 0.0),
            name=act_data.get("name", ""),
            is_dummy=act_data.get("is_dummy", False),
            optimistic_time=act_data.get("optimistic_time"),
            most_likely_time=act_data.get("most_likely_time"),
            pessimistic_time=act_data.get("pessimistic_time"),
        )

    for source, target in dependencies:
        builder.add_dependency(source, target)

    return builder.build()


def build_aoa_graph(
    nodes: List[Dict[str, Any]],
    activities: List[Dict[str, Any]],
    dependencies: List[Tuple[str, str]],
    project_name: str = "",
) -> GraphModel:
    """
    Build an AOA graph from structured data.

    This is a convenience function for creating AOA graphs from
    dictionaries and tuples.

    Args:
        nodes: List of node/event dictionaries with keys:
            - node_id (required)
            - label (optional)
        activities: List of activity dictionaries with keys:
            - activity_id (required)
            - duration (required)
            - name (optional)
            - is_dummy (optional)
            - source_node (required)
            - target_node (required)
        dependencies: List of (source_node_id, target_node_id) tuples.
        project_name: Optional project name for metadata.

    Returns:
        A validated GraphModel.

    Example:
        nodes = [
            {"node_id": "1", "label": "Start"},
            {"node_id": "2", "label": "End"},
        ]
        activities = [
            {"activity_id": "A", "duration": 5, "source_node": "1", "target_node": "2"},
        ]
        deps = [("1", "2")]
        graph = build_aoa_graph(nodes, activities, deps)
    """
    builder = GraphBuilder(diagram_type=DiagramType.AOA)

    if project_name:
        builder.graph.metadata["project_name"] = project_name

    for node_data in nodes:
        builder.add_node(
            node_id=node_data["node_id"],
            label=node_data.get("label", ""),
        )

    for act_data in activities:
        builder.add_activity(
            activity_id=act_data["activity_id"],
            duration=act_data.get("duration", 0.0),
            name=act_data.get("name", ""),
            is_dummy=act_data.get("is_dummy", False),
            source_node=act_data.get("source_node"),
            target_node=act_data.get("target_node"),
        )

    for source, target in dependencies:
        builder.add_dependency(source, target)

    return builder.build()

"""
NetworkX adapter for converting between GraphModel and NetworkX DiGraph.

This module provides a clean conversion layer between the semantic
GraphModel and NetworkX's graph representation, isolating NetworkX
implementation details from the rest of the application.

Usage:
    from pert_analyzer.graph.adapter import NetworkXAdapter

    adapter = NetworkXAdapter()
    nx_graph = adapter.to_networkx(graph_model)
    # ... use NetworkX operations ...
    graph_model = adapter.from_networkx(nx_graph)
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DiagramType,
    GraphModel,
    Node,
)
from pert_analyzer.graph.exceptions import (
    EmptyGraphError,
    InvalidGraphDataError,
    MissingDependencyRefError,
)

logger = logging.getLogger(__name__)


class NetworkXAdapter:
    """
    Adapter for converting between GraphModel and NetworkX DiGraph.

    In AON representation:
    - NetworkX nodes = activities (identified by activity_id)
    - NetworkX edges = dependencies (precedence relationships)

    In AOA representation:
    - NetworkX nodes = events/nodes
    - NetworkX edges = activities (with activity metadata on edges)

    The adapter preserves all semantic information during conversion.
    """

    def __init__(self):
        """Initialize the adapter."""
        pass

    def to_networkx(self, graph: GraphModel) -> nx.DiGraph:
        """
        Convert a GraphModel to a NetworkX DiGraph.

        For AON graphs:
        - Each activity becomes a node with duration, name, is_dummy attributes.
        - Each dependency becomes an edge with dependency_id, dependency_type.

        For AOA graphs:
        - Each node/event becomes a node.
        - Each activity becomes an edge with activity_id, duration, is_dummy.

        Args:
            graph: The GraphModel to convert.

        Returns:
            A NetworkX DiGraph.

        Raises:
            EmptyGraphError: If the graph has no activities or nodes.
        """
        if graph.diagram_type == DiagramType.AON:
            return self._aon_to_networkx(graph)
        elif graph.diagram_type == DiagramType.AOA:
            return self._aoa_to_networkx(graph)
        else:
            # For unknown type, try AON first
            if graph.activities:
                return self._aon_to_networkx(graph)
            elif graph.nodes:
                return self._aoa_to_networkx(graph)
            else:
                raise EmptyGraphError()

    def _aon_to_networkx(self, graph: GraphModel) -> nx.DiGraph:
        """Convert AON GraphModel to NetworkX DiGraph."""
        if not graph.activities:
            raise EmptyGraphError()

        nx_graph = nx.DiGraph()

        # Add activities as nodes
        for aid, activity in graph.activities.items():
            nx_graph.add_node(
                aid,
                duration=activity.duration,
                name=activity.name,
                is_dummy=activity.is_dummy,
                node_type="activity",
            )

        # Add dependencies as edges
        for dep in graph.dependencies:
            if dep.source in graph.activities and dep.target in graph.activities:
                nx_graph.add_edge(
                    dep.source,
                    dep.target,
                    dependency_id=dep.dependency_id,
                    dependency_type=dep.dependency_type,
                )

        logger.debug(
            "Converted AON to NetworkX: %d nodes, %d edges",
            nx_graph.number_of_nodes(),
            nx_graph.number_of_edges(),
        )
        return nx_graph

    def _aoa_to_networkx(self, graph: GraphModel) -> nx.DiGraph:
        """Convert AOA GraphModel to NetworkX DiGraph."""
        if not graph.nodes:
            raise EmptyGraphError()

        nx_graph = nx.DiGraph()

        # Add nodes/events
        for nid, node in graph.nodes.items():
            nx_graph.add_node(
                nid,
                label=node.label,
                node_type="event",
            )

        # Add activities as edges
        for aid, activity in graph.activities.items():
            source = activity.source_node
            target = activity.target_node
            if source and target and source in graph.nodes and target in graph.nodes:
                nx_graph.add_edge(
                    source,
                    target,
                    activity_id=aid,
                    duration=activity.duration,
                    name=activity.name,
                    is_dummy=activity.is_dummy,
                )

        logger.debug(
            "Converted AOA to NetworkX: %d nodes, %d edges",
            nx_graph.number_of_nodes(),
            nx_graph.number_of_edges(),
        )
        return nx_graph

    def from_networkx(
        self,
        nx_graph: nx.DiGraph,
        diagram_type: DiagramType = DiagramType.AON,
    ) -> GraphModel:
        """
        Convert a NetworkX DiGraph back to a GraphModel.

        Args:
            nx_graph: The NetworkX graph to convert.
            diagram_type: The diagram type to assign.

        Returns:
            A GraphModel.
        """
        graph = GraphModel(diagram_type=diagram_type)

        if diagram_type == DiagramType.AON:
            self._networkx_to_aon(nx_graph, graph)
        elif diagram_type == DiagramType.AOA:
            self._networkx_to_aoa(nx_graph, graph)
        else:
            # Default to AON interpretation
            self._networkx_to_aon(nx_graph, graph)

        return graph

    def _networkx_to_aon(self, nx_graph: nx.DiGraph, graph: GraphModel) -> None:
        """Convert NetworkX DiGraph to AON GraphModel."""
        for node_id, attrs in nx_graph.nodes(data=True):
            activity = Activity(
                activity_id=str(node_id),
                name=attrs.get("name", str(node_id)),
                duration=attrs.get("duration", 0.0),
                is_dummy=attrs.get("is_dummy", False),
            )
            graph.activities[node_id] = activity

        for source, target, attrs in nx_graph.edges(data=True):
            dependency = Dependency(
                source=str(source),
                target=str(target),
                dependency_type=attrs.get("dependency_type", "finish_to_start"),
            )
            graph.dependencies.append(dependency)

    def _networkx_to_aoa(self, nx_graph: nx.DiGraph, graph: GraphModel) -> None:
        """Convert NetworkX DiGraph to AOA GraphModel."""
        for node_id, attrs in nx_graph.nodes(data=True):
            node = Node(
                node_id=str(node_id),
                label=attrs.get("label", str(node_id)),
            )
            graph.nodes[node_id] = node

        edge_count = 0
        for source, target, attrs in nx_graph.edges(data=True):
            edge_count += 1
            activity_id = attrs.get("activity_id", f"act_{source}_{target}")
            activity = Activity(
                activity_id=activity_id,
                name=attrs.get("name", ""),
                duration=attrs.get("duration", 0.0),
                is_dummy=attrs.get("is_dummy", False),
                source_node=str(source),
                target_node=str(target),
            )
            graph.activities[activity_id] = activity

            dependency = Dependency(
                source=str(source),
                target=str(target),
            )
            graph.dependencies.append(dependency)

    def get_predecessors(
        self, nx_graph: nx.DiGraph, node_id: str
    ) -> List[str]:
        """Get predecessor node IDs from a NetworkX graph."""
        return list(nx_graph.predecessors(node_id))

    def get_successors(
        self, nx_graph: nx.DiGraph, node_id: str
    ) -> List[str]:
        """Get successor node IDs from a NetworkX graph."""
        return list(nx_graph.successors(node_id))

    def get_sources(self, nx_graph: nx.DiGraph) -> List[str]:
        """Get source nodes (nodes with no predecessors)."""
        return [n for n in nx_graph.nodes() if nx_graph.in_degree(n) == 0]

    def get_sinks(self, nx_graph: nx.DiGraph) -> List[str]:
        """Get sink nodes (nodes with no successors)."""
        return [n for n in nx_graph.nodes() if nx_graph.out_degree(n) == 0]

    def is_dag(self, nx_graph: nx.DiGraph) -> bool:
        """Check if the graph is a Directed Acyclic Graph."""
        return nx.is_directed_acyclic_graph(nx_graph)

    def topological_sort(self, nx_graph: nx.DiGraph) -> List[str]:
        """Get nodes in topological order."""
        return list(nx.topological_sort(nx_graph))

    def get_all_paths(
        self, nx_graph: nx.DiGraph, source: str, target: str
    ) -> List[List[str]]:
        """Get all simple paths from source to target."""
        try:
            return list(nx.all_simple_paths(nx_graph, source, target))
        except nx.NetworkXError:
            return []

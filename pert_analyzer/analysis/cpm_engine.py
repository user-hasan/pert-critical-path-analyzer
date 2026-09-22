"""
CPM (Critical Path Method) Engine.

Implements the complete CPM analysis for project networks:
- Forward pass (ES, EF calculation)
- Backward pass (LS, LF calculation)
- Total float and free float calculation
- Critical activity identification
- ALL critical path extraction

Uses NetworkX for graph operations while keeping the mathematical
business logic clean and testable.

Reference:
    The CPM algorithm operates on a Directed Acyclic Graph (DAG)
    where nodes represent activities and edges represent precedence
    dependencies (finish-to-start relationships).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx

from pert_analyzer.analysis.exceptions import (
    CyclicGraphError,
    DisconnectedGraphError,
    DuplicateActivityError,
    EmptyGraphError,
    InvalidDurationError,
    MissingActivityError,
    MissingDurationError,
)
from pert_analyzer.core.interfaces import AnalysisEngine
from pert_analyzer.core.models import (
    Activity,
    ActivityAnalysis,
    AnalysisResult,
    Dependency,
    DiagramType,
    GraphModel,
)

logger = logging.getLogger(__name__)

# Default tolerance for floating-point comparison
DEFAULT_FLOAT_TOLERANCE = 1e-9


class CPMEngine(AnalysisEngine):
    """
    Critical Path Method analysis engine.

    Performs forward pass, backward pass, float calculation,
    and critical path identification on project activity networks.

    The engine operates on a GraphModel containing activities and
    their dependency relationships. It builds an internal NetworkX
    directed graph for efficient graph operations.

    Usage:
        engine = CPMEngine()
        result = engine.analyze(graph_model)

    The engine validates the graph before analysis and raises
    domain-specific exceptions for invalid inputs.
    """

    def __init__(self, float_tolerance: float = DEFAULT_FLOAT_TOLERANCE):
        """
        Initialize the CPM engine.

        Args:
            float_tolerance: Tolerance for floating-point comparison
                when determining if a value is zero (e.g., for float
                calculation). Default is 1e-9.
        """
        self._float_tolerance = float_tolerance

    @property
    def float_tolerance(self) -> float:
        """Get the current float tolerance."""
        return self._float_tolerance

    @float_tolerance.setter
    def float_tolerance(self, value: float) -> None:
        """Set the float tolerance."""
        if value < 0:
            raise ValueError("Float tolerance must be non-negative")
        self._float_tolerance = value

    def analyze(self, graph: GraphModel) -> AnalysisResult:
        """
        Perform complete CPM analysis on the project graph.

        Executes:
            1. Graph validation
            2. NetworkX graph construction
            3. Forward pass (ES, EF)
            4. Backward pass (LS, LF)
            5. Float calculation
            6. Critical path identification

        Args:
            graph: The GraphModel to analyze.

        Returns:
            AnalysisResult containing all CPM analysis results.

        Raises:
            EmptyGraphError: If the graph has no activities.
            CyclicGraphError: If the graph contains cycles.
            InvalidDurationError: If any activity has invalid duration.
            MissingDurationError: If activities are missing durations.
            MissingActivityError: If dependencies reference nonexistent activities.
            DuplicateActivityError: If duplicate activity IDs exist.
        """
        logger.info("Starting CPM analysis")

        # Step 1: Validate
        validation_errors = self._validate_graph(graph)
        if validation_errors:
            # Raise the first error found
            raise validation_errors[0]

        # Step 2: Build NetworkX graph
        nx_graph = self._build_nx_graph(graph)

        # Step 3: Forward pass
        es, ef = self._forward_pass(graph, nx_graph)

        # Step 4: Backward pass
        ls, lf = self._backward_pass(graph, nx_graph, ef)

        # Step 5: Float calculation
        activity_analyses = self._calculate_floats(
            graph, es, ef, ls, lf, nx_graph
        )

        # Step 6: Critical path identification
        project_duration = max(ef.values()) if ef else 0.0
        critical_activities = {
            aid for aid, aa in activity_analyses.items() if aa.is_critical
        }
        critical_paths = self._find_all_critical_paths(
            nx_graph, critical_activities, graph, project_duration
        )

        # Use the first critical path as the primary path
        primary_path = critical_paths[0] if critical_paths else []

        result = AnalysisResult(
            project_duration=project_duration,
            critical_path=primary_path,
            critical_paths=critical_paths,
            activity_analyses=activity_analyses,
            analysis_type="CPM",
            is_valid=True,
        )

        logger.info(
            "CPM analysis complete: duration=%.2f, %d critical activities, "
            "%d critical paths",
            project_duration,
            len(critical_activities),
            len(critical_paths),
        )

        return result

    def get_analysis_type(self) -> str:
        """Return the analysis type."""
        return "CPM"

    def can_analyze(self, graph: GraphModel) -> bool:
        """Check if this engine can analyze the given graph."""
        errors = self._validate_graph(graph)
        return len(errors) == 0

    def get_critical_path(self, graph: GraphModel) -> List[str]:
        """Extract the primary critical path from the graph."""
        result = self.analyze(graph)
        return result.critical_path

    def get_all_critical_paths(self, graph: GraphModel) -> List[List[str]]:
        """Extract ALL critical paths from the graph."""
        result = self.analyze(graph)
        return result.critical_paths

    # =========================================================================
    # Graph Validation
    # =========================================================================

    def _validate_graph(self, graph: GraphModel) -> List[Exception]:
        """
        Validate the graph for CPM analysis.

        Returns a list of exceptions found. An empty list means valid.
        """
        errors: List[Exception] = []

        # Check for empty graph
        if not graph.activities:
            errors.append(EmptyGraphError())
            return errors

        # Check for invalid (negative) durations first
        for aid, act in graph.activities.items():
            if act.duration < 0:
                errors.append(InvalidDurationError(aid, act.duration))

        # Check for missing durations (zero or missing for non-dummy activities)
        missing_duration_ids = [
            aid for aid, act in graph.activities.items()
            if act.duration <= 0 and not act.is_dummy
        ]
        if missing_duration_ids:
            errors.append(MissingDurationError(missing_duration_ids))

        # Check for duplicate activity IDs (check the activity_id field, not dict keys)
        seen_ids: Set[str] = set()
        duplicate_ids: List[str] = []
        for aid, act in graph.activities.items():
            if act.activity_id in seen_ids:
                duplicate_ids.append(act.activity_id)
            seen_ids.add(act.activity_id)
        if duplicate_ids:
            errors.append(DuplicateActivityError(duplicate_ids))

        # Check for invalid dependency references
        all_activity_ids = set(graph.activities.keys())
        missing_refs: List[str] = []
        for dep in graph.dependencies:
            if dep.source not in all_activity_ids:
                missing_refs.append(dep.source)
            if dep.target not in all_activity_ids:
                missing_refs.append(dep.target)
        if missing_refs:
            errors.append(MissingActivityError(list(set(missing_refs))))

        # Check for cycles using NetworkX
        if not errors:  # Only check cycles if no other errors
            nx_graph = self._build_nx_graph(graph)
            if not nx.is_directed_acyclic_graph(nx_graph):
                try:
                    cycle = nx.find_cycle(nx_graph)
                    cycle_path = [str(n) for n, _ in cycle]
                    cycle_path.append(cycle_path[0])
                    errors.append(CyclicGraphError(cycle_path))
                except nx.NetworkXNoCycle:
                    # This shouldn't happen if is_directed_acyclic_graph returned False
                    errors.append(CyclicGraphError())

        return errors

    # =========================================================================
    # NetworkX Graph Construction
    # =========================================================================

    def _build_nx_graph(self, graph: GraphModel) -> nx.DiGraph:
        """
        Build a NetworkX directed graph from the GraphModel.

        In the AON representation:
        - Nodes = activities (identified by activity_id)
        - Edges = dependencies (precedence relationships)

        Returns:
            A NetworkX DiGraph.
        """
        nx_graph = nx.DiGraph()

        # Add all activities as nodes
        for aid, activity in graph.activities.items():
            nx_graph.add_node(
                aid,
                duration=activity.duration,
                name=activity.name,
                is_dummy=activity.is_dummy,
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

        return nx_graph

    # =========================================================================
    # Forward Pass
    # =========================================================================

    def _forward_pass(
        self, graph: GraphModel, nx_graph: nx.DiGraph
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform the forward pass to calculate Early Start (ES) and Early Finish (EF).

        Algorithm:
            1. Process activities in topological order
            2. For each activity:
               - ES = max(EF of all predecessors), or 0 if no predecessors
               - EF = ES + duration

        Args:
            graph: The original GraphModel.
            nx_graph: The NetworkX directed graph.

        Returns:
            Tuple of (es_dict, ef_dict) mapping activity_id to values.
        """
        es: Dict[str, float] = {}
        ef: Dict[str, float] = {}

        # Get topological order
        topo_order = list(nx.topological_sort(nx_graph))

        for activity_id in topo_order:
            activity = graph.activities[activity_id]
            predecessors = list(nx_graph.predecessors(activity_id))

            if not predecessors:
                # No predecessors — start at time 0
                es[activity_id] = 0.0
            else:
                # ES = max(EF of all predecessors)
                es[activity_id] = max(ef[pred] for pred in predecessors)

            ef[activity_id] = es[activity_id] + activity.duration

        return es, ef

    # =========================================================================
    # Backward Pass
    # =========================================================================

    def _backward_pass(
        self,
        graph: GraphModel,
        nx_graph: nx.DiGraph,
        ef: Dict[str, float],
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Perform the backward pass to calculate Late Start (LS) and Late Finish (LF).

        Algorithm:
            1. Determine project duration from terminal activities
            2. Process activities in reverse topological order
            3. For each activity:
               - LF = min(LS of all successors), or project_duration if no successors
               - LS = LF - duration

        Args:
            graph: The original GraphModel.
            nx_graph: The NetworkX directed graph.
            ef: Early finish values from forward pass.

        Returns:
            Tuple of (ls_dict, lf_dict) mapping activity_id to values.
        """
        ls: Dict[str, float] = {}
        lf: Dict[str, float] = {}

        # Project duration is the maximum EF
        project_duration = max(ef.values()) if ef else 0.0

        # Get reverse topological order
        topo_order = list(nx.topological_sort(nx_graph))
        reverse_topo = list(reversed(topo_order))

        for activity_id in reverse_topo:
            successors = list(nx_graph.successors(activity_id))

            if not successors:
                # No successors — terminal activity
                lf[activity_id] = project_duration
            else:
                # LF = min(LS of all successors)
                lf[activity_id] = min(ls[succ] for succ in successors)

            ls[activity_id] = lf[activity_id] - graph.activities[activity_id].duration

        return ls, lf

    # =========================================================================
    # Float Calculation
    # =========================================================================

    def _calculate_floats(
        self,
        graph: GraphModel,
        es: Dict[str, float],
        ef: Dict[str, float],
        ls: Dict[str, float],
        lf: Dict[str, float],
        nx_graph: nx.DiGraph,
    ) -> Dict[str, ActivityAnalysis]:
        """
        Calculate total float, free float, and identify critical activities.

        Total Float = LS - ES = LF - EF
        Free Float = min(ES of successors) - EF

        An activity is critical when Total Float == 0 (within tolerance).

        For terminal activities, free float = 0 (they finish at project end).

        Args:
            graph: The original GraphModel.
            es, ef, ls, lf: Calculated timing values.
            nx_graph: The NetworkX directed graph.

        Returns:
            Dict mapping activity_id to ActivityAnalysis.
        """
        activity_analyses: Dict[str, ActivityAnalysis] = {}

        for activity_id, activity in graph.activities.items():
            total_float = ls[activity_id] - es[activity_id]

            successors = list(nx_graph.successors(activity_id))
            if successors:
                free_float = min(es[succ] for succ in successors) - ef[activity_id]
            else:
                # Terminal activity: free float = 0 (no successor to constrain)
                free_float = 0.0

            is_critical = abs(total_float) < self._float_tolerance

            analysis = ActivityAnalysis(
                activity_id=activity_id,
                early_start=es[activity_id],
                early_finish=ef[activity_id],
                late_start=ls[activity_id],
                late_finish=lf[activity_id],
                total_float=total_float,
                free_float=free_float,
                is_critical=is_critical,
            )
            activity_analyses[activity_id] = analysis

        return activity_analyses

    # =========================================================================
    # Critical Path Identification
    # =========================================================================

    def _find_all_critical_paths(
        self,
        nx_graph: nx.DiGraph,
        critical_activities: Set[str],
        graph: GraphModel,
        project_duration: float,
    ) -> List[List[str]]:
        """
        Find ALL critical paths in the network.

        A critical path is a path that spans the full project duration
        (sum of activity durations equals the project duration) made up
        entirely of critical activities (total float = 0).

        Uses DFS to enumerate all valid critical paths.

        Args:
            nx_graph: The NetworkX directed graph.
            critical_activities: Set of critical activity IDs.
            graph: The original GraphModel.
            project_duration: The overall project duration.

        Returns:
            List of critical paths, each path is a list of activity IDs.
        """
        if not critical_activities:
            return []

        # Build a subgraph of only critical activities and edges between them
        critical_subgraph = nx.DiGraph()
        for aid in critical_activities:
            critical_subgraph.add_node(aid)
        for u, v in nx_graph.edges():
            if u in critical_activities and v in critical_activities:
                critical_subgraph.add_edge(u, v)

        # Find source nodes (no predecessors in the subgraph)
        sources = [
            n for n in critical_subgraph.nodes()
            if critical_subgraph.in_degree(n) == 0
        ]

        # Find sink nodes (no successors in the subgraph)
        sinks = [
            n for n in critical_subgraph.nodes()
            if critical_subgraph.out_degree(n) == 0
        ]

        # Find all paths from each source to each sink
        all_paths: List[List[str]] = []
        for source in sources:
            for sink in sinks:
                try:
                    paths = list(
                        nx.all_simple_paths(critical_subgraph, source, sink)
                    )
                    all_paths.extend(paths)
                except nx.NetworkXError:
                    # No path exists between this source-sink pair
                    continue

        # Sort paths for consistent output
        all_paths.sort(key=lambda p: (len(p), p))

        # A critical path must span the whole project: drop zero-float
        # walks that are only sub-paths of a longer critical path.
        all_paths = [
            p
            for p in all_paths
            if abs(
                sum(float(graph.activities[n].duration) for n in p)
                - project_duration
            ) < self._float_tolerance
        ]

        return all_paths

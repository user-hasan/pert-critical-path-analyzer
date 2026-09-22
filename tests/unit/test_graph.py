"""
Tests for graph builder, adapter, and exceptions.
"""

import pytest
import networkx as nx

from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DiagramType,
    GraphModel,
    Node,
)
from pert_analyzer.graph.builder import GraphBuilder, build_aon_graph, build_aoa_graph
from pert_analyzer.graph.adapter import NetworkXAdapter
from pert_analyzer.graph.exceptions import (
    DuplicateActivityError,
    DuplicateDependencyError,
    DuplicateNodeError,
    EmptyGraphError,
    GraphError,
    InvalidDurationError,
    InvalidGraphDataError,
    MissingDependencyRefError,
    SelfDependencyError,
)


class TestGraphBuilder:
    """Tests for GraphBuilder."""

    def test_create_aon_builder(self):
        builder = GraphBuilder(diagram_type=DiagramType.AON)
        assert builder.graph.diagram_type == DiagramType.AON

    def test_create_aoa_builder(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        assert builder.graph.diagram_type == DiagramType.AOA

    def test_add_single_activity(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        assert "A" in builder.graph.activities
        assert builder.graph.activities["A"].duration == 5.0

    def test_add_multiple_activities(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        assert len(builder.graph.activities) == 3

    def test_add_activity_with_name(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0, name="Design")
        assert builder.graph.activities["A"].name == "Design"

    def test_add_dummy_activity(self):
        builder = GraphBuilder()
        builder.add_activity("DUMMY", duration=0.0, is_dummy=True)
        assert builder.graph.activities["DUMMY"].is_dummy
        assert builder.graph.activities["DUMMY"].duration == 0.0

    def test_add_activity_pert_times(self):
        builder = GraphBuilder()
        builder.add_activity(
            "A", duration=5.0, optimistic_time=3.0, most_likely_time=5.0, pessimistic_time=8.0
        )
        act = builder.graph.activities["A"]
        assert act.optimistic_time == 3.0
        assert act.most_likely_time == 5.0
        assert act.pessimistic_time == 8.0

    def test_add_activity_metadata(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0, metadata={"team": "backend"})
        assert builder.graph.activities["A"].metadata["team"] == "backend"

    def test_duplicate_activity_raises_error(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        with pytest.raises(DuplicateActivityError):
            builder.add_activity("A", duration=3.0)

    def test_negative_duration_raises_error(self):
        builder = GraphBuilder()
        with pytest.raises(InvalidDurationError):
            builder.add_activity("A", duration=-1.0)

    def test_zero_duration_non_dummy_raises_error(self):
        builder = GraphBuilder()
        with pytest.raises(InvalidDurationError):
            builder.add_activity("A", duration=0.0, is_dummy=False)

    def test_add_node(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        builder.add_node("1", label="Start")
        assert "1" in builder.graph.nodes
        assert builder.graph.nodes["1"].label == "Start"

    def test_add_nodes(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        builder.add_node("1")
        builder.add_node("2")
        builder.add_node("3")
        assert len(builder.graph.nodes) == 3

    def test_duplicate_node_raises_error(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        builder.add_node("1")
        with pytest.raises(DuplicateNodeError):
            builder.add_node("1")

    def test_add_dependency_aon(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_dependency("A", "B")
        assert len(builder.graph.dependencies) == 1
        assert builder.graph.dependencies[0].source == "A"
        assert builder.graph.dependencies[0].target == "B"

    def test_add_dependency_aoa(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        builder.add_node("1")
        builder.add_node("2")
        builder.add_dependency("1", "2")
        assert len(builder.graph.dependencies) == 1

    def test_self_dependency_raises_error(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        with pytest.raises(SelfDependencyError):
            builder.add_dependency("A", "A")

    def test_self_dependency_allowed(self):
        builder = GraphBuilder()
        builder.allow_self_dependency()
        builder.add_activity("A", duration=5.0)
        builder.add_dependency("A", "A")
        assert len(builder.graph.dependencies) == 1

    def test_duplicate_dependency_raises_error(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_dependency("A", "B")
        with pytest.raises(DuplicateDependencyError):
            builder.add_dependency("A", "B")

    def test_missing_source_dependency_raises_error(self):
        builder = GraphBuilder()
        builder.add_activity("B", duration=3.0)
        with pytest.raises(MissingDependencyRefError):
            builder.add_dependency("A", "B")

    def test_missing_target_dependency_raises_error(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        with pytest.raises(MissingDependencyRefError):
            builder.add_dependency("A", "B")

    def test_build_returns_graph(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        graph = builder.build()
        assert isinstance(graph, GraphModel)
        assert "A" in graph.activities

    def test_build_empty_graph_raises_error(self):
        builder = GraphBuilder()
        with pytest.raises(EmptyGraphError):
            builder.build()

    def test_build_aoa_empty_nodes_raises_error(self):
        builder = GraphBuilder(diagram_type=DiagramType.AOA)
        with pytest.raises(EmptyGraphError):
            builder.build()

    def test_method_chaining(self):
        builder = GraphBuilder()
        result = (
            builder
            .add_activity("A", duration=5.0)
            .add_activity("B", duration=3.0)
            .add_dependency("A", "B")
        )
        assert result is builder
        assert len(builder.graph.activities) == 2

    def test_reset(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.reset()
        assert len(builder.graph.activities) == 0
        assert builder.graph.diagram_type == DiagramType.AON


class TestBuildAONGraph:
    """Tests for build_aon_graph convenience function."""

    def test_simple_graph(self):
        activities = [
            {"activity_id": "A", "duration": 5, "name": "Design"},
            {"activity_id": "B", "duration": 3, "name": "Build"},
        ]
        deps = [("A", "B")]
        graph = build_aon_graph(activities, deps)

        assert graph.diagram_type == DiagramType.AON
        assert len(graph.activities) == 2
        assert len(graph.dependencies) == 1
        assert graph.activities["A"].name == "Design"
        assert graph.activities["B"].name == "Build"

    def test_with_project_name(self):
        activities = [{"activity_id": "A", "duration": 5}]
        graph = build_aon_graph(activities, [], project_name="Test Project")
        assert graph.metadata.get("project_name") == "Test Project"

    def test_with_pert_times(self):
        activities = [
            {
                "activity_id": "A",
                "duration": 5,
                "optimistic_time": 3,
                "most_likely_time": 5,
                "pessimistic_time": 8,
            }
        ]
        graph = build_aon_graph(activities, [])
        act = graph.activities["A"]
        assert act.optimistic_time == 3
        assert act.most_likely_time == 5
        assert act.pessimistic_time == 8

    def test_empty_activities_raises_error(self):
        with pytest.raises(EmptyGraphError):
            build_aon_graph([], [])


class TestBuildAOAGraph:
    """Tests for build_aoa_graph convenience function."""

    def test_simple_graph(self):
        nodes = [
            {"node_id": "1", "label": "Start"},
            {"node_id": "2", "label": "End"},
        ]
        activities = [
            {"activity_id": "A", "duration": 5, "source_node": "1", "target_node": "2"},
        ]
        deps = [("1", "2")]
        graph = build_aoa_graph(nodes, activities, deps)

        assert graph.diagram_type == DiagramType.AOA
        assert len(graph.nodes) == 2
        assert len(graph.activities) == 1
        assert len(graph.dependencies) == 1

    def test_with_project_name(self):
        nodes = [{"node_id": "1"}]
        graph = build_aoa_graph(nodes, [], [], project_name="AOA Project")
        assert graph.metadata.get("project_name") == "AOA Project"


class TestNetworkXAdapter:
    """Tests for NetworkXAdapter."""

    def setup_method(self):
        self.adapter = NetworkXAdapter()

    def test_aon_to_networkx(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        graph.activities["A"] = Activity(activity_id="A", duration=5.0)
        graph.activities["B"] = Activity(activity_id="B", duration=3.0)
        graph.dependencies.append(Dependency(source="A", target="B"))

        nx_graph = self.adapter.to_networkx(graph)

        assert nx_graph.number_of_nodes() == 2
        assert nx_graph.number_of_edges() == 1
        assert nx_graph.nodes["A"]["duration"] == 5.0
        assert nx_graph.has_edge("A", "B")

    def test_aoa_to_networkx(self):
        graph = GraphModel(diagram_type=DiagramType.AOA)
        graph.nodes["1"] = Node(node_id="1", label="Start")
        graph.nodes["2"] = Node(node_id="2", label="End")
        graph.activities["A"] = Activity(
            activity_id="A", duration=5.0, source_node="1", target_node="2"
        )
        graph.dependencies.append(Dependency(source="1", target="2"))

        nx_graph = self.adapter.to_networkx(graph)

        assert nx_graph.number_of_nodes() == 2
        assert nx_graph.number_of_edges() == 1
        assert nx_graph.nodes["1"]["label"] == "Start"
        assert nx_graph.has_edge("1", "2")

    def test_from_networkx_aon(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_node("A", duration=5.0, name="Design")
        nx_graph.add_node("B", duration=3.0, name="Build")
        nx_graph.add_edge("A", "B", dependency_type="finish_to_start")

        graph = self.adapter.from_networkx(nx_graph, DiagramType.AON)

        assert graph.diagram_type == DiagramType.AON
        assert len(graph.activities) == 2
        assert len(graph.dependencies) == 1
        assert graph.activities["A"].name == "Design"
        assert graph.activities["A"].duration == 5.0

    def test_from_networkx_aoa(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_node("1", label="Start")
        nx_graph.add_node("2", label="End")
        nx_graph.add_edge("1", "2", activity_id="A", duration=5.0)

        graph = self.adapter.from_networkx(nx_graph, DiagramType.AOA)

        assert graph.diagram_type == DiagramType.AOA
        assert len(graph.nodes) == 2
        assert len(graph.activities) == 1
        assert graph.nodes["1"].label == "Start"
        assert graph.activities["A"].duration == 5.0

    def test_empty_graph_raises_error(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        with pytest.raises(EmptyGraphError):
            self.adapter.to_networkx(graph)

    def test_get_predecessors(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D")])
        predecessors = self.adapter.get_predecessors(nx_graph, "D")
        assert "B" in predecessors

    def test_get_successors(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("A", "C")])
        successors = self.adapter.get_successors(nx_graph, "A")
        assert "B" in successors
        assert "C" in successors

    def test_get_sources(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("A", "C")])
        sources = self.adapter.get_sources(nx_graph)
        assert sources == ["A"]

    def test_get_sinks(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("A", "C")])
        sinks = self.adapter.get_sinks(nx_graph)
        assert set(sinks) == {"B", "C"}

    def test_is_dag(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("B", "C")])
        assert self.adapter.is_dag(nx_graph)

    def test_is_not_dag(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("B", "A")])
        assert not self.adapter.is_dag(nx_graph)

    def test_topological_sort(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("B", "C"), ("A", "C")])
        topo = self.adapter.topological_sort(nx_graph)
        assert topo.index("A") < topo.index("B")
        assert topo.index("B") < topo.index("C")

    def test_get_all_paths(self):
        nx_graph = nx.DiGraph()
        nx_graph.add_edges_from([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
        paths = self.adapter.get_all_paths(nx_graph, "A", "D")
        assert len(paths) == 2
        assert ["A", "B", "D"] in paths
        assert ["A", "C", "D"] in paths

    def test_roundtrip_aon(self):
        original = GraphModel(diagram_type=DiagramType.AON)
        original.activities["A"] = Activity(activity_id="A", duration=5.0, name="Design")
        original.activities["B"] = Activity(activity_id="B", duration=3.0, name="Build")
        original.dependencies.append(Dependency(source="A", target="B"))

        nx_graph = self.adapter.to_networkx(original)
        roundtripped = self.adapter.from_networkx(nx_graph, DiagramType.AON)

        assert roundtripped.diagram_type == DiagramType.AON
        assert len(roundtripped.activities) == 2
        assert roundtripped.activities["A"].duration == 5.0
        assert roundtripped.activities["B"].name == "Build"

    def test_roundtrip_aoa(self):
        original = GraphModel(diagram_type=DiagramType.AOA)
        original.nodes["1"] = Node(node_id="1", label="Start")
        original.nodes["2"] = Node(node_id="2", label="End")
        original.activities["A"] = Activity(
            activity_id="A", duration=5.0, source_node="1", target_node="2"
        )
        original.dependencies.append(Dependency(source="1", target="2"))

        nx_graph = self.adapter.to_networkx(original)
        roundtripped = self.adapter.from_networkx(nx_graph, DiagramType.AOA)

        assert roundtripped.diagram_type == DiagramType.AOA
        assert len(roundtripped.nodes) == 2
        assert len(roundtripped.activities) == 1
        assert roundtripped.activities["A"].duration == 5.0


class TestGraphExceptions:
    """Tests for graph exception hierarchy."""

    def test_all_exceptions_inherit_from_graph_error(self):
        assert issubclass(DuplicateActivityError, GraphError)
        assert issubclass(DuplicateDependencyError, GraphError)
        assert issubclass(DuplicateNodeError, GraphError)
        assert issubclass(EmptyGraphError, GraphError)
        assert issubclass(InvalidDurationError, GraphError)
        assert issubclass(InvalidGraphDataError, GraphError)
        assert issubclass(MissingDependencyRefError, GraphError)
        assert issubclass(SelfDependencyError, GraphError)

    def test_duplicate_activity_error_message(self):
        error = DuplicateActivityError("A")
        assert "A" in str(error)

    def test_duplicate_dependency_error_message(self):
        error = DuplicateDependencyError("A", "B")
        assert "A" in str(error)
        assert "B" in str(error)

    def test_duplicate_node_error_message(self):
        error = DuplicateNodeError("1")
        assert "1" in str(error)

    def test_self_dependency_error_message(self):
        error = SelfDependencyError("A")
        assert "A" in str(error)

    def test_invalid_duration_error_message(self):
        error = InvalidDurationError("A", -1.0, "negative")
        assert "A" in str(error)

    def test_missing_dependency_ref_error_message(self):
        error = MissingDependencyRefError(["A", "B"], "build")
        assert "A" in str(error)
        assert "B" in str(error)

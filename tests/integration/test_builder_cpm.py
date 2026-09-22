"""
Integration tests for the CPM engine with graph builder.

Tests end-to-end workflows:
- Build graph → CPM analysis
- CSV import → CPM analysis
- JSON save/load → CPM analysis
- Builder + NetworkX adapter + CPM engine
"""

import json
import os
import tempfile

import pytest

from pert_analyzer.analysis.cpm_engine import CPMEngine
from pert_analyzer.core.models import (
    DiagramType,
    GraphModel,
)
from pert_analyzer.graph.builder import GraphBuilder, build_aon_graph
from pert_analyzer.graph.adapter import NetworkXAdapter
from pert_analyzer.persistence.serializer import save_graph, load_graph, graph_to_json
from pert_analyzer.persistence.csv_handler import import_csv_from_string, export_csv_to_string


class TestBuilderToCPM:
    """Integration tests for GraphBuilder → CPMEngine workflow."""

    def test_simple_linear_chain(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        builder.add_dependency("A", "B")
        builder.add_dependency("B", "C")

        graph = builder.build()
        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == 12.0
        assert len(result.critical_path) == 3
        assert result.critical_path == ["A", "B", "C"]

    def test_branching_network(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        builder.add_activity("D", duration=2.0)
        builder.add_dependency("A", "B")
        builder.add_dependency("A", "C")
        builder.add_dependency("B", "D")
        builder.add_dependency("C", "D")

        graph = builder.build()
        engine = CPMEngine()
        result = engine.analyze(graph)

        # A(5)+B(3)+D(2)=10 vs A(5)+C(4)+D(2)=11 → critical path A→C→D=11
        assert result.project_duration == 11.0
        assert "A" in result.critical_path
        assert "C" in result.critical_path
        assert "D" in result.critical_path

    def test_multiple_critical_paths(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=5.0)
        builder.add_activity("C", duration=3.0)
        builder.add_activity("D", duration=3.0)
        builder.add_dependency("A", "C")
        builder.add_dependency("B", "D")

        graph = builder.build()
        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == 8.0
        assert result.critical_paths is not None
        assert len(result.critical_paths) >= 1


class TestCSVImportToCPM:
    """Integration tests for CSV import → CPM analysis."""

    def test_csv_to_cpm(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,3,A
C,Test,4,B
D,Deploy,2,C
"""
        graph = import_csv_from_string(csv_content)
        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == 14.0
        assert result.critical_path == ["A", "B", "C", "D"]

    def test_csv_branching_to_cpm(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build Frontend,3,A
C,Build Backend,4,A
D,Integration,2,B C
E,Deploy,1,D
"""
        graph = import_csv_from_string(csv_content)
        engine = CPMEngine()
        result = engine.analyze(graph)

        # A(5)+C(4)+D(2)+E(1)=12 vs A(5)+B(3)+D(2)+E(1)=11
        assert result.project_duration == 12.0
        assert len(result.critical_path) == 4
        assert "A" in result.critical_path
        assert "E" in result.critical_path

    def test_csv_with_pert_times_to_cpm(self):
        csv_content = """activity_id,name,duration,predecessors,optimistic_time,most_likely_time,pessimistic_time
A,Design,5,,3,5,8
B,Build,3,A,2,3,5
C,Test,4,B,3,4,6
"""
        graph = import_csv_from_string(csv_content)

        assert graph.activities["A"].optimistic_time == 3.0
        assert graph.activities["A"].most_likely_time == 5.0
        assert graph.activities["A"].pessimistic_time == 8.0

        engine = CPMEngine()
        result = engine.analyze(graph)
        assert result.project_duration == 12.0


class TestJSONSerializationToCPM:
    """Integration tests for JSON save/load → CPM analysis."""

    def test_save_load_cpm_roundtrip(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        builder.add_dependency("A", "B")
        builder.add_dependency("B", "C")

        graph = builder.build()

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "project.json")
            save_graph(graph, file_path)
            loaded_graph = load_graph(file_path)

            engine = CPMEngine()
            result = engine.analyze(loaded_graph)

            assert result.project_duration == 12.0
            assert result.critical_path == ["A", "B", "C"]

    def test_json_string_roundtrip(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_dependency("A", "B")

        graph = builder.build()
        json_str = graph_to_json(graph)

        from pert_analyzer.persistence.serializer import graph_from_json
        loaded_graph = graph_from_json(json_str)

        engine = CPMEngine()
        result = engine.analyze(loaded_graph)

        assert result.project_duration == 8.0


class TestNetworkXAdapterCPM:
    """Integration tests for NetworkXAdapter → CPMEngine workflow."""

    def test_adapter_aon_to_cpm(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        builder.add_dependency("A", "B")
        builder.add_dependency("B", "C")

        graph = builder.build()

        adapter = NetworkXAdapter()
        nx_graph = adapter.to_networkx(graph)

        assert nx_graph.number_of_nodes() == 3
        assert nx_graph.number_of_edges() == 2

        # Convert back and analyze
        roundtripped = adapter.from_networkx(nx_graph, DiagramType.AON)
        engine = CPMEngine()
        result = engine.analyze(roundtripped)

        assert result.project_duration == 12.0

    def test_adapter_preserves_graph_structure(self):
        builder = GraphBuilder()
        builder.add_activity("A", duration=5.0)
        builder.add_activity("B", duration=3.0)
        builder.add_activity("C", duration=4.0)
        builder.add_activity("D", duration=2.0)
        builder.add_dependency("A", "B")
        builder.add_dependency("A", "C")
        builder.add_dependency("B", "D")
        builder.add_dependency("C", "D")

        graph = builder.build()

        adapter = NetworkXAdapter()
        nx_graph = adapter.to_networkx(graph)
        roundtripped = adapter.from_networkx(nx_graph, DiagramType.AON)

        engine = CPMEngine()
        original_result = engine.analyze(graph)
        roundtripped_result = engine.analyze(roundtripped)

        assert original_result.project_duration == roundtripped_result.project_duration
        assert original_result.critical_path == roundtripped_result.critical_path


class TestEndToEndWorkflow:
    """Full end-to-end workflow tests."""

    def test_manual_input_to_cpm_report(self):
        """Simulate user manually entering project data → CPM analysis."""
        builder = GraphBuilder()
        builder.add_activity("REQ", duration=2.0, name="Requirements")
        builder.add_activity("DES", duration=3.0, name="Design")
        builder.add_activity("DEV", duration=5.0, name="Development")
        builder.add_activity("TEST", duration=3.0, name="Testing")
        builder.add_activity("DEP", duration=1.0, name="Deployment")
        builder.add_dependency("REQ", "DES")
        builder.add_dependency("DES", "DEV")
        builder.add_dependency("DEV", "TEST")
        builder.add_dependency("TEST", "DEP")

        graph = builder.build()
        graph.metadata["project_name"]="Simple Project"

        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == 14.0
        assert result.critical_path == ["REQ", "DES", "DEV", "TEST", "DEP"]
        assert len(result.critical_paths) == 1

    def test_complex_project_workflow(self):
        """Simulate a more complex project with parallel paths."""
        builder = GraphBuilder()
        builder.add_activity("A", duration=3.0)
        builder.add_activity("B", duration=4.0)
        builder.add_activity("C", duration=2.0)
        builder.add_activity("D", duration=5.0)
        builder.add_activity("E", duration=3.0)
        builder.add_activity("F", duration=2.0)
        builder.add_activity("G", duration=4.0)

        # A → B → D → G (critical path)
        # A → C → E → G
        # B → F → G
        builder.add_dependency("A", "B")
        builder.add_dependency("A", "C")
        builder.add_dependency("B", "D")
        builder.add_dependency("B", "F")
        builder.add_dependency("C", "E")
        builder.add_dependency("D", "G")
        builder.add_dependency("E", "G")
        builder.add_dependency("F", "G")

        graph = builder.build()
        engine = CPMEngine()
        result = engine.analyze(graph)

        assert result.project_duration == 16.0
        assert result.critical_path == ["A", "B", "D", "G"]
        assert result.activity_analyses["C"].total_float > 0
        assert result.activity_analyses["E"].total_float > 0
        assert result.activity_analyses["F"].total_float > 0

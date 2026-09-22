"""
Tests for JSON serialization/deserialization and CSV import/export.
"""

import json
import os
import tempfile

import pytest

from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DiagramType,
    GraphModel,
    Node,
    Project,
)
from pert_analyzer.persistence.serializer import (
    dict_to_graph,
    dict_to_project,
    graph_from_json,
    graph_to_dict,
    graph_to_json,
    load_graph,
    load_project,
    save_graph,
    save_project,
)
from pert_analyzer.persistence.csv_handler import (
    export_csv,
    export_csv_to_file,
    export_csv_to_string,
    import_csv,
    import_csv_from_file,
    import_csv_from_string,
)


class TestJSONSerialization:
    """Tests for JSON serialization/deserialization."""

    def setup_method(self):
        self.graph = GraphModel(diagram_type=DiagramType.AON, confidence=0.85)
        self.graph.activities["A"] = Activity(
            activity_id="A", duration=5.0, name="Design"
        )
        self.graph.activities["B"] = Activity(
            activity_id="B",
            duration=3.0,
            name="Build",
            optimistic_time=2.0,
            most_likely_time=3.0,
            pessimistic_time=5.0,
        )
        self.graph.dependencies.append(Dependency(source="A", target="B"))

    def test_graph_to_dict(self):
        data = graph_to_dict(self.graph)

        assert data["project"]["diagram_type"] == "AON"
        assert data["project"]["confidence"] == 0.85
        assert len(data["activities"]) == 2
        assert len(data["dependencies"]) == 1

    def test_dict_to_graph(self):
        data = graph_to_dict(self.graph)
        graph = dict_to_graph(data)

        assert graph.diagram_type == DiagramType.AON
        assert graph.confidence == 0.85
        assert len(graph.activities) == 2
        assert graph.activities["A"].name == "Design"
        assert graph.activities["B"].optimistic_time == 2.0

    def test_graph_to_json(self):
        json_str = graph_to_json(self.graph)
        data = json.loads(json_str)

        assert data["version"] == "1.0"
        assert data["project"]["diagram_type"] == "AON"

    def test_graph_from_json(self):
        json_str = graph_to_json(self.graph)
        graph = graph_from_json(json_str)

        assert graph.diagram_type == DiagramType.AON
        assert len(graph.activities) == 2

    def test_aoa_roundtrip(self):
        graph = GraphModel(diagram_type=DiagramType.AOA)
        graph.nodes["1"] = Node(node_id="1", label="Start")
        graph.nodes["2"] = Node(node_id="2", label="End")
        graph.activities["A"] = Activity(
            activity_id="A", duration=5.0, source_node="1", target_node="2"
        )
        graph.dependencies.append(Dependency(source="1", target="2"))

        data = graph_to_dict(graph)
        roundtripped = dict_to_graph(data)

        assert roundtripped.diagram_type == DiagramType.AOA
        assert len(roundtripped.nodes) == 2
        assert roundtripped.nodes["1"].label == "Start"

    def test_save_load_graph(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_graph.json")
            save_graph(self.graph, file_path)
            loaded = load_graph(file_path)

            assert loaded.diagram_type == DiagramType.AON
            assert len(loaded.activities) == 2
            assert loaded.activities["A"].name == "Design"

    def test_save_load_project(self):
        project = Project(name="Test Project", description="A test project")
        project.graph = self.graph

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "test_project.json")
            save_project(project, file_path)
            loaded = load_project(file_path)

            assert loaded.name == "Test Project"
            assert loaded.graph is not None
            assert len(loaded.graph.activities) == 2

    def test_metadata_preserved(self):
        self.graph.metadata["author"] = "Test Author"
        data = graph_to_dict(self.graph)
        graph = dict_to_graph(data)

        assert graph.metadata["author"] == "Test Author"

    def test_missing_project_field_raises_error(self):
        with pytest.raises(Exception):
            dict_to_graph({})

    def test_missing_activity_id_raises_error(self):
        data = {
            "project": {"diagram_type": "AON"},
            "activities": [{"name": "Test"}],
        }
        with pytest.raises(Exception):
            dict_to_graph(data)

    def test_missing_node_id_raises_error(self):
        data = {
            "project": {"diagram_type": "AOA"},
            "nodes": [{"label": "Test"}],
        }
        with pytest.raises(Exception):
            dict_to_graph(data)

    def test_invalid_diagram_type_defaults_to_unknown(self):
        data = {
            "project": {"diagram_type": "INVALID"},
            "activities": [],
        }
        graph = dict_to_graph(data)
        assert graph.diagram_type == DiagramType.UNKNOWN

    def test_dependency_with_type(self):
        dep = Dependency(source="A", target="B", dependency_type="start_to_start")
        self.graph.dependencies = [dep]

        data = graph_to_dict(self.graph)
        graph = dict_to_graph(data)

        assert graph.dependencies[0].dependency_type == "start_to_start"


class TestCSVImportExport:
    """Tests for CSV import/export."""

    def test_import_csv_basic(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,3,A
C,Test,4,B
"""
        graph = import_csv_from_string(csv_content)

        assert graph.diagram_type == DiagramType.AON
        assert len(graph.activities) == 3
        assert len(graph.dependencies) == 2
        assert graph.activities["A"].name == "Design"
        assert graph.activities["A"].duration == 5.0

    def test_import_csv_with_pert_times(self):
        csv_content = """activity_id,name,duration,predecessors,optimistic_time,most_likely_time,pessimistic_time
A,Design,5,,3,5,8
B,Build,3,A,2,3,5
"""
        graph = import_csv_from_string(csv_content)

        assert graph.activities["A"].optimistic_time == 3.0
        assert graph.activities["A"].most_likely_time == 5.0
        assert graph.activities["A"].pessimistic_time == 8.0

    def test_import_csv_multiple_predecessors(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,3,A
C,Test,4,A
D,Deploy,2,B C
"""
        graph = import_csv_from_string(csv_content)

        assert len(graph.dependencies) == 4
        # D has two predecessors: B and C
        d_deps = [d for d in graph.dependencies if d.target == "D"]
        assert len(d_deps) == 2

    def test_import_csv_comma_predecessors(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,3,A
C,Test,4,A
D,Deploy,2,"B,C"
"""
        graph = import_csv_from_string(csv_content)

        d_deps = [d for d in graph.dependencies if d.target == "D"]
        assert len(d_deps) == 2

    def test_export_csv_basic(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        graph.activities["A"] = Activity(activity_id="A", duration=5.0, name="Design")
        graph.activities["B"] = Activity(activity_id="B", duration=3.0, name="Build")
        graph.dependencies.append(Dependency(source="A", target="B"))

        csv_str = export_csv_to_string(graph)

        assert "activity_id" in csv_str
        assert "A" in csv_str
        assert "B" in csv_str
        assert "Design" in csv_str

    def test_csv_roundtrip(self):
        csv_content = """activity_id,name,duration,predecessors,optimistic_time,most_likely_time,pessimistic_time
A,Design,5,,3,5,8
B,Build,3,A,2,3,5
C,Test,4,B,3,4,6
"""
        graph = import_csv_from_string(csv_content)
        exported_csv = export_csv_to_string(graph)
        roundtripped = import_csv_from_string(exported_csv)

        assert len(roundtripped.activities) == 3
        assert len(roundtripped.dependencies) == 2
        assert roundtripped.activities["A"].optimistic_time == 3.0

    def test_import_csv_with_file(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,3,A
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name
        try:
            graph = import_csv(temp_path)
            assert len(graph.activities) == 2
        finally:
            os.unlink(temp_path)

    def test_export_csv_with_file(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        graph.activities["A"] = Activity(activity_id="A", duration=5.0, name="Design")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
        try:
            export_csv(graph, temp_path)
            with open(temp_path, "r") as read_f:
                content = read_f.read()
                assert "A" in content
                assert "Design" in content
        finally:
            os.unlink(temp_path)

    def test_import_csv_missing_activity_id_raises_error(self):
        csv_content = """name,duration
Design,5
"""
        with pytest.raises(Exception):
            import_csv_from_string(csv_content)

    def test_import_csv_missing_duration_uses_zero(self):
        csv_content = """activity_id,name,duration,predecessors
A,Design,5,
B,Build,,A
"""
        graph = import_csv_from_string(csv_content)
        assert graph.activities["B"].duration == 0.0
        assert graph.activities["B"].name == "Build"

    def test_export_csv_pert_times(self):
        graph = GraphModel(diagram_type=DiagramType.AON)
        graph.activities["A"] = Activity(
            activity_id="A",
            duration=5.0,
            name="Design",
            optimistic_time=3.0,
            most_likely_time=5.0,
            pessimistic_time=8.0,
        )

        csv_str = export_csv_to_string(graph)
        assert "3.0" in csv_str
        assert "5.0" in csv_str
        assert "8.0" in csv_str

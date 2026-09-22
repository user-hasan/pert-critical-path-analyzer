"""
JSON serialization/deserialization for project data.

Provides save/load functions for persisting GraphModel and Project
data in a stable JSON format.

JSON Schema:
{
    "version": "1.0",
    "project": {
        "name": "...",
        "description": "...",
        "diagram_type": "AON|AOA"
    },
    "activities": [...],
    "nodes": [...],
    "dependencies": [...]
}
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from pert_analyzer.core.models import (
    Activity,
    Dependency,
    DiagramType,
    GraphModel,
    Node,
    Project,
)
from pert_analyzer.graph.exceptions import (
    InvalidGraphDataError,
)

logger = logging.getLogger(__name__)

JSON_VERSION = "1.0"


def save_graph(
    graph: GraphModel,
    file_path: Union[str, os.PathLike],
    indent: int = 2,
) -> None:
    """
    Save a GraphModel to a JSON file.

    Args:
        graph: The GraphModel to save.
        file_path: Path to the output JSON file.
        indent: JSON indentation level.
    """
    data = graph_to_dict(graph)
    data["version"] = JSON_VERSION

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info("Saved graph to %s", file_path)


def load_graph(file_path: Union[str, os.PathLike]) -> GraphModel:
    """
    Load a GraphModel from a JSON file.

    Args:
        file_path: Path to the input JSON file.

    Returns:
        A GraphModel.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidGraphDataError: If the JSON structure is invalid.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return dict_to_graph(data)


def save_project(
    project: Project,
    file_path: Union[str, os.PathLike],
    indent: int = 2,
) -> None:
    """
    Save a Project to a JSON file.

    Args:
        project: The Project to save.
        file_path: Path to the output JSON file.
        indent: JSON indentation level.
    """
    data = project_to_dict(project)
    data["version"] = JSON_VERSION

    os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)

    logger.info("Saved project to %s", file_path)


def load_project(file_path: Union[str, os.PathLike]) -> Project:
    """
    Load a Project from a JSON file.

    Args:
        file_path: Path to the input JSON file.

    Returns:
        A Project.

    Raises:
        FileNotFoundError: If the file doesn't exist.
        InvalidGraphDataError: If the JSON structure is invalid.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return dict_to_project(data)


def graph_to_dict(graph: GraphModel) -> Dict[str, Any]:
    """
    Convert a GraphModel to a dictionary.

    Args:
        graph: The GraphModel to convert.

    Returns:
        A dictionary representation.
    """
    activities = []
    for aid, activity in graph.activities.items():
        act_dict: Dict[str, Any] = {
            "activity_id": activity.activity_id,
            "name": activity.name,
            "duration": activity.duration,
            "is_dummy": activity.is_dummy,
        }
        if activity.source_node:
            act_dict["source_node"] = activity.source_node
        if activity.target_node:
            act_dict["target_node"] = activity.target_node
        if activity.optimistic_time is not None:
            act_dict["optimistic_time"] = activity.optimistic_time
        if activity.most_likely_time is not None:
            act_dict["most_likely_time"] = activity.most_likely_time
        if activity.pessimistic_time is not None:
            act_dict["pessimistic_time"] = activity.pessimistic_time
        if activity.metadata:
            act_dict["metadata"] = activity.metadata
        activities.append(act_dict)

    nodes = []
    for nid, node in graph.nodes.items():
        node_dict: Dict[str, Any] = {
            "node_id": node.node_id,
            "label": node.label,
        }
        if node.node_type != "unknown":
            node_dict["node_type"] = node.node_type
        if node.metadata:
            node_dict["metadata"] = node.metadata
        nodes.append(node_dict)

    dependencies = []
    for dep in graph.dependencies:
        dep_dict: Dict[str, Any] = {
            "source": dep.source,
            "target": dep.target,
        }
        if dep.dependency_type != "finish_to_start":
            dep_dict["dependency_type"] = dep.dependency_type
        if dep.activity_id:
            dep_dict["activity_id"] = dep.activity_id
        if dep.metadata:
            dep_dict["metadata"] = dep.metadata
        dependencies.append(dep_dict)

    result: Dict[str, Any] = {
        "project": {
            "diagram_type": graph.diagram_type.value,
            "confidence": graph.confidence,
        },
        "activities": activities,
        "nodes": nodes,
        "dependencies": dependencies,
    }

    if graph.metadata:
        result["project"]["metadata"] = graph.metadata

    return result


def dict_to_graph(data: Dict[str, Any]) -> GraphModel:
    """
    Convert a dictionary to a GraphModel.

    Args:
        data: Dictionary representation of a graph.

    Returns:
        A GraphModel.

    Raises:
        InvalidGraphDataError: If the data is invalid.
    """
    if "project" not in data:
        raise InvalidGraphDataError("Missing 'project' field in JSON data")

    project_info = data["project"]
    diagram_type_str = project_info.get("diagram_type", "UNKNOWN")
    try:
        diagram_type = DiagramType(diagram_type_str)
    except ValueError:
        diagram_type = DiagramType.UNKNOWN

    graph = GraphModel(
        diagram_type=diagram_type,
        confidence=project_info.get("confidence", 0.0),
        metadata=project_info.get("metadata", {}),
    )

    # Load activities
    for act_data in data.get("activities", []):
        if "activity_id" not in act_data:
            raise InvalidGraphDataError(
                "Activity missing 'activity_id'", "activities"
            )
        activity = Activity(
            activity_id=act_data["activity_id"],
            name=act_data.get("name", ""),
            duration=act_data.get("duration", 0.0),
            is_dummy=act_data.get("is_dummy", False),
            source_node=act_data.get("source_node"),
            target_node=act_data.get("target_node"),
            optimistic_time=act_data.get("optimistic_time"),
            most_likely_time=act_data.get("most_likely_time"),
            pessimistic_time=act_data.get("pessimistic_time"),
            metadata=act_data.get("metadata", {}),
        )
        graph.activities[activity.activity_id] = activity

    # Load nodes
    for node_data in data.get("nodes", []):
        if "node_id" not in node_data:
            raise InvalidGraphDataError("Node missing 'node_id'", "nodes")
        node = Node(
            node_id=node_data["node_id"],
            label=node_data.get("label", ""),
            node_type=node_data.get("node_type", "unknown"),
            metadata=node_data.get("metadata", {}),
        )
        graph.nodes[node.node_id] = node

    # Load dependencies
    for dep_data in data.get("dependencies", []):
        if "source" not in dep_data or "target" not in dep_data:
            raise InvalidGraphDataError(
                "Dependency missing 'source' or 'target'", "dependencies"
            )
        dependency = Dependency(
            source=dep_data["source"],
            target=dep_data["target"],
            dependency_type=dep_data.get("dependency_type", "finish_to_start"),
            activity_id=dep_data.get("activity_id"),
            metadata=dep_data.get("metadata", {}),
        )
        graph.dependencies.append(dependency)

    logger.info(
        "Loaded graph: %s, %d activities, %d nodes, %d dependencies",
        diagram_type.value,
        len(graph.activities),
        len(graph.nodes),
        len(graph.dependencies),
    )

    return graph


def project_to_dict(project: Project) -> Dict[str, Any]:
    """
    Convert a Project to a dictionary.

    Args:
        project: The Project to convert.

    Returns:
        A dictionary representation.
    """
    result: Dict[str, Any] = {
        "project_info": {
            "project_id": project.project_id,
            "name": project.name,
            "description": project.description,
            "created_at": project.created_at,
            "modified_at": project.modified_at,
            "status": project.status,
        },
    }

    if project.graph:
        graph_data = graph_to_dict(project.graph)
        result["graph"] = graph_data

    if project.metadata:
        result["project_info"]["metadata"] = project.metadata

    return result


def dict_to_project(data: Dict[str, Any]) -> Project:
    """
    Convert a dictionary to a Project.

    Args:
        data: Dictionary representation of a project.

    Returns:
        A Project.

    Raises:
        InvalidGraphDataError: If the data is invalid.
    """
    if "project_info" not in data:
        raise InvalidGraphDataError("Missing 'project_info' field in JSON data")

    info = data["project_info"]
    project = Project(
        project_id=info.get("project_id", ""),
        name=info.get("name", "Untitled Project"),
        description=info.get("description", ""),
        created_at=info.get("created_at", ""),
        modified_at=info.get("modified_at", ""),
        status=info.get("status", "draft"),
        metadata=info.get("metadata", {}),
    )

    if "graph" in data:
        project.graph = dict_to_graph(data["graph"])

    return project


def graph_to_json(graph: GraphModel) -> str:
    """
    Convert a GraphModel to a JSON string.

    Args:
        graph: The GraphModel to convert.

    Returns:
        A JSON string.
    """
    data = graph_to_dict(graph)
    data["version"] = JSON_VERSION
    return json.dumps(data, indent=2, ensure_ascii=False)


def graph_from_json(json_str: str) -> GraphModel:
    """
    Parse a JSON string into a GraphModel.

    Args:
        json_str: JSON string representation.

    Returns:
        A GraphModel.
    """
    data = json.loads(json_str)
    return dict_to_graph(data)

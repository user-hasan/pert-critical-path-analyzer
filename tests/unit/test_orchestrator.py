"""
Tests for the application orchestrator.
"""

import pytest

from pert_analyzer.application import AnalysisOrchestrator
from pert_analyzer.core.models import Project


class TestAnalysisOrchestrator:
    """Tests for AnalysisOrchestrator."""

    def test_creation(self):
        orchestrator = AnalysisOrchestrator()
        assert not orchestrator.is_project_loaded

    def test_create_project(self):
        orchestrator = AnalysisOrchestrator()
        project = orchestrator.create_project(name="Test Project")
        assert project.name == "Test Project"
        assert orchestrator.is_project_loaded

    def test_load_project(self):
        orchestrator = AnalysisOrchestrator()
        project = Project(name="Loaded Project")
        orchestrator.load_project(project)
        assert orchestrator.project.name == "Loaded Project"

    def test_get_project_summary_no_project(self):
        orchestrator = AnalysisOrchestrator()
        summary = orchestrator.get_project_summary()
        assert summary["status"] == "no_project"

    def test_get_project_summary_with_project(self):
        orchestrator = AnalysisOrchestrator()
        orchestrator.create_project(name="Summary Test")
        summary = orchestrator.get_project_summary()
        assert summary["name"] == "Summary Test"
        assert summary["status"] == "draft"

    def test_reset(self):
        orchestrator = AnalysisOrchestrator()
        orchestrator.create_project(name="To Reset")
        orchestrator.reset()
        assert not orchestrator.is_project_loaded

    def test_callback_registration(self):
        orchestrator = AnalysisOrchestrator()
        events = []
        orchestrator.register_callback("test_event", lambda d: events.append(d))
        orchestrator._emit("test_event", "data")
        assert events == ["data"]

"""Tests for Gorgon Dashboard.

These tests mock Streamlit components to verify dashboard logic
without requiring a running Streamlit server.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


# Helper functions for creating mock context managers
def _create_context_manager():
    """Create a mock context manager."""
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cm)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


def _create_columns(n):
    """Create mock columns for st.columns."""
    count = n if isinstance(n, int) else len(n)
    return [_create_context_manager() for _ in range(count)]


def _create_tabs(labels):
    """Create mock tabs for st.tabs."""
    return [_create_context_manager() for _ in labels]


def _create_expander(label, **kwargs):
    """Create mock expander for st.expander."""
    return _create_context_manager()


# Modules to clear between tests to ensure fresh imports with mocks
_DASHBOARD_MODULES = [
    "animus_forge.dashboard",
    "animus_forge.dashboard.app",
    "animus_forge.dashboard.monitoring_pages",
    "animus_forge.dashboard.cost_dashboard",
    "animus_forge.dashboard.workflow_visualizer",
    "animus_forge.dashboard.workflow_builder",
]


# Create mock streamlit module before importing dashboard
@pytest.fixture(autouse=True)
def mock_streamlit():
    """Mock streamlit module for all tests."""
    # Clear any cached dashboard modules to ensure fresh imports with mock
    for mod_name in list(sys.modules.keys()):
        if mod_name in _DASHBOARD_MODULES or mod_name.startswith("animus_forge.dashboard."):
            del sys.modules[mod_name]

    mock_st = MagicMock()

    # Create a dict-like object that also supports attribute access
    class SessionState(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError:
                raise AttributeError(name)

        def __setattr__(self, name, value):
            self[name] = value

        def __delattr__(self, name):
            try:
                del self[name]
            except KeyError:
                raise AttributeError(name)

    mock_st.session_state = SessionState()
    mock_st.cache_resource = lambda f: f
    mock_st.columns.side_effect = _create_columns
    mock_st.tabs.side_effect = _create_tabs
    mock_st.expander.side_effect = _create_expander

    # Mock all streamlit submodules that might be imported internally
    mock_submodules = {
        "streamlit": mock_st,
        "streamlit.emojis": MagicMock(),
        "streamlit.components": MagicMock(),
        "streamlit.components.v1": MagicMock(),
        "streamlit.runtime": MagicMock(),
        "streamlit.runtime.scriptrunner": MagicMock(),
        "streamlit.web": MagicMock(),
        "streamlit.delta_generator": MagicMock(),
    }

    with patch.dict(sys.modules, mock_submodules):
        yield mock_st


class TestDashboardHelpers:
    """Test dashboard helper functions."""

    def test_get_workflow_engine(self, mock_streamlit):
        """get_workflow_engine returns WorkflowEngine instance."""
        with patch("animus_forge.orchestrator.WorkflowEngineAdapter") as mock_engine_class:
            mock_engine = MagicMock()
            mock_engine_class.return_value = mock_engine

            from animus_forge.dashboard.app import get_workflow_engine

            result = get_workflow_engine()

            assert result is not None
            mock_engine_class.assert_called_once()

    def test_get_prompt_manager(self, mock_streamlit):
        """get_prompt_manager returns PromptTemplateManager instance."""
        with patch("animus_forge.prompts.PromptTemplateManager") as mock_manager_class:
            mock_manager = MagicMock()
            mock_manager_class.return_value = mock_manager

            from animus_forge.dashboard.app import get_prompt_manager

            result = get_prompt_manager()

            assert result is not None
            mock_manager_class.assert_called_once()

    def test_get_openai_client(self, mock_streamlit):
        """get_openai_client returns OpenAIClient instance."""
        with patch("animus_forge.api_clients.OpenAIClient") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            from animus_forge.dashboard.app import get_openai_client

            result = get_openai_client()

            assert result is not None
            mock_client_class.assert_called_once()


class TestRenderSidebar:
    """Test sidebar rendering."""

    def test_render_sidebar_shows_title(self, mock_streamlit):
        """Sidebar displays Gorgon title."""
        with patch("animus_forge.dashboard.monitoring_pages.render_system_status"):
            from animus_forge.dashboard.app import render_sidebar

            render_sidebar()

            mock_streamlit.sidebar.title.assert_called_once()
            title_call = mock_streamlit.sidebar.title.call_args[0][0]
            assert "Gorgon" in title_call

    def test_render_sidebar_shows_navigation(self, mock_streamlit):
        """Sidebar displays navigation radio buttons."""
        mock_streamlit.sidebar.radio.return_value = "Dashboard"

        with patch("animus_forge.dashboard.monitoring_pages.render_system_status"):
            from animus_forge.dashboard.app import render_sidebar

            page = render_sidebar()

            mock_streamlit.sidebar.radio.assert_called_once()
            assert page == "Dashboard"

    def test_render_sidebar_pages(self, mock_streamlit):
        """Sidebar includes all expected pages."""
        mock_streamlit.sidebar.radio.return_value = "Workflows"

        with patch("animus_forge.dashboard.monitoring_pages.render_system_status"):
            from animus_forge.dashboard.app import render_sidebar

            render_sidebar()

            # Check that all pages are present
            call_args = mock_streamlit.sidebar.radio.call_args
            pages = call_args[0][1]  # Second positional arg is the options list

            expected_pages = [
                "Dashboard",
                "Monitoring",
                "Agents",
                "Metrics",
                "Workflows",
                "Prompts",
                "Execute",
                "Logs",
            ]
            for page in expected_pages:
                assert page in pages


class TestDashboardPage:
    """Test main dashboard page rendering."""

    def test_render_dashboard_shows_title(self, mock_streamlit):
        """Dashboard page shows title."""
        with (
            patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get_engine,
            patch("animus_forge.dashboard.app.get_prompt_manager") as mock_get_prompts,
        ):
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = []
            mock_get_engine.return_value = mock_engine

            mock_prompts = MagicMock()
            mock_prompts.list_templates.return_value = []
            mock_get_prompts.return_value = mock_prompts

            from animus_forge.dashboard.app import render_dashboard_page

            render_dashboard_page()

            mock_streamlit.title.assert_called_once()
            assert "Dashboard" in mock_streamlit.title.call_args[0][0]

    def test_render_dashboard_shows_metrics(self, mock_streamlit):
        """Dashboard page displays workflow and prompt metrics."""
        with (
            patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get_engine,
            patch("animus_forge.dashboard.app.get_prompt_manager") as mock_get_prompts,
        ):
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = [{"id": "wf1"}, {"id": "wf2"}]
            mock_get_engine.return_value = mock_engine

            mock_prompts = MagicMock()
            mock_prompts.list_templates.return_value = [{"id": "p1"}]
            mock_get_prompts.return_value = mock_prompts

            from animus_forge.dashboard.app import render_dashboard_page

            render_dashboard_page()

            # Should call st.metric for counts
            metric_calls = mock_streamlit.metric.call_args_list
            assert len(metric_calls) >= 2

    def test_render_dashboard_quick_actions(self, mock_streamlit):
        """Dashboard shows quick action buttons."""
        mock_streamlit.button.return_value = False

        with (
            patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get_engine,
            patch("animus_forge.dashboard.app.get_prompt_manager") as mock_get_prompts,
        ):
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = []
            mock_get_engine.return_value = mock_engine

            mock_prompts = MagicMock()
            mock_prompts.list_templates.return_value = []
            mock_get_prompts.return_value = mock_prompts

            from animus_forge.dashboard.app import render_dashboard_page

            render_dashboard_page()

            # Should have at least 2 button calls (Create Workflow, Create Prompt)
            assert mock_streamlit.button.call_count >= 2


class TestWorkflowsPage:
    """Test workflows management page."""

    def test_render_workflows_shows_list(self, mock_streamlit):
        """Workflows page shows list of workflows."""
        mock_streamlit.button.return_value = False
        # Setup text_area to return valid JSON
        mock_streamlit.text_area.return_value = "{}"
        mock_streamlit.number_input.return_value = 1
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.selectbox.return_value = "openai"

        with patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = [
                {"id": "wf1", "name": "Workflow 1", "description": "Test"}
            ]
            mock_get_engine.return_value = mock_engine

            from animus_forge.dashboard.app import render_workflows_page

            render_workflows_page()

            # Should display the workflow
            mock_streamlit.expander.assert_called()

    def test_render_workflows_empty_state(self, mock_streamlit):
        """Workflows page shows empty state message."""
        # Setup inputs with valid values
        mock_streamlit.text_area.return_value = "{}"
        mock_streamlit.number_input.return_value = 1
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.selectbox.return_value = "openai"

        with patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get_engine:
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = []
            mock_get_engine.return_value = mock_engine

            from animus_forge.dashboard.app import render_workflows_page

            render_workflows_page()

            # Should show info message about creating first workflow
            mock_streamlit.info.assert_called()


class TestMonitoringPages:
    """Test monitoring dashboard pages."""

    def test_get_tracker_lazy_loads(self, mock_streamlit):
        """get_tracker lazily imports the tracker module."""
        with patch("animus_forge.monitoring.get_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_get.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import get_tracker

            result = get_tracker()

            assert result == mock_tracker
            mock_get.assert_called_once()

    def test_get_agent_tracker_creates_once(self, mock_streamlit):
        """get_agent_tracker creates tracker in session state."""
        # Use the proper SessionState object from fixture
        mock_streamlit.session_state.clear()

        with patch("animus_forge.monitoring.tracker.AgentTracker") as mock_tracker_class:
            mock_tracker = MagicMock()
            mock_tracker_class.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import get_agent_tracker

            result1 = get_agent_tracker()
            result2 = get_agent_tracker()

            # Should create only once, return cached instance
            mock_tracker_class.assert_called_once()
            assert result1 == result2

    def test_render_monitoring_handles_tracker_error(self, mock_streamlit):
        """Monitoring page handles tracker errors gracefully."""
        mock_streamlit.toggle.return_value = False  # Disable auto-refresh

        with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_get:
            mock_get.side_effect = Exception("Tracker unavailable")

            from animus_forge.dashboard.monitoring_pages import render_monitoring_page

            # Should not raise, should show warning
            render_monitoring_page()

            mock_streamlit.warning.assert_called()

    def test_render_monitoring_shows_metrics(self, mock_streamlit):
        """Monitoring page displays summary metrics."""
        mock_streamlit.toggle.return_value = False

        with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_data.return_value = {
                "summary": {
                    "active_workflows": 5,
                    "total_executions": 100,
                    "success_rate": 95.0,
                    "avg_duration_ms": 500,
                },
                "active_workflows": [],
                "recent_executions": [],
            }
            mock_get.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import render_monitoring_page

            render_monitoring_page()

            # Should display metrics
            assert mock_streamlit.metric.call_count >= 4

    def test_render_agents_page(self, mock_streamlit):
        """Agents page renders agent status."""
        mock_streamlit.toggle.return_value = False

        with patch("animus_forge.dashboard.monitoring_pages.get_agent_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_all_status.return_value = {}
            mock_get.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import render_agents_page

            render_agents_page()

            # Should show title
            mock_streamlit.title.assert_called()

    def test_render_metrics_page(self, mock_streamlit):
        """Metrics page renders charts."""
        with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_data.return_value = {
                "summary": {},
                "recent_executions": [],
            }
            mock_get.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import render_metrics_page

            render_metrics_page()

            mock_streamlit.title.assert_called()

    def test_render_system_status(self, mock_streamlit):
        """System status renders in sidebar."""
        with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_get:
            mock_tracker = MagicMock()
            mock_tracker.get_dashboard_data.return_value = {
                "summary": {
                    "active_workflows": 2,
                    "success_rate": 90.0,
                },
            }
            mock_get.return_value = mock_tracker

            from animus_forge.dashboard.monitoring_pages import render_system_status

            render_system_status()

            # Should write to sidebar
            mock_streamlit.sidebar.markdown.assert_called()


class TestPromptsPage:
    """Test prompts management page."""

    def test_render_prompts_shows_list(self, mock_streamlit):
        """Prompts page shows list of templates."""
        mock_streamlit.button.return_value = False
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.text_area.return_value = ""
        mock_streamlit.selectbox.return_value = "planner"

        with patch("animus_forge.dashboard.app.get_prompt_manager") as mock_get:
            mock_manager = MagicMock()
            # Use 'id' and 'description' as expected by the dashboard code
            mock_manager.list_templates.return_value = [
                {
                    "id": "t1",
                    "name": "Test",
                    "role": "planner",
                    "description": "Test desc",
                }
            ]
            mock_manager.get_template.return_value = MagicMock(
                template_id="t1",
                name="Test",
                role="planner",
                template="Test template",
                description="Test desc",
            )
            mock_get.return_value = mock_manager

            from animus_forge.dashboard.app import render_prompts_page

            render_prompts_page()

            mock_streamlit.expander.assert_called()

    def test_render_prompts_empty_state(self, mock_streamlit):
        """Prompts page shows empty state."""
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.text_area.return_value = ""
        mock_streamlit.selectbox.return_value = "planner"

        with patch("animus_forge.dashboard.app.get_prompt_manager") as mock_get:
            mock_manager = MagicMock()
            mock_manager.list_templates.return_value = []
            mock_get.return_value = mock_manager

            from animus_forge.dashboard.app import render_prompts_page

            render_prompts_page()

            mock_streamlit.info.assert_called()


class TestExecutePage:
    """Test workflow execution page."""

    def test_render_execute_no_workflow(self, mock_streamlit):
        """Execute page shows selection when no workflow selected."""
        mock_streamlit.session_state.clear()
        mock_streamlit.selectbox.return_value = None

        with patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = []
            mock_get.return_value = mock_engine

            from animus_forge.dashboard.app import render_execute_page

            render_execute_page()

            mock_streamlit.title.assert_called()

    def test_render_execute_with_workflow(self, mock_streamlit):
        """Execute page shows workflow details when selected."""
        mock_streamlit.session_state.clear()
        mock_streamlit.session_state["execute_workflow_id"] = "wf1"
        mock_streamlit.button.return_value = False
        mock_streamlit.text_input.return_value = ""
        # Return the actual workflow id from selectbox
        mock_streamlit.selectbox.return_value = "wf1"

        with patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.list_workflows.return_value = [{"id": "wf1", "name": "Test"}]
            mock_workflow = MagicMock()
            mock_workflow.id = "wf1"
            mock_workflow.name = "Test Workflow"
            mock_workflow.description = "Test"
            mock_workflow.steps = []
            mock_workflow.variables = {}
            mock_engine.load_workflow.return_value = mock_workflow
            mock_get.return_value = mock_engine

            from animus_forge.dashboard.app import render_execute_page

            render_execute_page()

            # load_workflow is called with selectbox value
            mock_engine.load_workflow.assert_called_with("wf1")


class TestLogsPage:
    """Test logs page."""

    def test_render_logs_empty(self, mock_streamlit):
        """Logs page shows empty state."""
        with patch("animus_forge.dashboard.app.get_workflow_engine") as mock_get:
            mock_engine = MagicMock()
            mock_engine.settings.logs_dir.glob.return_value = []
            mock_get.return_value = mock_engine

            from animus_forge.dashboard.app import render_logs_page

            render_logs_page()

            mock_streamlit.info.assert_called()


class TestMainApp:
    """Test main app routing."""

    def test_main_routes_to_dashboard(self, mock_streamlit):
        """Main app routes Dashboard page correctly."""
        mock_streamlit.session_state.clear()
        mock_render = MagicMock()
        with (
            patch("animus_forge.dashboard.app.render_sidebar") as mock_sidebar,
            patch.dict("animus_forge.dashboard.app._PAGE_RENDERERS", {"Dashboard": mock_render}),
        ):
            mock_sidebar.return_value = "Dashboard"

            from animus_forge.dashboard.app import main

            main()

            mock_render.assert_called_once()

    def test_main_routes_to_workflows(self, mock_streamlit):
        """Main app routes Workflows page correctly."""
        mock_streamlit.session_state.clear()
        mock_render = MagicMock()
        with (
            patch("animus_forge.dashboard.app.render_sidebar") as mock_sidebar,
            patch.dict("animus_forge.dashboard.app._PAGE_RENDERERS", {"Workflows": mock_render}),
        ):
            mock_sidebar.return_value = "Workflows"

            from animus_forge.dashboard.app import main

            main()

            mock_render.assert_called_once()

    def test_main_routes_to_monitoring(self, mock_streamlit):
        """Main app routes Monitoring page correctly."""
        mock_streamlit.session_state.clear()
        mock_render = MagicMock()
        with (
            patch("animus_forge.dashboard.app.render_sidebar") as mock_sidebar,
            patch.dict("animus_forge.dashboard.app._PAGE_RENDERERS", {"Monitoring": mock_render}),
        ):
            mock_sidebar.return_value = "Monitoring"

            from animus_forge.dashboard.app import main

            main()

            mock_render.assert_called_once()

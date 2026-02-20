"""Tests for cost dashboard page."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest


def _create_context_manager():
    mock_cm = MagicMock()
    mock_cm.__enter__ = MagicMock(return_value=mock_cm)
    mock_cm.__exit__ = MagicMock(return_value=False)
    return mock_cm


def _create_columns(n):
    count = n if isinstance(n, int) else len(n)
    return [_create_context_manager() for _ in range(count)]


def _create_tabs(labels):
    return [_create_context_manager() for _ in labels]


def _create_expander(label, **kwargs):
    return _create_context_manager()


@pytest.fixture(autouse=True)
def mock_streamlit():
    mock_st = MagicMock()

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

    # Remove cached module so re-import picks up mock streamlit
    mod_key = "animus_forge.dashboard.cost_dashboard"
    cached = sys.modules.pop(mod_key, None)

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st

    # Remove again so other tests aren't affected
    sys.modules.pop(mod_key, None)
    if cached is not None:
        sys.modules[mod_key] = cached


def _make_tracker():
    """Create a mock CostTracker with sensible defaults."""
    tracker = MagicMock()
    tracker.get_summary.return_value = {
        "total_cost_usd": 12.50,
        "total_tokens": 500000,
        "total_calls": 200,
        "avg_cost_per_call": 0.0625,
        "by_provider": {
            "openai": {"cost": 10.0, "tokens": 400000, "calls": 150},
            "anthropic": {"cost": 2.50, "tokens": 100000, "calls": 50},
        },
        "budget": {},
    }
    tracker.get_daily_cost.return_value = 1.25
    tracker.get_monthly_cost.return_value = 12.50
    tracker.get_agent_costs.return_value = {
        "planner": {"cost": 5.0, "tokens": 200000, "calls": 80},
        "builder": {"cost": 7.5, "tokens": 300000, "calls": 120},
    }
    tracker.get_model_costs.return_value = {
        "gpt-4": {"cost": 10.0, "tokens": 400000, "calls": 150},
    }
    tracker.entries = []
    tracker.PRICING = {"gpt-4": {"input": 30.0, "output": 60.0}}
    return tracker


class TestRenderCostDashboard:
    def test_renders_title(self, mock_streamlit):
        tracker = _make_tracker()
        with patch("animus_forge.dashboard.cost_dashboard._get_tracker", return_value=tracker):
            from animus_forge.dashboard.cost_dashboard import render_cost_dashboard

            render_cost_dashboard()

            mock_streamlit.title.assert_called_once()
            assert "Cost" in mock_streamlit.title.call_args[0][0]

    def test_creates_four_tabs(self, mock_streamlit):
        tracker = _make_tracker()
        with patch("animus_forge.dashboard.cost_dashboard._get_tracker", return_value=tracker):
            from animus_forge.dashboard.cost_dashboard import render_cost_dashboard

            render_cost_dashboard()

            mock_streamlit.tabs.assert_called_once()
            labels = mock_streamlit.tabs.call_args[0][0]
            assert len(labels) == 4

    def test_handles_missing_cost_tracker_module(self, mock_streamlit):
        with patch("animus_forge.dashboard.cost_dashboard.CostTracker", None):
            from animus_forge.dashboard.cost_dashboard import render_cost_dashboard

            render_cost_dashboard()

            mock_streamlit.error.assert_called()

    def test_renders_top_metrics(self, mock_streamlit):
        tracker = _make_tracker()
        with patch("animus_forge.dashboard.cost_dashboard._get_tracker", return_value=tracker):
            from animus_forge.dashboard.cost_dashboard import render_cost_dashboard

            render_cost_dashboard()

            assert mock_streamlit.metric.call_count >= 4


class TestRenderCostWidget:
    def test_renders_sidebar_content(self, mock_streamlit):
        tracker = _make_tracker()
        with patch("animus_forge.dashboard.cost_dashboard._get_tracker", return_value=tracker):
            from animus_forge.dashboard.cost_dashboard import render_cost_widget

            render_cost_widget()

            mock_streamlit.sidebar.markdown.assert_called()
            mock_streamlit.sidebar.write.assert_called()

    def test_shows_daily_and_monthly(self, mock_streamlit):
        tracker = _make_tracker()
        with patch("animus_forge.dashboard.cost_dashboard._get_tracker", return_value=tracker):
            from animus_forge.dashboard.cost_dashboard import render_cost_widget

            render_cost_widget()

            write_calls = [str(c) for c in mock_streamlit.sidebar.write.call_args_list]
            combined = " ".join(write_calls)
            assert "1.25" in combined
            assert "12.50" in combined

    def test_handles_no_cost_tracker(self, mock_streamlit):
        with patch("animus_forge.dashboard.cost_dashboard.CostTracker", None):
            from animus_forge.dashboard.cost_dashboard import render_cost_widget

            render_cost_widget()

            mock_streamlit.sidebar.markdown.assert_not_called()


class TestOverviewTab:
    def test_provider_breakdown_rendered(self, mock_streamlit):
        tracker = _make_tracker()

        from animus_forge.dashboard.cost_dashboard import _render_overview_tab

        _render_overview_tab(tracker)

        mock_streamlit.subheader.assert_called()
        assert mock_streamlit.progress.call_count >= 1

    def test_daily_trend_rendered(self, mock_streamlit):
        tracker = _make_tracker()
        tracker.get_daily_cost.return_value = 2.0

        from animus_forge.dashboard.cost_dashboard import _render_overview_tab

        _render_overview_tab(tracker)

        # Daily trend creates 7 columns
        calls = mock_streamlit.columns.call_args_list
        assert any(c[0] == (7,) for c in calls)


class TestAgentTab:
    def test_agent_costs_displayed(self, mock_streamlit):
        tracker = _make_tracker()

        from animus_forge.dashboard.cost_dashboard import _render_agent_tab

        _render_agent_tab(tracker)

        mock_streamlit.subheader.assert_called()
        assert mock_streamlit.metric.call_count >= 1

    def test_handles_no_agents(self, mock_streamlit):
        tracker = _make_tracker()
        tracker.get_agent_costs.return_value = {}

        from animus_forge.dashboard.cost_dashboard import _render_agent_tab

        _render_agent_tab(tracker)

        mock_streamlit.info.assert_called()


class TestModelTab:
    def test_model_costs_displayed(self, mock_streamlit):
        tracker = _make_tracker()
        # Patch PRICING on the class
        with patch("animus_forge.dashboard.cost_dashboard.CostTracker") as mock_ct_class:
            mock_ct_class.PRICING = {"gpt-4": {"input": 30.0, "output": 60.0}}

            from animus_forge.dashboard.cost_dashboard import _render_model_tab

            _render_model_tab(tracker)

            mock_streamlit.expander.assert_called()

    def test_handles_no_models(self, mock_streamlit):
        tracker = _make_tracker()
        tracker.get_model_costs.return_value = {}

        from animus_forge.dashboard.cost_dashboard import _render_model_tab

        _render_model_tab(tracker)

        mock_streamlit.info.assert_called()


class TestDetailsTab:
    def test_entries_listed(self, mock_streamlit):
        tracker = _make_tracker()
        entry = MagicMock()
        entry.timestamp = datetime(2025, 1, 15, 10, 30)
        entry.model = "gpt-4"
        entry.cost_usd = 0.005
        entry.provider.value = "openai"
        entry.tokens.input_tokens = 100
        entry.tokens.output_tokens = 50
        entry.tokens.total_tokens = 150
        entry.workflow_id = None
        entry.agent_role = None
        tracker.entries = [entry]

        from animus_forge.dashboard.cost_dashboard import _render_details_tab

        _render_details_tab(tracker)

        mock_streamlit.expander.assert_called()

    def test_export_csv_button(self, mock_streamlit):
        tracker = _make_tracker()
        entry = MagicMock()
        entry.timestamp = datetime(2025, 1, 15, 10, 30)
        entry.model = "gpt-4"
        entry.cost_usd = 0.005
        entry.provider.value = "openai"
        entry.tokens.input_tokens = 100
        entry.tokens.output_tokens = 50
        entry.tokens.total_tokens = 150
        entry.workflow_id = None
        entry.agent_role = None
        tracker.entries = [entry]

        from animus_forge.dashboard.cost_dashboard import _render_details_tab

        _render_details_tab(tracker)

        mock_streamlit.button.assert_called()

    def test_clear_old_entries_button(self, mock_streamlit):
        tracker = _make_tracker()

        from animus_forge.dashboard.cost_dashboard import _render_details_tab

        _render_details_tab(tracker)

        # Should have number_input for days and button for clear
        mock_streamlit.number_input.assert_called()

    def test_empty_entries(self, mock_streamlit):
        tracker = _make_tracker()

        from animus_forge.dashboard.cost_dashboard import _render_details_tab

        _render_details_tab(tracker)

        mock_streamlit.info.assert_called()


class TestGetAgentIcon:
    def test_known_roles(self, mock_streamlit):
        from animus_forge.dashboard.cost_dashboard import _get_agent_icon

        assert _get_agent_icon("planner") == "📋"
        assert _get_agent_icon("builder") == "🔨"
        assert _get_agent_icon("tester") == "🧪"
        assert _get_agent_icon("reviewer") == "👁️"

    def test_unknown_role_returns_default(self, mock_streamlit):
        from animus_forge.dashboard.cost_dashboard import _get_agent_icon

        assert _get_agent_icon("unknown_agent") == "🤖"

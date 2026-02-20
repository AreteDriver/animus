"""Tests for dashboard monitoring pages.

Tests all rendering functions with mocked Streamlit components.
"""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")


class _SessionState(dict):
    """Dict that also supports attribute access, like Streamlit session_state."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


class _ColumnContext:
    """Mock context manager for st.columns items."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __getattr__(self, name):
        return MagicMock()


class _MockSt:
    """Mock Streamlit module that handles columns() correctly."""

    def __init__(self):
        self._mock = MagicMock()
        self.session_state = _SessionState()
        self.sidebar = MagicMock()

    def columns(self, *args, **kwargs):
        # Return the right number of column contexts
        if args and isinstance(args[0], (list, int)):
            n = args[0] if isinstance(args[0], int) else len(args[0])
        else:
            n = 4
        return [_ColumnContext() for _ in range(n)]

    def container(self, *args, **kwargs):
        return _ColumnContext()

    def expander(self, *args, **kwargs):
        return _ColumnContext()

    def toggle(self, *args, **kwargs):
        return False

    def button(self, *args, **kwargs):
        return False

    def selectbox(self, *args, **kwargs):
        return "workflow_metrics"

    def number_input(self, *args, **kwargs):
        return 24

    def __getattr__(self, name):
        return getattr(self._mock, name)


def _make_mock_st():
    return _MockSt()


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_MockSt)
@patch("animus_forge.dashboard.monitoring_pages.get_tracker")
class TestRenderMonitoringPage:
    def test_renders_with_data(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_workflows": 2,
                "total_executions": 50,
                "success_rate": 95.0,
                "avg_duration_ms": 500,
            },
            "active_workflows": [
                {
                    "workflow_name": "build",
                    "execution_id": "abc123",
                    "completed_steps": 3,
                    "total_steps": 5,
                    "status": "running",
                }
            ],
            "recent_executions": [
                {
                    "workflow_name": "test",
                    "execution_id": "def456abcdef1234",
                    "status": "completed",
                    "duration_ms": 1200,
                    "completed_steps": 4,
                    "total_steps": 4,
                    "started_at": "2026-01-27T10:00:00",
                    "completed_at": "2026-01-27T10:01:00",
                    "total_tokens": 500,
                    "error": None,
                    "steps": [
                        {
                            "step_id": "s1",
                            "step_type": "claude_code",
                            "action": "execute",
                            "status": "success",
                            "duration_ms": 300,
                        }
                    ],
                }
            ],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_monitoring_page()

    def test_renders_with_tracker_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker_fn.side_effect = Exception("No tracker")
        render_monitoring_page()

    def test_renders_empty_workflows(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_workflows": 0,
                "total_executions": 0,
                "success_rate": 0,
                "avg_duration_ms": 0,
            },
            "active_workflows": [],
            "recent_executions": [],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_monitoring_page()

    def test_renders_execution_with_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_workflows": 0,
                "total_executions": 1,
                "success_rate": 0,
                "avg_duration_ms": 1500,
            },
            "active_workflows": [],
            "recent_executions": [
                {
                    "workflow_name": "fail",
                    "execution_id": "err123456789abcdef",
                    "status": "failed",
                    "duration_ms": 1500,
                    "completed_steps": 1,
                    "total_steps": 3,
                    "started_at": "2026-01-27",
                    "error": "Step 2 failed",
                    "steps": [],
                }
            ],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_monitoring_page()

    def test_low_success_rate(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_workflows": 0,
                "total_executions": 10,
                "success_rate": 50.0,
                "avg_duration_ms": 2000,
            },
            "active_workflows": [],
            "recent_executions": [],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_monitoring_page()

    def test_duration_in_seconds(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_monitoring_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_workflows": 0,
                "total_executions": 1,
                "success_rate": 100,
                "avg_duration_ms": 5000,
            },
            "active_workflows": [],
            "recent_executions": [],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_monitoring_page()


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
class TestRenderAgentsPage:
    def test_renders_with_active_agents(self, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_agents_page

        mock_tracker = MagicMock()
        mock_tracker.get_agent_summary.return_value = {
            "active_count": 2,
            "recent_count": 5,
            "by_role": {"planner": 1, "builder": 1},
        }
        mock_tracker.get_active_agents.return_value = [
            {
                "role": "planner",
                "agent_id": "agent_abc12345",
                "status": "active",
                "tasks_completed": 3,
            },
            {
                "role": "builder",
                "agent_id": "agent_def67890",
                "status": "active",
                "tasks_completed": 1,
            },
        ]
        mock_tracker.get_agent_history.return_value = [
            {
                "role": "planner",
                "agent_id": "agent_old12345",
                "status": "completed",
                "completed_at": "2026-01-27",
            },
        ]
        mock_st.session_state = _SessionState(agent_tracker=mock_tracker)
        render_agents_page()

    def test_renders_empty_agents(self, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_agents_page

        mock_tracker = MagicMock()
        mock_tracker.get_agent_summary.return_value = {
            "active_count": 0,
            "recent_count": 0,
            "by_role": {},
        }
        mock_tracker.get_active_agents.return_value = []
        mock_tracker.get_agent_history.return_value = []
        mock_st.session_state = _SessionState(agent_tracker=mock_tracker)
        render_agents_page()


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
@patch("animus_forge.dashboard.monitoring_pages.get_tracker")
class TestRenderMetricsPage:
    def test_renders_with_data(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_metrics_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "total_executions": 100,
                "failed_executions": 5,
                "total_steps_executed": 500,
                "total_tokens_used": 10000,
                "success_rate": 95.0,
                "avg_duration_ms": 800,
            },
            "step_performance": {
                "claude_code": {"count": 200, "avg_ms": 500, "failure_rate": 2.0},
            },
            "recent_executions": [
                {"execution_id": "abc12345", "total_tokens": 500, "duration_ms": 800},
            ],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_metrics_page()

    def test_renders_with_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_metrics_page

        mock_tracker_fn.side_effect = Exception("No tracker")
        render_metrics_page()

    def test_renders_empty(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_metrics_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {},
            "step_performance": {},
            "recent_executions": [],
        }
        mock_tracker_fn.return_value = mock_tracker
        render_metrics_page()


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
@patch("animus_forge.dashboard.monitoring_pages.get_tracker")
class TestRenderSystemStatus:
    def test_active(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_system_status

        mock_tracker = MagicMock()
        mock_tracker.store.get_summary.return_value = {
            "active_workflows": 2,
            "success_rate": 90.0,
        }
        mock_tracker_fn.return_value = mock_tracker
        render_system_status()

    def test_idle(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_system_status

        mock_tracker = MagicMock()
        mock_tracker.store.get_summary.return_value = {
            "active_workflows": 0,
            "success_rate": 100,
        }
        mock_tracker_fn.return_value = mock_tracker
        render_system_status()

    def test_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_system_status

        mock_tracker_fn.side_effect = Exception("fail")
        render_system_status()


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
class TestRenderAnalyticsPage:
    def test_renders(self, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_analytics_page

        mock_st.session_state = _SessionState()
        render_analytics_page()

    def test_with_cached_result(self, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_analytics_page

        mock_result = MagicMock()
        mock_result.status = "completed"
        mock_result.stages = []
        mock_result.errors = []
        mock_result.final_output = None
        mock_st.session_state = _SessionState(
            analytics_result=mock_result,
            analytics_pipeline="workflow_metrics",
        )
        render_analytics_page()


class TestRunAnalyticsPipeline:
    def test_unknown_pipeline(self):
        import pytest

        from animus_forge.dashboard.monitoring_pages import _run_analytics_pipeline

        with pytest.raises(ValueError, match="Unknown pipeline"):
            _run_analytics_pipeline("nonexistent")


class TestRenderPipelineResult:
    def test_completed(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_pipeline_result

            stage = MagicMock()
            stage.stage.value = "collect"
            stage.duration_ms = 100.0
            stage.status = "success"
            stage.error = None
            stage.output = MagicMock()

            result = MagicMock()
            result.status = "completed"
            result.stages = [stage]
            result.errors = []
            result.final_output = None

            _render_pipeline_result(result, "test", {"test": {"name": "Test"}})

    def test_with_errors(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_pipeline_result

            stage = MagicMock()
            stage.stage.value = "analyze"
            stage.duration_ms = 50.0
            stage.status = "failed"
            stage.error = "Something broke"
            stage.output = None

            result = MagicMock()
            result.status = "failed"
            result.stages = [stage]
            result.errors = ["Error 1"]
            result.final_output = None

            _render_pipeline_result(result, "unknown", {})

    def test_with_final_output(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_pipeline_result

            result = MagicMock()
            result.status = "completed"
            result.stages = []
            result.errors = []
            result.final_output = {"metrics": {"counters": {"total": 10}}}

            _render_pipeline_result(result, "test", {"test": {"name": "Test"}})


class TestStageRenderers:
    def test_collect_with_source(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            output = SimpleNamespace(
                source="tracker",
                collected_at="2026-01-27",
                data={
                    "metrics": {"counters": {"a": 1, "b": 2}},
                    "summary": {"key": "val"},
                },
            )
            _render_collect_output(output)

    def test_collect_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            _render_collect_output({"raw": "data"})

    def test_collect_no_data(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            _render_collect_output(SimpleNamespace(source="tracker", collected_at="now"))

    def test_analyze_with_findings(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_analyze_output

            output = SimpleNamespace(
                severity="warning",
                findings=[
                    {"severity": "warning", "message": "High latency"},
                    {"severity": "critical", "message": "Errors detected"},
                    {"severity": "info", "message": "OK"},
                ],
            )
            _render_analyze_output(output)

    def test_analyze_no_findings(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_analyze_output

            _render_analyze_output(SimpleNamespace(severity="info", findings=[]))

    def test_analyze_no_severity(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_analyze_output

            # Object without severity attr
            obj = MagicMock(spec=[])
            _render_analyze_output(obj)

    def test_visualize(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_visualize_output

            chart = SimpleNamespace(title="My Chart", chart_type="bar")
            _render_visualize_output(
                SimpleNamespace(charts=[chart], streamlit_code="st.write('hi')")
            )

    def test_visualize_empty(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_visualize_output

            _render_visualize_output(SimpleNamespace(charts=[], streamlit_code=""))

    def test_report(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_report_output

            _render_report_output(SimpleNamespace(report_type="summary", summary="All good"))

    def test_alert_with_alerts(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_alert_output

            _render_alert_output(
                SimpleNamespace(
                    alerts=[
                        {
                            "severity": "warning",
                            "title": "Warn",
                            "message": "Something",
                        },
                        {"severity": "critical", "title": "Crit", "message": "Bad"},
                    ]
                )
            )

    def test_alert_empty(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_alert_output

            _render_alert_output(SimpleNamespace(alerts=[]))

    def test_fallback_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_fallback_output

            _render_fallback_output({"key": "val"})

    def test_fallback_to_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_fallback_output

            obj = MagicMock()
            obj.to_dict.return_value = {"a": 1}
            _render_fallback_output(obj)

    def test_fallback_string(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_fallback_output

            _render_fallback_output("some string")

    def test_stage_dispatch(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_stage_output

            stage = MagicMock()
            stage.stage.value = "report"
            stage.output = SimpleNamespace(report_type="summary", summary="ok")
            _render_stage_output(stage)

    def test_stage_unknown(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_stage_output

            stage = MagicMock()
            stage.stage.value = "unknown_type"
            stage.output = {"raw": "data"}
            _render_stage_output(stage)


class TestRenderFinalOutput:
    def test_with_metrics_object(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_final_output

            _render_final_output(
                SimpleNamespace(
                    metrics={"counters": {"total_requests": 100, "errors": 5}},
                    findings=[{"severity": "warning", "message": "High error rate"}],
                )
            )

    def test_with_data_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_final_output

            _render_final_output(SimpleNamespace(data={"metrics": {"counters": {"x": 1}}}))

    def test_with_plain_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_final_output

            _render_final_output({"metrics": {"counters": {}}, "some": "data"})

    def test_with_string(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_final_output

            _render_final_output("plain string output")


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
@patch("animus_forge.dashboard.monitoring_pages.get_parallel_tracker")
class TestRenderParallelExecutionPage:
    def test_renders_with_data(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_execution_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_executions": 1,
                "total_executions": 10,
                "success_rate": 90.0,
                "active_branches": 3,
                "counters": {"branches_completed": 20, "branches_failed": 2},
                "rate_limit_waits": {"count": 5, "avg": 100.0, "max": 500.0},
                "rate_limit_states": {},
                "execution_duration": {"count": 10, "avg": 500, "p50": 450, "p95": 900},
                "branch_duration": {"count": 20, "avg": 100, "p50": 90, "p95": 200},
                "execution_tokens": {"count": 10, "avg": 500, "min": 100, "max": 1000},
            },
            "active_executions": [
                {
                    "pattern_type": "fan_out",
                    "step_id": "step_1",
                    "execution_id": "exec123456789abc",
                    "total_items": 10,
                    "completed_count": 5,
                    "failed_count": 1,
                    "active_branch_count": 4,
                    "duration_ms": 2000,
                    "total_tokens": 500,
                    "branches": [
                        {
                            "branch_id": "branch_1_long_name",
                            "status": "completed",
                            "item_index": 0,
                            "duration_ms": 100,
                        },
                        {
                            "branch_id": "branch_2_long_name",
                            "status": "running",
                            "item_index": 1,
                            "duration_ms": 50,
                        },
                    ],
                }
            ],
            "recent_executions": [
                {
                    "pattern_type": "map_reduce",
                    "step_id": "step_2",
                    "execution_id": "exec_old_12345678",
                    "status": "completed",
                    "total_items": 5,
                    "completed_count": 5,
                    "failed_count": 0,
                    "duration_ms": 3000,
                    "total_tokens": 1000,
                    "started_at": "2026-01-27",
                    "completed_at": "2026-01-27",
                    "branches": [],
                }
            ],
            "rate_limits": {
                "openai": {
                    "base_limit": 10,
                    "current_limit": 5,
                    "total_429s": 3,
                    "is_throttled": True,
                },
            },
        }
        mock_tracker_fn.return_value = mock_tracker
        render_parallel_execution_page()

    def test_renders_with_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_execution_page

        mock_tracker_fn.side_effect = Exception("No tracker")
        render_parallel_execution_page()

    def test_renders_empty(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_execution_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_executions": 0,
                "total_executions": 0,
                "success_rate": 0,
                "active_branches": 0,
                "counters": {},
            },
            "active_executions": [],
            "recent_executions": [],
            "rate_limits": {},
        }
        mock_tracker_fn.return_value = mock_tracker
        render_parallel_execution_page()

    def test_renders_failed(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_execution_page

        mock_tracker = MagicMock()
        mock_tracker.get_dashboard_data.return_value = {
            "summary": {
                "active_executions": 0,
                "total_executions": 1,
                "success_rate": 0,
                "active_branches": 0,
                "counters": {"branches_completed": 0, "branches_failed": 3},
            },
            "active_executions": [],
            "recent_executions": [
                {
                    "pattern_type": "fan_out",
                    "step_id": "s1",
                    "execution_id": "fail123456789abc",
                    "status": "failed",
                    "total_items": 3,
                    "completed_count": 0,
                    "failed_count": 3,
                    "duration_ms": 100,
                    "total_tokens": 0,
                    "started_at": "2026-01-27",
                    "completed_at": "2026-01-27",
                    "branches": [{"branch_id": "b1", "status": "failed", "error": "timeout"}],
                }
            ],
            "rate_limits": {},
        }
        mock_tracker_fn.return_value = mock_tracker
        render_parallel_execution_page()


class TestRenderRateLimitSection:
    def test_no_limits(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_rate_limit_section

            _render_rate_limit_section({}, {})

    def test_with_limits(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_rate_limit_section

            _render_rate_limit_section(
                {
                    "openai": {
                        "base_limit": 10,
                        "current_limit": 5,
                        "total_429s": 3,
                        "is_throttled": True,
                    },
                    "anthropic": {
                        "base_limit": 10,
                        "current_limit": 10,
                        "total_429s": 0,
                        "is_throttled": False,
                    },
                },
                {
                    "rate_limit_waits": {"count": 5, "avg": 100.0, "max": 500.0},
                    "rate_limit_states": {},
                },
            )

    def test_only_states(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_rate_limit_section

            _render_rate_limit_section(
                {},
                {
                    "rate_limit_waits": {"count": 0},
                    "rate_limit_states": {
                        "openai": {
                            "base_limit": 10,
                            "current_limit": 10,
                            "total_429s": 1,
                        }
                    },
                },
            )

    def test_warning_status(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_rate_limit_section

            _render_rate_limit_section(
                {
                    "provider": {
                        "base_limit": 10,
                        "current_limit": 10,
                        "total_429s": 1,
                        "is_throttled": False,
                    }
                },
                {},
            )


class TestRenderActiveExecution:
    def test_renders(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_active_execution

            _render_active_execution(
                {
                    "pattern_type": "auto_parallel",
                    "step_id": "step_1",
                    "execution_id": "exec1234567890ab",
                    "total_items": 5,
                    "completed_count": 3,
                    "failed_count": 0,
                    "active_branch_count": 2,
                    "duration_ms": 500,
                    "total_tokens": 100,
                    "branches": [],
                }
            )

    def test_unknown_pattern(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_active_execution

            _render_active_execution(
                {
                    "pattern_type": "unknown",
                    "step_id": "s",
                    "execution_id": "x" * 20,
                    "total_items": 0,
                    "completed_count": 0,
                    "failed_count": 0,
                    "active_branch_count": 0,
                    "duration_ms": 0,
                    "total_tokens": 0,
                    "branches": [],
                }
            )


class TestRenderPerformanceStats:
    def test_full_stats(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_performance_stats

            _render_performance_stats(
                {
                    "execution_duration": {
                        "count": 10,
                        "avg": 500,
                        "p50": 450,
                        "p95": 900,
                    },
                    "branch_duration": {"count": 20, "avg": 100, "p50": 90, "p95": 200},
                    "execution_tokens": {
                        "count": 10,
                        "avg": 500,
                        "min": 100,
                        "max": 1000,
                    },
                    "counters": {
                        "executions_started_fan_out": 5,
                        "executions_started_map_reduce": 3,
                        "executions_started": 8,
                    },
                }
            )

    def test_empty_stats(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_performance_stats

            _render_performance_stats(
                {
                    "execution_duration": {"count": 0},
                    "branch_duration": {"count": 0},
                    "execution_tokens": {"count": 0},
                    "counters": {},
                }
            )


@patch("animus_forge.dashboard.monitoring_pages.st", new_callable=_make_mock_st)
@patch("animus_forge.dashboard.monitoring_pages.get_parallel_tracker")
class TestRenderParallelStatusSidebar:
    def test_active(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_status_sidebar

        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {
            "active_executions": 2,
            "active_branches": 5,
            "rate_limit_states": {"openai": {"is_throttled": True}},
        }
        mock_tracker_fn.return_value = mock_tracker
        render_parallel_status_sidebar()

    def test_idle(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_status_sidebar

        mock_tracker = MagicMock()
        mock_tracker.get_summary.return_value = {
            "active_executions": 0,
            "active_branches": 0,
            "rate_limit_states": {},
        }
        mock_tracker_fn.return_value = mock_tracker
        render_parallel_status_sidebar()

    def test_error(self, mock_tracker_fn, mock_st):
        from animus_forge.dashboard.monitoring_pages import render_parallel_status_sidebar

        mock_tracker_fn.side_effect = Exception("No tracker")
        render_parallel_status_sidebar()


# =============================================================================
# Additional coverage tests for previously uncovered lines
# =============================================================================


class TestGetTrackerFunction:
    """Test the lazy-load get_tracker function (lines 17-19)."""

    def test_get_tracker_returns_tracker(self):
        with patch("animus_forge.monitoring.get_tracker", return_value=MagicMock()) as mock_inner:
            from animus_forge.dashboard.monitoring_pages import get_tracker

            result = get_tracker()
            mock_inner.assert_called_once()
            assert result is mock_inner.return_value


class TestGetAgentTrackerFunction:
    """Test get_agent_tracker session_state creation path (line 26)."""

    def test_creates_tracker_when_not_in_session(self):
        mock_st_module = _make_mock_st()
        # Ensure agent_tracker is NOT in session_state
        mock_st_module.session_state = _SessionState()
        assert "agent_tracker" not in mock_st_module.session_state

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch(
                "animus_forge.monitoring.tracker.AgentTracker", return_value=MagicMock()
            ) as mock_cls:
                from animus_forge.dashboard.monitoring_pages import get_agent_tracker

                _result = get_agent_tracker()
                mock_cls.assert_called_once()
                assert "agent_tracker" in mock_st_module.session_state


class TestGetParallelTrackerFunction:
    """Test the lazy-load get_parallel_tracker function (lines 32-36)."""

    def test_get_parallel_tracker_returns_tracker(self):
        with patch(
            "animus_forge.monitoring.parallel_tracker.get_parallel_tracker",
            return_value=MagicMock(),
        ) as mock_inner:
            from animus_forge.dashboard.monitoring_pages import get_parallel_tracker

            result = get_parallel_tracker()
            mock_inner.assert_called_once()
            assert result is mock_inner.return_value


class TestAutoRefreshBranch:
    """Test auto-refresh rerun paths (lines 51-52, 718-719)."""

    def test_monitoring_page_auto_refresh_rerun(self):
        mock_st_module = _make_mock_st()
        # Make toggle return True to trigger auto-refresh
        mock_st_module.toggle = lambda *a, **kw: True
        mock_st_module.rerun = MagicMock()

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_dashboard_data.return_value = {
                    "summary": {
                        "active_workflows": 0,
                        "total_executions": 0,
                        "success_rate": 0,
                        "avg_duration_ms": 0,
                    },
                    "active_workflows": [],
                    "recent_executions": [],
                }
                mock_tracker_fn.return_value = mock_tracker
                with patch("animus_forge.dashboard.monitoring_pages.time"):
                    from animus_forge.dashboard.monitoring_pages import (
                        render_monitoring_page,
                    )

                    render_monitoring_page()
                    mock_st_module.rerun.assert_called()

    def test_parallel_page_auto_refresh_rerun(self):
        mock_st_module = _make_mock_st()
        mock_st_module.toggle = lambda *a, **kw: True
        mock_st_module.rerun = MagicMock()

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch(
                "animus_forge.dashboard.monitoring_pages.get_parallel_tracker"
            ) as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_dashboard_data.return_value = {
                    "summary": {
                        "active_executions": 0,
                        "total_executions": 0,
                        "success_rate": 0,
                        "active_branches": 0,
                        "counters": {},
                    },
                    "active_executions": [],
                    "recent_executions": [],
                    "rate_limits": {},
                }
                mock_tracker_fn.return_value = mock_tracker
                with patch("animus_forge.dashboard.monitoring_pages.time"):
                    from animus_forge.dashboard.monitoring_pages import (
                        render_parallel_execution_page,
                    )

                    render_parallel_execution_page()
                    mock_st_module.rerun.assert_called()


class TestSimulateAgentActivity:
    """Test the simulate agent button path (lines 232-240)."""

    def test_simulate_button_pressed(self):
        mock_st_module = _make_mock_st()
        # Make button return True to trigger simulation
        mock_st_module.button = lambda *a, **kw: True
        mock_st_module.rerun = MagicMock()

        mock_tracker = MagicMock()
        mock_tracker.get_agent_summary.return_value = {
            "active_count": 0,
            "recent_count": 0,
            "by_role": {},
        }
        mock_tracker.get_active_agents.return_value = []
        mock_tracker.get_agent_history.return_value = []
        mock_st_module.session_state = _SessionState(agent_tracker=mock_tracker)

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            from animus_forge.dashboard.monitoring_pages import render_agents_page

            render_agents_page()
            # register_agent should have been called 3 times
            assert mock_tracker.register_agent.call_count == 3
            mock_st_module.rerun.assert_called()


class TestMetricsPageTokenBranches:
    """Test metrics page token/duration branches (lines 367, 384)."""

    def test_token_data_all_zero(self):
        """Test branch where token_data exists but all tokens are 0 (line 367)."""
        mock_st_module = _make_mock_st()
        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_dashboard_data.return_value = {
                    "summary": {
                        "total_executions": 1,
                        "failed_executions": 0,
                        "total_steps_executed": 1,
                        "total_tokens_used": 0,
                        "success_rate": 100,
                        "avg_duration_ms": 100,
                    },
                    "step_performance": {},
                    "recent_executions": [
                        {
                            "execution_id": "abc12345",
                            "total_tokens": 0,
                            "duration_ms": 0,
                        },
                    ],
                }
                mock_tracker_fn.return_value = mock_tracker
                from animus_forge.dashboard.monitoring_pages import render_metrics_page

                render_metrics_page()

    def test_duration_data_missing(self):
        """Test branch where recent execs exist but duration_ms is falsy (line 384)."""
        mock_st_module = _make_mock_st()
        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_dashboard_data.return_value = {
                    "summary": {
                        "total_executions": 1,
                        "failed_executions": 0,
                        "total_steps_executed": 1,
                        "total_tokens_used": 0,
                        "success_rate": 100,
                        "avg_duration_ms": 0,
                    },
                    "step_performance": {},
                    "recent_executions": [
                        {
                            "execution_id": "abc12345",
                            "total_tokens": 100,
                            "duration_ms": 0,
                        },
                    ],
                }
                mock_tracker_fn.return_value = mock_tracker
                from animus_forge.dashboard.monitoring_pages import render_metrics_page

                render_metrics_page()


class TestRunAnalyticsPipelineKnownPipelines:
    """Test _run_analytics_pipeline for each known pipeline (lines 483-493)."""

    def test_workflow_metrics(self):
        with patch(
            "animus_forge.dashboard.monitoring_pages.PipelineBuilder",
            create=True,
        ) as _mock_pb_import:
            # We need to patch the import inside the function
            mock_pipeline = MagicMock()
            mock_pipeline.execute.return_value = "result"
            mock_builder = MagicMock()
            mock_builder.workflow_metrics_pipeline.return_value = mock_pipeline

            with patch(
                "animus_forge.analytics.pipeline.PipelineBuilder",
                mock_builder,
            ):
                from animus_forge.dashboard.monitoring_pages import _run_analytics_pipeline

                result = _run_analytics_pipeline("workflow_metrics")
                mock_builder.workflow_metrics_pipeline.assert_called_once()
                assert result == "result"

    def test_historical_trends(self):
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = "trends"
        mock_builder = MagicMock()
        mock_builder.historical_trends_pipeline.return_value = mock_pipeline

        with patch(
            "animus_forge.analytics.pipeline.PipelineBuilder",
            mock_builder,
        ):
            from animus_forge.dashboard.monitoring_pages import _run_analytics_pipeline

            _result = _run_analytics_pipeline("historical_trends", hours=48)
            mock_builder.historical_trends_pipeline.assert_called_once_with(hours=48)

    def test_api_health(self):
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = "health"
        mock_builder = MagicMock()
        mock_builder.api_health_pipeline.return_value = mock_pipeline

        with patch(
            "animus_forge.analytics.pipeline.PipelineBuilder",
            mock_builder,
        ):
            from animus_forge.dashboard.monitoring_pages import _run_analytics_pipeline

            _result = _run_analytics_pipeline("api_health")
            mock_builder.api_health_pipeline.assert_called_once()

    def test_operations_dashboard(self):
        mock_pipeline = MagicMock()
        mock_pipeline.execute.return_value = "ops"
        mock_builder = MagicMock()
        mock_builder.operations_dashboard_pipeline.return_value = mock_pipeline

        with patch(
            "animus_forge.analytics.pipeline.PipelineBuilder",
            mock_builder,
        ):
            from animus_forge.dashboard.monitoring_pages import _run_analytics_pipeline

            _result = _run_analytics_pipeline("operations_dashboard")
            mock_builder.operations_dashboard_pipeline.assert_called_once()


class TestAnalyticsPageButtonPress:
    """Test analytics page button press and error handling (lines 457-466)."""

    def test_button_press_success(self):
        mock_st_module = _make_mock_st()
        mock_st_module.button = lambda *a, **kw: True
        mock_st_module.spinner = MagicMock(return_value=_ColumnContext())
        mock_st_module.session_state = _SessionState()

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch(
                "animus_forge.dashboard.monitoring_pages._run_analytics_pipeline"
            ) as mock_run:
                mock_result = MagicMock()
                mock_result.status = "completed"
                mock_result.stages = []
                mock_result.errors = []
                mock_result.final_output = None
                mock_run.return_value = mock_result

                from animus_forge.dashboard.monitoring_pages import render_analytics_page

                render_analytics_page()
                assert "analytics_result" in mock_st_module.session_state

    def test_button_press_pipeline_error(self):
        mock_st_module = _make_mock_st()
        mock_st_module.button = lambda *a, **kw: True
        mock_st_module.spinner = MagicMock(return_value=_ColumnContext())
        mock_st_module.session_state = _SessionState()

        with patch("animus_forge.dashboard.monitoring_pages.st", mock_st_module):
            with patch(
                "animus_forge.dashboard.monitoring_pages._run_analytics_pipeline"
            ) as mock_run:
                mock_run.side_effect = RuntimeError("Pipeline crashed")

                from animus_forge.dashboard.monitoring_pages import render_analytics_page

                render_analytics_page()
                mock_st_module._mock.error.assert_called()


class TestRenderFinalOutputToDictBranch:
    """Test _render_final_output with to_dict method (line 688)."""

    def test_output_with_to_dict(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_final_output

            output = MagicMock()
            # Has to_dict but no metrics/data/findings attrs
            output.to_dict.return_value = {"key": "value"}
            del output.metrics
            del output.data
            del output.findings
            _render_final_output(output)


class TestRenderRateLimitSectionEmptyProviders:
    """Test _render_rate_limit_section with non-empty dicts but no providers (line 833)."""

    def test_both_dicts_set_but_empty_providers(self):
        """When rate_limits and rate_limit_states are non-empty dicts at top
        level but combined provider set is empty (degenerate case)."""
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_rate_limit_section

            # Both truthy but rate_limit_states keys used for providers
            _render_rate_limit_section(
                {},
                {
                    "rate_limit_waits": {"count": 0},
                    "rate_limit_states": {},
                },
            )


class TestPerformanceStatsPatternBarChart:
    """Test pattern count bar chart in _render_performance_stats (line 833-834)."""

    def test_pattern_counts_with_pandas(self):
        """Verify pattern count bar chart renders (relies on pandas import)."""
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_performance_stats

            _render_performance_stats(
                {
                    "execution_duration": {"count": 0},
                    "branch_duration": {"count": 0},
                    "execution_tokens": {"count": 0},
                    "counters": {
                        "executions_started_fan_out": 5,
                        "executions_started_map_reduce": 3,
                    },
                }
            )


class TestRecentExecutionWithStepStatuses:
    """Test recent execution rendering with various step statuses."""

    def test_execution_with_failed_step(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            with patch("animus_forge.dashboard.monitoring_pages.get_tracker") as mock_tracker_fn:
                mock_tracker = MagicMock()
                mock_tracker.get_dashboard_data.return_value = {
                    "summary": {
                        "active_workflows": 0,
                        "total_executions": 1,
                        "success_rate": 75.0,
                        "avg_duration_ms": 200,
                    },
                    "active_workflows": [],
                    "recent_executions": [
                        {
                            "workflow_name": "mixed",
                            "execution_id": "mix123456789abcdef",
                            "status": "failed",
                            "duration_ms": 200,
                            "completed_steps": 1,
                            "total_steps": 2,
                            "started_at": "2026-01-27",
                            "completed_at": "2026-01-27",
                            "total_tokens": 0,
                            "error": "Step failed",
                            "steps": [
                                {
                                    "step_id": "s1",
                                    "step_type": "shell",
                                    "action": "run",
                                    "status": "success",
                                    "duration_ms": 100,
                                },
                                {
                                    "step_id": "s2",
                                    "step_type": "openai",
                                    "action": "generate",
                                    "status": "failed",
                                    "duration_ms": 100,
                                },
                            ],
                        }
                    ],
                }
                mock_tracker_fn.return_value = mock_tracker
                from animus_forge.dashboard.monitoring_pages import render_monitoring_page

                render_monitoring_page()


class TestActiveExecutionWithBranches:
    """Test active execution rendering with branch details."""

    def test_with_all_branch_statuses(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_active_execution

            _render_active_execution(
                {
                    "pattern_type": "map_reduce",
                    "step_id": "step_map",
                    "execution_id": "exec_map_123456789",
                    "total_items": 5,
                    "completed_count": 2,
                    "failed_count": 1,
                    "active_branch_count": 2,
                    "duration_ms": 1500,
                    "total_tokens": 250,
                    "branches": [
                        {
                            "branch_id": "b1_pending_branch",
                            "status": "pending",
                            "item_index": 0,
                            "duration_ms": 0,
                        },
                        {
                            "branch_id": "b2_running_branch",
                            "status": "running",
                            "item_index": 1,
                            "duration_ms": 50,
                        },
                        {
                            "branch_id": "b3_completed_branch",
                            "status": "completed",
                            "item_index": 2,
                            "duration_ms": 200,
                        },
                        {
                            "branch_id": "b4_failed_branch",
                            "status": "failed",
                            "item_index": 3,
                            "duration_ms": 100,
                        },
                        {
                            "branch_id": "b5_cancelled_branch",
                            "status": "cancelled",
                            "item_index": 4,
                            "duration_ms": 10,
                        },
                    ],
                }
            )

    def test_with_no_duration_no_tokens(self):
        """Test execution with 0 duration and 0 tokens (no metric rendered)."""
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_active_execution

            _render_active_execution(
                {
                    "pattern_type": "parallel_group",
                    "step_id": "step_pg",
                    "execution_id": "exec_pg_1234567890",
                    "total_items": 2,
                    "completed_count": 0,
                    "failed_count": 0,
                    "active_branch_count": 2,
                    "duration_ms": 0,
                    "total_tokens": 0,
                    "branches": [],
                }
            )


class TestRecentExecutionVariants:
    """Test _render_recent_execution with different patterns and statuses."""

    def test_fan_in_pattern(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_recent_execution

            _render_recent_execution(
                {
                    "pattern_type": "fan_in",
                    "step_id": "merge_step",
                    "execution_id": "exec_fanin_12345678",
                    "status": "completed",
                    "total_items": 3,
                    "completed_count": 3,
                    "failed_count": 0,
                    "duration_ms": 500,
                    "total_tokens": 100,
                    "started_at": "2026-01-27",
                    "completed_at": "2026-01-27",
                    "branches": [],
                }
            )

    def test_failed_with_error_branches(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_recent_execution

            _render_recent_execution(
                {
                    "pattern_type": "auto_parallel",
                    "step_id": "auto_step",
                    "execution_id": "exec_auto_12345678",
                    "status": "failed",
                    "total_items": 3,
                    "completed_count": 1,
                    "failed_count": 2,
                    "duration_ms": 800,
                    "total_tokens": 50,
                    "started_at": "2026-01-27",
                    "completed_at": "2026-01-27",
                    "branches": [
                        {
                            "branch_id": "b1",
                            "status": "failed",
                            "error": "Connection timeout",
                        },
                        {
                            "branch_id": "b2",
                            "status": "failed",
                            "error": "Rate limited",
                        },
                        {
                            "branch_id": "b3",
                            "status": "completed",
                            "error": None,
                        },
                    ],
                }
            )


class TestCollectOutputEdgeCases:
    """Test _render_collect_output edge cases."""

    def test_collect_with_empty_counters(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            output = SimpleNamespace(
                source="tracker",
                collected_at="2026-01-27",
                data={"metrics": {"counters": {}}, "summary": {}},
            )
            _render_collect_output(output)

    def test_collect_with_string_output(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            _render_collect_output("raw string output")

    def test_collect_with_many_counters(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_collect_output

            output = SimpleNamespace(
                source="tracker",
                collected_at="2026-01-27",
                data={
                    "metrics": {"counters": {f"counter_{i}": i * 10 for i in range(10)}},
                },
            )
            _render_collect_output(output)


class TestAnalyzeOutputEdgeCases:
    """Test _render_analyze_output edge cases."""

    def test_findings_with_unknown_severity(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_analyze_output

            output = SimpleNamespace(
                severity="critical",
                findings=[
                    {"severity": "unknown_sev", "message": "Something"},
                    {"message": "No severity key"},
                ],
            )
            _render_analyze_output(output)


class TestVisualizeOutputEdgeCases:
    """Test _render_visualize_output when no charts or no streamlit_code."""

    def test_no_charts_attr(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_visualize_output

            # Object without charts attr
            obj = MagicMock(spec=[])
            _render_visualize_output(obj)

    def test_no_streamlit_code(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_visualize_output

            output = SimpleNamespace(charts=[], streamlit_code=None)
            _render_visualize_output(output)


class TestReportOutputEdgeCases:
    """Test _render_report_output edge cases."""

    def test_no_report_type(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_report_output

            obj = MagicMock(spec=[])
            _render_report_output(obj)

    def test_no_summary(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_report_output

            output = SimpleNamespace(report_type="detailed")
            _render_report_output(output)


class TestAlertOutputEdgeCases:
    """Test _render_alert_output edge cases."""

    def test_alerts_with_missing_keys(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_alert_output

            _render_alert_output(
                SimpleNamespace(
                    alerts=[
                        {},  # No severity, title, or message
                        {"severity": "info"},  # No title or message
                    ]
                )
            )

    def test_no_alerts_attr(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_alert_output

            obj = MagicMock(spec=[])
            _render_alert_output(obj)


class TestRenderPipelineResultWithStages:
    """Test _render_pipeline_result with various stage types."""

    def test_stage_with_output_for_each_type(self):
        with patch("animus_forge.dashboard.monitoring_pages.st", _make_mock_st()):
            from animus_forge.dashboard.monitoring_pages import _render_pipeline_result

            stages = []
            for stage_type in ["collect", "analyze", "visualize", "report", "alert"]:
                stage = MagicMock()
                stage.stage.value = stage_type
                stage.duration_ms = 50.0
                stage.status = "success"
                stage.error = None
                stage.output = MagicMock()
                stages.append(stage)

            result = MagicMock()
            result.status = "completed"
            result.stages = stages
            result.errors = []
            result.final_output = SimpleNamespace(
                metrics={"counters": {"total": 10, "errors": 2}},
                findings=[{"severity": "info", "message": "All good"}],
                to_dict=lambda: {"key": "val"},
            )

            _render_pipeline_result(result, "test", {"test": {"name": "Test"}})

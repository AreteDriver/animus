"""Tests for workflow visualizer component."""

import sys
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
    mod_key = "animus_forge.dashboard.workflow_visualizer"
    cached = sys.modules.pop(mod_key, None)

    with patch.dict(sys.modules, {"streamlit": mock_st}):
        yield mock_st

    # Remove again so other tests aren't affected
    sys.modules.pop(mod_key, None)
    if cached is not None:
        sys.modules[mod_key] = cached


class TestDataClasses:
    def test_step_status_enum_values(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import StepStatus

        assert StepStatus.PENDING.value == "pending"
        assert StepStatus.RUNNING.value == "running"
        assert StepStatus.COMPLETED.value == "completed"
        assert StepStatus.FAILED.value == "failed"
        assert StepStatus.SKIPPED.value == "skipped"

    def test_visual_step_defaults(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import StepStatus, VisualStep

        step = VisualStep(id="s1", name="Test", type="shell")

        assert step.id == "s1"
        assert step.name == "Test"
        assert step.type == "shell"
        assert step.status == StepStatus.PENDING
        assert step.duration_ms is None
        assert step.error is None
        assert step.output_preview is None

    def test_visual_step_all_fields(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import StepStatus, VisualStep

        step = VisualStep(
            id="s1",
            name="Build",
            type="claude_code",
            status=StepStatus.COMPLETED,
            duration_ms=1500,
            error=None,
            output_preview="success",
        )

        assert step.status == StepStatus.COMPLETED
        assert step.duration_ms == 1500
        assert step.output_preview == "success"


class TestRenderWorkflowVisualizer:
    def _make_steps(self):
        return [
            {"id": "plan", "type": "claude_code", "params": {"role": "planner"}},
            {"id": "build", "type": "openai", "params": {"role": "builder"}},
            {"id": "test", "type": "shell", "params": {}},
        ]

    def test_renders_with_steps(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        render_workflow_visualizer(self._make_steps())

        mock_streamlit.markdown.assert_called()
        mock_streamlit.progress.assert_called()

    def test_compact_mode(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        render_workflow_visualizer(self._make_steps(), compact=True)

        mock_streamlit.progress.assert_called()
        # Compact mode creates columns for steps
        mock_streamlit.columns.assert_called()

    def test_empty_steps(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        render_workflow_visualizer([])

        mock_streamlit.progress.assert_called_once()

    def test_current_step_highlight(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        render_workflow_visualizer(self._make_steps(), current_step="build")

        # Should render without error; the running step gets highlighted via markdown
        mock_streamlit.markdown.assert_called()

    def test_with_step_results(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        results = {
            "plan": {"status": "completed", "duration_ms": 500},
            "build": {"status": "completed", "duration_ms": 1200},
        }

        render_workflow_visualizer(self._make_steps(), step_results=results)

        # Progress should reflect 2/3 completed
        progress_call = mock_streamlit.progress.call_args
        assert progress_call[0][0] == pytest.approx(2 / 3, abs=0.01)


class TestRenderWorkflowSummary:
    def test_renders_metric_columns(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Test WF",
            total_steps=5,
            completed_steps=4,
            failed_steps=1,
            total_duration_ms=3000,
            total_cost_usd=0.05,
        )

        assert mock_streamlit.metric.call_count == 4
        mock_streamlit.columns.assert_called_with(4)

    def test_zero_values(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Empty",
            total_steps=0,
            completed_steps=0,
            failed_steps=0,
        )

        mock_streamlit.metric.assert_called()

    def test_optional_duration_and_cost(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Basic",
            total_steps=3,
            completed_steps=3,
            failed_steps=0,
        )

        # Duration and cost should show "N/A"
        metric_calls = mock_streamlit.metric.call_args_list
        na_calls = [c for c in metric_calls if "N/A" in str(c)]
        assert len(na_calls) >= 1

    def test_failed_steps_shows_error(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Fail",
            total_steps=3,
            completed_steps=2,
            failed_steps=1,
        )

        mock_streamlit.error.assert_called()

    def test_all_completed_shows_success(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Done",
            total_steps=3,
            completed_steps=3,
            failed_steps=0,
        )

        mock_streamlit.success.assert_called()

    def test_in_progress_shows_info(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Running",
            total_steps=5,
            completed_steps=2,
            failed_steps=0,
        )

        mock_streamlit.info.assert_called()


class TestRenderStepTimeline:
    def test_renders_timeline(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_step_timeline

        steps = [
            {"id": "plan", "type": "claude_code"},
            {"id": "build", "type": "openai"},
        ]
        results = {
            "plan": {"status": "completed", "duration_ms": 500},
            "build": {"status": "completed", "duration_ms": 1200},
        }

        render_step_timeline(steps, results)

        mock_streamlit.markdown.assert_called()

    def test_empty_results(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_step_timeline

        steps = [{"id": "plan", "type": "claude_code"}]

        render_step_timeline(steps, {})

        mock_streamlit.markdown.assert_called()


class TestRenderAgentActivity:
    def test_renders_activities(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        agents = [
            {
                "role": "planner",
                "action": "planning",
                "timestamp": "10:00",
                "status": "completed",
            },
            {
                "role": "builder",
                "action": "coding",
                "timestamp": "10:05",
                "status": "running",
            },
        ]

        render_agent_activity(agents)

        mock_streamlit.markdown.assert_called()

    def test_empty_list(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        render_agent_activity([])

        mock_streamlit.info.assert_called()

    def test_show_live_flag(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        render_agent_activity([], show_live=True)

        # Live indicator rendered via markdown
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        assert any("Live" in c for c in calls)

    def test_show_live_false(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        render_agent_activity([], show_live=False)

        # No live indicator — only the title markdown and info
        calls = [str(c) for c in mock_streamlit.markdown.call_args_list]
        live_calls = [c for c in calls if "pulse" in c]
        assert len(live_calls) == 0


# =============================================================================
# Additional coverage tests for previously uncovered lines
# =============================================================================


class TestCompactViewStatusBranches:
    """Test compact view status determination paths (lines 111-120)."""

    def test_compact_all_statuses(self, mock_streamlit):
        """Test compact view with all status variants: running, completed, failed, skipped, pending."""
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [
            {"id": "s_running", "type": "claude_code"},
            {"id": "s_completed", "type": "openai"},
            {"id": "s_failed", "type": "shell"},
            {"id": "s_skipped", "type": "github"},
            {"id": "s_pending", "type": "notion"},
        ]
        results = {
            "s_completed": {"status": "completed", "duration_ms": 100},
            "s_failed": {"status": "failed", "error": "boom"},
            "s_skipped": {"status": "skipped"},
        }

        render_workflow_visualizer(
            steps, current_step="s_running", step_results=results, compact=True
        )

        # Should have rendered all steps via markdown
        assert mock_streamlit.markdown.call_count >= 5

    def test_compact_with_unknown_type(self, mock_streamlit):
        """Test compact view with unknown step type (falls back to default icon)."""
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [{"id": "mystery", "type": "custom_unknown_type"}]
        render_workflow_visualizer(steps, compact=True)

        # Should still render without error
        mock_streamlit.markdown.assert_called()


class TestDetailedViewAllStatuses:
    """Test detailed view with all status variants (lines 160-169)."""

    def test_detailed_failed_status(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [{"id": "fail_step", "type": "shell", "params": {"role": "tester"}}]
        results = {
            "fail_step": {
                "status": "failed",
                "duration_ms": 500,
                "error": "Command failed with exit code 1",
            }
        }

        render_workflow_visualizer(steps, step_results=results)
        mock_streamlit.markdown.assert_called()

    def test_detailed_skipped_status(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [{"id": "skip_step", "type": "condition", "params": {}}]
        results = {"skip_step": {"status": "skipped"}}

        render_workflow_visualizer(steps, step_results=results)
        mock_streamlit.markdown.assert_called()

    def test_detailed_running_status(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [{"id": "run_step", "type": "claude_code", "params": {"role": "builder"}}]

        render_workflow_visualizer(steps, current_step="run_step")
        mock_streamlit.markdown.assert_called()


class TestDetailedViewPromptPreview:
    """Test detailed view prompt preview truncation (lines 207-212)."""

    def test_long_prompt_truncated(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        long_prompt = "x" * 200  # Longer than 100 chars
        steps = [
            {
                "id": "prompt_step",
                "type": "claude_code",
                "params": {"role": "planner", "prompt": long_prompt},
            }
        ]
        results = {"prompt_step": {"status": "completed", "duration_ms": 100}}

        render_workflow_visualizer(steps, step_results=results)

        # Check that write was called with truncated prompt
        write_calls = [str(c) for c in mock_streamlit.write.call_args_list]
        prompt_calls = [c for c in write_calls if "Prompt:" in c]
        assert len(prompt_calls) >= 1
        assert any("..." in c for c in prompt_calls)

    def test_short_prompt_not_truncated(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        short_prompt = "Do something simple"
        steps = [
            {
                "id": "short_step",
                "type": "openai",
                "params": {"role": "builder", "prompt": short_prompt},
            }
        ]
        results = {"short_step": {"status": "completed", "duration_ms": 50}}

        render_workflow_visualizer(steps, step_results=results)

        write_calls = [str(c) for c in mock_streamlit.write.call_args_list]
        prompt_calls = [c for c in write_calls if "Prompt:" in c]
        assert len(prompt_calls) >= 1


class TestDetailedViewErrorAndOutput:
    """Test detailed view error display and output code (lines 219-223)."""

    def test_step_with_error(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [
            {
                "id": "err_step",
                "type": "shell",
                "params": {"role": "tester"},
            }
        ]
        results = {
            "err_step": {
                "status": "failed",
                "duration_ms": 200,
                "error": "Connection refused",
            }
        }

        render_workflow_visualizer(steps, step_results=results)
        mock_streamlit.error.assert_called_with("Connection refused")

    def test_step_with_output(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        steps = [
            {
                "id": "out_step",
                "type": "claude_code",
                "params": {"role": "builder"},
            }
        ]
        results = {
            "out_step": {
                "status": "completed",
                "duration_ms": 300,
                "output": "Generated 5 files successfully",
            }
        }

        render_workflow_visualizer(steps, step_results=results)
        mock_streamlit.code.assert_called()

    def test_step_with_long_output_truncated(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_visualizer

        long_output = "A" * 300  # > 200 chars
        steps = [
            {
                "id": "long_step",
                "type": "openai",
                "params": {},
            }
        ]
        results = {
            "long_step": {
                "status": "completed",
                "duration_ms": 100,
                "output": long_output,
            }
        }

        render_workflow_visualizer(steps, step_results=results)
        # output_preview = str(result["output"])[:200]
        code_calls = mock_streamlit.code.call_args_list
        assert len(code_calls) >= 1
        # The output should be truncated to 200 chars
        actual_output = code_calls[0][0][0]
        assert len(actual_output) == 200


class TestStepTimelineStatuses:
    """Test step timeline with various status values."""

    def test_timeline_all_statuses(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_step_timeline

        steps = [
            {"id": "s1", "type": "claude_code"},
            {"id": "s2", "type": "openai"},
            {"id": "s3", "type": "shell"},
            {"id": "s4", "type": "github"},
            {"id": "s5", "type": "notion"},
        ]
        results = {
            "s1": {"status": "completed", "duration_ms": 500},
            "s2": {"status": "failed", "duration_ms": 200},
            "s3": {"status": "running", "duration_ms": 100},
            "s4": {"status": "skipped", "duration_ms": 0},
            "s5": {"status": "pending", "duration_ms": 0},
        }

        render_step_timeline(steps, results)
        mock_streamlit.markdown.assert_called()

    def test_timeline_unknown_status_falls_back(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_step_timeline

        steps = [{"id": "s1"}]
        results = {"s1": {"status": "something_unknown", "duration_ms": 100}}

        render_step_timeline(steps, results)
        mock_streamlit.markdown.assert_called()


class TestAgentActivityRoles:
    """Test agent activity with all role icon branches."""

    def test_all_role_icons(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        agents = [
            {
                "role": "builder",
                "action": "coding",
                "timestamp": "10:00",
                "status": "running",
            },
            {
                "role": "planner",
                "action": "planning",
                "timestamp": "10:01",
                "status": "completed",
            },
            {
                "role": "tester",
                "action": "testing",
                "timestamp": "10:02",
                "status": "failed",
            },
            {
                "role": "reviewer",
                "action": "reviewing",
                "timestamp": "10:03",
                "status": "running",
            },
        ]

        render_agent_activity(agents)

        # All 4 agents should render, each with their icon
        markdown_calls = mock_streamlit.markdown.call_args_list
        # Title + live indicator + 4 agents = at least 6 markdown calls
        assert len(markdown_calls) >= 6

    def test_more_than_10_agents_shows_last_10(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_agent_activity

        agents = [
            {
                "role": "builder",
                "action": f"task_{i}",
                "timestamp": f"10:{i:02d}",
                "status": "completed",
            }
            for i in range(15)
        ]

        render_agent_activity(agents, show_live=False)

        # Title markdown + 10 agent markdowns = 11
        agent_calls = [
            c for c in mock_streamlit.markdown.call_args_list if "border-bottom" in str(c)
        ]
        assert len(agent_calls) == 10


class TestWorkflowSummaryEdgeCases:
    """Test workflow summary edge cases."""

    def test_cost_zero(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import render_workflow_summary

        render_workflow_summary(
            workflow_name="Free",
            total_steps=2,
            completed_steps=2,
            failed_steps=0,
            total_cost_usd=0.0,
        )

        metric_calls = mock_streamlit.metric.call_args_list
        cost_calls = [c for c in metric_calls if "Cost" in str(c)]
        assert len(cost_calls) >= 1
        # Should show $0.0000 not N/A
        assert any("$0.0000" in str(c) for c in cost_calls)


class TestStatusDicts:
    """Test STATUS_COLORS and STATUS_ICONS completeness."""

    def test_all_statuses_have_colors(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import STATUS_COLORS, StepStatus

        for status in StepStatus:
            assert status in STATUS_COLORS

    def test_all_statuses_have_icons(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import STATUS_ICONS, StepStatus

        for status in StepStatus:
            assert status in STATUS_ICONS

    def test_step_type_icons(self, mock_streamlit):
        from animus_forge.dashboard.workflow_visualizer import STEP_TYPE_ICONS

        expected_types = [
            "claude_code",
            "openai",
            "shell",
            "github",
            "notion",
            "slack",
            "checkpoint",
            "condition",
        ]
        for t in expected_types:
            assert t in STEP_TYPE_ICONS

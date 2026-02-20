"""Tests for parallel execution dashboard page."""

import sys

sys.path.insert(0, "src")


class TestParallelDashboardImports:
    """Test that dashboard parallel page imports correctly."""

    def test_import_render_functions(self):
        """Can import render functions."""
        from animus_forge.dashboard.monitoring_pages import (
            get_parallel_tracker,
            render_parallel_execution_page,
            render_parallel_status_sidebar,
        )

        assert callable(render_parallel_execution_page)
        assert callable(render_parallel_status_sidebar)
        assert callable(get_parallel_tracker)

    def test_page_in_app_registry(self):
        """Parallel page is registered in app."""
        from animus_forge.dashboard.app import _PAGE_RENDERERS

        assert "Parallel" in _PAGE_RENDERERS
        assert callable(_PAGE_RENDERERS["Parallel"])


class TestParallelTrackerDashboardData:
    """Test that parallel tracker provides dashboard-compatible data."""

    def test_get_dashboard_data_structure(self):
        """get_dashboard_data returns expected structure."""
        from animus_forge.monitoring.parallel_tracker import get_parallel_tracker

        tracker = get_parallel_tracker()
        tracker.reset()  # Start fresh

        data = tracker.get_dashboard_data()

        assert "summary" in data
        assert "active_executions" in data
        assert "recent_executions" in data
        assert "rate_limits" in data

    def test_summary_has_required_fields(self):
        """Summary contains required fields for dashboard."""
        from animus_forge.monitoring.parallel_tracker import get_parallel_tracker

        tracker = get_parallel_tracker()
        tracker.reset()

        summary = tracker.get_summary()

        assert "active_executions" in summary
        assert "active_branches" in summary
        assert "total_executions" in summary
        assert "success_rate" in summary
        assert "counters" in summary
        assert "execution_duration" in summary
        assert "branch_duration" in summary
        assert "rate_limit_states" in summary

    def test_dashboard_data_with_execution(self):
        """Dashboard data reflects actual executions."""
        from animus_forge.monitoring.parallel_tracker import (
            ParallelPatternType,
            get_parallel_tracker,
        )

        tracker = get_parallel_tracker()
        tracker.reset()

        # Start an execution
        tracker.start_execution(
            execution_id="test_dashboard_exec",
            pattern_type=ParallelPatternType.FAN_OUT,
            step_id="test_step",
            total_items=5,
            max_concurrent=3,
        )

        # Add a branch
        tracker.start_branch("test_dashboard_exec", "branch_0", 0, "item_0")

        data = tracker.get_dashboard_data()

        assert data["summary"]["active_executions"] == 1
        assert len(data["active_executions"]) == 1

        active = data["active_executions"][0]
        assert active["execution_id"] == "test_dashboard_exec"
        assert active["pattern_type"] == "fan_out"
        assert active["total_items"] == 5

        # Complete and check history
        tracker.complete_branch("test_dashboard_exec", "branch_0", tokens=100)
        tracker.complete_execution("test_dashboard_exec")

        data = tracker.get_dashboard_data()
        assert data["summary"]["active_executions"] == 0
        assert len(data["recent_executions"]) >= 1

    def test_rate_limit_data_structure(self):
        """Rate limit data has correct structure."""
        from animus_forge.monitoring.parallel_tracker import get_parallel_tracker

        tracker = get_parallel_tracker()
        tracker.reset()

        # Add rate limit state
        tracker.update_rate_limit_state(
            provider="anthropic",
            current_limit=3,
            base_limit=5,
            total_429s=2,
            is_throttled=True,
        )

        data = tracker.get_dashboard_data()
        rate_limits = data["rate_limits"]

        assert "anthropic" in rate_limits
        assert rate_limits["anthropic"]["current_limit"] == 3
        assert rate_limits["anthropic"]["base_limit"] == 5
        assert rate_limits["anthropic"]["total_429s"] == 2
        assert rate_limits["anthropic"]["is_throttled"] is True


class TestParallelDashboardHelpers:
    """Test helper functions for parallel dashboard."""

    def test_get_parallel_tracker_returns_singleton(self):
        """get_parallel_tracker returns same instance."""
        from animus_forge.dashboard.monitoring_pages import get_parallel_tracker

        tracker1 = get_parallel_tracker()
        tracker2 = get_parallel_tracker()

        assert tracker1 is tracker2

    def test_parallel_tracker_from_monitoring_package(self):
        """Tracker from monitoring package matches."""
        from animus_forge.dashboard.monitoring_pages import get_parallel_tracker
        from animus_forge.monitoring.parallel_tracker import (
            get_parallel_tracker as direct_get_tracker,
        )

        dashboard_tracker = get_parallel_tracker()
        direct_tracker = direct_get_tracker()

        assert dashboard_tracker is direct_tracker

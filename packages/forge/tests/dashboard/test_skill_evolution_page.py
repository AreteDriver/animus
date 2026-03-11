"""Tests for skill evolution dashboard page."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import MagicMock, patch

import pytest


@dataclass(frozen=True)
class FakeMetrics:
    skill_name: str = "code_review"
    skill_version: str = "1.0"
    period_start: str = "2026-01-01"
    period_end: str = "2026-03-01"
    total_invocations: int = 100
    success_count: int = 90
    failure_count: int = 10
    success_rate: float = 0.9
    avg_quality_score: float = 0.85
    avg_cost_usd: float = 0.01
    avg_latency_ms: float = 250.0
    total_cost_usd: float = 1.0
    trend: str = "stable"
    computed_at: str = "2026-03-01T00:00:00"


def _make_ctx_mocks(n):
    """Create n MagicMock context managers."""
    mocks = [MagicMock() for _ in range(n)]
    for m in mocks:
        m.__enter__ = MagicMock(return_value=m)
        m.__exit__ = MagicMock(return_value=False)
    return mocks


@pytest.fixture
def mock_streamlit():
    """Mock streamlit for testing dashboard rendering."""
    with patch("animus_forge.dashboard.skill_evolution_page.st") as mock_st:
        mock_st.columns.side_effect = lambda n: _make_ctx_mocks(n)
        mock_st.tabs.return_value = _make_ctx_mocks(4)
        yield mock_st


class TestSkillEvolutionPage:
    """Test skill evolution page rendering."""

    def test_render_metrics_tab_no_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_metrics_tab,
        )

        aggregator = MagicMock()
        aggregator.get_all_skill_metrics.return_value = []

        _render_metrics_tab(aggregator)
        mock_streamlit.info.assert_called_once()

    def test_render_metrics_tab_with_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_metrics_tab,
        )

        mock_streamlit.slider.return_value = 30
        mock_streamlit.text_input.return_value = ""
        mock_streamlit.selectbox.return_value = "code_review"

        metrics = [FakeMetrics(), FakeMetrics(skill_name="debug", skill_version="2.0")]
        aggregator = MagicMock()
        aggregator.get_all_skill_metrics.return_value = metrics
        aggregator.get_skill_metrics.return_value = FakeMetrics()
        aggregator.get_skill_trend.return_value = "improving"

        _render_metrics_tab(aggregator)
        aggregator.get_all_skill_metrics.assert_called_once_with(days=30)
        mock_streamlit.dataframe.assert_called_once()

    def test_render_metrics_tab_filter(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_metrics_tab,
        )

        mock_streamlit.slider.return_value = 30
        mock_streamlit.text_input.return_value = "code"
        mock_streamlit.selectbox.return_value = "code_review"

        metrics = [FakeMetrics(), FakeMetrics(skill_name="debug")]
        aggregator = MagicMock()
        aggregator.get_all_skill_metrics.return_value = metrics
        aggregator.get_skill_metrics.return_value = FakeMetrics()
        aggregator.get_skill_trend.return_value = "stable"

        _render_metrics_tab(aggregator)
        # dataframe should only contain "code_review" (filtered)
        call_args = mock_streamlit.dataframe.call_args
        rows = call_args[0][0]
        assert len(rows) == 1
        assert rows[0]["Skill"] == "code_review"

    def test_render_experiments_tab_no_active(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_experiments_tab,
        )

        ab_manager = MagicMock()
        ab_manager.get_active_experiments.return_value = []
        aggregator = MagicMock()

        with patch("animus_forge.dashboard.skill_evolution_page.st", mock_streamlit):
            _render_experiments_tab(ab_manager, aggregator)
        mock_streamlit.info.assert_called()

    def test_render_experiments_tab_with_active(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_experiments_tab,
        )

        ab_manager = MagicMock()
        ab_manager.get_active_experiments.return_value = [
            {
                "id": "abc12345",
                "skill_name": "code_review",
                "control_version": "1.0",
                "variant_version": "1.1",
                "traffic_split": 0.5,
                "min_invocations": 100,
                "start_date": "2026-03-01T00:00:00",
            }
        ]
        ab_manager.evaluate_experiment.return_value = None
        aggregator = MagicMock()

        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_streamlit.expander.return_value = exp_mock

        with patch("animus_forge.dashboard.skill_evolution_page.st", mock_streamlit):
            _render_experiments_tab(ab_manager, aggregator)
        ab_manager.get_active_experiments.assert_called_once()

    def test_render_versions_tab_no_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_versions_tab,
        )

        db = MagicMock()
        db.fetchall.return_value = []

        _render_versions_tab(db)
        mock_streamlit.info.assert_called()

    def test_render_versions_tab_with_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_versions_tab,
        )

        db = MagicMock()
        db.fetchall.return_value = [
            {
                "skill_name": "code_review",
                "version": "1.1",
                "previous_version": "1.0",
                "change_type": "tune",
                "change_description": "Improved prompt",
                "created_at": "2026-03-01T00:00:00",
            }
        ]
        mock_streamlit.selectbox.return_value = "All"
        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_streamlit.expander.return_value = exp_mock

        _render_versions_tab(db)
        db.fetchall.assert_called_once()

    def test_render_deprecations_tab_no_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_deprecations_tab,
        )

        db = MagicMock()
        db.fetchall.return_value = []

        _render_deprecations_tab(db)
        mock_streamlit.info.assert_called()

    def test_render_deprecations_tab_with_data(self, mock_streamlit):
        from animus_forge.dashboard.skill_evolution_page import (
            _render_deprecations_tab,
        )

        db = MagicMock()
        db.fetchall.return_value = [
            {
                "skill_name": "old_skill",
                "status": "flagged",
                "reason": "Low success rate",
                "success_rate_at_flag": 0.3,
                "invocations_at_flag": 50,
                "flagged_at": "2026-02-01T00:00:00",
                "deprecated_at": None,
                "retired_at": None,
                "replacement_skill": None,
            },
            {
                "skill_name": "ancient_skill",
                "status": "retired",
                "reason": "Replaced by new_skill",
                "success_rate_at_flag": 0.1,
                "invocations_at_flag": 200,
                "flagged_at": "2026-01-01T00:00:00",
                "deprecated_at": "2026-01-15T00:00:00",
                "retired_at": "2026-02-01T00:00:00",
                "replacement_skill": "new_skill",
            },
        ]

        exp_mock = MagicMock()
        exp_mock.__enter__ = MagicMock(return_value=exp_mock)
        exp_mock.__exit__ = MagicMock(return_value=False)
        mock_streamlit.expander.return_value = exp_mock

        _render_deprecations_tab(db)
        db.fetchall.assert_called_once()


class TestSkillEvolutionPageWiring:
    """Test that the page is wired into the dashboard app."""

    def test_skills_in_page_renderers(self):
        from animus_forge.dashboard.app import _PAGE_RENDERERS

        assert "Skills" in _PAGE_RENDERERS

    def test_render_function_importable(self):
        from animus_forge.dashboard.skill_evolution_page import (
            render_skill_evolution_page,
        )

        assert callable(render_skill_evolution_page)

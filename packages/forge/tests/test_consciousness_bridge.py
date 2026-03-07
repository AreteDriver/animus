"""Tests for the consciousness-quorum bridge."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from animus_forge.budget.manager import BudgetConfig, BudgetManager
from animus_forge.coordination.consciousness_bridge import (
    _DEFAULT_PRINCIPLES,
    BudgetExhausted,
    ConsciousnessBridge,
    ConsciousnessConfig,
    ReflectionInput,
    ReflectionOutput,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_provider():
    """Mock LLM provider returning valid reflection JSON."""
    provider = MagicMock()
    response = MagicMock()
    response.content = json.dumps(
        {
            "summary": "Reviewed 5 recent actions. No anomalies.",
            "insights": ["Build times trending down after caching change"],
            "proposed_intent_updates": [
                {"content": "Cache hit rate improving", "tags": ["performance"]},
            ],
            "workflow_patch_ids": [],
            "principle_tensions": [],
            "next_reflection_in": 300,
        }
    )
    response.tokens_used = 450
    response.model = "claude-sonnet-4-6"
    provider.complete.return_value = response
    return provider


@pytest.fixture()
def budget_manager():
    """Budget manager with room to spare."""
    return BudgetManager(config=BudgetConfig(total_budget=100000))


@pytest.fixture()
def config(tmp_path):
    """Default consciousness config with tmp log paths."""
    return ConsciousnessConfig(
        enabled=True,
        min_idle_seconds=0,
        reflections_log_path=tmp_path / "reflections.jsonl",
        review_queue_path=tmp_path / "workflow_review_queue.jsonl",
    )


@pytest.fixture()
def bridge(mock_provider, budget_manager, config):
    """Ready-to-use bridge without Quorum graph."""
    return ConsciousnessBridge(
        provider=mock_provider,
        budget_manager=budget_manager,
        config=config,
    )


@pytest.fixture()
def mock_metrics():
    """Mock MetricsStore."""
    store = MagicMock()
    store.get_recent_executions.return_value = [
        {"workflow_id": "code-review", "status": "success", "tokens": 1200},
        {"workflow_id": "feature-build", "status": "failed", "error": "timeout"},
    ]
    return store


# ---------------------------------------------------------------------------
# ReflectionOutput parsing
# ---------------------------------------------------------------------------


class TestReflectionOutput:
    def test_default_values(self):
        output = ReflectionOutput()
        assert output.summary == ""
        assert output.insights == []
        assert output.next_reflection_in == 300

    def test_from_dict(self):
        data = {
            "summary": "All clear",
            "insights": ["x", "y"],
            "proposed_intent_updates": [],
            "workflow_patch_ids": ["code-review"],
            "principle_tensions": ["P5 nearly violated"],
            "next_reflection_in": 600,
        }
        output = ReflectionOutput(**data)
        assert output.summary == "All clear"
        assert len(output.insights) == 2
        assert output.workflow_patch_ids == ["code-review"]


class TestReflectionInput:
    def test_default_values(self):
        inp = ReflectionInput()
        assert inp.recent_actions == []
        assert inp.session_budget_remaining == 0


# ---------------------------------------------------------------------------
# Bridge initialization
# ---------------------------------------------------------------------------


class TestBridgeInit:
    def test_default_config(self, mock_provider, budget_manager):
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
        )
        assert not bridge.is_running
        assert bridge.reflection_count == 0
        assert bridge.total_tokens == 0

    def test_loads_principles_from_file(self, mock_provider, budget_manager, tmp_path):
        principles_file = tmp_path / "PRINCIPLES.md"
        principles_file.write_text(
            "# Principles\n"
            "### P1 — Sovereignty\nServe one user.\n"
            "### P2 — Continuity\nMemory persists.\n"
        )
        config = ConsciousnessConfig(principles_path=principles_file)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        assert len(bridge._principles) == 2
        assert "P1" in bridge._principles[0]

    def test_missing_principles_file_uses_defaults(
        self,
        mock_provider,
        budget_manager,
    ):
        config = ConsciousnessConfig(principles_path=Path("/nonexistent/file.md"))
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        assert bridge._principles == _DEFAULT_PRINCIPLES


# ---------------------------------------------------------------------------
# reflect_once()
# ---------------------------------------------------------------------------


class TestReflectOnce:
    def test_successful_reflection(self, bridge, mock_provider):
        output = bridge.reflect_once()
        assert output.summary == "Reviewed 5 recent actions. No anomalies."
        assert len(output.insights) == 1
        assert bridge.reflection_count == 1
        assert bridge.total_tokens == 450
        mock_provider.complete.assert_called_once()

    def test_records_budget_usage(self, bridge, budget_manager):
        bridge.reflect_once()
        assert budget_manager.used == 450
        history = budget_manager.get_usage_history(agent_id="consciousness_bridge")
        assert len(history) == 1
        assert history[0].operation == "reflect"

    def test_logs_to_file(self, bridge, config):
        bridge.reflect_once()
        log_path = config.reflections_log_path
        assert log_path.exists()
        lines = log_path.read_text().strip().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert "timestamp" in record
        assert record["model"] == "claude-sonnet-4-6"

    def test_second_reflection_increments_count(self, bridge):
        bridge.reflect_once()
        bridge.reflect_once()
        assert bridge.reflection_count == 2
        assert bridge.total_tokens == 900

    def test_gather_input_with_metrics(self, mock_provider, budget_manager, config, mock_metrics):
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
            metrics_store=mock_metrics,
        )
        output = bridge.reflect_once()
        mock_metrics.get_recent_executions.assert_called_once_with(limit=50)
        assert output.summary != ""

    def test_malformed_llm_response_returns_empty(self, bridge, mock_provider):
        mock_provider.complete.return_value.content = "not json at all {{"
        output = bridge.reflect_once()
        assert output.summary == "Parse failure — skipped"
        assert bridge.reflection_count == 1

    def test_markdown_fenced_json_parsed(self, bridge, mock_provider):
        mock_provider.complete.return_value.content = (
            "```json\n"
            '{"summary": "fenced", "insights": [], "proposed_intent_updates": [],'
            ' "workflow_patch_ids": [], "principle_tensions": [],'
            ' "next_reflection_in": 300}\n'
            "```"
        )
        output = bridge.reflect_once()
        assert output.summary == "fenced"

    def test_regex_fallback_for_preamble(self, bridge, mock_provider):
        mock_provider.complete.return_value.content = (
            "Here is my reflection:\n"
            '{"summary": "extracted", "insights": ["a"], "proposed_intent_updates": [],'
            ' "workflow_patch_ids": [], "principle_tensions": [],'
            ' "next_reflection_in": 300}'
        )
        output = bridge.reflect_once()
        assert output.summary == "extracted"


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_raises_budget_exhausted_when_no_tokens(self, mock_provider, config):
        budget = BudgetManager(config=BudgetConfig(total_budget=100))
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget,
            config=config,
        )
        with pytest.raises(BudgetExhausted):
            bridge.reflect_once()

    def test_should_reflect_false_at_warning_threshold(self, mock_provider, config):
        budget = BudgetManager(config=BudgetConfig(total_budget=1000))
        # Push past 75%
        budget.record_usage("test", 760, "test")
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget,
            config=config,
        )
        assert not bridge._should_reflect()

    def test_should_reflect_false_when_exceeded(self, mock_provider, config):
        budget = BudgetManager(config=BudgetConfig(total_budget=100))
        budget.record_usage("test", 101, "test")
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget,
            config=config,
        )
        assert not bridge._should_reflect()

    def test_should_reflect_false_when_disabled(self, mock_provider, budget_manager):
        config = ConsciousnessConfig(enabled=False)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        assert not bridge._should_reflect()


# ---------------------------------------------------------------------------
# Idle timer
# ---------------------------------------------------------------------------


class TestIdleTimer:
    def test_should_reflect_false_before_idle_threshold(
        self,
        mock_provider,
        budget_manager,
    ):
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=300)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        # Simulate a recent reflection
        bridge._last_reflection = datetime.now(UTC)
        assert not bridge._should_reflect()

    def test_should_reflect_true_after_idle_threshold(
        self,
        mock_provider,
        budget_manager,
    ):
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=0)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        assert bridge._should_reflect()

    def test_first_reflection_has_no_idle_constraint(
        self,
        mock_provider,
        budget_manager,
    ):
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=9999)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        # _last_reflection is None -> no idle check
        assert bridge._should_reflect()


# ---------------------------------------------------------------------------
# Quorum integration
# ---------------------------------------------------------------------------


class TestQuorumPublish:
    @pytest.fixture()
    def mock_graph(self):
        graph = MagicMock()
        snapshot = MagicMock()
        snapshot.intents = []
        graph.snapshot.return_value = snapshot
        graph.publish.return_value = 0.3
        return graph

    def test_publishes_insights_to_graph(
        self,
        mock_provider,
        budget_manager,
        config,
        mock_graph,
    ):
        with patch(
            "animus_forge.coordination.consciousness_bridge.HAS_CONVERGENT",
            True,
        ):
            bridge = ConsciousnessBridge(
                provider=mock_provider,
                budget_manager=budget_manager,
                config=config,
                graph=mock_graph,
            )
            bridge.reflect_once()
            # Provider returns 1 insight -> 1 publish call
            assert mock_graph.publish.call_count == 1
            published_intent = mock_graph.publish.call_args[0][0]
            assert published_intent.agent_id == "consciousness_bridge"
            assert "reflection" in published_intent.provides[0].tags

    def test_publishes_principle_tensions(
        self,
        mock_provider,
        budget_manager,
        config,
        mock_graph,
    ):
        mock_provider.complete.return_value.content = json.dumps(
            {
                "summary": "Tension detected",
                "insights": [],
                "proposed_intent_updates": [],
                "workflow_patch_ids": [],
                "principle_tensions": ["P5 nearly violated: Opus used for routing"],
                "next_reflection_in": 300,
            }
        )
        with patch(
            "animus_forge.coordination.consciousness_bridge.HAS_CONVERGENT",
            True,
        ):
            bridge = ConsciousnessBridge(
                provider=mock_provider,
                budget_manager=budget_manager,
                config=config,
                graph=mock_graph,
            )
            bridge.reflect_once()
            assert mock_graph.publish.call_count == 1
            intent = mock_graph.publish.call_args[0][0]
            assert "PRINCIPLE TENSION" in intent.intent
            assert "principle_tension" in intent.provides[0].tags

    def test_no_quorum_degrades_gracefully(self, bridge):
        # bridge has no graph -> should not raise
        output = bridge.reflect_once()
        assert output.summary != ""

    def test_graph_error_does_not_crash(
        self,
        mock_provider,
        budget_manager,
        config,
        mock_graph,
    ):
        mock_graph.publish.side_effect = RuntimeError("graph broken")
        with patch(
            "animus_forge.coordination.consciousness_bridge.HAS_CONVERGENT",
            True,
        ):
            bridge = ConsciousnessBridge(
                provider=mock_provider,
                budget_manager=budget_manager,
                config=config,
                graph=mock_graph,
            )
            # Should not raise
            output = bridge.reflect_once()
            assert output.summary != ""


# ---------------------------------------------------------------------------
# Workflow review queue
# ---------------------------------------------------------------------------


class TestWorkflowReviewQueue:
    def test_queues_workflow_reviews(self, mock_provider, budget_manager, config):
        mock_provider.complete.return_value.content = json.dumps(
            {
                "summary": "Found issues",
                "insights": [],
                "proposed_intent_updates": [],
                "workflow_patch_ids": ["code-review", "feature-build"],
                "principle_tensions": [],
                "next_reflection_in": 300,
            }
        )
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        bridge.reflect_once()
        queue_path = config.review_queue_path
        assert queue_path.exists()
        lines = queue_path.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["workflow_id"] == "code-review"
        assert json.loads(lines[1])["workflow_id"] == "feature-build"

    def test_no_queue_path_skips_silently(self, mock_provider, budget_manager):
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=0)
        mock_provider.complete.return_value.content = json.dumps(
            {
                "summary": "ok",
                "insights": [],
                "proposed_intent_updates": [],
                "workflow_patch_ids": ["x"],
                "principle_tensions": [],
                "next_reflection_in": 300,
            }
        )
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        # Should not raise
        bridge.reflect_once()


# ---------------------------------------------------------------------------
# Background loop
# ---------------------------------------------------------------------------


class TestBackgroundLoop:
    def test_start_and_stop(self, bridge):
        bridge.start()
        assert bridge.is_running
        bridge.stop()
        assert not bridge.is_running

    def test_start_disabled_is_noop(self, mock_provider, budget_manager):
        config = ConsciousnessConfig(enabled=False)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        bridge.start()
        assert not bridge.is_running

    def test_double_start_is_safe(self, bridge):
        bridge.start()
        bridge.start()  # should warn, not crash
        assert bridge.is_running
        bridge.stop()

    def test_stop_without_start_is_safe(self, bridge):
        bridge.stop()  # should not raise


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_should_reflect_false_when_cannot_allocate(
        self,
        mock_provider,
    ):
        """Budget can_allocate returns False -> should_reflect False."""
        budget = BudgetManager(config=BudgetConfig(total_budget=100000))
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=0, estimated_tokens=200000)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget,
            config=config,
        )
        assert not bridge._should_reflect()

    def test_loop_handles_reflection_error(self, mock_provider, budget_manager):
        """Background loop swallows exceptions from reflect_once."""
        config = ConsciousnessConfig(enabled=True, min_idle_seconds=0)
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
        )
        mock_provider.complete.side_effect = RuntimeError("provider down")
        # Simulate one loop iteration — should not raise
        bridge._stop_event.set()
        bridge._loop()

    def test_gather_input_metrics_exception(self, mock_provider, budget_manager, config):
        """Metrics store exception is swallowed."""
        bad_metrics = MagicMock()
        bad_metrics.get_recent_executions.side_effect = RuntimeError("db gone")
        bridge = ConsciousnessBridge(
            provider=mock_provider,
            budget_manager=budget_manager,
            config=config,
            metrics_store=bad_metrics,
        )
        output = bridge.reflect_once()
        assert output.summary != ""

    def test_gather_input_graph_with_intents(self, mock_provider, budget_manager, config):
        """Graph snapshot with low-stability intents."""
        mock_graph = MagicMock()
        intent_mock = MagicMock()
        intent_mock.stability = 0.2
        intent_mock.to_dict.return_value = {"id": "test", "stability": 0.2}
        snapshot = MagicMock()
        snapshot.intents = [intent_mock]
        mock_graph.snapshot.return_value = snapshot
        with patch(
            "animus_forge.coordination.consciousness_bridge.HAS_CONVERGENT",
            True,
        ):
            bridge = ConsciousnessBridge(
                provider=mock_provider,
                budget_manager=budget_manager,
                config=config,
                graph=mock_graph,
            )
            output = bridge.reflect_once()
            assert output.summary != ""

    def test_gather_input_graph_exception(self, mock_provider, budget_manager, config):
        """Graph snapshot exception is swallowed."""
        mock_graph = MagicMock()
        mock_graph.snapshot.side_effect = RuntimeError("graph broken")
        with patch(
            "animus_forge.coordination.consciousness_bridge.HAS_CONVERGENT",
            True,
        ):
            bridge = ConsciousnessBridge(
                provider=mock_provider,
                budget_manager=budget_manager,
                config=config,
                graph=mock_graph,
            )
            output = bridge.reflect_once()
            assert output.summary != ""

    def test_regex_fallback_invalid_json(self, bridge, mock_provider):
        """Regex extracts braces but content is still invalid JSON."""
        mock_provider.complete.return_value.content = "Look: {not: valid: json:}"
        output = bridge.reflect_once()
        assert output.summary == "Parse failure — skipped"


class TestStatus:
    def test_status_structure(self, bridge):
        s = bridge.status()
        assert "running" in s
        assert "enabled" in s
        assert "reflection_count" in s
        assert "total_tokens" in s
        assert s["last_reflection"] is None

    def test_status_after_reflection(self, bridge):
        bridge.reflect_once()
        s = bridge.status()
        assert s["reflection_count"] == 1
        assert s["total_tokens"] == 450
        assert s["last_reflection"] is not None

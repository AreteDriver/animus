"""Tests for the autoresearch-style evolution loop."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest

from animus_forge.coordination.evolution_loop import (
    BetterMdMissing,
    BudgetExhausted,
    EvolutionConfig,
    EvolutionLoop,
    IterationRecord,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@dataclass
class FakeCompletionResponse:
    content: str
    model: str = "test-model"
    provider: str = "test"
    tokens_used: int = 100
    input_tokens: int = 50
    output_tokens: int = 50
    finish_reason: str = "stop"
    latency_ms: float = 10.0


@pytest.fixture()
def tmp_better(tmp_path: Path) -> Path:
    """Create a temporary better.md."""
    p = tmp_path / "better.md"
    p.write_text(
        "## Definition of Better\n\n"
        "- Reduce workflow execution time by 10%\n"
        "- Improve output quality score above 0.85\n"
    )
    return p


@pytest.fixture()
def tmp_audit(tmp_path: Path) -> Path:
    return tmp_path / "evolution_audit.jsonl"


@pytest.fixture()
def mock_budget() -> MagicMock:
    budget = MagicMock()
    budget.can_allocate.return_value = True
    budget.remaining = 50000
    budget.status = MagicMock()
    budget.status.value = "ok"
    budget.usage_percent = 0.1
    # Make status comparison work
    type(budget).status = PropertyMock(return_value=MagicMock(value="ok"))
    return budget


@pytest.fixture()
def mock_provider() -> MagicMock:
    provider = MagicMock()
    return provider


def _make_hypothesis_response(tokens: int = 100) -> FakeCompletionResponse:
    return FakeCompletionResponse(
        content=json.dumps(
            {
                "hypothesis": "Caching repeated LLM calls will reduce execution time",
                "experiment_plan": "Add memoization to provider.complete() for identical prompts",
                "expected_outcome": "10% fewer LLM calls on repeated workflows",
                "estimated_tokens": 500,
            }
        ),
        tokens_used=tokens,
    )


def _make_eval_response(outcome: str = "keep", tokens: int = 80) -> FakeCompletionResponse:
    return FakeCompletionResponse(
        content=json.dumps(
            {
                "outcome": outcome,
                "rationale": "Caching reduced calls by 15%, exceeding target",
                "confidence": 0.85,
                "suggestions_for_next": "Try cache invalidation strategies",
            }
        ),
        tokens_used=tokens,
    )


def _make_loop(
    provider: MagicMock,
    budget: MagicMock,
    better_path: Path,
    audit_path: Path,
    max_iterations: int = 10,
    experiment_runner=None,
) -> EvolutionLoop:
    config = EvolutionConfig(
        enabled=True,
        max_iterations=max_iterations,
        better_path=better_path,
        audit_log_path=audit_path,
        model="test-model",
    )
    return EvolutionLoop(
        provider=provider,
        budget_manager=budget,
        config=config,
        experiment_runner=experiment_runner,
    )


# ---------------------------------------------------------------------------
# Tests: Initialization & Config
# ---------------------------------------------------------------------------


class TestEvolutionConfig:
    def test_defaults(self):
        config = EvolutionConfig()
        assert config.enabled is False
        assert config.max_iterations == 10
        assert config.budget_pause_threshold == 0.80

    def test_custom_config(self):
        config = EvolutionConfig(
            enabled=True,
            max_iterations=5,
            model="claude-opus-4-6",
        )
        assert config.enabled is True
        assert config.max_iterations == 5
        assert config.model == "claude-opus-4-6"


class TestEvolutionLoopInit:
    def test_init_defaults(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        assert loop.iteration_count == 0
        assert loop.total_tokens == 0
        assert loop.is_running is False
        assert loop.history == []

    def test_status(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        s = loop.status()
        assert s["running"] is False
        assert s["enabled"] is True
        assert s["iteration"] == 0
        assert s["max_iterations"] == 10

    def test_load_custom_principles(
        self, mock_provider, mock_budget, tmp_better, tmp_audit, tmp_path
    ):
        principles_file = tmp_path / "principles.md"
        principles_file.write_text("### P1 — Sovereignty\n### P2 — Continuity\n")
        config = EvolutionConfig(
            enabled=True,
            better_path=tmp_better,
            audit_log_path=tmp_audit,
            principles_path=principles_file,
        )
        loop = EvolutionLoop(mock_provider, mock_budget, config)
        assert len(loop._principles) == 2


# ---------------------------------------------------------------------------
# Tests: better.md validation
# ---------------------------------------------------------------------------


class TestBetterMd:
    def test_missing_better_md_raises(self, mock_provider, mock_budget, tmp_path, tmp_audit):
        missing = tmp_path / "nonexistent.md"
        loop = _make_loop(mock_provider, mock_budget, missing, tmp_audit)
        with pytest.raises(BetterMdMissing, match="not found"):
            loop.run_one()

    def test_empty_better_md_raises(self, mock_provider, mock_budget, tmp_path, tmp_audit):
        empty = tmp_path / "better.md"
        empty.write_text("   \n  ")
        loop = _make_loop(mock_provider, mock_budget, empty, tmp_audit)
        with pytest.raises(BetterMdMissing, match="empty"):
            loop.run_one()

    def test_valid_better_md_loads(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        record = loop.run_one()
        assert record.outcome in ("keep", "discard")


# ---------------------------------------------------------------------------
# Tests: Single iteration
# ---------------------------------------------------------------------------


class TestSingleIteration:
    def test_run_one_keep(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(100),
            _make_eval_response("keep", 80),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        record = loop.run_one()

        assert record.iteration == 0
        assert record.outcome == "keep"
        assert record.budget_used == 180
        assert "Caching" in record.hypothesis
        assert loop.iteration_count == 1
        assert loop.total_tokens == 180

    def test_run_one_discard(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response("discard"),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        record = loop.run_one()
        assert record.outcome == "discard"

    def test_budget_recorded(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        loop.run_one()

        assert mock_budget.record_usage.call_count == 2
        calls = mock_budget.record_usage.call_args_list
        assert calls[0].kwargs["agent_id"] == "evolution_loop"
        assert calls[0].kwargs["operation"] == "generate_hypothesis"
        assert calls[1].kwargs["operation"] == "evaluate"

    def test_audit_log_written(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        loop.run_one()

        assert tmp_audit.exists()
        lines = tmp_audit.read_text().strip().splitlines()
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["iteration"] == 0
        assert entry["outcome"] in ("keep", "discard")
        assert "hypothesis" in entry
        assert "budget_used" in entry
        assert "timestamp" in entry

    def test_history_tracked(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        loop.run_one()

        assert len(loop.history) == 1
        assert loop.history[0].iteration == 0


# ---------------------------------------------------------------------------
# Tests: Budget enforcement
# ---------------------------------------------------------------------------


class TestBudgetEnforcement:
    def test_budget_exhausted_halts(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_budget.can_allocate.return_value = False
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        with pytest.raises(BudgetExhausted):
            loop.run_one()

    def test_budget_threshold_halts_loop(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_budget.usage_percent = 0.85  # above 0.80 threshold
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        assert loop._can_continue() is False

    def test_budget_exceeded_halts_loop(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        from animus_forge.budget.manager import BudgetStatus

        type(mock_budget).status = PropertyMock(return_value=BudgetStatus.EXCEEDED)
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        assert loop._can_continue() is False


# ---------------------------------------------------------------------------
# Tests: Experiment runner
# ---------------------------------------------------------------------------


class TestExperimentRunner:
    def test_custom_runner(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        def runner(hypothesis, plan):
            return f"Result: tested '{hypothesis}' via '{plan}'"

        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(
            mock_provider,
            mock_budget,
            tmp_better,
            tmp_audit,
            experiment_runner=runner,
        )
        record = loop.run_one()
        assert "Result:" in record.experiment_summary

    def test_runner_error_captured(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        def bad_runner(hypothesis, plan):
            raise RuntimeError("experiment failed")

        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(
            mock_provider,
            mock_budget,
            tmp_better,
            tmp_audit,
            experiment_runner=bad_runner,
        )
        record = loop.run_one()
        assert "error" in record.experiment_summary.lower()

    def test_dry_run_default(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        record = loop.run_one()
        assert "[dry run]" in record.experiment_summary


# ---------------------------------------------------------------------------
# Tests: Multi-iteration
# ---------------------------------------------------------------------------


class TestMultiIteration:
    def test_multiple_iterations(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(100),
            _make_eval_response("keep", 80),
            _make_hypothesis_response(120),
            _make_eval_response("discard", 90),
            _make_hypothesis_response(110),
            _make_eval_response("keep", 85),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit, max_iterations=3)
        for _ in range(3):
            loop.run_one()

        assert loop.iteration_count == 3
        assert len(loop.history) == 3
        assert loop.history[0].outcome == "keep"
        assert loop.history[1].outcome == "discard"
        assert loop.history[2].outcome == "keep"

    def test_audit_log_append_only(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        loop.run_one()
        loop.run_one()

        lines = tmp_audit.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["iteration"] == 0
        assert json.loads(lines[1])["iteration"] == 1


# ---------------------------------------------------------------------------
# Tests: JSON parsing
# ---------------------------------------------------------------------------


class TestJsonParsing:
    def test_parse_clean_json(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        result = loop._parse_json('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parse_fenced_json(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        result = loop._parse_json('```json\n{"key": "value"}\n```')
        assert result == {"key": "value"}

    def test_parse_embedded_json(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        result = loop._parse_json('Here is the result:\n{"key": "value"}\nDone.')
        assert result == {"key": "value"}

    def test_parse_invalid_returns_fallback(
        self, mock_provider, mock_budget, tmp_better, tmp_audit
    ):
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        result = loop._parse_json("not json at all")
        assert result["hypothesis"] == "not json at all"
        assert result["_parse_fallback"] is True


# ---------------------------------------------------------------------------
# Tests: Background thread
# ---------------------------------------------------------------------------


class TestBackgroundLoop:
    def test_start_stop(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ] * 10  # enough for max_iterations
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit, max_iterations=2)
        loop.start()
        time.sleep(0.5)
        loop.stop()
        assert loop.is_running is False
        assert loop.iteration_count <= 2

    def test_disabled_does_not_start(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        config = EvolutionConfig(
            enabled=False,
            better_path=tmp_better,
            audit_log_path=tmp_audit,
        )
        loop = EvolutionLoop(mock_provider, mock_budget, config)
        loop.start()
        assert loop.is_running is False

    def test_start_without_better_md(self, mock_provider, mock_budget, tmp_path, tmp_audit):
        missing = tmp_path / "missing.md"
        loop = _make_loop(mock_provider, mock_budget, missing, tmp_audit)
        with pytest.raises(BetterMdMissing):
            loop.start()

    def test_double_start_no_crash(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            _make_hypothesis_response(),
            _make_eval_response(),
        ] * 10
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit, max_iterations=1)
        loop.start()
        loop.start()  # should warn, not crash
        time.sleep(0.3)
        loop.stop()


# ---------------------------------------------------------------------------
# Tests: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_llm_error_does_not_crash_loop(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            RuntimeError("LLM down"),
            _make_hypothesis_response(),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit, max_iterations=2)
        loop.start()
        time.sleep(0.5)
        loop.stop()
        # Should have logged error and continued
        assert loop.iteration_count >= 1

    def test_malformed_llm_output_handled(self, mock_provider, mock_budget, tmp_better, tmp_audit):
        mock_provider.complete.side_effect = [
            FakeCompletionResponse(content="not valid json", tokens_used=50),
            _make_eval_response(),
        ]
        loop = _make_loop(mock_provider, mock_budget, tmp_better, tmp_audit)
        record = loop.run_one()
        # Should complete without crashing, hypothesis will be empty
        assert record.iteration == 0


# ---------------------------------------------------------------------------
# Tests: IterationRecord
# ---------------------------------------------------------------------------


class TestIterationRecord:
    def test_fields(self):
        record = IterationRecord(
            iteration=0,
            hypothesis="test",
            experiment_summary="summary",
            outcome="keep",
            rationale="because",
            budget_used=100,
        )
        assert record.iteration == 0
        assert record.outcome == "keep"
        assert record.timestamp  # auto-generated

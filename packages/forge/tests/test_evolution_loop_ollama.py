"""Live Ollama smoke tests for the evolution loop.

Requires a running Ollama instance. Excluded from CI via conftest.py collect_ignore.
Run manually: pytest tests/test_evolution_loop_ollama.py -v -s
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# Skip entire module if Ollama is not reachable
try:
    import urllib.request

    urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

pytestmark = pytest.mark.skipif(not OLLAMA_AVAILABLE, reason="Ollama not running")


@pytest.fixture()
def better_md(tmp_path: Path) -> Path:
    p = tmp_path / "better.md"
    p.write_text(
        "# Definition of Better\n\n"
        "- Reduce unnecessary verbosity in workflow step descriptions\n"
        "- Ensure all steps have clear, actionable instructions\n"
        "- Token efficiency: say more with less\n"
    )
    return p


@pytest.fixture()
def audit_log(tmp_path: Path) -> Path:
    return tmp_path / "evolution_audit.jsonl"


@pytest.fixture()
def ollama_provider():
    """Create a real Ollama provider."""
    from animus_forge.providers.ollama_provider import OllamaProvider

    host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
    model = os.environ.get("OLLAMA_MODEL", "llama3.1:8b")
    return OllamaProvider(host=host, model=model)


@pytest.fixture()
def budget_manager():
    from animus_forge.budget.manager import BudgetConfig, BudgetManager

    config = BudgetConfig(total_budget=50000)
    return BudgetManager(config=config)


class TestEvolutionLoopOllama:
    """Live integration tests with Ollama."""

    def test_single_iteration_live(self, ollama_provider, budget_manager, better_md, audit_log):
        """Run one real evolution iteration against Ollama."""
        from animus_forge.coordination.evolution_loop import EvolutionConfig, EvolutionLoop

        config = EvolutionConfig(
            enabled=True,
            max_iterations=1,
            model="llama3.1:8b",
            better_path=better_md,
            audit_log_path=audit_log,
            estimated_tokens_per_iteration=5000,
        )

        loop = EvolutionLoop(
            provider=ollama_provider,
            budget_manager=budget_manager,
            config=config,
        )

        start = time.time()
        record = loop.run_one()
        elapsed = time.time() - start

        # Verify structured output
        assert record.iteration == 0
        assert record.outcome in ("keep", "discard", "")
        # hypothesis may be empty if LLM returned non-JSON (parse fallback)
        assert record.budget_used > 0
        assert loop.total_tokens > 0

        # Verify audit log
        assert audit_log.exists()
        entry = json.loads(audit_log.read_text().strip())
        assert entry["iteration"] == 0
        assert "hypothesis" in entry

        print(f"\n--- Evolution Iteration 0 ({elapsed:.1f}s) ---")
        print(f"Hypothesis: {record.hypothesis}")
        print(f"Outcome: {record.outcome}")
        print(f"Rationale: {record.rationale}")
        print(f"Tokens used: {record.budget_used}")
        print(f"Experiment: {record.experiment_summary[:200]}")

    def test_two_iterations_with_history(self, ollama_provider, budget_manager, better_md, audit_log):
        """Verify the loop passes prior results to subsequent iterations."""
        from animus_forge.coordination.evolution_loop import EvolutionConfig, EvolutionLoop

        config = EvolutionConfig(
            enabled=True,
            max_iterations=2,
            model="llama3.1:8b",
            better_path=better_md,
            audit_log_path=audit_log,
            estimated_tokens_per_iteration=5000,
        )

        loop = EvolutionLoop(
            provider=ollama_provider,
            budget_manager=budget_manager,
            config=config,
        )

        r1 = loop.run_one()
        r2 = loop.run_one()

        assert r1.iteration == 0
        assert r2.iteration == 1
        assert len(loop.history) == 2
        assert loop.total_tokens > r1.budget_used  # second iter added tokens

        # Audit log should have 2 lines
        lines = audit_log.read_text().strip().splitlines()
        assert len(lines) == 2

        print(f"\n--- Iteration 0: {r1.outcome} ---")
        print(f"  {r1.hypothesis[:100]}")
        print(f"--- Iteration 1: {r2.outcome} ---")
        print(f"  {r2.hypothesis[:100]}")
        print(f"Total tokens: {loop.total_tokens}")

    def test_budget_tracking_accurate(self, ollama_provider, budget_manager, better_md, audit_log):
        """Verify budget manager reflects actual usage."""
        from animus_forge.coordination.evolution_loop import EvolutionConfig, EvolutionLoop

        config = EvolutionConfig(
            enabled=True,
            max_iterations=1,
            model="llama3.1:8b",
            better_path=better_md,
            audit_log_path=audit_log,
            estimated_tokens_per_iteration=5000,
        )

        loop = EvolutionLoop(
            provider=ollama_provider,
            budget_manager=budget_manager,
            config=config,
        )

        loop.run_one()

        # Budget manager should have recorded usage
        assert budget_manager._total_used > 0
        assert len(budget_manager._usage_history) >= 2  # hypothesis + evaluate
        operations = [r.operation for r in budget_manager._usage_history]
        assert "generate_hypothesis" in operations
        assert "evaluate" in operations

        print(f"\nBudget used: {budget_manager._total_used} / {budget_manager.config.total_budget}")
        print(f"Usage records: {len(budget_manager._usage_history)}")

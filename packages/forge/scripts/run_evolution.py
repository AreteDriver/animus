#!/usr/bin/env python3
"""Run one iteration of the Animus Forge evolution loop.

Usage:
    cd /home/arete/projects/animus/packages/forge
    .venv/bin/python scripts/run_evolution.py [--iterations N]
"""

import argparse
import sys
from pathlib import Path

# Ensure forge package is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from animus_forge.budget.manager import BudgetConfig, BudgetManager
from animus_forge.coordination.evolution_loop import EvolutionConfig, EvolutionLoop
from animus_forge.providers.ollama_provider import OllamaProvider


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Animus Forge evolution loop")
    parser.add_argument("--iterations", type=int, default=1, help="Number of iterations (default: 1)")
    parser.add_argument("--model", default="qwen2.5:14b", help="Ollama model (default: qwen2.5:14b)")
    parser.add_argument("--budget", type=int, default=100_000, help="Token budget (default: 100000)")
    args = parser.parse_args()

    forge_root = Path(__file__).parent.parent
    better_path = forge_root / "forge" / "better.md"
    audit_path = forge_root / "forge" / "forge_audit.jsonl"

    if not better_path.exists():
        print(f"ERROR: better.md not found at {better_path}")
        sys.exit(1)

    print(f"Evolution Loop Configuration:")
    print(f"  Model: {args.model}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Budget: {args.budget:,} tokens")
    print(f"  Better: {better_path}")
    print(f"  Audit: {audit_path}")
    print()

    provider = OllamaProvider(model=args.model)
    budget = BudgetManager(config=BudgetConfig(total_budget=args.budget))
    config = EvolutionConfig(
        enabled=True,
        max_iterations=args.iterations,
        model=args.model,
        better_path=better_path,
        audit_log_path=audit_path,
    )

    loop = EvolutionLoop(provider=provider, budget_manager=budget, config=config)

    for i in range(args.iterations):
        print(f"--- Iteration {i + 1}/{args.iterations} ---")
        try:
            record = loop.run_one()
            print(f"  Hypothesis: {record.hypothesis[:100]}...")
            print(f"  Outcome: {record.outcome}")
            print(f"  Budget used: {record.budget_used} tokens")
            print()
        except Exception as e:
            print(f"  ERROR: {e}")
            break

    print(f"Budget status: {budget.status.value}")
    print(f"Budget remaining: {budget.remaining:,} / {args.budget:,} tokens")
    print(f"Audit log: {audit_path}")


if __name__ == "__main__":
    main()

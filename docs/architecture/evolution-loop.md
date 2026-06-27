# Evolution Loop — Forge Autonomous Improvement

## Overview

The evolution loop is Forge's autoresearch-style improvement engine. It runs
autonomous hypothesis-experiment-evaluate cycles, constrained by a human-authored
definition of "better" and gated by the budget manager.

**Module**: `animus_forge.coordination.evolution_loop`

## Architecture

```
load_better_md() →
  generate_hypothesis() →      [LLM call, budget-gated]
    run_experiment() →          [pluggable runner]
      evaluate_against_better() → [LLM call, budget-gated]
        keep_or_discard() →
          append_to_audit_jsonl() →
            repeat_or_halt()
```

## TPS Mapping

The evolution loop maps directly onto Toyota Production System principles:

| TPS Concept | Forge Implementation | Purpose |
|---|---|---|
| **Standard Work** | `better.md` | Human-authored definition of "better" — required input, hard stop if missing |
| **Kaizen loop** | The iteration cycle | Continuous improvement through hypothesis → test → evaluate |
| **Jidoka** | Budget gate + halt conditions | Authority to stop the line when something is wrong (budget exceeded, better.md stale) |
| **Poka-yoke** | Max iterations, threshold caps | Guard rails preventing runaway iteration or budget drain |
| **Andon cord** | `stop()` + error logging | Any component can signal halt; errors are logged, not swallowed |

The human's role: **write Standard Work (better.md) and define what "better" means.**
The agent's role: **iterate until better is achieved or constraints are hit.**

## Key Constraints

1. `better.md` must exist and be non-empty — hard stop otherwise
2. All LLM calls route through `BudgetManager` — no exceptions
3. Forge may never write to Core files directly
4. `evolution_audit.jsonl` is append-only — never overwrite, never delete entries
5. Max iterations per session: configurable (default 10)
6. Budget pause threshold: configurable (default 80%)

## Configuration

```python
from animus_forge.coordination import EvolutionConfig, EvolutionLoop

config = EvolutionConfig(
    enabled=True,              # opt-in only
    max_iterations=10,         # safety cap
    model="claude-sonnet-4-6",
    better_path=Path("forge/better.md"),
    audit_log_path=Path("forge/evolution_audit.jsonl"),
    budget_pause_threshold=0.80,
)
```

## Audit Log Format

Each iteration produces a structured JSONL entry:

```json
{
  "iteration": 0,
  "hypothesis": "Caching repeated LLM calls will reduce execution time",
  "experiment_summary": "[dry run] Plan executed: Add memoization...",
  "outcome": "keep",
  "rationale": "Caching reduced calls by 15%, exceeding target",
  "budget_used": 180,
  "timestamp": "2026-03-10T04:30:00+00:00"
}
```

## Related

- [Consciousness-Quorum Bridge](CONSCIOUSNESS_QUORUM_BRIDGE.md) — reflection loop (structural template)
- [Workflow Evolution Constraints](WORKFLOW_EVOLUTION_CONSTRAINTS.md) — YAML-only fast path
- [Constitutional Principles](CONSTITUTIONAL_PRINCIPLES.md) — P4 (Constraint), P6 (Budget Sovereignty), P8 (Jidoka)

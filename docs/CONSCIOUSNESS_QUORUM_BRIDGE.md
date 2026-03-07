# Consciousness-Quorum Bridge — Integration Spec

> Wiring the existing reflection loop to Quorum's intent graph. This is the novel integration — reflection outputs become coordination signals.

## What This Adds

Bootstrap already has a reflection loop (APScheduler cron, `LEARNED.md` updates). Core has a proactive engine with scheduled checks. What's **missing** is the bridge: reflection outputs should flow into Quorum's intent graph as low-stability nodes that need agent consensus to stabilize.

This turns Animus from "reflects and logs" into "reflects, coordinates, and surfaces unresolved tensions."

---

## Architecture (Existing Components)

| Component | Location | Role |
|-----------|----------|------|
| Bootstrap reflection | `packages/bootstrap/` | Cron-triggered, updates LEARNED.md |
| Core proactive engine | `packages/core/animus/proactive.py` | Scheduled checks, quiet hours |
| Core learning system | `packages/core/animus/learning/` | Pattern detection, approval queues |
| AnimusIdentity | `packages/core/animus/identity.py` | `record_reflection()`, reflection_count |
| Quorum intent graph | `packages/quorum/python/convergent/intent.py` | InterfaceSpec, Constraint, Evidence |
| Quorum event log | `packages/quorum/python/convergent/event_log.py` | Coordination event timeline |
| Quorum stigmergy | `packages/quorum/python/convergent/stigmergy.py` | Indirect coordination via environment |
| Quorum stability/scoring | `packages/quorum/python/convergent/scoring.py` | Phi-weighted scoring, decay |
| Forge monitoring | `packages/forge/src/animus_forge/monitoring/` | Structured execution logs |
| Forge budget | `packages/forge/src/animus_forge/budget/manager.py` | BudgetManager, threshold callbacks |

---

## The Bridge: ReflectionOutput -> Quorum IntentNodes

### Data Flow

```
Forge audit logs + open IntentNodes + P1-P9 principles
        |
        v
  [Reflection LLM call]  (Sonnet — never Opus for routine reflection)
        |
        v
  ReflectionOutput
    ├── insights[]          --> write_intent(node_type="reflection_surface", stability=0.3)
    ├── principle_tensions[] --> write_intent(node_type="principle_tension", stability=0.1)
    ├── workflow_patch_ids[] --> append to workflow_review_queue (not auto-applied)
    └── summary             --> append to logs/reflections.jsonl
```

### Trigger Conditions

The bridge fires when ALL of these are true:
- No active Forge workers (task queue empty)
- `BudgetManager.status` is `OK` (not WARNING/CRITICAL/EXCEEDED)
- At least `MIN_IDLE_SECONDS` since last reflection (default: 300s)
- Consciousness mode is enabled (off by default — user opts in)

The bridge does NOT fire when:
- Self-improve orchestrator is in any stage other than IDLE
- Budget is at or above warning threshold (75%)
- System is shutting down

### Proposed Models

These extend the existing reflection tracking — they don't replace it.

```python
from pydantic import BaseModel

class ReflectionInput(BaseModel):
    """Assembled from existing sources — no new data stores needed."""
    recent_actions: list[dict]       # From Forge monitoring logs
    open_intent_nodes: list[dict]    # Quorum nodes below stability threshold
    identity_principles: list[str]   # P1-P9 from CONSTITUTIONAL_PRINCIPLES.md
    session_budget_remaining: float  # From BudgetManager.remaining

class ReflectionOutput(BaseModel):
    """Structured output from reflection LLM call."""
    summary: str
    insights: list[str]
    proposed_intent_updates: list[dict]   # {content, tags, stability}
    workflow_patch_ids: list[str]         # Workflow IDs flagged for review
    principle_tensions: list[str]         # Actions close to violating P1-P9
    next_reflection_in: int              # Suggested seconds until next cycle
```

### Quorum Integration

Reflection insights become low-stability intent nodes:

```python
from convergent.intent import Intent, InterfaceSpec, InterfaceKind

# Each insight becomes a coordination signal
intent = Intent(
    agent_id="consciousness_bridge",
    intent=insight_text,
    provides=[InterfaceSpec(
        name="reflection_surface",
        kind=InterfaceKind.CONFIG,
        signature="",
        tags=["reflection", "auto-generated"],
    )],
)
# Published via VersionedGraph.publish(intent)
# Initial stability is LOW — needs agent consensus to stabilize
```

Principle tensions get even lower stability and a `principle_tension` tag — these are surfaced on `animus status` as unresolved items requiring human attention.

### Workflow Review Queue

Workflow IDs flagged by reflection are NOT auto-patched. They're appended to a review queue and surfaced on next `animus status` call:

```
Pending workflow reviews (from reflection):
  - code-review v1.0: "12% failure rate on security_review step — consider retry"
  - feature-build v1.0: "Estimated tokens consistently 40% under actual"
```

The owner decides whether to evolve these workflows via the self-improve pipeline.

---

## System Prompt for Reflection Call

```
You are the reflective layer of Animus, a sovereign personal AI exocortex.

Review recent actions and identify patterns. You do not execute tasks — you observe
and synthesize.

Constraints:
- You serve one user. Your insights are private.
- You may propose intent updates, but never execute them directly.
- You may flag principle tensions, but never resolve them unilaterally.
- Be concise. One reflection pass, not an essay. You're on a budget.

Current principles:
{principles}

Recent actions ({n} records):
{audit_records}

Open intent nodes (unresolved coordination):
{open_nodes}

Budget remaining this session: ${budget_remaining:.4f}

Respond in the ReflectionOutput JSON schema.
```

---

## Implementation Notes

- Use Sonnet for reflection, not Opus. Proportionality (P5).
- If LLM returns malformed JSON, log the error and skip — don't crash the loop.
- Reflection must go through `BudgetManager.can_allocate()` before the LLM call.
- The bridge thread must be registered so `animus stop` terminates it cleanly.
- `logs/reflections.jsonl` is separate from Forge execution logs — different consumers.
- Principle tensions with no resolution after 3 reflection cycles should be escalated (surfaced more prominently in CLI status output).

---

## CLI Surface

These extend existing CLI commands — not new sub-commands:

```
animus status          # Already exists. Add: pending reflections, open tensions, review queue
animus bg on/off       # Toggle consciousness bridge (off by default)
animus bg status       # Running state, last reflection time, cycle count, total cost
animus bg reflect      # Trigger one reflection cycle immediately (bypass idle timer)
```

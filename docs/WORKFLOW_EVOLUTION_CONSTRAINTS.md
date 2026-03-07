# Workflow Evolution Constraints

> Safety rules and changelog pattern for YAML workflow self-modification. This supplements the existing `self_improve/` pipeline — it does NOT replace it.

## Context

The `self_improve/orchestrator.py` handles the full pipeline: analyze, plan, sandbox, test, apply, PR, merge — with 3 human approval gates and rollback. That pipeline is designed for **Python code changes**.

YAML workflow evolution is a **fast path** — lower risk, tighter scope, faster iteration. Forge can propose patches to its own workflow definitions without the full sandbox/PR pipeline, because YAML changes can't introduce code-level bugs.

---

## Scope Boundary

- Forge MAY propose changes to `packages/forge/workflows/*.yaml`
- Forge MAY NOT modify any `.py` file via this path (use `self_improve/` for that)
- Forge MAY NOT modify `packages/core/` or `packages/quorum/` via any self-modification path
- Forge MAY NOT modify `packages/bootstrap/` identity files (use `IdentityProposalManager`)

---

## YAML Validation Rules

Before any workflow patch is staged, validate:

```python
def validate_workflow_patch(current_yaml: dict, proposed_yaml: str) -> None:
    """Raise WorkflowPatchInvalid on any violation."""
    parsed = yaml.safe_load(proposed_yaml)

    # Required fields present
    assert all(k in parsed for k in ["name", "version", "steps"])

    # Version must increment (semver string comparison)
    assert _version_gt(parsed["version"], current_yaml["version"])

    # No code injection in YAML
    assert "```python" not in proposed_yaml
    assert "exec(" not in proposed_yaml
    assert "eval(" not in proposed_yaml
    assert "import " not in proposed_yaml
    assert "__import__" not in proposed_yaml

    # Tool names must be snake_case (prevents shell injection)
    for step in parsed["steps"]:
        assert re.match(r'^[a-z_]+$', step["type"]), \
            f"Step type must be snake_case: {step['type']}"

    # Budget estimate present if LLM steps exist
    llm_types = {"claude_code", "openai", "ollama"}
    if any(s["type"] in llm_types for s in parsed["steps"]):
        assert "token_budget" in parsed, "Workflows with LLM steps require token_budget"
```

---

## Evolution Notes

When a workflow is patched (via any path — self-improve or fast-path), append to its `evolution_notes` field. This is the workflow's own changelog, readable by Forge during task planning.

### Format

Add this field to workflow YAML files:

```yaml
# At bottom of workflow file
evolution_notes:
  - version: "1.1"
    date: 2026-03-07
    change: "Added retry on security_review step timeout"
    reasoning: "Audit log showed 12% failure rate due to LLM timeouts"
    proposed_by: "consciousness_bridge"  # or "human" or task_id
```

### Why This Matters

- Forge can read a workflow's evolution history before executing it
- The consciousness-quorum bridge (see `CONSCIOUSNESS_QUORUM_BRIDGE.md`) uses this to detect stale or underperforming workflows
- It creates an audit trail without requiring git log archaeology

---

## Staging Pattern

Workflow patches use a `.pending.yaml` staging file:

1. Proposed patch written to `workflows/{id}.pending.yaml`
2. Surfaced on next `animus status` or `animus evolve status`
3. Owner approves (`animus evolve approve {id}`) or rejects (`animus evolve reject {id} "reason"`)
4. On approve: replace live file, append to `evolution_notes`, log to Forge monitoring
5. On reject: delete `.pending.yaml`, log rejection reason

If `.pending.yaml` files exist at startup, surface them before any task execution — an interrupted evolution needs resolution.

---

## Integration with Existing Self-Improve

The YAML fast path and the full `SelfImproveOrchestrator` are complementary:

| Aspect | YAML Fast Path | Full Pipeline (`self_improve/`) |
|--------|---------------|--------------------------------|
| Scope | `workflows/*.yaml` only | Any Python/config file |
| Sandbox | Not needed (YAML can't break code) | Required (`sandbox.py`) |
| Approval | Single gate (approve/reject) | 3 gates (plan/apply/merge) |
| Rollback | Git revert on the YAML file | `RollbackManager` snapshots |
| PR | Not needed | `PRManager` creates branch + PR |
| Trigger | Consciousness bridge or manual | `SelfImproveOrchestrator.run()` |
| Budget | Minimal (validation only, no LLM) | Full pre-flight + execution budget |

The consciousness-quorum bridge may propose YAML patches based on audit log analysis. These go through the YAML fast path. If a workflow problem requires Python-level changes, it gets routed to the full pipeline.

---

## CLI Commands

```
animus evolve status               # List pending .pending.yaml files
animus evolve approve [id]         # Apply pending patch, append evolution_notes
animus evolve reject [id] [reason] # Delete pending patch, log rejection
animus evolve history [id]         # Show evolution_notes for a workflow
animus evolve list                 # List all workflows with version and last-evolved date
```

---

## Safety Notes

- Forge should never propose a patch to a workflow it is currently executing
- Evolution mode (`--evolve` flag) enables autonomous patch proposals — off by default
- The consciousness bridge is the primary driver of evolution proposals — it reads audit logs for performance evidence
- All evolution actions emit to Forge monitoring logs (Transparency — P3)

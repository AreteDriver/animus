# Forge / Kernel WorkflowExecutor Deduplication Plan

**ADL:** ADL-20260723-001 (Bootstrap → Operations Center)
**P0 Finding:** Forge and Kernel have competing `WorkflowExecutor` implementations
**Status:** Dependency declared; file migration planned
**Author:** Claude Code session 2026-07-23

---

## Current State

| Metric | Value |
|--------|-------|
| Duplicated modules | 15 |
| Forge has counterpart | 15/15 (100%) |
| Identical modulo imports | 4/15 (26.7%) |
| Total duplicated bytes | 178,774 |
| Identical modules | `executor_arete.py`, `executor_error.py`, `executor_loop.py`, `executor_results.py` |

### Structural Similarity

Every module differs **only** in:
1. **Import paths** — `animus_kernel.*` → `animus_forge.*`
2. **Minor syntax** — `contextlib.suppress(Exception)` vs `try/except Exception: pass` (2 occurrences in `executor_agents.py`)
3. **Type annotations** — `dict | None = None` vs `dict = None` (3 occurrences)

There is **no behavioral divergence** in the execution framework itself.

### Why Re-export Doesn’t Work (Yet)

The `WorkflowExecutor` class is a mixin-composition:

```python
class WorkflowExecutor(
    StepExecutionMixin,      # _execute_step, _execute_shell, …
    ErrorHandlerMixin,       # _handle_error, fallback logic
    AIHandlersMixin,         # _execute_claude_code, _execute_openai, …
    …
):
```

Each mixin imports package-specific modules:
- `executor_agents.py` → `animus_kernel.agents.autonomy` (or Forge equivalent)
- `executor_ai.py` → `animus_kernel.providers.base` (or Forge equivalent)
- `executor_core.py` → `animus_kernel.db.get_task_store` (or Forge equivalent)

If Forge simply re-exported Kernel’s `WorkflowExecutor`, Forge tests that patch `animus_forge.db.get_task_store` would silently fail because the class would be importing from `animus_kernel.db`.

---

## Migration Path (3 Phases)

### Phase 1: Dependency + Audit (DONE)

- [x] Add `animus-kernel` to Forge `pyproject.toml` dependencies
- [x] Create `scripts/audit_executor_duplication.py` — quantifies duplication surface
- [x] Verify no regressions in Forge executor tests (14/14 `test_executor_streaming.py` green)

### Phase 2: Extract Shared Base (Target: 2026-08)

Goal: Move execution framework into a **shared package** (`packages/executor/`) that:
- Defines `WorkflowExecutor` with all mixins
- Uses **dependency injection** or a **plugin registry** for package-specific handlers
- Has **zero** imports of `animus_kernel.*` or `animus_forge.*`
- Is depended on by both Kernel and Forge

**Why a new shared package instead of Kernel-only?**
The mixins import package-specific implementations (agents, providers, db, budget). Moving them all into Kernel would create an unacceptable coupling direction (Kernel → Forge internals). A shared base with a plugin/registry pattern keeps the boundary clean.

**Concrete steps:**
1. Create `packages/executor/` with the shared execution framework
2. Refactor mixin imports to use a `HandlerRegistry` instead of direct module imports
3. Both Kernel and Forge register their handlers at runtime:
   ```python
   from animus_executor import WorkflowExecutor, HandlerRegistry
   registry = HandlerRegistry()
   registry.register("shell", my_shell_handler)
   executor = WorkflowExecutor(handler_registry=registry)
   ```
4. Update both packages’ `pyproject.toml` to depend on `animus-executor`
5. Delete duplicated modules from both Kernel and Forge

### Phase 3: Test Patch Migration (Target: 2026-08)

Forge tests that patch `animus_forge.db.get_task_store` (e.g., `test_executor_history.py`) must be updated to:
- Patch the shared registry entry, OR
- Patch the new shared module path

Estimated test files affected: 6–8 files.

---

## Decision Record

**We chose NOT to do a simple re-export today** because:
1. Import path differences mean test patches would silently break
2. The mixin architecture embeds package-specific imports deep in the class hierarchy
3. A shared package with a plugin registry is the architecturally correct fix
4. The risk of a rushed migration (10,431 Forge tests) outweighs the benefit of an immediate cosmetic change

**What we DID do today:**
1. Added the dependency arrow (`animus-forge` → `animus-kernel`) in `pyproject.toml`
2. Quantified the duplication surface (15 modules, 178 KB)
3. Verified no regressions
4. Documented the blocker and the 3-phase path

---

## Acceptance Criteria (Done Definition)

- [ ] `packages/executor/` exists with shared execution framework
- [ ] Kernel and Forge both depend on `animus-executor`
- [ ] Zero `executor_*.py` modules remain in `packages/kernel/src/animus_kernel/executor/`
- [ ] Zero `executor_*.py` modules remain in `packages/forge/src/animus_forge/workflow/`
- [ ] All Forge executor tests green (10,431 total)
- [ ] All Kernel tests green (493 total)
- [ ] ADL-20260723-001 updated to mark this P0 as resolved

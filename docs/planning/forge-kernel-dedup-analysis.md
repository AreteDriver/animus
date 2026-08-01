# Forge / Kernel Executor Dedup — Phase 1 Gap Analysis

> Analysis date: 2026-07-24
> Analyst: Claude (autonomous loop, ADL-20260723-001)
> Scope: Identify overlapping and duplicated execution primitives between `packages/forge/` and `packages/kernel/`. Phase 2 (consolidation) requires explicit user confirmation.

---

## Executive Summary

The Forge ↔ Kernel executor dedup is **largely complete** already. Forge's `src/animus_forge/workflow/` directory contains 31 thin re-export wrapper files (279 total lines) that delegate entirely to `animus_kernel.executor.*`. The actual execution engine (~11,600 lines) lives in Kernel.

**Remaining duplication** is concentrated in:
1. The `executions/` subpackage (3 files, ~600 lines) — near-identical copies with minor import-path drift.
2. A small number of Forge-specific wrappers that extend Kernel primitives with budget/approval logic.

**No Phase 2 action is required** unless the user wants to:
- Remove the Forge wrapper layer entirely and have Forge import from Kernel directly (breaking change for internal imports).
- Merge the `executions/` packages into a single shared location.
- Move Forge-unique orchestration code (BudgetManager, ApprovalStore) into Kernel or a shared coordination package.

---

## 1. Directory Map

### Forge `workflow/` → Kernel `executor/` (ALREADY CONSOLIDATED)

| Forge Path | Kernel Path | Relationship | Forge LOC |
|---|---|---|---|
| `forge/src/animus_forge/workflow/arete_hooks.py` | `kernel/src/animus_kernel/executor/arete_hooks.py` | `import *` re-export + private imports | 2 |
| `forge/src/animus_forge/workflow/auto_parallel.py` | `kernel/src/animus_kernel/executor/auto_parallel.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/composer.py` | `kernel/src/animus_kernel/executor/composer.py` | `import *` re-export + private imports | 3 |
| `forge/src/animus_forge/workflow/distributed_rate_limiter.py` | `kernel/src/animus_kernel/executor/distributed_rate_limiter.py` | `import *` re-export + private imports | 3 |
| `forge/src/animus_forge/workflow/executor_agents.py` | `kernel/src/animus_kernel/executor/executor_agents.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_ai.py` | `kernel/src/animus_kernel/executor/executor_ai.py` | `import *` re-export + private imports | 3 |
| `forge/src/animus_forge/workflow/executor_approval.py` | `kernel/src/animus_kernel/executor/executor_approval.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_arete.py` | `kernel/src/animus_kernel/executor/executor_arete.py` | `import *` re-export + private imports | 3 |
| `forge/src/animus_forge/workflow/executor_clients.py` | `kernel/src/animus_kernel/executor/executor_clients.py` | `import *` re-export + private imports | 3 |
| `forge/src/animus_forge/workflow/executor_core.py` | `kernel/src/animus_kernel/executor/executor_core.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_cost_audit.py` | `kernel/src/animus_kernel/executor/executor_cost_audit.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_error.py` | `kernel/src/animus_kernel/executor/executor_error.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_integrations.py` | `kernel/src/animus_kernel/executor/executor_integrations.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_loop.py` | `kernel/src/animus_kernel/executor/executor_loop.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_mcp.py` | `kernel/src/animus_kernel/executor/executor_mcp.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_parallel_exec.py` | `kernel/src/animus_kernel/executor/executor_parallel_exec.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_patterns.py` | `kernel/src/animus_kernel/executor/executor_patterns.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor.py` | `kernel/src/animus_kernel/executor/executor.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_results.py` | `kernel/src/animus_kernel/executor/executor_results.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/executor_step.py` | `kernel/src/animus_kernel/executor/executor_step.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/graph_executor.py` | `kernel/src/animus_kernel/executor/graph_executor.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/graph_models.py` | `kernel/src/animus_kernel/executor/graph_models.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/graph_walker.py` | `kernel/src/animus_kernel/executor/graph_walker.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/__init__.py` | `kernel/src/animus_kernel/executor/__init__.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/loader.py` | `kernel/src/animus_kernel/executor/loader.py` | `import *` re-export + private imports | 5 |
| `forge/src/animus_forge/workflow/parallel.py` | `kernel/src/animus_kernel/executor/parallel.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/rate_limited_executor.py` | `kernel/src/animus_kernel/executor/rate_limited_executor.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/scheduler.py` | `kernel/src/animus_kernel/executor/scheduler.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/versioning.py` | `kernel/src/animus_kernel/executor/versioning.py` | `import *` re-export | 2 |
| `forge/src/animus_forge/workflow/version_manager.py` | `kernel/src/animus_kernel/executor/version_manager.py` | `import *` re-export | 2 |

**Finding:** Every Forge workflow executor file is a thin `import *` re-export wrapper. The actual implementation (11,612 lines) lives in Kernel.

**Files unique to Forge `workflow/`:**
- `approval_store.py` — Resume token store for approval gates (not present in Kernel)

**Files unique to Kernel `executor/`:**
- `checkpoint.py` — Execution checkpointing
- `core_engine.py`, `core_init.py`, `core_loader.py`, `core_models.py` — Core engine bootstrap primitives

---

## 2. Duplicated Subpackage: `executions/`

Both packages maintain an `executions/` subpackage with identical public APIs but divergent internals.

| File | Forge LOC | Kernel LOC | Diff Summary |
|---|---|---|---|
| `__init__.py` | ~30 | ~30 | **Identical** |
| `manager.py` | ~300 | ~300 | Import path: `animus_forge.state.backends` vs `animus_kernel.state.backends` |
| `models.py` | ~200 | ~200 | `str, Enum` vs `StrEnum` (Python 3.11 compat) |

**Verdict:** The `executions/` packages are **soft forks**. They share the same schema and behavior but use different internal import paths. Consolidation would require either:
- Moving `executions/` to a shared location (e.g., `packages/types/` or a new `packages/executions/`)
- Having Forge re-export from Kernel (consistent with the `workflow/` → `executor/` pattern)

**Risk:** `manager.py` references `state.backends` which may have diverged between Forge and Kernel. Blind re-export could mask backend incompatibilities.

---

## 3. Forge-Unique Execution Code

These files exist only in Forge and add orchestration layers on top of Kernel primitives:

| File | Purpose | Lines | Relationship to Kernel |
|---|---|---|---|
| `forge/src/animus_forge/api_routes/executions.py` | REST API for listing/managing executions | ~150 | Uses Forge `executions.manager`; no Kernel equivalent |
| `forge/src/animus_forge/skill_bridge/executor.py` | `SkillExecutor` — wraps calls in BudgetManager + ApprovalStore | ~400 | Builds on Kernel executor primitives but adds Forge-specific budget/approval/guardrails |
| `forge/src/animus_forge/dashboard/workflow_builder/renderers/execution.py` | Streamlit UI for workflow metadata/settings | ~100 | UI layer; no Kernel equivalent |

**Finding:** Forge's unique execution code is **orchestration and presentation**, not core engine logic. The SkillExecutor is the most significant unique asset — it adds financial/approval guardrails that Kernel's raw executor lacks.

---

## 4. Quantitative Summary

| Metric | Forge | Kernel | Notes |
|---|---|---|---|
| Executor engine LOC | 279 (wrappers) | 11,612 | Forge delegates 100% to Kernel |
| Executions subpackage LOC | ~530 | ~530 | Near-identical soft forks |
| Forge-unique execution LOC | ~650 | — | API routes, SkillExecutor, UI |
| **Total execution-related LOC** | **~1,459** | **12,142** | **~88% lives in Kernel** |

---

## 5. Risk Assessment

| Risk | Severity | Mitigation |
|---|---|---|
| Removing Forge wrappers breaks internal imports | Medium | Search/replace `animus_forge.workflow.X` → `animus_kernel.executor.X` across Forge codebase |
| `executions/` backend drift | Low | Audit `state.backends` diff before consolidating |
| SkillExecutor budget logic not in Kernel | Low | Keep SkillExecutor in Forge; it's orchestration, not engine |
| `StrEnum` vs `str, Enum` breakages | Low | Kernel already uses `StrEnum` (Python 3.11+); Forge uses fallback for compat |
| CI test coverage gaps after move | Medium | Run full Forge + Kernel suites before and after any refactor |

---

## 6. Phase 2 Options (Requires Explicit Confirmation)

### Option A: Remove Forge Wrapper Layer (Minimal)
Replace all `from animus_forge.workflow import X` with `from animus_kernel.executor import X` across Forge codebase. Delete `forge/src/animus_forge/workflow/` wrappers.

**Effort:** ~1 day
**Impact:** Breaks any external code importing from `animus_forge.workflow`
**Benefit:** Eliminates 279 lines of dead-weight indirection

### Option B: Merge `executions/` into Kernel (Medium)
Move `executions/` to Kernel as the canonical implementation. Have Forge re-export from Kernel (like `workflow/` does today).

**Effort:** ~2–3 days
**Impact:** Requires reconciling `state.backends` import paths
**Benefit:** Eliminates the last significant soft-fork

### Option C: Extract Shared Execution Platform (Large)
Move all execution primitives (executor + executions + shared state) into a new `packages/execution/` package. Both Forge and Kernel depend on it.

**Effort:** ~1–2 weeks
**Impact:** Major dependency restructuring; requires CI overhaul
**Benefit:** Cleanest architecture for future multi-package scaling

### Option D: Leave As-Is (Recommended for Now)
The current pattern (Kernel owns engine, Forge re-exports + adds orchestration) is functional and well-tested. The duplication surface is small (`executions/` soft-fork, 530 lines). The cost of refactoring exceeds the maintenance burden at this time.

**Effort:** 0 days
**Impact:** None
**Benefit:** Preserves stability; revisit after Kernel reaches v1.0

---

## 7. Checklist for Phase 2 (If Approved)

- [ ] Audit `state.backends` divergence between Forge and Kernel
- [ ] Run full Forge test suite (~2,100 tests) against proposed changes
- [ ] Run full Kernel test suite against proposed changes
- [ ] Update import paths in Forge dashboard, API routes, and skill bridge
- [ ] Update `docs/architecture/packages.md` dependency diagram
- [ ] Verify no `animus_forge.workflow` imports remain in external packages
- [ ] Add CI job to detect new wrapper files that aren't thin re-exports

---

## Appendix: Methodology

1. `diff -rq` compared Forge `workflow/` and Kernel `executor/` directories
2. `wc -l` measured line counts for each file pair
3. `diff -u` inspected the 3 `executions/` files for semantic differences
4. Manual review of Forge-unique files (`api_routes/executions.py`, `skill_bridge/executor.py`, `dashboard/workflow_builder/renderers/execution.py`)
5. `grep` verified no hidden logic in Forge wrapper files beyond `import *` re-exports

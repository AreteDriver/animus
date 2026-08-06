# Forge Regression Sweep — 2026-08-06

Post-loop-governor-integration sweep covering:

- `test_governor/*` — 116 tests, 5 skipped (all pass)
- `test_citizens`, `test_api_citizens`, `test_api_mission_scheduler`,
  `test_missions` — 67/68 pass; 1 fail
- `test_scheduler_lease`, `test_run_store` — 42/42 pass
- `test_scheduler_phase5::TestSchedulerMetrics` — 5/5 pass
- `test_scheduler_phase5::TestCheckpointPersistence` (sync) — 2/2 pass
- 5 mission-specific test files — all pass

## Pre-existing failures (NOT caused by 6b92c7d / governor integration)

### 1. `test_mission_delete_cascades_to_tasks` (`tests/test_missions.py:374`)

Test enables `PRAGMA foreign_keys=ON` inside a `with ledger._backend.transaction():`
block. Per SQLite docs (https://www.sqlite.org/pragma.html#pragma_foreign_keys),
the pragma is a no-op inside a transaction. The mission gets deleted, but the
orphan task remains because FK enforcement never actually turned on.

Introduced in 5d55146 (Phase 4 Mission Domain, 2026-07-26). Tracked as task #33.

### 2. Four budget tests — `animus_forge.budget.BudgetManager` vs `animus_kernel.budget.BudgetManager`

After the 203791d refactor (consolidate Forge execution primitives into Kernel
imports), the executor (`packages/forge/src/animus_forge/workflow/executor.py:_check_budget_exceeded`)
imports `BudgetStatus` from `animus_kernel.budget`, but tests instantiate
`BudgetManager` from `animus_forge.budget.manager`. The two `BudgetStatus` enums
are distinct — `mgr.status == BudgetStatus.EXCEEDED` evaluates `False`, and
the executor falls through to the `can_allocate` branch producing
"Token budget exceeded" instead of "Budget exceeded (effective-tokens)".

Failing tests (4 total):

- `test_budget_effective_tokens.py:271::test_executor_halts_on_effective_token_overspend`
- `test_budget_passthrough.py:141::test_daily_limit_blocks_when_exceeded`
  (and 2 others in the same class — all daily-budget tests patch
  `animus_forge.db.get_task_store` instead of `animus_kernel.db.get_task_store`)
- `test_budget_integration.py::TestDailyLimitWithPersistence::test_daily_limit_blocks_after_threshold`

Affected lines all have the pattern: forge-side test setup, kernel-side executor
import.

Tracked as task #34.

### 3. `test_scheduler_phase5::TestCheckpointPersistence::test_checkpoint_saved_on_completion`

Single hung test (no output within 45s). Calls `await scheduler.start()` and
`scheduler.run_once()` then `asyncio.sleep(3.0)`. Async worker-pool test
unrelated to governor integration. To be re-investigated with `--timeout`
flag once pytest-timeout dep is enabled in pyproject.toml.

## Cleanup required

These are real regressions from the kernel/forge split, but they pre-date
the governor integration and need their own fix PRs. Adding a "broken" skip
marker would be dishonest; instead each is tracked as a separate task with
a real fix scoped out.
